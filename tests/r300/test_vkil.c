/*
 * The Vulkan backend's ordered pixel interlock (R300_GLSL_FB_INTERLOCK),
 * run through ppc_mac_gpu_vulkan.c itself: random overlapping triangles
 * drawn as one draw, as a draw each in one render pass, and as a draw
 * each with a render pass boundary between them must leave VRAM (colour,
 * depth and stencil) and the Z-pass counter exactly as drawing them one
 * at a time, waiting for each, does.  That reference cannot race; a CPU
 * model of the blend checks the reference itself for plain RGBA8.
 *
 * Covered: blending (and separate alpha), ROP, the channel mask, the
 * alpha test killing some or all fragments, depth and two-sided stencil
 * with their fail ops, polygon offset, Z16 and Z24S8, every colour
 * format the interlock stores (rgba8, r8ui, r16ui, r32ui, rg32ui,
 * rgba32ui), a second render target, and pixels outside the scissor.
 *
 * Linux (Vulkan) only, against a QEMU build: ./run-vulkan.sh [builddir].
 * PPCGPU_VK_INTERLOCK=0 runs the same draws without the interlock, as a
 * diagnostic: differences are reported but are not failures.
 */
#include "qemu/osdep.h"
#include <math.h>
#include "../../hw/display/ppc_mac_gpu_vulkan.c"

#define VRAM_SIZE   (16u << 20)
#define PITCH       256u            /* pixels, every buffer */
#define RT_A        0x100000u
#define RT_B        0x200000u
#define RT_X        0x300000u       /* the other target: pass boundaries */
#define ZBUF        0x380000u
#define BUF_BYTES   0x80000u        /* PITCH * 16 bytes * ROWS, rounded up */
#define SC_X0       24u             /* scissor, inclusive; it sets the height */
#define SC_Y0       20u
#define SC_X1       235u
#define SC_Y1       215u
#define ROWS        (SC_Y1 + 1)
#define NPRIM       200
#define BIAS        1440u           /* SC_COORD_BIAS */

static uint8_t *vram;
static void *op;
static PPCMacGPURenderer *R;
static R300State st;
static unsigned long cpu_dirty[BITS_TO_LONGS(VRAM_SIZE / PPC_MAC_GPU_DIRTY_PAGE)];
static int fails, diffs;

#define CHECK(c, ...) do { if (!(c)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); \
    printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

/* The device's dirty log: what the test wrote into VRAM itself. */
static void dirty_fn(void *arg, unsigned long *bm, uint64_t npages)
{
    bitmap_or(bm, bm, cpu_dirty, npages);
    bitmap_zero(cpu_dirty, npages);
}

static void cpu_wrote(uint32_t lo, uint32_t len)
{
    uint32_t p0 = lo / PPC_MAC_GPU_DIRTY_PAGE;
    bitmap_set(cpu_dirty, p0, DIV_ROUND_UP(lo + len, PPC_MAC_GPU_DIRTY_PAGE) - p0);
}

static uint32_t rng;
static uint32_t rnd(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}
static float frnd(void) { return (rnd() >> 8) / 16777216.0f; }

typedef struct Case {
    const char *name;
    uint32_t cf, of, endian;        /* target A: COLORPITCH format and endian, US_OUT_FMT */
    uint32_t bcf, bof;              /* target B the same way (MRT), bcf 0 for none */
    uint32_t cblend, ablend, rop, chanmask, alpha_func;
    uint32_t zcntl, zfmt, zs, zref; /* ZB_CNTL (0: no depth buffer), ZB_FORMAT,
                                       ZSTENCILCNTL, STENCILREFMASK */
    bool poly;                      /* polygon offset */
    float amin, amax;               /* the triangles' alpha */
    bool model;                     /* check the reference with the CPU blend model */
} Case;

#define OVER    (1u | (38u << 16) | (39u << 24))     /* src.a, 1 - src.a, add */
#define SEPA    2u
#define ADD     (1u | (33u << 16) | (33u << 24))     /* 1, 1 */
#define MUL     (1u | (36u << 16) | (32u << 24))     /* dst colour, 0 */
#define ROPXOR  (4u | (6u << 8))
#define AGREATER(ref) ((1u << 11) | (4u << 8) | (ref))
#define SEL     0x1B00u                               /* C0..C3 = B, G, R, A */

static const Case cases[] = {
    { "over rgba8", 6, 1, 2, .cblend = OVER, .chanmask = 15, .amin = .25, .amax = .75,
      .model = true },
    { "over rgba8, no swap", 6, 0, 0, .cblend = OVER, .chanmask = 15, .amin = .25, .amax = .75,
      .model = true },
    { "separate alpha, mask, alpha test", 6, 1, 2, .cblend = OVER | SEPA, .ablend = ADD,
      .chanmask = 0xB, .alpha_func = AGREATER(0x60), .amin = 0, .amax = 1 },
    { "mul", 6, 1, 2, .cblend = MUL, .chanmask = 15, .amin = 0, .amax = 1 },
    { "rop xor", 6, 1, 2, .rop = ROPXOR, .chanmask = 15, .amin = 0, .amax = 1 },
    { "all killed", 6, 1, 2, .cblend = OVER, .chanmask = 15, .alpha_func = AGREATER(0xF0),
      .zcntl = 1 | 2 | 4, .zfmt = 2, .zs = 1 | (7u << 3) | (6u << 9), .zref = 0xFFFF00,
      .amin = 0, .amax = .9 },
    { "z24s8 less, stencil two-sided, offset", 6, 1, 2, .cblend = OVER, .chanmask = 15,
      .zcntl = 1 | 2 | 4 | 16, .zfmt = 2,
      .zs = 1 | (7u << 3) | (6u << 9) | (7u << 12) | (7u << 15) | (5u << 21) | (3u << 24),
      .zref = 0xFFFF00, .poly = true, .amin = .25, .amax = .75 },
    { "z24s8 stencil fail ops", 6, 1, 2, .cblend = OVER, .chanmask = 15,
      .zcntl = 1 | 2 | 4, .zfmt = 2,
      .zs = 5 | (1u << 3) | (5u << 6) | (3u << 9) | (4u << 12), .zref = 0xFF7F80,
      .amin = .25, .amax = .75 },
    { "z16 lequal, no blend", 6, 1, 2, .chanmask = 15, .zcntl = 2 | 4, .zfmt = 0, .zs = 2,
      .amin = 0, .amax = 1 },
    { "i8 over", 9, 0, 0, .cblend = OVER, .chanmask = 15, .amin = .25, .amax = .75 },
    { "565 over", 4, 0, 0, .cblend = OVER, .chanmask = 15, .amin = .25, .amax = .75 },
    { "1555 over", 3, 0, 1, .cblend = OVER, .chanmask = 15, .amin = .25, .amax = .75 },
    { "4444 over", 15, 0, 0, .cblend = OVER, .chanmask = 15, .amin = .25, .amax = .75 },
    { "uv88 xor", 13, 0, 0, .rop = ROPXOR, .chanmask = 15, .amin = 0, .amax = 1 },
    { "r32 (C_16) last wins", 6, 3, 0, .chanmask = 15, .amin = 0, .amax = 1 },
    { "rg16f last wins, z", 6, 17, 0, .chanmask = 15, .zcntl = 2 | 4, .zfmt = 2, .zs = 1,
      .amin = 0, .amax = 1 },
    { "rgba16 last wins", 10, 5, 0, .chanmask = 15, .amin = 0, .amax = 1 },
    { "rgba16f, z16 greater", 10, 18, 0, .chanmask = 15, .zcntl = 2 | 4, .zfmt = 0, .zs = 5,
      .amin = 0, .amax = 1 },
    { "rgba32f last wins", 7, 21, 0, .chanmask = 15, .amin = 0, .amax = 1 },
    { "mrt rgba8 + 565 over", 6, 1, 2, 4, 0, .cblend = OVER, .chanmask = 15,
      .amin = .25, .amax = .75 },
    { "mrt + z24s8", 6, 1, 2, 10, 18, .cblend = OVER, .chanmask = 15, .zcntl = 2 | 4,
      .zfmt = 2, .zs = 1, .amin = .25, .amax = .75 },
};

static uint32_t cbpp(uint32_t cf, uint32_t of)
{
    uint32_t b;
    r300_cb_view(cf, of, &b);
    return b;
}

static void load_state(void)
{
    FILE *f = fopen("qe_draw1_full.txt", "r");
    char line[256];
    int mode = 0;

    if (!f) {
        printf("run from tests/r300 (qe_draw1_full.txt)\n");
        exit(1);
    }
    r300_state_reset(&st);
    while (fgets(line, sizeof(line), f)) {
        unsigned a, v, i, d[4];
        float c[4];
        if (strstr(line, "PVS code")) { mode = 1; continue; }
        if (strstr(line, "PVS constants")) { mode = 2; continue; }
        if (mode == 0 && sscanf(line, " %x %x", &a, &v) == 2) r300_state_write(&st, a, v);
        if (mode == 1 && sscanf(line, " %u: %x %x %x %x", &i, &d[0], &d[1], &d[2], &d[3]) == 5)
            memcpy(&st.pvs_mem[i * 4], d, 16);
        if (mode == 2 && sscanf(line, " c%u %f %f %f %f", &i, &c[0], &c[1], &c[2], &c[3]) == 5)
            memcpy(&st.pvs_mem[(512 + i) * 4], c, 16);
    }
    fclose(f);
}

/*
 * The captured Quartz Extreme state draws window-space positions; its
 * colour (vertex dwords 4-7) reaches the program as t0.  The program:
 * out A = t0, and with mrt out B = t0 too.
 */
static void set_case(const Case *c)
{
    bool mrt = c->bcf != 0;

    r300_state_write(&st, 0x4104, 0);                   /* TX_ENABLE */
    r300_state_write(&st, 0x4600, 0);                   /* US_CONFIG */
    r300_state_write(&st, 0x4608, 0);                   /* US_CODE_OFFSET */
    r300_state_write(&st, 0x461C, (mrt ? 1u : 0u) << 6);
    for (unsigned i = 0; i < 1u + mrt; i++) {
        r300_state_write(&st, 0x46C0 + 4 * i, (7u << 26) | (i << 29));
        r300_state_write(&st, 0x48C0 + 4 * i, 0 | (21u << 7) | (20u << 14));
        r300_state_write(&st, 0x47C0 + 4 * i, (1u << 24) | (i << 25));
        r300_state_write(&st, 0x49C0 + 4 * i, 9 | (17u << 7) | (16u << 14));
    }
    r300_state_write(&st, 0x46A4, SEL | c->of);
    r300_state_write(&st, 0x46A8, mrt ? SEL | c->bof : 0x1B0F);
    r300_state_write(&st, 0x4E28, RT_A);
    r300_state_write(&st, 0x4E38, (c->cf << 21) | (c->endian << 19) | PITCH);
    r300_state_write(&st, 0x4E2C, RT_B);
    r300_state_write(&st, 0x4E3C, (c->bcf << 21) | PITCH);
    r300_state_write(&st, 0x4E04, c->cblend);
    r300_state_write(&st, 0x4E08, c->ablend);
    r300_state_write(&st, 0x4E18, c->rop);
    r300_state_write(&st, 0x4E0C, c->chanmask);
    r300_state_write(&st, 0x4BD4, c->alpha_func);
    r300_state_write(&st, 0x4F00, c->zcntl);
    r300_state_write(&st, 0x4F04, c->zs);
    r300_state_write(&st, 0x4F08, c->zref);
    r300_state_write(&st, 0x4F10, c->zfmt);
    r300_state_write(&st, 0x4F20, ZBUF);
    r300_state_write(&st, 0x4F24, PITCH);
    r300_state_write(&st, 0x42B4, c->poly ? 3 : 0);     /* SU_POLY_OFFSET_ENABLE */
    float ps = 2.0f, po = 64.0f;
    uint32_t psb, pob;
    memcpy(&psb, &ps, 4);
    memcpy(&pob, &po, 4);
    for (int i = 0; i < 2; i++) {
        r300_state_write(&st, 0x42A4 + 8 * i, psb);
        r300_state_write(&st, 0x42A8 + 8 * i, pob);
    }
    r300_state_write(&st, 0x42B8, 0);                   /* SU_CULL_MODE: none */
    r300_state_write(&st, 0x43E0, (SC_X0 + BIAS) | ((SC_Y0 + BIAS) << 13));
    r300_state_write(&st, 0x43E4, (SC_X1 + BIAS) | ((SC_Y1 + BIAS) << 13));
    /* Clip rectangles off: every pixel passes SC_CLIP_RULE. */
    r300_state_write(&st, 0x43D0, 0xFFFF);
    st.glsl_flags = R->r300_glsl_flags;
}

typedef struct Prim {
    float x[3], y[3], z[3];
    float c[4];
} Prim;

static Prim prims[NPRIM];

/* Triangles of all sizes over (and past) the scissor, flat colours. */
static void gen_prims(const Case *c)
{
    for (int i = 0; i < NPRIM; i++) {
        Prim *p = &prims[i];
        float cx = frnd() * PITCH, cy = frnd() * ROWS, s = 4 + frnd() * 90;
        for (int k = 0; k < 3; k++) {
            p->x[k] = cx + (frnd() - 0.5f) * 2 * s;
            p->y[k] = cy + (frnd() - 0.5f) * 2 * s;
            p->z[k] = (frnd() - 0.5f) * 1.6f;
        }
        for (int k = 0; k < 3; k++) {
            p->c[k] = frnd();
        }
        p->c[3] = c->amin + frnd() * (c->amax - c->amin);
    }
}

static bool no_read(void *o, uint32_t a, void *d, uint32_t l)
{
    return false;
}

/* One DRAW_IMMD_2 of triangles prims[i0, i1). */
static void draw(int i0, int i1, bool count)
{
    static uint32_t d[1 + 36 * NPRIM];
    uint32_t n = 3 * (i1 - i0);
    R300Arrays arr = { 0 };
    R300DrawPacket p;
    const char *err;

    d[0] = (n << 16) | (3u << 4) | 4;                   /* triangle list */
    for (int i = i0; i < i1; i++) {
        for (int k = 0; k < 3; k++) {
            const Prim *pr = &prims[i];
            float v[12] = { pr->x[k], pr->y[k], pr->z[k], 1,
                            pr->c[0], pr->c[1], pr->c[2], pr->c[3], 0, 0, 0, 1 };
            memcpy(&d[1 + 12 * (3 * (i - i0) + k)], v, sizeof(v));
        }
    }
    if (!r300_draw_build(&st, &arr, 0x35, d, 1 + 12 * n, no_read, NULL, &p, &err)) {
        CHECK(0, "draw build: %s", err);
        return;
    }
    CHECK(!p.warn, "draw warnings %x", p.warn);
    p.uniforms.zpass_count = count;
    CHECK(R->draw_r300(op, vram, VRAM_SIZE, &p) == 0, "draw_r300 failed");
    r300_draw_free(&p);
}

/* A small triangle into RT_X with no depth buffer: the next draw into
 * RT_A starts a new render pass. */
static void draw_elsewhere(const Case *c)
{
    Prim keep = prims[0];
    prims[0] = (Prim){ { 0, 8, 0 }, { 0, 0, 8 }, { 0, 0, 0 }, { 1, 1, 1, 1 } };
    r300_state_write(&st, 0x4E28, RT_X);
    r300_state_write(&st, 0x4E38, (6u << 21) | PITCH);
    r300_state_write(&st, 0x46A4, SEL | 1);
    r300_state_write(&st, 0x46A8, 0x1B0F);
    r300_state_write(&st, 0x461C, 0);
    r300_state_write(&st, 0x4F00, 0);
    draw(0, 1, false);
    prims[0] = keep;
    set_case(c);
}

/* The buffers before every run: the same noise, written by the "CPU". */
static void fill_buffers(uint32_t seed)
{
    uint32_t save = rng;
    rng = seed;
    for (uint32_t a = RT_A; a < ZBUF + BUF_BYTES; a += 4) {
        uint32_t v = rnd();
        memcpy(vram + a, &v, 4);
    }
    rng = save;
    cpu_wrote(RT_A, ZBUF + BUF_BYTES - RT_A);
}

typedef struct Snap {
    uint8_t a[BUF_BYTES], b[BUF_BYTES], z[BUF_BYTES];
    uint32_t zpass;
} Snap;

static void snap(Snap *s)
{
    R->flush_r200(op);
    memcpy(s->a, vram + RT_A, BUF_BYTES);
    memcpy(s->b, vram + RT_B, BUF_BYTES);
    memcpy(s->z, vram + ZBUF, BUF_BYTES);
    s->zpass = R->zpass_r300(op, true, 0);
}

/* Pixels (of bpp bytes) that differ; prints the first few. */
static int compare(const char *what, const char *mode, const uint8_t *got,
                   const uint8_t *want, uint32_t bpp)
{
    int n = 0;
    for (uint32_t y = 0; y < ROWS; y++) {
        for (uint32_t x = 0; x < PITCH; x++) {
            uint32_t o = (y * PITCH + x) * bpp;
            if (memcmp(got + o, want + o, bpp)) {
                if (n < 3) {
                    printf("    %s %s: pixel %u,%u differs:", mode, what, x, y);
                    for (uint32_t i = 0; i < bpp; i++) printf(" %02x", got[o + i]);
                    printf(" want");
                    for (uint32_t i = 0; i < bpp; i++) printf(" %02x", want[o + i]);
                    printf("\n");
                }
                n++;
            }
        }
    }
    return n;
}

/*
 * The reference against the blend done on the CPU, in order, rounding
 * after every step: RGBA8 over, no depth.  Pixels a triangle edge passes
 * near are skipped (the rasteriser's rules decide those).
 */
static void check_model(const Case *c, const uint8_t *init, const uint8_t *ref)
{
    int checked = 0, bad = 0;
    for (uint32_t y = SC_Y0; y <= SC_Y1; y++) {
        for (uint32_t x = SC_X0; x <= SC_X1; x++) {
            float px = x + 0.5f, py = y + 0.5f;
            const uint8_t *s = init + (y * PITCH + x) * 4;
            /* bytes A R G B with the swap, else B G R A */
            int ia = c->endian == 2 ? 0 : 3, ir = c->endian == 2 ? 1 : 2;
            int ig = c->endian == 2 ? 2 : 1, ib = c->endian == 2 ? 3 : 0;
            float d[4] = { s[ir] / 255.0f, s[ig] / 255.0f, s[ib] / 255.0f, s[ia] / 255.0f };
            bool skip = false;
            for (int i = 0; i < NPRIM && !skip; i++) {
                const Prim *p = &prims[i];
                float e[3];
                int pos = 0, neg = 0;
                for (int k = 0; k < 3; k++) {
                    int l = (k + 1) % 3;
                    float ex = p->x[l] - p->x[k], ey = p->y[l] - p->y[k];
                    float len = sqrtf(ex * ex + ey * ey);
                    e[k] = (ex * (py - p->y[k]) - ey * (px - p->x[k])) / (len ? len : 1);
                    if (fabsf(e[k]) < 0.05f) skip = true;
                    pos += e[k] > 0;
                    neg += e[k] < 0;
                }
                if (skip || (pos != 3 && neg != 3)) {
                    continue;
                }
                float a = p->c[3];
                for (int k = 0; k < 4; k++) {
                    float r = p->c[k] * a + d[k] * (1 - a);
                    r = fminf(fmaxf(r, 0), 1);
                    d[k] = rintf(r * 255) / 255;
                }
            }
            if (skip) {
                continue;
            }
            const uint8_t *g = ref + (y * PITCH + x) * 4;
            int gi[4] = { ir, ig, ib, ia };
            checked++;
            for (int k = 0; k < 4; k++) {
                if (abs((int)g[gi[k]] - (int)rintf(d[k] * 255)) > 3) {
                    if (bad < 3) {
                        printf("    model: pixel %u,%u channel %d: %d, want %d\n", x, y, k,
                               g[gi[k]], (int)rintf(d[k] * 255));
                    }
                    bad++;
                    break;
                }
            }
        }
    }
    printf("    CPU model: %d pixels checked, %d off by more than 3\n", checked, bad);
    CHECK(checked > 10000 && !bad, "%s: the reference disagrees with the CPU blend", c->name);
}

/* Pixels outside the scissor must keep what the CPU wrote. */
static void check_outside(const Case *c, const uint8_t *init, const uint8_t *got, uint32_t bpp,
                          const char *what)
{
    int n = 0;
    for (uint32_t y = 0; y < ROWS; y++) {
        for (uint32_t x = 0; x < PITCH; x++) {
            if (x >= SC_X0 && x <= SC_X1 && y >= SC_Y0) {
                continue;
            }
            uint32_t o = (y * PITCH + x) * bpp;
            n += memcmp(got + o, init + o, bpp) != 0;
        }
    }
    CHECK(!n, "%s %s: %d pixels outside the scissor changed", c->name, what, n);
}

static Snap ref, got, init;

static void run_case(const Case *c, unsigned idx)
{
    uint32_t seed = 0x9E3779B9u * (idx + 1);
    uint32_t abpp = cbpp(c->cf, c->of), bbpp = c->bcf ? cbpp(c->bcf, c->bof) : 0;
    uint32_t zbpp = c->zfmt == 2 ? 4 : 2;
    static const char *modes[] = { "one draw", "a draw each", "a pass each" };
    int case_diffs = 0;

    printf("%s\n", c->name);
    rng = seed;
    gen_prims(c);
    set_case(c);

    fill_buffers(seed);
    snap(&init);

    /* The reference: one triangle at a time, each finished before the next. */
    fill_buffers(seed);
    R->zpass_r300(op, true, 0);
    for (int i = 0; i < NPRIM; i++) {
        draw(i, i + 1, true);
        R->flush_r200(op);
    }
    snap(&ref);
    check_outside(c, init.a, ref.a, abpp, "colour");
    if (c->bcf) {
        check_outside(c, init.b, ref.b, bbpp, "target B");
    }
    if (c->zcntl) {
        check_outside(c, init.z, ref.z, zbpp, "depth");
    }
    if (c->alpha_func == AGREATER(0xF0)) {
        CHECK(!memcmp(ref.a, init.a, BUF_BYTES) && !memcmp(ref.z, init.z, BUF_BYTES) &&
              ref.zpass == 0, "%s: killed fragments changed something (zpass %u)",
              c->name, ref.zpass);
    } else {
        CHECK(ref.zpass > 0, "%s: no fragment counted", c->name);
        CHECK(memcmp(ref.a, init.a, BUF_BYTES), "%s: nothing drawn", c->name);
    }
    if (c->model) {
        check_model(c, init.a, ref.a);
    }

    for (int m = 0; m < 3; m++) {
        fill_buffers(seed);
        R->zpass_r300(op, true, 0);
        if (m == 0) {
            draw(0, NPRIM, true);
        } else {
            for (int i = 0; i < NPRIM; i++) {
                draw(i, i + 1, true);
                if (m == 2) {
                    draw_elsewhere(c);
                }
            }
        }
        snap(&got);
        int n = compare("colour", modes[m], got.a, ref.a, abpp);
        if (c->bcf) {
            n += compare("target B", modes[m], got.b, ref.b, bbpp);
        }
        if (c->zcntl) {
            n += compare("depth/stencil", modes[m], got.z, ref.z, zbpp);
        }
        if (got.zpass != ref.zpass) {
            printf("    %s: Z-pass count %u, want %u\n", modes[m], got.zpass, ref.zpass);
            n++;
        }
        printf("  %-12s %s (%d differences)\n", modes[m], n ? "DIFFERS" : "same", n);
        case_diffs += n;
    }
    if (V.interlock) {
        CHECK(!case_diffs, "%s: not drawn in order", c->name);
    }
    diffs += case_diffs;
}

int main(int argc, char **argv)
{
    vram = ppc_mac_gpu_vulkan_alloc_vram(VRAM_SIZE, &op);
    if (!vram) {
        printf("no Vulkan: %s\n", ppc_mac_gpu_vulkan_error());
        return 77;
    }
    R = ppc_mac_gpu_renderer_vulkan();
    if (!R->init(vram, VRAM_SIZE)) {
        printf("renderer init failed\n");
        return 1;
    }
    R->set_dirty_source(op, dirty_fn, NULL);
    printf("%s, interlock %s\n", V.props.deviceName, V.interlock ? "on" : "OFF");
    load_state();
    for (unsigned i = 0; i < ARRAY_SIZE(cases); i++) {
        if (argc > 1 && !strstr(cases[i].name, argv[1])) {
            continue;
        }
        run_case(&cases[i], i);
    }
    if (!V.interlock) {
        printf("interlock off (diagnostic): %d differences in all\n", diffs);
    }
    if (fails) {
        printf("%d FAILED\n", fails);
        return 1;
    }
    printf("interlock test passed\n");
    return 0;
}
