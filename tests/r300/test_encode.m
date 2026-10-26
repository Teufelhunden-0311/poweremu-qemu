/*
 * The production R300 encoder, hw/display/ppc_mac_gpu_metal.m (included
 * whole, for its statics, with QEMU's headers from encshim/), driven by a
 * fixed sequence of draws built from the captured Quartz Extreme state
 * through r300_draw_build, as the device builds them.
 *
 * Each step ends with a flush and prints one line: hashes of the colour
 * buffers and the depth buffer, and the Z-pass counter.  run.sh runs it in
 * several configurations (render queue, PPCGPU_R300_EXTENT, the setter
 * cache PPCGPU_R300_ENC_REUSE) and requires the same lines from all of
 * them.  With the setter cache on it also checks the setter counts
 * (PPCGPU_R300_ENC_STATS): repeated state is skipped, every change and
 * every new encoder is set.
 *
 * The steps follow the encoder-reuse checklist (Codex, 2026-10-24, A5):
 * each binding family changed on its own; R300 -> R200 -> R300; a fence
 * commit; another colour buffer; a hazard flush; GPU -> CPU -> GPU vertex
 * programs; culled triangles and polygon-mode lines; AA samples; more than
 * one arena buffer in a pass; a texture rewritten at the same address.
 * Not covered: view-pressure flushes, and the one-off VSU buffer of a
 * failed arena allocation (neither is changed by the cache).
 *
 * AI-generated (Claude).  This work is licensed under the terms of the
 * GNU GPL, version 2 or later.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

/* The renderer keeps a pipeline archive in ~/Library/Caches: the test's
 * goes to its own directory (ENC_TEST_DIR) instead. */
static NSArray<NSString *> *test_search_path(NSSearchPathDirectory d,
                                             NSSearchPathDomainMask m, BOOL e);
#define NSSearchPathForDirectoriesInDomains test_search_path
#ifndef ENC_METAL_M    /* another copy, e.g. a deliberately broken one */
#define ENC_METAL_M "../../hw/display/ppc_mac_gpu_metal.m"
#endif
#include ENC_METAL_M
#undef NSSearchPathForDirectoriesInDomains

FrameTracker *g_frame_tracker;     /* the device's (hw/display/ppc_mac_gpu.c); unused here */
bool encshim_bql = true;           /* draws come from the device with the BQL held */

static NSArray<NSString *> *test_search_path(NSSearchPathDirectory d,
                                             NSSearchPathDomainMask m, BOOL e)
{
    const char *dir = getenv("ENC_TEST_DIR");
    return @[ @(dir ? dir : "/tmp") ];
}

#define VRAM_SIZE   (32u << 20)
#define RTW         832u            /* the capture's colour buffer: 832 x 600 */
#define RTH         600u
#define RT1         0x1EE000u       /* the capture's */
#define RT2         0x800000u
#define RT3         0x1000000u      /* two samples per row */
#define ZBUF        0xC00000u
#define TEX0        0x3DC000u       /* the capture's: 256 x 256, 32 bpp */
#define TEX1        0x500000u

/* Registers (as r300_draw.c names them) */
#define H_SU_CULL_MODE        0x42B8
#define H_SC_SCISSORS_BR      0x43E4
#define H_TX_FILTER0_0        0x4400
#define H_TX_OFFSET_0         0x4540
#define H_RB3D_COLOROFFSET0   0x4E28
#define H_ZB_CNTL             0x4F00
#define H_ZB_ZSTENCILCNTL     0x4F04
#define H_ZB_FORMAT           0x4F10
#define H_ZB_DEPTHOFFSET      0x4F20
#define H_ZB_DEPTHPITCH       0x4F24

static PPCMacGPURenderer *R;
static void *ST;
static uint8_t *V;
static R300State S;
static bool reuse;
static int fails, step;
static uint32_t br0;            /* the capture's SC_SCISSORS_BR (bottom row 599) */

#define CHECK(c, ...) do { if (!(c)) { printf("FAIL step %d: ", step); \
    printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

static bool no_read(void *o, uint32_t a, void *d, uint32_t l) { return false; }

static void load_capture(void)
{
    FILE *f = fopen("qe_draw1_full.txt", "r");
    char line[256];
    int mode = 0;

    if (!f) {
        printf("qe_draw1_full.txt: %s\n", strerror(errno));
        exit(1);
    }
    r300_state_reset(&S);
    while (fgets(line, sizeof(line), f)) {
        unsigned a, v, i, d[4];
        float c[4];
        if (strstr(line, "PVS code")) { mode = 1; continue; }
        if (strstr(line, "PVS constants")) { mode = 2; continue; }
        if (mode == 0 && sscanf(line, " %x %x", &a, &v) == 2) r300_state_write(&S, a, v);
        if (mode == 1 && sscanf(line, " %u: %x %x %x %x", &i, &d[0], &d[1], &d[2], &d[3]) == 5)
            memcpy(&S.pvs_mem[i * 4], d, 16);
        if (mode == 2 && sscanf(line, " c%u %f %f %f %f", &i, &c[0], &c[1], &c[2], &c[3]) == 5)
            memcpy(&S.pvs_mem[(512 + i) * 4], c, 16);
    }
    fclose(f);
    S.pvs_gen++;
    br0 = r300_reg(&S, H_SC_SCISSORS_BR);
}

/* Two textures with different, position-dependent texels. */
static void fill_textures(uint32_t salt)
{
    for (uint32_t y = 0; y < 256; y++) {
        for (uint32_t x = 0; x < 256; x++) {
            uint32_t a = 0xFF000000u | ((x * 7 + salt) & 0xFF) << 16 |
                         ((y * 13) & 0xFF) << 8 | ((x ^ y) & 0xFF);
            uint32_t b = 0xFF000000u | ((y * 5) & 0xFF) << 16 |
                         ((x * 3 + y + salt) & 0xFF) << 8 | ((x * y) & 0xFF);
            memcpy(V + TEX0 + y * 1024 + x * 4, &a, 4);
            memcpy(V + TEX1 + y * 1024 + x * 4, &b, 4);
        }
    }
}

/* One draw: a quad, and the registers that differ from the capture. */
typedef struct Draw {
    bool gpu;                   /* vertex program on the GPU */
    float x, y, size;           /* window pixels */
    float tc;                   /* texcoord extent in texels (256: 1:1) */
    uint32_t rt;                /* colour buffer; 0: RT1 */
    bool depth;                 /* ZBUF, Z test LESS, write */
    uint32_t cull;              /* SU_CULL_MODE */
    uint32_t tex;               /* TX_OFFSET_0; 0: TEX0 */
    bool linear;                /* TX_FILTER0_0 magnification/minification */
    float c0x;                  /* PVS constant c0.x; 0: the capture's */
    uint32_t h;                 /* the packet's height (the scissor's bottom edge);
                                   0: the capture's 600 */
} Draw;

static bool build(const Draw *d, R300DrawPacket *p)
{
    static const float col[4][4] = {
        { 0, 0, 0, 1 }, { 1, 0, 0, 1 }, { 1, 1, 0, 1 }, { 0, 1, 0, 1 },
    };
    float x0 = d->x, y0 = d->y, x1 = d->x + d->size, y1 = d->y + d->size;
    float tc = d->tc ? d->tc : 256;
    float pos[4][2] = { { x0, y0 }, { x1, y0 }, { x1, y1 }, { x0, y1 } };
    float st[4][2] = { { 0, 0 }, { tc, 0 }, { tc, tc }, { 0, tc } };
    uint32_t pkt3[49] = { 0x0004003D };
    static float c0x_capture;
    R300Arrays arr = { 0 };
    const char *err;

    if (!c0x_capture) {
        memcpy(&c0x_capture, &S.pvs_mem[512 * 4], 4);
    }
    for (int i = 0; i < 4; i++) {
        float v[12] = { pos[i][0], pos[i][1], 0, 1,
                        col[i][0], col[i][1], col[i][2], col[i][3],
                        st[i][0], st[i][1], 0, 1 };
        memcpy(&pkt3[1 + 12 * i], v, sizeof(v));
    }
    S.glsl_flags = d->gpu ? R300_GLSL_GPU_VS : 0;
    r300_state_write(&S, H_RB3D_COLOROFFSET0, d->rt ? d->rt : RT1);
    r300_state_write(&S, H_ZB_CNTL, d->depth ? 0x6 : 0);        /* Z_ENABLE, Z_WRITE */
    r300_state_write(&S, H_ZB_ZSTENCILCNTL, 1);                 /* LESS */
    r300_state_write(&S, H_ZB_FORMAT, 2);                       /* Z24S8 */
    r300_state_write(&S, H_ZB_DEPTHOFFSET, ZBUF);
    r300_state_write(&S, H_ZB_DEPTHPITCH, RTW);
    r300_state_write(&S, H_SU_CULL_MODE, d->cull);
    r300_state_write(&S, H_TX_OFFSET_0, d->tex ? d->tex : TEX0);
    /* MAG_FILTER / MIN_FILTER: 1 point (the capture), 2 linear */
    r300_state_write(&S, H_TX_FILTER0_0, d->linear ? 0x1412 : 0xA12);
    /* SC_SCISSORS_BR: x in bits 0-12, y in 13-25, both biased by 1440 */
    r300_state_write(&S, H_SC_SCISSORS_BR,
                     d->h ? ((1440 + d->h - 1) << 13) | (br0 & 0x1FFF) : br0);
    float c0x = d->c0x ? d->c0x : c0x_capture;
    if (memcmp(&S.pvs_mem[512 * 4], &c0x, 4)) {
        memcpy(&S.pvs_mem[512 * 4], &c0x, 4);
        S.pvs_gen++;
    }
    if (!r300_draw_build(&S, &arr, 0x35, pkt3, 49, no_read, NULL, p, &err)) {
        CHECK(false, "build: %s", err);
        return false;
    }
    return true;
}

static void submit(R300DrawPacket *p)
{
    int r = R->draw_r300(ST, V, VRAM_SIZE, p);
    CHECK(r == 0, "draw_r300 returned %d", r);
    r300_draw_free(p);
}

static void draw(const Draw *d)
{
    R300DrawPacket p;
    if (build(d, &p)) {
        submit(&p);
    }
}

/* Everything queued so far encoded (the encoder stays open). */
static void drained(void)
{
    if (rqueue_enabled()) {
        rqueue_drain();
    }
}

/* Setter counts since snap(). */
static uint64_t m0[R300R_N], s0[R300R_N];
static void snap(void)
{
    drained();
    memcpy(m0, g_r300_reuse_made, sizeof(m0));
    memcpy(s0, g_r300_reuse_skipped, sizeof(s0));
}
static uint64_t made(int f)    { drained(); return g_r300_reuse_made[f] - m0[f]; }
static uint64_t skipped(int f) { drained(); return g_r300_reuse_skipped[f] - s0[f]; }

/* Render passes opened since snap_passes(). */
static uint64_t p0;
static void snap_passes(void) { drained(); p0 = g_r200_stat_passes; }
static uint64_t passes(void)  { drained(); return g_r200_stat_passes - p0; }

/* With the cache on, a draw on the same encoder that changes only family
 * f (or nothing, f < 0) sets f and skips everything else. */
static void expect_only(int f, const char *what)
{
    if (!reuse) {
        return;
    }
    for (int k = 0; k < R300R_N; k++) {
        if (k == R300R_TEX || k == R300R_SMP) {
            continue;           /* per unit: checked by the caller */
        }
        if (k == f) {
            CHECK(made(k) == 1, "%s: family %d set %llu times", what, k,
                  (unsigned long long)made(k));
        } else {
            CHECK(made(k) == 0, "%s: family %d set again (%llu)", what, k,
                  (unsigned long long)made(k));
        }
    }
}

/* With the cache on, the first draw of a new encoder sets everything. */
static void expect_fresh(const char *what)
{
    if (!reuse) {
        return;
    }
    for (int k = 0; k < R300R_N; k++) {
        CHECK(skipped(k) == 0, "%s: family %d skipped on a new encoder", what, k);
    }
    CHECK(made(R300R_PIPE) == 1 && made(R300R_TEX) == R300_NUM_TEX_UNITS &&
          made(R300R_SMP) == R300_NUM_TEX_UNITS && made(R300R_ZPASS) == 1 &&
          made(R300R_FSU) == 1 && made(R300R_RECT) == 1,
          "%s: not everything set on a new encoder", what);
}

static uint64_t fnv(const uint8_t *p, size_t n, uint64_t h)
{
    for (size_t i = 0; i < n; i++) {
        h = (h ^ p[i]) * 0x100000001B3ull;
    }
    return h;
}

/* Flush, and print what the step left in memory. */
static void end_step(const char *name)
{
    R->flush_r200(ST);
    uint32_t zp = R->zpass_r300(ST, false, 0);
    uint64_t gf = R->gpu_failures(ST);
    CHECK(gf == 0, "%llu GPU command buffer failure(s)", (unsigned long long)gf);
    uint64_t h1 = fnv(V + RT1, (size_t)RTW * 4 * RTH, 0xCBF29CE484222325ull);
    uint64_t h2 = fnv(V + RT2, (size_t)RTW * 4 * RTH, 0xCBF29CE484222325ull);
    uint64_t h3 = fnv(V + RT3, (size_t)RTW * 4 * RTH * 2, 0xCBF29CE484222325ull);
    uint64_t hz = fnv(V + ZBUF, (size_t)RTW * 4 * RTH, 0xCBF29CE484222325ull);
    printf("step %2d %-22s rt1 %016llx rt2 %016llx rt3 %016llx z %016llx zpass %u\n",
           step, name, (unsigned long long)h1, (unsigned long long)h2,
           (unsigned long long)h3, (unsigned long long)hz, zp);
    step++;
}

static void clear_targets(void)
{
    memset(V + RT1, 0x20, (size_t)RTW * 4 * RTH);
    memset(V + RT2, 0x40, (size_t)RTW * 4 * RTH);
    memset(V + RT3, 0x60, (size_t)RTW * 4 * RTH * 2);
    memset(V + ZBUF, 0xFF, (size_t)RTW * 4 * RTH);     /* far */
    R->zpass_r300(ST, true, 0);
}

/* An R200 draw into RT2: ends any R300 pass. */
static void r200_quad(float x, float y)
{
    static R200Vertex v[4];
    static const uint32_t idx[6] = { 0, 1, 2, 0, 2, 3 };
    R200DrawPacket q;

    memset(&q, 0, sizeof(q));
    for (int i = 0; i < 4; i++) {
        memset(&v[i], 0, sizeof(v[i]));
        v[i].pos[0] = x + (i == 1 || i == 2 ? 64 : 0);
        v[i].pos[1] = y + (i >= 2 ? 64 : 0);
        v[i].pos[3] = 1;
        v[i].color[0] = 0.25f * i;
        v[i].color[1] = 0.5f;
        v[i].color[3] = 1;
    }
    q.rt_offset = RT2;
    q.rt_pitch = RTW;
    q.rt_width = RTW;
    q.rt_height = RTH;
    q.rb3d_cntl = 6u << 10;         /* ARGB8888 */
    q.scissor[2] = RTW;
    q.scissor[3] = RTH;
    q.plane_mask = 0xFFFFFFFFu;
    q.verts = v;
    q.num_verts = 4;
    q.indices = idx;
    q.num_indices = 6;
    int r = R->draw_r200(ST, V, VRAM_SIZE, &q);
    CHECK(r == 0, "draw_r200 returned %d", r);
}

static void fence_done(void *arg, uint32_t seq) { }

int main(void)
{
    @autoreleasepool {
        const char *e = getenv("PPCGPU_R300_ENC_REUSE");
        void *va;
        reuse = e && e[0] == '1';
        setenv("PPCGPU_R300_ENC_STATS", "1", 1);
        load_capture();
        V = ppc_mac_gpu_metal_alloc_vram(VRAM_SIZE, &va);
        R = ppc_mac_gpu_renderer_metal();
        ST = V ? R->init(V, VRAM_SIZE) : NULL;
        if (!ST) {
            printf("no Metal renderer\n");
            return 1;
        }
        printf("render queue %s, reuse %s, extent %s\n", rqueue_enabled() ? "on" : "off",
               reuse ? "on" : "off", r300_extent_on() ? "on" : "off");
        fill_textures(0);
        clear_targets();

        /* 1. Repeated state, then each family on its own (one pass). */
        Draw g = { .gpu = true, .x = 16, .y = 16, .size = 128 };
        draw(&g);
        snap();
        g.x = 160;
        draw(&g);                               /* same state, new geometry */
        expect_only(-1, "repeat");
        if (reuse) {
            CHECK(skipped(R300R_PIPE) == 1 && skipped(R300R_TEX) == R300_NUM_TEX_UNITS &&
                  skipped(R300R_SMP) == R300_NUM_TEX_UNITS && skipped(R300R_VSU) == 1 &&
                  skipped(R300R_FSU) == 1 && skipped(R300R_RECT) == 1,
                  "repeat: nothing skipped");
        }
        snap();
        g.x = 304;
        g.tex = TEX1;
        draw(&g);                               /* one texture */
        expect_only(-1, "texture");
        if (reuse) {
            CHECK(made(R300R_TEX) == 1 && made(R300R_SMP) == 0, "texture: tex %llu smp %llu",
                  (unsigned long long)made(R300R_TEX), (unsigned long long)made(R300R_SMP));
        }
        snap();
        g.x = 448;
        g.size = 256;
        g.linear = true;
        draw(&g);                               /* its sampler (magnified 2x: visible) */
        expect_only(-1, "sampler");
        if (reuse) {
            CHECK(made(R300R_SMP) == 1 && made(R300R_TEX) == 0, "sampler: smp %llu tex %llu",
                  (unsigned long long)made(R300R_SMP), (unsigned long long)made(R300R_TEX));
        }
        /* Unit 1 (which this shader does not sample): a texture, then
         * another one; each must be set (counts only, no pixels). */
        for (int k = 0; k < 2; k++) {
            R300DrawPacket p;
            g.x = 16 + 140 * k;
            g.y = 450;
            g.size = 64;
            if (build(&g, &p)) {
                p.tex[1] = p.tex[0];
                p.tex[1].gpu_addr = k ? TEX1 : TEX0;
                memcpy(p.uniforms.tex_info[1], p.uniforms.tex_info[0], sizeof(p.uniforms.tex_info[1]));
                snap();
                submit(&p);
                if (reuse) {
                    CHECK(made(R300R_TEX) == 1, "unit 1 texture %d: set %llu times", k,
                          (unsigned long long)made(R300R_TEX));
                }
            }
        }
        snap();
        g = (Draw){ .gpu = true, .x = 16, .y = 300, .size = 128, .c0x = 0.004f };
        draw(&g);                               /* a VS constant: buffer 2 only */
        if (reuse) {
            CHECK(made(R300R_VSU) == 1 && made(R300R_PIPE) == 0, "vsu: vsu %llu pipe %llu",
                  (unsigned long long)made(R300R_VSU), (unsigned long long)made(R300R_PIPE));
        }
        end_step("families-a");

        /* 2. Uniforms, scissor, pipeline, winding and cull (one pass). */
        g = (Draw){ .gpu = true, .x = 16, .y = 16, .size = 128 };
        draw(&g);
        snap();
        {
            R300DrawPacket p;
            g.x = 160;
            if (build(&g, &p)) {
                uint32_t t = p.uniforms.out_sel[1];     /* swap R and B on output */
                p.uniforms.out_sel[1] = p.uniforms.out_sel[3];
                p.uniforms.out_sel[3] = t;
                submit(&p);
            }
            expect_only(R300R_FSU, "fs uniforms");
            snap();
            g.x = 304;
            if (build(&g, &p)) {
                uint32_t t = p.uniforms.out_sel[1];     /* the uniforms as before */
                p.uniforms.out_sel[1] = p.uniforms.out_sel[3];
                p.uniforms.out_sel[3] = t;
                p.scissor[0] = 320;                     /* clips the quad's left part */
                submit(&p);
            }
            expect_only(R300R_RECT, "scissor");
        }
        snap();
        g = (Draw){ .gpu = false, .x = 448, .y = 16, .size = 128 };
        draw(&g);                               /* CPU vertex program: another pipeline */
        if (reuse) {
            CHECK(made(R300R_PIPE) == 1, "cpu vs: pipeline not set");
        }
        snap();
        g = (Draw){ .gpu = true, .x = 16, .y = 300, .size = 128, .cull = R300_CULL_BACK };
        draw(&g);                               /* back faces culled */
        if (reuse) {
            CHECK(made(R300R_PIPE) == 1 && made(R300R_VSU) == 1 && made(R300R_CULL) == 1,
                  "gpu again: pipe %llu vsu %llu cull %llu",
                  (unsigned long long)made(R300R_PIPE), (unsigned long long)made(R300R_VSU),
                  (unsigned long long)made(R300R_CULL));
        }
        snap();
        g.x = 160;
        g.cull = R300_CULL_BACK | R300_FACE_CW;
        draw(&g);                               /* the winding flips which faces cull */
        expect_only(R300R_WIND, "winding");
        snap();
        g.x = 304;
        g.cull = R300_CULL_FRONT | R300_FACE_CW;
        draw(&g);                               /* front culled instead */
        expect_only(R300R_CULL, "cull");
        end_step("families-b");

        /* 3. New encoders: R200 in between, fence commit, another colour
         * buffer, a hazard flush; each first draw sets everything. */
        g = (Draw){ .gpu = true, .x = 600, .y = 16, .size = 64 };
        draw(&g);
        r200_quad(16, 16);
        drained();
        CHECK(!g_r300_enc_mine, "r200 draw did not take the pass");
        snap();
        g.y = 100;
        draw(&g);
        expect_fresh("after r200");
        R->submit_r200(ST, fence_done, NULL);
        snap();
        g.y = 184;
        draw(&g);
        expect_fresh("after fence");
        snap();
        g.rt = RT2;                             /* (the R200 quad wrote it: also a flush) */
        draw(&g);
        expect_fresh("other colour buffer");
        snap();
        g.tex = RT2;                            /* reads the buffer this pass writes */
        g.y = 268;
        draw(&g);                               /* same attachments: the hazard flushes */
        expect_fresh("hazard flush");
        end_step("encoders");

        /* 4. GPU -> CPU -> GPU, constants unchanged and changed. */
        g = (Draw){ .gpu = true, .x = 16, .y = 16, .size = 96 };
        draw(&g);
        g.gpu = false;
        g.x = 120;
        draw(&g);
        snap();
        g.gpu = true;
        g.x = 224;
        draw(&g);
        if (reuse) {
            CHECK(made(R300R_VSU) == 1, "gpu after cpu: vsu not set again");
        }
        g.gpu = false;
        g.x = 328;
        g.c0x = 0.003f;
        draw(&g);
        g.gpu = true;
        g.x = 432;
        draw(&g);
        end_step("gpu-cpu-gpu");

        /* 5. Polygon-mode edges (lines, never culled) between culled
         * triangles: the cull mode has to come back after the lines. */
        {
            R300DrawPacket p;
            g = (Draw){ .gpu = false, .x = 16, .y = 16, .size = 128, .cull = R300_CULL_BACK };
            if (build(&g, &p)) {
                uint32_t n = p.num_verts;
                p.verts = realloc(p.verts, (n + 2) * sizeof(R300Vertex));
                p.verts[n] = p.verts[0];
                p.verts[n + 1] = p.verts[2];
                p.num_line_verts = 2;
                submit(&p);
            }
            snap();
            g = (Draw){ .gpu = true, .x = 160, .y = 16, .size = 128,
                        .cull = R300_CULL_BACK | R300_FACE_CW };
            draw(&g);                           /* culled triangles again */
            if (reuse) {
                CHECK(made(R300R_CULL) == 1, "after lines: cull not restored");
            }
            g.x = 304;
            g.cull = R300_CULL_BACK;            /* the other winding: one of the two culls */
            draw(&g);
        }
        end_step("lines-cull");

        /* 6. Two samples per pixel, then back to one. */
        {
            R300DrawPacket p;
            g = (Draw){ .gpu = true, .x = 16, .y = 16, .size = 128, .rt = RT3 };
            if (build(&g, &p)) {
                p.aa_samples = 2;
                p.aa_pos[0][0] = -0.25f;
                p.aa_pos[0][1] = -0.25f;
                p.aa_pos[1][0] = 0.25f;
                p.aa_pos[1][1] = 0.25f;
                submit(&p);
            }
            g.x = 160;
            draw(&g);
        }
        end_step("aa");

        /* 7. Depth, Z test and the Z-pass counter: an overlapping second
         * quad at the same depth fails LESS. */
        {
            R300DrawPacket p;
            g = (Draw){ .gpu = true, .x = 16, .y = 16, .size = 128, .depth = true };
            for (int k = 0; k < 3; k++) {
                if (build(&g, &p)) {
                    p.uniforms.zpass_count = 1;
                    submit(&p);
                }
                g.x += 64;
            }
        }
        end_step("depth-zpass");

        /* 8. More than one 8 MiB arena buffer in a pass (CPU vertices). */
        for (int k = 0; k < 3; k++) {
            R300DrawPacket p;
            g = (Draw){ .gpu = false, .x = 16 + 100 * k, .y = 16, .size = 96 };
            if (build(&g, &p)) {
                uint32_t n = p.num_verts, reps = (6u << 20) / (n * sizeof(R300Vertex));
                R300Vertex *big = malloc((size_t)n * reps * sizeof(R300Vertex));
                for (uint32_t r = 0; r < reps; r++) {
                    memcpy(big + r * n, p.verts, n * sizeof(R300Vertex));
                }
                free(p.verts);
                p.verts = big;
                p.num_verts = n * reps;
                submit(&p);
            }
        }
        end_step("arena");

        /* 9. PPCGPU_R300_EXTENT keeps one pass while the packet's height
         * (its scissor's bottom edge) changes: the viewport follows the
         * packet, the attachments stay. */
        {
            static const uint32_t hs[4] = { 600, 400, 500, 300 };
            g = (Draw){ .gpu = true, .x = 16, .y = 16, .size = 128 };
            for (int k = 0; k < 4; k++) {
                g.h = hs[k];
                g.x = 16 + 140 * k;
                draw(&g);
                if (k == 0) {
                    snap_passes();
                }
            }
            if (r300_extent_on() && rqueue_enabled()) {
                CHECK(passes() == 0, "extent: %llu new passes for height changes",
                      (unsigned long long)passes());
            }
        }
        end_step("extent-height");

        /* 10. Depth, then a depth-less draw kept in the depth pass (test and
         * write off), then depth again: one pass throughout. */
        {
            R300DrawPacket p;
            bool dz[3] = { true, false, true };
            for (int k = 0; k < 3; k++) {
                g = (Draw){ .gpu = true, .x = 16 + 64 * k, .y = 16, .size = 128,
                            .depth = dz[k] };
                if (build(&g, &p)) {
                    p.uniforms.zpass_count = 1;
                    submit(&p);
                }
                if (k == 0) {
                    snap_passes();
                }
            }
            CHECK(passes() == 0, "depth-less draw: %llu new passes", (unsigned long long)passes());
        }
        end_step("depth-depthless-depth");

        /* 11. GPU vertex uniforms kept bound while the index staging runs
         * through more than one 8 MiB arena buffer; a fence commit, then a
         * new encoder binds them again. */
        {
            R300DrawPacket p;
            uint32_t n = 6u * 131072u;          /* 3 MiB of indices per draw */
            for (int k = 0; k < 4; k++) {
                g = (Draw){ .gpu = true, .x = 16 + 100 * k, .y = 16, .size = 96 };
                if (!build(&g, &p)) {
                    continue;
                }
                uint32_t *idx = malloc((size_t)n * 4);
                for (uint32_t i = 0; i < n; i++) {
                    idx[i] = p.vs_idx[i % p.num_verts];
                }
                free(p.vs_idx);
                p.vs_idx = idx;
                p.num_verts = n;
                if (k == 1) {
                    snap();
                    snap_passes();
                }
                submit(&p);
            }
            drained();
            CHECK(g_r200_arena_used && g_r200_arena_used.count >= 1,
                  "arena: no buffer turnover in the pass");
            CHECK(passes() == 0, "arena: %llu new passes", (unsigned long long)passes());
            if (reuse) {
                CHECK(made(R300R_VSU) == 0 && skipped(R300R_VSU) == 3,
                      "arena: vsu set %llu, skipped %llu", (unsigned long long)made(R300R_VSU),
                      (unsigned long long)skipped(R300R_VSU));
            }
            R->submit_r200(ST, fence_done, NULL);
            snap();
            g = (Draw){ .gpu = true, .x = 16, .y = 140, .size = 96 };
            draw(&g);
            if (reuse) {
                CHECK(made(R300R_VSU) == 1, "after the fence: vsu not bound again");
            }
            expect_fresh("arena, after the fence");
        }
        end_step("vsu-arena");

        /* 12. Polygon-mode lines in a two-sample pass, then culled triangles
         * of both windings: the cull mode comes back in each sample. */
        {
            R300DrawPacket p;
            g = (Draw){ .gpu = false, .x = 16, .y = 16, .size = 128, .rt = RT3,
                        .cull = R300_CULL_BACK };
            if (build(&g, &p)) {
                uint32_t nv = p.num_verts;
                p.verts = realloc(p.verts, (nv + 2) * sizeof(R300Vertex));
                p.verts[nv] = p.verts[0];
                p.verts[nv + 1] = p.verts[2];
                p.num_line_verts = 2;
                p.aa_samples = 2;
                p.aa_pos[0][0] = -0.25f;
                p.aa_pos[0][1] = -0.25f;
                p.aa_pos[1][0] = 0.25f;
                p.aa_pos[1][1] = 0.25f;
                submit(&p);
            }
            for (int k = 0; k < 2; k++) {
                g = (Draw){ .gpu = true, .x = 160 + 140 * k, .y = 16, .size = 128, .rt = RT3,
                            .cull = R300_CULL_BACK | (k ? 0 : R300_FACE_CW) };
                if (build(&g, &p)) {
                    p.aa_samples = 2;
                    memcpy(p.aa_pos, (float[2][2]){ { -0.25f, -0.25f }, { 0.25f, 0.25f } },
                           sizeof(float[2][2]));
                    submit(&p);
                }
            }
        }
        end_step("aa-lines");

        /* 13. A texture rewritten in place between two draws of a pass. */
        g = (Draw){ .gpu = true, .x = 16, .y = 16, .size = 128 };
        draw(&g);
        drained();
        fill_textures(77);
        g.x = 160;
        draw(&g);
        end_step("texture-rewrite");

        R->fini(ST);
    }
    printf("%s: %d failure(s)\n", fails ? "FAIL" : "ok", fails);
    return fails != 0;
}
