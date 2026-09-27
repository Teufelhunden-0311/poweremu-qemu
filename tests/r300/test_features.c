/*
 * Draw assembly for the 3D features games need: depth/stencil state,
 * culling, and DRAW_INDX_2 with indices from memory (INDX_BUFFER).
 * Built on the first Quartz Extreme draw's state (qe_draw1_full.txt).
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../../hw/display/r300/r300_draw.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); \
    printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

static R300State st;

static void load_state(void)
{
    FILE *f = fopen("qe_draw1_full.txt", "r");
    char line[256];
    int mode = 0;

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

/* Guest memory for vertex arrays: big-endian dwords, as the PPC wrote them
 * (the captured state has VC_SWAP = 2). */
static uint8_t mem[0x10000];
#define MEM_BASE 0x08000000u

static bool rd(void *o, uint32_t a, void *d, uint32_t l)
{
    if (a < MEM_BASE || a - MEM_BASE + l > sizeof(mem)) return false;
    memcpy(d, mem + (a - MEM_BASE), l);
    return true;
}

static void put_be(uint32_t off, uint32_t v)
{
    mem[off] = v >> 24; mem[off + 1] = v >> 16; mem[off + 2] = v >> 8; mem[off + 3] = v;
}

static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static const uint32_t immd[49] = { 0x0004003D };    /* 4 vertices, quads */

/* A DRAW_IMMD_2 packet of the captured QE state: n vertices (pos, colour,
 * texcoord) at window positions xy, colour k/n. */
static uint32_t *immd_pkt(unsigned prim, unsigned n, const float (*xy)[2], uint32_t *ndw)
{
    static uint32_t d[1 + 12 * 16];
    d[0] = (n << 16) | (3u << 4) | prim;
    for (unsigned i = 0; i < n; i++) {
        float v[12] = { xy[i][0], xy[i][1], 0, 1,  (float)i / n, 0.5f, 0.25f, 1,
                        xy[i][0], xy[i][1], 0, 1 };
        memcpy(&d[1 + 12 * i], v, sizeof(v));
    }
    *ndw = 1 + 12 * n;
    return d;
}

/* Window position of a packet vertex (y down). */
static void win(const R300DrawPacket *p, const R300Vertex *v, float *x, float *y)
{
    *x = (v->pos[0] / v->pos[3] + 1) * p->rt_width / 2;
    *y = (1 - v->pos[1] / v->pos[3]) * p->rt_height / 2;
}

#define FEQ(a, b) (fabsf((a) - (b)) < 1e-3f)

static void rendering_completeness(void)
{
    R300Arrays none = { 0 };
    R300DrawPacket p;
    const char *err;
    uint32_t n;
    uint32_t *d;
    static const float quad4[4][2] = { { 0, 0 }, { 256, 0 }, { 256, 256 }, { 0, 256 } };

    /* ---- texture layout (Mesa r300_texture_desc.c rules) ---- */
    R300TexDesc t;
    memset(&t, 0, sizeof(t));
    t.width = t.height = 64; t.levels = 7; t.dim = R300_TEXDIM_2D; t.depth = 1;
    r300_tex_layout(&t, 4, false, false);
    CHECK(t.lvl_off[1] == 16384 && t.lvl_pitch[1] == 128 && t.lvl_pitch[5] == 32 &&
          t.lvl_rows[5] == 2 && t.lvl_off[6] == 16384 + 4096 + 1024 + 256 + 128 + 64 &&
          t.size_bytes == t.lvl_off[6] + 32, "64x64 chain: off1 %u pitch5 %u size %u",
          t.lvl_off[1], t.lvl_pitch[5], t.size_bytes);
    memset(&t, 0, sizeof(t));
    t.width = t.height = 16; t.levels = 1; t.dim = R300_TEXDIM_CUBE; t.depth = 1;
    r300_tex_layout(&t, 4, false, false);
    CHECK(t.size_bytes == 6 * 64 * 16, "cube size %u", t.size_bytes);
    memset(&t, 0, sizeof(t));
    t.width = t.height = 8; t.depth = 4; t.levels = 2; t.dim = R300_TEXDIM_3D;
    r300_tex_layout(&t, 4, false, false);
    CHECK(t.lvl_off[1] == 32 * 8 * 4 && t.size_bytes == 1024 + 32 * 4 * 2, "3D %u %u",
          t.lvl_off[1], t.size_bytes);
    memset(&t, 0, sizeof(t));
    t.width = 6; t.height = 5; t.levels = 3; t.depth = 1;
    r300_tex_layout(&t, 1, false, false);           /* NPOT mipmapped X8: POT rows */
    CHECK(t.lvl_rows[0] == 8 && t.lvl_pitch[0] == 32 && t.lvl_off[1] == 256 &&
          t.lvl_off[2] == 256 + 64 && t.size_bytes == 352, "npot %u %u %u", t.lvl_rows[0],
          t.lvl_off[2], t.size_bytes);
    memset(&t, 0, sizeof(t));
    t.width = t.height = 16; t.levels = 3; t.depth = 1;
    r300_tex_layout(&t, 8, true, false);            /* DXT1 */
    CHECK(t.lvl_pitch[0] == 32 && t.lvl_rows[0] == 4 && t.lvl_off[1] == 128 &&
          t.lvl_off[2] == 192 && t.size_bytes == 224, "dxt1 %u %u %u", t.lvl_off[1],
          t.lvl_off[2], t.size_bytes);
    memset(&t, 0, sizeof(t));
    t.width = 100; t.height = 10; t.levels = 1; t.depth = 1; t.pitch_bytes = 512;
    r300_tex_layout(&t, 4, false, true);            /* TX_PITCH_EN keeps its pitch */
    CHECK(t.lvl_pitch[0] == 512 && t.size_bytes == 5120, "pitch_en %u", t.lvl_pitch[0]);

    /* ---- border colours: the texel's own packing ---- */
    float bc[4];
    r300_border_color(0x0C, 0x80FF0040, bc);
    CHECK(FEQ(bc[0], 0x40 / 255.0f) && bc[1] == 0 && bc[2] == 1 && FEQ(bc[3], 0x80 / 255.0f),
          "8888 border %g %g %g %g", bc[0], bc[1], bc[2], bc[3]);
    r300_border_color(0x06, 0xF800, bc);
    CHECK(bc[0] == 0 && bc[1] == 0 && bc[2] == 1, "565 border %g %g %g", bc[0], bc[1], bc[2]);
    r300_border_color(0x0F, 0x00FF0000, bc);        /* DXT: B8G8R8A8 */
    CHECK(bc[0] == 1 && bc[2] == 0, "dxt border %g %g", bc[0], bc[2]);

    /* ---- texture unit state: mip chain, LOD, 3D, cube, sign, gamma ---- */
    r300_state_write(&st, 0x4104, 1);
    r300_state_write(&st, 0x4540, 0x00100000);
    r300_state_write(&st, 0x4480, 63 | (63u << 11) | (6u << 26));          /* 64x64, 7 levels */
    r300_state_write(&st, 0x44C0, 0x0C | (1u << 8) | (1u << 21));          /* 8888, X signed, gamma */
    r300_state_write(&st, 0x4400, (2u << 9) | (2u << 11) | (2u << 13) | (1u << 17));
    r300_state_write(&st, 0x4440, (uint32_t)(-16 & 0x3FF) << 3);           /* LOD bias -0.5 */
    d = immd_pkt(13, 4, quad4, &n);
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(p.tex[0].levels == 7 && p.tex[0].size_bytes == 16384 + 4096 + 1024 + 256 + 128 + 64 + 32,
          "levels %u size %u", p.tex[0].levels, p.tex[0].size_bytes);
    CHECK(FEQ(p.uniforms.tex_lod[0][0], -0.5f) && p.uniforms.tex_lod[0][1] == 1 &&
          p.uniforms.tex_lod[0][2] == 6 && p.uniforms.tex_lod[0][3] == 2,
          "lod %g %g %g %g", p.uniforms.tex_lod[0][0], p.uniforms.tex_lod[0][1],
          p.uniforms.tex_lod[0][2], p.uniforms.tex_lod[0][3]);
    CHECK(p.uniforms.tex_dim[0][3] == (R300_TEXF_SIGNED_X | R300_TEXF_GAMMA | R300_TEXF_POT_ROWS),
          "tex flags %x", p.uniforms.tex_dim[0][3]);
    r300_draw_free(&p);
    r300_state_write(&st, 0x4480, 7 | (7u << 11) | (2u << 22));            /* 8x8x4 */
    r300_state_write(&st, 0x44C0, 0x0C | (1u << 25));                      /* 3D */
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(p.tex[0].dim == R300_TEXDIM_3D && p.tex[0].depth == 4 && p.tex[0].size_bytes == 1024,
          "3D unit dim %u depth %u size %u", p.tex[0].dim, p.tex[0].depth, p.tex[0].size_bytes);
    r300_draw_free(&p);
    r300_state_write(&st, 0x44C0, 0x0C | (2u << 25));                      /* cube */
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(p.tex[0].dim == R300_TEXDIM_CUBE && p.tex[0].size_bytes == 6 * 32 * 8,
          "cube unit %u", p.tex[0].size_bytes);
    r300_draw_free(&p);
    r300_state_write(&st, 0x4104, 0);
    r300_state_write(&st, 0x4440, 0);

    /* ---- polygon mode: quad edges only (no diagonal), points ---- */
    r300_state_write(&st, 0x4288, 1 | (1u << 4) | (1u << 7));
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(p.prim_class == 1 && p.num_verts == 8, "poly lines cls %u verts %u",
          p.prim_class, p.num_verts);
    r300_draw_free(&p);
    r300_state_write(&st, 0x4288, 1 | (0u << 4) | (0u << 7));
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(p.prim_class == 0 && p.num_verts == 6 * 6, "poly points %u", p.num_verts);
    r300_draw_free(&p);
    /* back faces culled on the CPU when polygon mode is on (the quad is
     * clockwise on screen: a back face with FACE_CW clear) */
    r300_state_write(&st, 0x4288, 1 | (1u << 4) | (1u << 7));
    r300_state_write(&st, 0x42B8, R300_CULL_BACK);
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(p.num_verts == 0 && p.num_line_verts == 0, "poly cull %u", p.num_verts);
    r300_draw_free(&p);
    r300_state_write(&st, 0x42B8, 0);
    r300_state_write(&st, 0x4288, 0);

    /* ---- flat shading, provoking vertex LAST: the quad's vertex 3 ---- */
    r300_state_write(&st, 0x4278, 0x5555 | (3u << 16));
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    bool flat = p.num_verts == 6;
    for (unsigned i = 0; i < p.num_verts; i++) {
        flat &= FEQ(p.verts[i].v[1][0], 0.75f);
    }
    CHECK(flat, "flat shading");
    r300_draw_free(&p);
    /* FIRST: vertex 0 of the quad */
    r300_state_write(&st, 0x4278, 0x5555);
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(p.num_verts == 6 && FEQ(p.verts[5].v[1][0], 0.0f), "flat first %g", p.verts[5].v[1][0]);
    r300_draw_free(&p);
    /* solid: GA_SOLID_RG/BA in S3.12 */
    r300_state_write(&st, 0x4278, 0);
    r300_state_write(&st, 0x427C, (0x0800u << 16) | 0x1000);
    r300_state_write(&st, 0x4280, (0x0000u << 16) | 0x1000);
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(FEQ(p.verts[0].v[1][0], 0.5f) && FEQ(p.verts[0].v[1][1], 1.0f) &&
          FEQ(p.verts[0].v[1][2], 0.0f), "solid %g %g %g", p.verts[0].v[1][0],
          p.verts[0].v[1][1], p.verts[0].v[1][2]);
    r300_draw_free(&p);
    r300_state_write(&st, 0x4278, 0x3AAAA);

    /* ---- two-sided colours: colour 2 present, the quad is a back face:
     * colour 0 comes from output slot 2 (the texcoord output here) ---- */
    r300_state_write(&st, 0x2090, 3 | (1u << 3));
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    {
        float x, y;     /* the texcoord output is window position / 256 */
        win(&p, &p.verts[1], &x, &y);
        CHECK(FEQ(p.verts[1].v[1][0], x / 256) && FEQ(p.verts[1].v[1][1], y / 256),
              "back colour %g %g at %g %g", p.verts[1].v[1][0], p.verts[1].v[1][1], x, y);
    }
    r300_draw_free(&p);
    r300_state_write(&st, 0x42B8, R300_FACE_CW);     /* now it is front-facing */
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(FEQ(p.verts[1].v[1][1], 0.5f), "front colour %g", p.verts[1].v[1][1]);
    r300_draw_free(&p);
    r300_state_write(&st, 0x42B8, 0);
    r300_state_write(&st, 0x2090, 3);

    /* ---- wide lines: 3 px (half width 18/12), x-major: widened in y ---- */
    static const float seg[2][2] = { { 10, 20 }, { 110, 20 } };
    r300_state_write(&st, 0x4234, 18 | (2u << 16));
    d = immd_pkt(2, 2, seg, &n);
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    {
        float x0, y0, x1, y1;
        CHECK(p.prim_class == 0 && p.num_verts == 6, "wide line %u %u", p.prim_class, p.num_verts);
        win(&p, &p.verts[0], &x0, &y0);
        win(&p, &p.verts[1], &x1, &y1);
        CHECK(FEQ(x0, 10) && FEQ(y0, 18.5f) && FEQ(y1, 21.5f), "wide corners %g,%g %g,%g",
              x0, y0, x1, y1);
    }
    r300_draw_free(&p);
    /* stipple coordinates: s = accumulated subpixels * STIPPLE_SCALE past
     * the vertex's own texture components (RS reads one past the texcoord) */
    r300_state_write(&st, 0x4234, 6 | (2u << 16));
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(p.prim_class == 1 && p.num_verts == 2, "thin line %u", p.num_verts);
    r300_draw_free(&p);

    /* ---- rectangle list: the card completes the parallelogram ---- */
    static const float rect3[3][2] = { { 0, 0 }, { 100, 0 }, { 100, 50 } };
    d = immd_pkt(8, 3, rect3, &n);
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    {
        float x, y;
        CHECK(p.num_verts == 6, "rect verts %u", p.num_verts);
        win(&p, &p.verts[5], &x, &y);
        CHECK(FEQ(x, 0) && FEQ(y, 50), "rect 4th corner %g,%g", x, y);
    }
    r300_draw_free(&p);

    /* ---- points: per-vertex size (output slot 1 .x = colour r) clamped
     * to GA_POINT_MINMAX (4..8 px) ---- */
    static const float pt[1][2] = { { 50, 60 } };
    d = immd_pkt(1, 1, pt, &n);
    r300_state_write(&st, 0x2090, 3 | (1u << 16));
    r300_state_write(&st, 0x4230, 24 | (48u << 16));
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    {
        float x0, y0, x1, y1;
        win(&p, &p.verts[0], &x0, &y0);
        win(&p, &p.verts[2], &x1, &y1);
        CHECK(p.num_verts == 6 && FEQ(x1 - x0, 4) && FEQ(y1 - y0, 4), "point size %g x %g",
              x1 - x0, y1 - y0);
    }
    r300_draw_free(&p);
    r300_state_write(&st, 0x2090, 3);

    /* ---- user clip planes: distance to plane 0 = clip-space x ---- */
    d = immd_pkt(13, 4, quad4, &n);
    float plane[4] = { 1, 0, 0, 0 };
    memcpy(&st.pvs_mem[1024 * 4], plane, sizeof(plane));
    r300_state_write(&st, 0x221C, 1);
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(FEQ(p.verts[0].ucp[0], -1.0f) && FEQ(p.verts[1].ucp[0], -0.36f) &&
          p.verts[0].ucp[1] == 1.0f, "ucp %g %g %g", p.verts[0].ucp[0], p.verts[1].ucp[0],
          p.verts[0].ucp[1]);
    CHECK(p.glsl && strstr(p.glsl, "gl_ClipDistance[6]"), "vertex shader lacks clip distances");
    r300_draw_free(&p);
    r300_state_write(&st, 0x221C, 0);

    /* ---- fog value: C0A (1.0) * GA_FOG_SCALE + GA_FOG_OFFSET ---- */
    r300_state_write(&st, 0x4BC0, 1);
    r300_state_write(&st, 0x401C, 0);
    r300_state_write(&st, 0x4294, fbits(2.0f));
    r300_state_write(&st, 0x4298, fbits(0.5f));
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(FEQ(p.verts[0].aux[0], 2.5f) && p.uniforms.fog_blend == 1, "fog %g",
          p.verts[0].aux[0]);
    r300_draw_free(&p);
    r300_state_write(&st, 0x4BC0, 0);

    /* ---- render targets: RB3D_CCTL multiwrites ---- */
    CHECK(r300_us_num_targets(&st) == 1, "targets %u", r300_us_num_targets(&st));
    r300_state_write(&st, 0x4E00, 2u << 5);
    r300_state_write(&st, 0x4E2C, 0x00200000);
    r300_state_write(&st, 0x4E30, 0x00300000);
    r300_state_write(&st, 0x4E3C, r300_reg(&st, 0x4E38));
    r300_state_write(&st, 0x4E40, r300_reg(&st, 0x4E38));
    r300_draw_build(&st, &none, 0x35, d, n, rd, NULL, &p, &err);
    CHECK(p.num_cb == 3 && p.cb[2].gpu_addr == 0x00300000 && p.cb[1].view == p.rt_view,
          "multiwrite cb %u %08x", p.num_cb, p.cb[2].gpu_addr);
    CHECK(p.glsl && strstr(p.glsl, "location = 2) out r300_cb2_o o_c2") &&
          strstr(p.glsl, "location = 3) out uvec4 o_z"),
          "MRT outputs missing");
    r300_draw_free(&p);
    r300_state_write(&st, 0x4E00, 0);
}

int main(void)
{
    R300Arrays none = { 0 };
    R300DrawPacket p;
    const char *err;

    load_state();

    /* The compositor's own state: no depth, no culling. */
    CHECK(r300_draw_build(&st, &none, 0x35, immd, 49, rd, NULL, &p, &err), "build: %s", err);
    CHECK(!p.depth.attach && p.cull == 0, "QE draw has depth %d cull %x", p.depth.attach, p.cull);
    r300_draw_free(&p);

    /* ZB_CNTL.STENCIL_FRONT_BACK alone (what Apple leaves set) is not a test. */
    r300_state_write(&st, 0x4F00, 0x10);
    r300_draw_build(&st, &none, 0x35, immd, 49, rd, NULL, &p, &err);
    CHECK(!p.depth.attach, "FRONT_BACK alone attached depth");
    r300_draw_free(&p);

    /* Z24S8, Z test + write, dword-swapped, macro-tiled pitch 704. */
    r300_state_write(&st, 0x4F00, 0x06);
    r300_state_write(&st, 0x4F04, 0x00000001);
    r300_state_write(&st, 0x4F08, 0x00FFFF00);
    r300_state_write(&st, 0x4F10, 2);
    r300_state_write(&st, 0x4F20, 0x01000000 | 0x3);
    r300_state_write(&st, 0x4F24, 704 | (1u << 16) | (2u << 19));
    r300_draw_build(&st, &none, 0x35, immd, 49, rd, NULL, &p, &err);
    CHECK(p.depth.attach && p.depth.gpu_addr == 0x01000000 && p.depth.pitch == 704 &&
          p.depth.bpp == 4, "depth %d %08x %u %u", p.depth.attach, p.depth.gpu_addr,
          p.depth.pitch, p.depth.bpp);
    CHECK(p.uniforms.zinfo[0] == 6 && p.uniforms.zinfo[1] == 1 &&
          p.uniforms.zinfo[2] == 0x00FFFF00 && p.uniforms.zinfo[3] == 2,
          "zinfo %x %x %x %x", p.uniforms.zinfo[0], p.uniforms.zinfo[1],
          p.uniforms.zinfo[2], p.uniforms.zinfo[3]);
    CHECK(!(p.warn & R300_WARN_DEPTH), "Z24S8 warned");
    r300_draw_free(&p);

    /* Z16 */
    r300_state_write(&st, 0x4F10, 0);
    r300_state_write(&st, 0x4F24, 1024);
    r300_draw_build(&st, &none, 0x35, immd, 49, rd, NULL, &p, &err);
    CHECK(p.depth.bpp == 2 && p.uniforms.zinfo[3] == R300_ZFMT_Z16, "z16 %u %x",
          p.depth.bpp, p.uniforms.zinfo[3]);
    r300_draw_free(&p);
    r300_state_write(&st, 0x4F00, 0);

    /* Culling: quads cull, points and lines never do. */
    r300_state_write(&st, 0x42B8, R300_CULL_BACK | R300_FACE_CW);
    r300_draw_build(&st, &none, 0x35, immd, 49, rd, NULL, &p, &err);
    CHECK(p.cull == (R300_CULL_BACK | R300_FACE_CW), "quad cull %x", p.cull);
    r300_draw_free(&p);
    uint32_t lines[49];
    memcpy(lines, immd, sizeof(lines));
    lines[0] = (lines[0] & ~0xFu) | 2;
    r300_draw_build(&st, &none, 0x35, lines, 49, rd, NULL, &p, &err);
    CHECK(p.cull == 0 && p.prim_class == 1, "lines cull %x cls %u", p.cull, p.prim_class);
    r300_draw_free(&p);
    r300_state_write(&st, 0x42B8, 0);

    /*
     * DRAW_INDX_2 + INDX_BUFFER: 4 vertices (pos, colour, texcoord; 12
     * dwords each) in memory, drawn as a triangle list 2,1,0, 0,3,2 of
     * 16-bit indices.  The vertex cache applies VC_SWAP to index dwords
     * like any fetched data; the first index is the low half.
     */
    static const float vtx[4][12] = {
        { 0, 0, 0, 1,  1, 0, 0, 1,  0, 0, 0, 1 },
        { 100, 0, 0, 1,  0, 1, 0, 1,  100, 0, 0, 1 },
        { 100, 100, 0, 1,  0, 0, 1, 1,  100, 100, 0, 1 },
        { 0, 100, 0, 1,  1, 1, 1, 1,  0, 100, 0, 1 },
    };
    for (int i = 0; i < 4; i++)
        for (int k = 0; k < 12; k++)
            put_be(0x100 + (i * 12 + k) * 4, fbits(vtx[i][k]));
    uint16_t ix[6] = { 2, 1, 0, 0, 3, 2 };
    uint32_t idw[3];
    for (int k = 0; k < 3; k++) {
        uint32_t w = ix[2 * k] | (uint32_t)ix[2 * k + 1] << 16;
        put_be(0x1000 + k * 4, w);
        memcpy(&idw[k], mem + 0x1000 + k * 4, 4);    /* as fetched */
    }
    R300Arrays arr = { 1 };
    arr.a[0].addr = MEM_BASE + 0x100;
    arr.a[0].size_dw = 12;
    arr.a[0].stride_dw = 12;
    R300Indices idx = { idw, 3 };
    uint32_t vf = (6u << 16) | (1u << 4) | 4;         /* 6 indices, walk indices, tris */
    CHECK(r300_draw_build_indexed(&st, &arr, vf, &idx, rd, NULL, &p, &err),
          "indexed: %s", err ? err : "");
    if (p.verts) {
        CHECK(p.num_verts == 6 && p.prim_class == 0, "indexed verts %u", p.num_verts);
        /* colour varying (v[1] in this state) identifies the source vertex */
        static const int want[6] = { 2, 1, 0, 0, 3, 2 };
        for (int i = 0; i < 6 && i < (int)p.num_verts; i++) {
            const float *c = vtx[want[i]] + 4;
            CHECK(!memcmp(p.verts[i].v[1], c, 16), "index %d: colour %.0f %.0f %.0f", i,
                  p.verts[i].v[1][0], p.verts[i].v[1][1], p.verts[i].v[1][2]);
        }
    }
    r300_draw_free(&p);

    /* 32-bit indices */
    uint32_t i32[3] = { 3, 2, 1 };
    for (int k = 0; k < 3; k++) {
        put_be(0x2000 + k * 4, i32[k]);
        memcpy(&idw[k], mem + 0x2000 + k * 4, 4);
    }
    vf = (3u << 16) | (1u << 11) | (1u << 4) | 4;
    CHECK(r300_draw_build_indexed(&st, &arr, vf, &idx, rd, NULL, &p, &err), "i32: %s", err);
    CHECK(p.num_verts == 3 && !memcmp(p.verts[0].v[1], vtx[3] + 4, 16) &&
          !memcmp(p.verts[2].v[1], vtx[1] + 4, 16), "i32 order");
    r300_draw_free(&p);

    /* An index buffer shorter than the draw is refused. */
    idx.ndw = 1;
    CHECK(!r300_draw_build_indexed(&st, &arr, vf, &idx, rd, NULL, &p, &err), "short ib");

    /* Multisample layout: every sample of a 64x16 buffer lands on a
     * distinct slot inside the buffer, for 2 and 4 samples. */
    for (unsigned ns = 2; ns <= 4; ns += 2) {
        static uint8_t used[64 * 16 * 4 * 4];
        memset(used, 0, sizeof(used));
        bool ok = true;
        for (unsigned y = 0; y < 16; y++)
            for (unsigned x = 0; x < 64; x++)
                for (unsigned k = 0; k < ns; k++) {
                    uint32_t o = r300_msaa_offset(x, y, ns, 64, 4, k);
                    if (o % 4 || o / 4 >= 64 * 16 * ns || used[o / 4]++) ok = false;
                }
        CHECK(ok, "msaa layout ns=%u not a permutation", ns);
    }
    /* The driver's own formula at a picked pixel (2 samples, 704 wide). */
    CHECK(r300_msaa_offset(592, 496, 2, 704, 4, 0) == 0x2b3400, "msaa sample 0 %x",
          r300_msaa_offset(592, 496, 2, 704, 4, 0));
    rendering_completeness();
    printf(fails ? "test_features: %d FAILED\n" : "test_features: PASS\n", fails);
    return fails != 0;
}
