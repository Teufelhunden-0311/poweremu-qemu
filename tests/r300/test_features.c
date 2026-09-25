/*
 * Draw assembly for the 3D features games need: depth/stencil state,
 * culling, and DRAW_INDX_2 with indices from memory (INDX_BUFFER).
 * Built on the first Quartz Extreme draw's state (qe_draw1_full.txt).
 */
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

    printf(fails ? "test_features: %d FAILED\n" : "test_features: PASS\n", fails);
    return fails != 0;
}
