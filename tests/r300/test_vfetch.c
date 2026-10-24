/*
 * r300_vfetch differential test: random PSC layouts, arrays, immediate,
 * inline-indexed and index-buffer draws, every VC_SWAP mode, CPU and GPU
 * vertex shading, PVS bypass.  Each draw is built with r300_vfetch off and on;
 * the packets must be identical (vertices, GPU-VS inputs and indices, warnings,
 * result and error).  ./test_vfetch [draws] [seed]
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../hw/display/r300/r300_draw.h"

#define MEM_SIZE (1u << 20)
static uint8_t mem[MEM_SIZE];
static uint64_t rng;

static uint32_t rnd(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (uint32_t)(rng >> 16);
}

static uint32_t pick(uint32_t n)
{
    return rnd() % n;
}

/* Guest memory at GPU address 0..MEM_SIZE; anything else is unmapped. */
static bool mem_read(void *o, uint32_t a, void *d, uint32_t l)
{
    if ((uint64_t)a + l > MEM_SIZE) {
        return false;
    }
    memcpy(d, mem + a, l);
    return true;
}

static void load_state(R300State *st)
{
    FILE *f = fopen("qe_draw1_full.txt", "r");
    char line[256];
    int mode = 0;

    assert(f);
    r300_state_reset(st);
    while (fgets(line, sizeof(line), f)) {
        unsigned a, v, i, d[4];
        float c[4];
        if (strstr(line, "PVS code")) { mode = 1; continue; }
        if (strstr(line, "PVS constants")) { mode = 2; continue; }
        if (mode == 0 && sscanf(line, " %x %x", &a, &v) == 2) r300_state_write(st, a, v);
        if (mode == 1 && sscanf(line, " %u: %x %x %x %x", &i, &d[0], &d[1], &d[2], &d[3]) == 5)
            memcpy(&st->pvs_mem[i * 4], d, 16);
        if (mode == 2 && sscanf(line, " c%u %f %f %f %f", &i, &c[0], &c[1], &c[2], &c[3]) == 5)
            memcpy(&st->pvs_mem[(512 + i) * 4], c, 16);
    }
    fclose(f);
}

/* Words of float-ish and arbitrary data, mostly finite floats. */
static uint32_t word(void)
{
    if (pick(4) == 0) {
        return rnd() << 16 | rnd();
    }
    float x = (float)((int)pick(20001) - 10000) / 100.0f;
    uint32_t u;
    memcpy(&u, &x, 4);
    return u;
}

static bool same_packet(const R300DrawPacket *a, const R300DrawPacket *b)
{
    if (a->num_verts != b->num_verts || a->num_line_verts != b->num_line_verts ||
        a->prim_class != b->prim_class || a->warn != b->warn ||
        !a->verts != !b->verts || !a->vs_glsl != !b->vs_glsl ||
        a->vs_in_vecs != b->vs_in_vecs) {
        return false;
    }
    if (a->verts && memcmp(a->verts, b->verts, sizeof(R300Vertex) *
                           (a->num_verts + a->num_line_verts))) {
        return false;
    }
    if (a->vs_glsl) {
        if (memcmp(a->vs_in, b->vs_in, sizeof(float) * 4 * a->vs_in_vecs) ||
            memcmp(a->vs_idx, b->vs_idx, sizeof(uint32_t) * a->num_verts) ||
            strcmp(a->vs_glsl, b->vs_glsl)) {
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv)
{
    static R300State st;
    unsigned draws = argc > 1 ? atoi(argv[1]) : 20000;
    rng = argc > 2 ? strtoull(argv[2], NULL, 0) : 0x9E3779B97F4A7C15ull;
    unsigned built = 0, gpu = 0, failed = 0, imm = 0, idxbuf = 0, inl = 0, vbuf = 0;
    static const uint32_t prims[] = { 1, 2, 3, 4, 5, 6, 8 };

    load_state(&st);
    for (uint32_t i = 0; i < MEM_SIZE / 4; i++) {
        uint32_t w = word();
        memcpy(mem + 4 * i, &w, 4);
    }
    for (unsigned t = 0; t < draws; t++) {
        /* PSC: 1-16 elements, LAST_VEC on the last (sometimes on none). */
        unsigned ne = 1 + pick(16), dwords = 0;
        bool last = pick(8) != 0;
        for (unsigned k = 0; k < 16; k += 2) {
            uint32_t reg = 0, ext = 0;
            for (unsigned h = 0; h < 2; h++) {
                unsigned e = k + h;
                uint32_t v = pick(16) | pick(4) << 4 | pick(32) << 8 |
                             pick(2) << 14 | pick(2) << 15;
                if (e == ne - 1 && last) {
                    v |= 1u << 13;
                } else if (pick(40) == 0) {
                    v |= 1u << 13;              /* an early LAST_VEC */
                }
                reg |= v << (16 * h);
                ext |= (rnd() & 0xFFFF) << (16 * h);
            }
            r300_state_write(&st, 0x2150 + 4 * (k / 2), reg);
            r300_state_write(&st, 0x21E0 + 4 * (k / 2), ext);
        }
        dwords = 16 + pick(28);
        r300_state_write(&st, 0x20B4, dwords);                  /* VAP_VTX_SIZE */
        uint32_t cs = r300_reg(&st, 0x2140) & ~0x103u;         /* VAP_CNTL_STATUS */
        cs |= pick(4) | (pick(6) == 0) << 8;                    /* VC_SWAP, PVS bypass */
        r300_state_write(&st, 0x2140, cs);
        st.glsl_flags = pick(2) ? R300_GLSL_GPU_VS : 0;

        R300Arrays arr = { 0 };
        arr.count = pick(20) ? 1 + pick(R300_MAX_ARRAYS) : 0;
        for (unsigned a = 0; a < arr.count; a++) {
            arr.a[a].size_dw = 1 + pick(8);
            arr.a[a].stride_dw = arr.a[a].size_dw + pick(8);
            arr.a[a].addr = 4 * pick(MEM_SIZE / 8);
        }
        if (arr.count && pick(16) == 0) {
            arr.a[pick(arr.count)].addr = MEM_SIZE - 64;    /* runs off the end */
        }

        uint32_t n = 1 + pick(48), prim = prims[pick(7)];
        uint32_t vf = n << 16 | prim, body[1 + 48 * 44];
        unsigned kind = pick(4), nd = 1;
        R300Indices idx;
        uint32_t ib[48];
        body[0] = vf;
        if (kind == 0) {                            /* DRAW_IMMD_2 */
            nd = 1 + n * dwords - (pick(10) == 0);  /* sometimes one short */
            for (unsigned k = 1; k < nd; k++) {
                body[k] = word();
            }
            imm++;
        } else if (kind == 1) {                     /* DRAW_INDX_2, inline 16-bit */
            for (unsigned k = 0; k < (n + 1) / 2; k++) {
                body[1 + k] = pick(64) | pick(64) << 16;
            }
            nd = 1 + (n + 1) / 2;
            inl++;
        } else if (kind == 2) {                     /* index buffer, 32-bit */
            vf |= 1u << 11;
            for (unsigned k = 0; k < n; k++) {
                ib[k] = pick(64);
            }
            idx.dw = ib;
            idx.ndw = n;
            idxbuf++;
        } else {
            vbuf++;
        }

        R300DrawPacket p[2];
        const char *err[2];
        bool ok[2];
        for (int on = 0; on < 2; on++) {
            r300_vfetch = on;
            if (kind == 2) {
                ok[on] = r300_draw_build_indexed(&st, &arr, vf, &idx, mem_read, NULL,
                                                 &p[on], &err[on]);
            } else {
                ok[on] = r300_draw_build(&st, &arr, kind == 0 ? 0x35 : kind == 1 ? 0x36 : 0x34,
                                         body, nd, mem_read, NULL, &p[on], &err[on]);
            }
        }
        r300_vfetch = false;
        if (ok[0] != ok[1] || (!ok[0] && strcmp(err[0] ? err[0] : "", err[1] ? err[1] : "")) ||
            (ok[0] && !same_packet(&p[0], &p[1]))) {
            printf("test_vfetch: draw %u differs (kind %u, n %u, prim %u, %u elements, "
                   "arrays %u, gpu %u): ok %d/%d err %s / %s\n", t, kind, n, prim, ne,
                   arr.count, st.glsl_flags != 0, ok[0], ok[1],
                   err[0] ? err[0] : "-", err[1] ? err[1] : "-");
            return 1;
        }
        if (ok[0]) {
            built++;
            gpu += p[0].vs_glsl != NULL;
            r300_draw_free(&p[0]);
            r300_draw_free(&p[1]);
        } else {
            failed++;
        }
    }
    printf("test_vfetch: PASS, %u draws identical with r300_vfetch off and on "
           "(%u built, %u with GPU vertex shading, %u refused alike; immediate %u, "
           "inline-indexed %u, index buffer %u, arrays %u)\n",
           draws, built, gpu, failed, imm, inl, idxbuf, vbuf);
    return 0;
}
