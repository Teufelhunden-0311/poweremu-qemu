/*
 * Captured builds (PPCGPU_BUILD_UNLOCK): r300_draw_fetch_plan and
 * r300_capture_read.  Random PSC layouts (every element type, zero-width
 * ones, skips, early LAST_VEC, more elements than arrays), arrays with
 * strides incl. 0, inline-indexed, index-buffer (every VC_SWAP mode, 16-
 * and 32-bit, sparse indices) and VBUF draws; both vertex-fetch paths.
 * Each draw is built from guest memory with a recording reader (the usual
 * build), and, when the plan accepts it, again from the planned capture
 * between r300_draw_parked_begin/end.  Checks:
 *  - every read of a usual build that took the GPU vertex-shader path lies
 *    inside the plan (and such builds did read);
 *  - the captured build of such a draw is the same packet: every field,
 *    and the shader texts, vertex inputs, indices and uniforms it points to;
 *  - a draw the usual build takes on the CPU path comes back "build it the
 *    usual way" from the captured build, never as a packet;
 *  - a failure is either the same failure or "the usual way";
 *  - wrapping addresses, the size cap and refused forms make the plan say no.
 * ./test_buildcap [draws] [seed]   AI-generated (Claude).
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

/* Guest memory at GPU address 0..MEM_SIZE; each read is recorded. */
#define MAX_READS 100000
static struct { uint32_t a, l; } reads[MAX_READS];
static unsigned nreads;

static bool mem_read(void *o, uint32_t a, void *d, uint32_t l)
{
    if (nreads < MAX_READS) {
        reads[nreads].a = a;
        reads[nreads].l = l;
    }
    nreads++;
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

/* Every field, and what the pointers point to (not the pointers). */
static bool same_packet(const R300DrawPacket *a, const R300DrawPacket *b)
{
    R300DrawPacket x = *a, y = *b;

    x.glsl = y.glsl = NULL;
    x.verts = y.verts = NULL;
    x.vs_glsl = y.vs_glsl = NULL;
    x.vs_in = y.vs_in = NULL;
    x.vs_idx = y.vs_idx = NULL;
    x.vs_u = y.vs_u = NULL;
    for (int t = 0; t < R300_NUM_TEX_UNITS; t++) {
        x.tex[t].host_data = y.tex[t].host_data = NULL;
    }
    if (memcmp(&x, &y, sizeof(x))) {
        return false;
    }
    if (!a->glsl != !b->glsl || (a->glsl && strcmp(a->glsl, b->glsl)) ||
        !a->vs_glsl != !b->vs_glsl || (a->vs_glsl && strcmp(a->vs_glsl, b->vs_glsl)) ||
        !a->verts != !b->verts || !a->vs_u != !b->vs_u) {
        return false;
    }
    if (a->verts && memcmp(a->verts, b->verts, sizeof(R300Vertex) *
                           (a->num_verts + a->num_line_verts))) {
        return false;
    }
    if (a->vs_glsl &&
        (memcmp(a->vs_in, b->vs_in, sizeof(float) * 4 * a->vs_in_vecs) ||
         memcmp(a->vs_idx, b->vs_idx, sizeof(uint32_t) * a->num_verts))) {
        return false;
    }
    if (a->vs_u && memcmp(a->vs_u, b->vs_u, sizeof(*a->vs_u))) {
        return false;
    }
    return true;
}

static bool covered(const R300FetchPlan *p, uint32_t a, uint32_t l)
{
    for (unsigned i = 0; i < p->n; i++) {
        if (a >= p->r[i].gpu && (uint64_t)a + l <= (uint64_t)p->r[i].gpu + p->r[i].len) {
            return true;
        }
    }
    return false;
}

static int fail(unsigned t, const char *what)
{
    printf("test_buildcap: FAIL draw %u: %s\n", t, what);
    return 1;
}

/* Fixed cases: the plan refuses what it must. */
static int edges(R300State *st)
{
    R300Arrays arr = { 0 };
    R300FetchPlan p;
    uint32_t body[1] = { 3u << 16 | 4 };            /* 3 vertices, triangles */

    st->glsl_flags = R300_GLSL_GPU_VS;
    /* one FLOAT_4 element (type 3), LAST_VEC */
    r300_state_write(st, 0x2150, 3 | 1u << 13);
    arr.count = 1;
    arr.a[0].size_dw = 4;
    arr.a[0].stride_dw = 4;
    arr.a[0].addr = 0x1000;
    if (!r300_draw_fetch_plan(st, &arr, 0x34, body, 1, NULL, 1u << 20, &p) ||
        p.n != 1 || p.r[0].gpu != 0x1000 || p.r[0].len != 48) {
        printf("test_buildcap: FAIL edge: plain draw (n %u gpu %x len %u)\n", p.n,
               p.n ? p.r[0].gpu : 0, p.n ? p.r[0].len : 0);
        return 1;
    }
    arr.a[0].addr = 0xFFFFFFD0u;                    /* ends exactly at 2^32 */
    if (!r300_draw_fetch_plan(st, &arr, 0x34, body, 1, NULL, 1u << 20, &p)) {
        printf("test_buildcap: FAIL edge: range ending at 2^32 refused\n");
        return 1;
    }
    arr.a[0].addr = 0xFFFFFFD4u;                    /* would wrap */
    if (r300_draw_fetch_plan(st, &arr, 0x34, body, 1, NULL, 1u << 20, &p)) {
        printf("test_buildcap: FAIL edge: wrapping range accepted\n");
        return 1;
    }
    arr.a[0].addr = 0x1000;
    arr.a[0].stride_dw = 0x40000000u;               /* vtx * stride wraps */
    if (r300_draw_fetch_plan(st, &arr, 0x34, body, 1, NULL, 1u << 20, &p)) {
        printf("test_buildcap: FAIL edge: wrapping stride accepted\n");
        return 1;
    }
    arr.a[0].stride_dw = 4;
    if (r300_draw_fetch_plan(st, &arr, 0x34, body, 1, NULL, 47, &p)) {
        printf("test_buildcap: FAIL edge: over the cap accepted\n");
        return 1;
    }
    if (r300_draw_fetch_plan(st, &arr, 0x35, body, 1, NULL, 1u << 20, &p) ||
        r300_draw_fetch_plan(st, &arr, 0x34, body, 0, NULL, 1u << 20, &p)) {
        printf("test_buildcap: FAIL edge: immediate or empty body accepted\n");
        return 1;
    }
    uint32_t rects[1] = { 3u << 16 | 8 };
    if (r300_draw_fetch_plan(st, &arr, 0x34, rects, 1, NULL, 1u << 20, &p)) {
        printf("test_buildcap: FAIL edge: rect list accepted\n");
        return 1;
    }
    arr.count = 0;
    if (r300_draw_fetch_plan(st, &arr, 0x34, body, 1, NULL, 1u << 20, &p)) {
        printf("test_buildcap: FAIL edge: no arrays accepted\n");
        return 1;
    }
    arr.count = 1;
    st->glsl_flags = 0;
    if (r300_draw_fetch_plan(st, &arr, 0x34, body, 1, NULL, 1u << 20, &p)) {
        printf("test_buildcap: FAIL edge: CPU vertex path accepted\n");
        return 1;
    }
    st->glsl_flags = R300_GLSL_GPU_VS;
    uint32_t ib[1] = { 0 };
    R300Indices idx = { ib, 1 };
    uint32_t vf[1] = { 3u << 16 | 4 | 1u << 11 };  /* 3 x 32-bit, 1 dword */
    if (r300_draw_fetch_plan(st, &arr, 0x36, vf, 1, &idx, 1u << 20, &p)) {
        printf("test_buildcap: FAIL edge: short index buffer accepted\n");
        return 1;
    }
    /* zero-width element (type 13) with a skip moves the next one */
    r300_state_write(st, 0x2150, (13 | 3 << 4) | (3u | 1u << 13) << 16);
    if (!r300_draw_fetch_plan(st, &arr, 0x34, body, 1, NULL, 1u << 20, &p) ||
        p.r[0].gpu != 0x1000 + 12 || p.r[0].len != (2 * 4 + 4) * 4) {
        printf("test_buildcap: FAIL edge: zero-width skip (gpu %x len %u)\n",
               p.r[0].gpu, p.r[0].len);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    static R300State st;
    unsigned draws = argc > 1 ? atoi(argv[1]) : 20000;
    rng = argc > 2 ? strtoull(argv[2], NULL, 0) : 0x2545F4914F6CDD1Dull;
    if (!rng) {
        rng = 1;                        /* xorshift stays at 0 */
    }
    unsigned planned = 0, gpu_checked = 0, gpu_reads = 0, cpu_redo = 0,
             fail_same = 0, fail_redo = 0, not_planned = 0, outside = 0;
    static const uint32_t prims[] = { 1, 2, 3, 4, 5, 6, 8 };

    load_state(&st);
    if (edges(&st)) {
        return 1;
    }
    load_state(&st);
    for (uint32_t i = 0; i < MEM_SIZE / 4; i++) {
        uint32_t w = word();
        memcpy(mem + 4 * i, &w, 4);
    }
    for (unsigned t = 0; t < draws; t++) {
        unsigned ne = 1 + pick(16);
        bool last = pick(8) != 0;
        for (unsigned k = 0; k < 16; k += 2) {
            uint32_t reg = 0, ext = 0;
            for (unsigned h = 0; h < 2; h++) {
                unsigned e = k + h;
                uint32_t v = pick(16) | pick(16) << 4 | pick(32) << 8 |
                             pick(2) << 14 | pick(2) << 15;
                if (e == ne - 1 && last) {
                    v |= 1u << 13;
                } else if (pick(40) == 0) {
                    v |= 1u << 13;
                }
                reg |= v << (16 * h);
                ext |= (rnd() & 0xFFFF) << (16 * h);
            }
            r300_state_write(&st, 0x2150 + 4 * (k / 2), reg);
            r300_state_write(&st, 0x21E0 + 4 * (k / 2), ext);
        }
        r300_state_write(&st, 0x20B4, 16 + pick(28));
        uint32_t cs = r300_reg(&st, 0x2140) & ~0x103u;
        cs |= pick(4) | (pick(12) == 0) << 8;           /* VC_SWAP, bypass */
        r300_state_write(&st, 0x2140, cs);
        st.glsl_flags = pick(6) ? R300_GLSL_GPU_VS : 0;

        R300Arrays arr = { 0 };
        arr.count = pick(30) ? 1 + pick(R300_MAX_ARRAYS) : 0;
        bool sparse = pick(4) == 0;
        for (unsigned a = 0; a < arr.count; a++) {
            arr.a[a].size_dw = 1 + pick(8);
            arr.a[a].stride_dw = pick(8) == 0 ? 0 : arr.a[a].size_dw + pick(8);
            arr.a[a].addr = 4 * pick(MEM_SIZE / 16);
        }
        if (arr.count && pick(16) == 0) {
            arr.a[pick(arr.count)].addr = MEM_SIZE - 64;
        }

        uint32_t n = 1 + pick(48), prim = prims[pick(7)];
        uint32_t vf = n << 16 | prim, body[1 + 48];
        unsigned kind = pick(3), nd = 1, imax = sparse ? 4096 : 64;
        R300Indices idx;
        uint32_t ib[48];
        uint32_t opcode = kind == 2 ? 0x34 : 0x36;
        body[0] = vf;
        if (kind == 0) {                            /* inline, 16- or 32-bit */
            bool i32 = pick(2);
            unsigned ndw = i32 ? n : (n + 1) / 2;
            if (i32) {
                vf |= 1u << 11;
                body[0] = vf;
            }
            for (unsigned k = 0; k < ndw; k++) {
                body[1 + k] = i32 ? pick(imax) : pick(imax) | pick(imax) << 16;
            }
            nd = 1 + ndw - (pick(10) == 0);         /* sometimes one short */
        } else if (kind == 1) {                     /* index buffer */
            bool i32 = pick(2);
            if (i32) {
                vf |= 1u << 11;
            }
            unsigned ndw = i32 ? n : (n + 1) / 2;
            for (unsigned k = 0; k < ndw; k++) {
                /* as the guest stores them: the build applies VC_SWAP */
                ib[k] = i32 ? pick(imax) : pick(imax) | pick(imax) << 16;
            }
            idx.dw = ib;
            idx.ndw = ndw - (pick(10) == 0);        /* sometimes short */
            body[0] = vf;
        }

        for (int vfetch = 0; vfetch < 2; vfetch++) {
            R300DrawPacket pa, pb;
            const char *ea, *eb;
            bool oka, okb, redo;
            R300FetchPlan plan;
            bool planned_ok = r300_draw_fetch_plan(&st, &arr, opcode, body, nd,
                                                   kind == 1 ? &idx : NULL,
                                                   4u << 20, &plan);
            r300_vfetch = vfetch;
            nreads = 0;
            oka = kind == 1 ?
                r300_draw_build_indexed(&st, &arr, vf, &idx, mem_read, NULL, &pa, &ea) :
                r300_draw_build(&st, &arr, opcode, body, nd, mem_read, NULL, &pa, &ea);
            if (!planned_ok) {
                not_planned++;
                /* (addresses here stay far below 2^32 and the spans below
                 * the cap: nothing the usual build draws on the GPU path
                 * may be refused) */
                if (oka && pa.vs_glsl) {
                    printf("kind %u n %u prim %u arrays %u reads %u plan.n %u total %llu\n",
                           kind, n, prim, arr.count, nreads, plan.n,
                           (unsigned long long)plan.total);
                    return fail(t, "plan refused a GPU vertex-shader draw");
                }
                if (oka) {
                    r300_draw_free(&pa);
                }
                continue;
            }
            planned++;
            if (oka && pa.vs_glsl) {
                gpu_checked++;
                gpu_reads += nreads > 0;
                for (unsigned r = 0; r < nreads && r < MAX_READS; r++) {
                    if (!covered(&plan, reads[r].a, reads[r].l)) {
                        printf("read %08x+%u not in plan (%u ranges)\n", reads[r].a,
                               reads[r].l, plan.n);
                        return fail(t, "a read of the usual build is outside the plan");
                    }
                }
            }
            /* The capture, as the device takes it (only inside guest memory). */
            R300Capture cap = { plan, NULL };
            uint64_t at = 0;
            bool inside = true;
            for (unsigned i = 0; i < plan.n; i++) {
                inside &= (uint64_t)plan.r[i].gpu + plan.r[i].len <= MEM_SIZE;
            }
            if (!inside) {
                outside++;
                if (oka) {
                    r300_draw_free(&pa);
                }
                continue;
            }
            cap.buf = malloc(plan.total ? plan.total : 1);
            for (unsigned i = 0; i < plan.n; i++) {
                memcpy(cap.buf + at, mem + plan.r[i].gpu, plan.r[i].len);
                at += plan.r[i].len;
            }
            r300_draw_parked_begin();
            okb = kind == 1 ?
                r300_draw_build_indexed(&st, &arr, vf, &idx, r300_capture_read, &cap, &pb, &eb) :
                r300_draw_build(&st, &arr, opcode, body, nd, r300_capture_read, &cap, &pb, &eb);
            redo = r300_draw_parked_end();
            free(cap.buf);
            if (oka && pa.vs_glsl) {
                if (redo || !okb || !same_packet(&pa, &pb)) {
                    printf("redo %d okb %d err %s\n", redo, okb, eb ? eb : "-");
                    return fail(t, "captured build differs from the usual one");
                }
            } else if (oka) {                       /* the CPU vertex path */
                if (!redo || okb) {
                    return fail(t, "captured build did not hand a CPU-path draw back");
                }
                cpu_redo++;
            } else {                                /* the usual build failed */
                if (redo) {
                    fail_redo++;
                } else if (okb || strcmp(ea ? ea : "", eb ? eb : "")) {
                    printf("usual: %s, captured: %s\n", ea ? ea : "-", okb ? "ok" : eb);
                    return fail(t, "different failure");
                } else {
                    fail_same++;
                }
            }
            if (oka) {
                r300_draw_free(&pa);
            }
            if (okb) {
                r300_draw_free(&pb);
            }
        }
    }
    r300_vfetch = false;
    if (gpu_reads < draws / 20) {          /* not vacuous: many that did read */
        printf("test_buildcap: FAIL too few GPU vertex-shader draws checked (%u, %u read)\n",
               gpu_checked, gpu_reads);
        return 1;
    }
    printf("test_buildcap: PASS, %u draws x 2 fetch paths: %u planned (%u GPU-VS packets "
           "identical, %u of them read, every read covered; %u CPU-path draws handed back; "
           "failures %u same, %u handed back; %u outside test memory), %u not planned; "
           "edge cases pass\n", draws, planned, gpu_checked, gpu_reads, cpu_redo, fail_same,
           fail_redo, outside, not_planned);
    return 0;
}
