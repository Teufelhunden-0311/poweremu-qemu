/*
 * r300_glsl_share: shader texts shared by reference (r300_text.h).  Random
 * draws vary the fragment program (texture formats, colour and depth
 * formats) and the GPU vertex program (clip planes, fog, perspective), so
 * both memos evict while packets still hold their texts.  Each draw is
 * built twice, sharing on and off (copy mode), and both packets go into a
 * queue; a second thread releases them, as the render thread does, in
 * turn lagging far behind and keeping up.  Checks: the shared and copied
 * texts are identical; a packet's texts are unchanged when released (hash
 * taken at build); sharing actually happened; both memos evicted; the
 * reference count survives threads taking and dropping references at once.
 * Built with AddressSanitizer and ThreadSanitizer by run.sh.
 * ./test_glslshare [draws] [seed]
 */
#include <assert.h>
#include <pthread.h>
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

static uint64_t hash(const char *s)
{
    uint64_t h = 1469598103934665603ull;
    for (; s && *s; s++) {
        h = (h ^ (uint8_t)*s) * 1099511628211ull;
    }
    return h;
}

/* The queue of built packets (two per draw) the releasing thread drains.
 * In alternate blocks of 1500 packets it lags LAG packets behind, longer
 * than 64 program changes (6 draws each), so the memos evict texts that
 * queued packets still hold; or releases at once, so it drops references
 * to the texts the builder is taking references to. */
#define QN 2048
#define LAG 1040
typedef struct Item {
    R300DrawPacket p;
    uint64_t hf, hv;
} Item;
static Item q[QN];
static unsigned q_head, q_tail;
static bool q_done;
static unsigned bad_release;
static pthread_mutex_t q_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t q_cond = PTHREAD_COND_INITIALIZER;

static void *releaser(void *arg)
{
    for (;;) {
        pthread_mutex_lock(&q_mtx);
        while (q_tail - q_head <= ((q_tail / 1500) % 2 ? 0 : LAG) && !q_done) {
            pthread_cond_wait(&q_cond, &q_mtx);
        }
        if (q_head == q_tail) {
            pthread_mutex_unlock(&q_mtx);
            return NULL;
        }
        Item it = q[q_head % QN];
        q_head++;
        pthread_cond_broadcast(&q_cond);
        pthread_mutex_unlock(&q_mtx);
        if (hash(it.p.glsl) != it.hf || hash(it.p.vs_glsl) != it.hv) {
            __atomic_fetch_add(&bad_release, 1, __ATOMIC_RELAXED);
        }
        r300_draw_free(&it.p);
    }
}

/* Reference count stress: threads taking and dropping references to one
 * text at once; the count must come back to exactly 1. */
#define STRESS_THREADS 4
#define STRESS_ROUNDS 200000
static char *stress_text;

static void *stress(void *arg)
{
    for (int i = 0; i < STRESS_ROUNDS; i++) {
        char *r = r300_text_ref(stress_text);
        if (r[0] != 'x') {
            abort();
        }
        r300_text_unref(r);
    }
    return NULL;
}

static bool stress_ok(void)
{
    pthread_t th[STRESS_THREADS];
    r300_glsl_share = true;
    stress_text = r300_text_new("x shared text");
    for (int i = 0; i < STRESS_THREADS; i++) {
        pthread_create(&th[i], NULL, stress, NULL);
    }
    for (int i = 0; i < STRESS_THREADS; i++) {
        pthread_join(th[i], NULL);
    }
    bool ok = __atomic_load_n(&r300_text_of(stress_text)->ref, __ATOMIC_ACQUIRE) == 1;
    r300_text_unref(stress_text);
    return ok;
}

static void push(const Item *it)
{
    pthread_mutex_lock(&q_mtx);
    while (q_tail - q_head == QN) {
        pthread_cond_wait(&q_cond, &q_mtx);
    }
    q[q_tail % QN] = *it;
    q_tail++;
    pthread_cond_broadcast(&q_cond);
    pthread_mutex_unlock(&q_mtx);
}

int main(int argc, char **argv)
{
    static R300State st;
    unsigned draws = argc > 1 ? atoi(argv[1]) : 6000;
    rng = argc > 2 ? strtoull(argv[2], NULL, 0) : 0x2545F4914F6CDD1Dull;
    static const uint32_t txfmt[] = { 0x00, 0x02, 0x05, 0x0C, 0x0C | (1u << 8), 0x01, 0x03 };
    static const uint32_t outfmt[] = { 0x00, 0x01, 0x02, 0x03 };
    unsigned built = 0, refused = 0, shared = 0, shared_vs = 0, gpu = 0;
    uint32_t max_fs = 0, max_vs = 0;
    const char *last_fs = NULL, *last_vs = NULL;
    pthread_t thr;

    if (!stress_ok()) {
        printf("test_glslshare: FAIL (reference count wrong after %d threads x %d "
               "references)\n", STRESS_THREADS, STRESS_ROUNDS);
        return 1;
    }
    load_state(&st);
    for (uint32_t i = 0; i < MEM_SIZE / 4; i++) {
        float x = (float)((int)pick(2001) - 1000) / 100.0f;
        memcpy(mem + 4 * i, &x, 4);
    }
    pthread_create(&thr, NULL, releaser, NULL);
    for (unsigned t = 0; t < draws; t++) {
        /* New program inputs every 6th draw (a game draws runs with one
         * program).  Fragment: texture formats of units 0-3, colour and
         * depth formats. */
        bool change = t % 6 == 0;
        for (unsigned u = 0; u < 4 && change; u++) {
            uint32_t f1 = r300_reg(&st, 0x44C0 + 4 * u) & ~0x11Fu;
            r300_state_write(&st, 0x44C0 + 4 * u, f1 | txfmt[pick(7)]);
        }
        if (change) {
            uint32_t of = r300_reg(&st, 0x46A4) & ~3u;
            r300_state_write(&st, 0x46A4, of | outfmt[pick(4)]);
            uint32_t zf = r300_reg(&st, 0x4F10) & ~0xFu;
            r300_state_write(&st, 0x4F10, zf | (pick(2) ? 2 : 0));
        }
        /* Vertex program inputs: a PSC layout (one of 48) and GPU or CPU
         * vertex shading. */
        uint32_t layout = pick(48);
        unsigned ne = 1 + layout % 6, dwords = 0;
        for (unsigned k = 0; k < 16; k += 2) {
            uint32_t reg = 0;
            for (unsigned h = 0; h < 2; h++) {
                unsigned e = k + h;
                uint32_t v = (e < ne ? (layout / 6 + e) % 4 : 0) | (e << 8) |
                             (e == ne - 1 ? 1u << 13 : 0);
                reg |= v << (16 * h);
            }
            r300_state_write(&st, 0x2150 + 4 * (k / 2), reg);
            r300_state_write(&st, 0x21E0 + 4 * (k / 2), 0xFAC6FAC6u);
        }
        dwords = 4 * ne;
        r300_state_write(&st, 0x20B4, dwords);
        st.glsl_flags = change ? (pick(4) ? R300_GLSL_GPU_VS : 0) : st.glsl_flags;
        /* The vertex memo's key: user clip planes, fog, fog source and
         * perspective (gpu_vs_get), a couple of thousand combinations. */
        if (change) {
            uint32_t cc = r300_reg(&st, 0x221C) & ~0x3Fu;       /* VAP_CLIP_CNTL */
            r300_state_write(&st, 0x221C, cc | pick(64));
            uint32_t fb = r300_reg(&st, 0x4BC0) & ~1u;          /* FG_FOG_BLEND */
            r300_state_write(&st, 0x4BC0, fb | pick(2));
            uint32_t gs = r300_reg(&st, 0x401C) & ~7u;          /* GB_SELECT */
            r300_state_write(&st, 0x401C, gs | pick(8));
            uint32_t vte = r300_reg(&st, 0x20B0) & ~(1u << 8);  /* VAP_VTE_CNTL */
            r300_state_write(&st, 0x20B0, vte | pick(2) << 8);
        }

        uint32_t n = 3 * (1 + pick(4)), body[1 + 12 * 24];
        body[0] = n << 16 | 4;                          /* triangle list */
        for (unsigned k = 1; k < 1 + n * dwords; k++) {
            float x = (float)((int)pick(201) - 100) / 100.0f;
            memcpy(&body[k], &x, 4);
        }
        R300Arrays arr = { 0 };
        Item it = { 0 };
        R300DrawPacket ref;
        const char *e1 = NULL, *e2 = NULL;

        r300_glsl_share = true;
        bool ok = r300_draw_build(&st, &arr, 0x35, body, 1 + n * dwords, mem_read, NULL,
                                  &it.p, &e1);
        r300_glsl_share = false;
        bool ok2 = r300_draw_build(&st, &arr, 0x35, body, 1 + n * dwords, mem_read, NULL,
                                   &ref, &e2);
        if (ok != ok2 || (ok && (!it.p.glsl != !ref.glsl || !it.p.vs_glsl != !ref.vs_glsl ||
                                 (it.p.glsl && strcmp(it.p.glsl, ref.glsl)) ||
                                 (it.p.vs_glsl && strcmp(it.p.vs_glsl, ref.vs_glsl)) ||
                                 it.p.glsl_id != ref.glsl_id || it.p.vs_id != ref.vs_id))) {
            printf("test_glslshare: draw %u differs between shared and copied texts "
                   "(ok %d/%d, err %s / %s)\n", t, ok, ok2, e1 ? e1 : "-", e2 ? e2 : "-");
            return 1;
        }
        if (!ok) {
            refused++;
            continue;
        }
        built++;
        gpu += it.p.vs_glsl != NULL;
        shared += it.p.glsl && it.p.glsl == last_fs;   /* the same text, not a copy */
        last_fs = it.p.glsl;
        shared_vs += it.p.vs_glsl && it.p.vs_glsl == last_vs;
        last_vs = it.p.vs_glsl;
        max_fs = it.p.glsl_id > max_fs ? it.p.glsl_id : max_fs;
        max_vs = it.p.vs_id > max_vs ? it.p.vs_id : max_vs;
        it.hf = hash(it.p.glsl);
        it.hv = hash(it.p.vs_glsl);
        push(&it);
        /* The copy-mode packet goes the same way (its own blocks). */
        Item cp = { .p = ref, .hf = hash(ref.glsl), .hv = hash(ref.vs_glsl) };
        push(&cp);
    }
    pthread_mutex_lock(&q_mtx);
    q_done = true;
    pthread_cond_broadcast(&q_cond);
    pthread_mutex_unlock(&q_mtx);
    pthread_join(thr, NULL);

    /* glsl_id and vs_id count the memos' generations: more than their
     * slots (64 fragment, 32 vertex) means entries were evicted. */
    if (bad_release || !shared || !shared_vs || max_fs <= 64 || max_vs <= 32) {
        printf("test_glslshare: FAIL (texts changed before release %u, shared %u/%u, "
               "fragment programs %u, vertex programs %u)\n", bad_release, shared,
               shared_vs, max_fs, max_vs);
        return 1;
    }
    printf("test_glslshare: PASS, %u draws (%u built, %u with GPU vertex shading, %u refused "
           "alike): shared and copied texts identical; consecutive draws shared one "
           "text %u times (fragment), %u (vertex); %u fragment and %u vertex programs "
           "generated; every text intact at release; reference count exact after "
           "%d threads x %d references\n", draws, built, gpu, refused, shared,
           shared_vs, max_fs, max_vs, STRESS_THREADS, STRESS_ROUNDS);
    return 0;
}
