/*
 * How the Vulkan renderer behaves when Vulkan calls fail.
 *
 * hw/display/ppc_mac_gpu_vulkan.c is compiled into this program as it is
 * (#include below), against stand-in QEMU headers (shim/) and a stand-in
 * Vulkan driver (mockvk.c) that can make any call fail and that checks
 * what the renderer does with handles, command buffers, fences and images.
 *
 *  - Helper contract: vk_image and vk_buffer with each of their calls
 *    failing, and vk_dummies around such a failure; a wait for the GPU
 *    that gives no answer, and one that reports the device lost.
 *  - Sweep: a scenario of draws, flushes, submissions and accesses to VRAM
 *    is run once per failable call it makes, with that one call failing
 *    (and again with every call from that one on failing), then once more
 *    with nothing failing.  A case passes if the driver saw no violation
 *    -- which includes the host touching memory that submitted work not
 *    known to have finished still uses -- no batch was reported complete
 *    before the driver had finished it, the renderer lost no object and
 *    holds no dead handle, it did not stop rendering for good unless the
 *    device was lost or a fence could not be reset, and every draw of the
 *    second, undisturbed run was accepted.
 *
 * Usage: test_vk [-v] [-i] [selftest | helpers | sweep |
 *                          case FN NTH PERSISTENT THREAD [RESULT] |
 *                          random SEED PERMILLE GENTLE]
 *   -i   the device offers the ordered pixel interlock (storage images)
 *   -v   print the renderer's log and each violation
 * With no command: selftest, helpers and sweep.  "case" and "random" repeat
 * one case of the sweep, as its failure summary names them (RESULT 0, or
 * none: the error the specification lists for the call).
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */
#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/error-report.h"
#include "qemu/thread.h"

#include <sched.h>
#include <signal.h>
#include <sys/wait.h>

#include "mockvk.h"

/* Processes here end with _exit(): a coverage build (build.sh COV=1) has to
 * be told to write its counts out first. */
#ifdef VKMOCK_COVERAGE
int __llvm_profile_write_file(void);
#define COVERAGE_WRITE() ((void)__llvm_profile_write_file())
#else
#define COVERAGE_WRITE() ((void)0)
#endif

static int g_verbose;

int qemu_log(const char *fmt, ...)
{
    if (g_verbose) {
        va_list ap;
        va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        va_end(ap);
    }
    return 0;
}

void error_report(const char *fmt, ...)
{
    if (g_verbose) {
        va_list ap;
        va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        va_end(ap);
        fputc('\n', stderr);
    }
}

/*
 * The renderer's pauses between attempts to get an answer from the GPU
 * (g_usleep, shim/qemu/osdep.h): counted, and slept only in tests that keep
 * it waiting for a while.
 */
static unsigned g_pauses;
static bool g_pause_sleeps;

void vkmock_pause(unsigned long us)
{
    __atomic_fetch_add(&g_pauses, 1, __ATOMIC_SEQ_CST);
    if (g_pause_sleeps) {
        usleep(us < 2000 ? us : 2000);
    } else {
        sched_yield();
    }
}

struct tramp {
    void *(*fn)(void *);
    void *arg;
};

static void *tramp(void *p)
{
    struct tramp t = *(struct tramp *)p;

    free(p);
    mock_thread = 1;
    return t.fn(t.arg);
}

void qemu_thread_create(QemuThread *thread, const char *name,
                        void *(*start_routine)(void *), void *arg, int mode)
{
    struct tramp *t = malloc(sizeof(*t));

    t->fn = start_routine;
    t->arg = arg;
    pthread_create(&thread->thread, NULL, tramp, t);
    if (mode == QEMU_THREAD_DETACHED) {
        pthread_detach(thread->thread);
    }
}

/* ------------------------------------------------------------------------ */
#include "ppc_mac_gpu_vulkan.c"
/* ------------------------------------------------------------------------ */

/* What the renderer needs from the R300 code (r300/r300_draw.c, r300_spirv.c). */

const char *r300_stage_entry(R300Stage stage)
{
    return "main";
}

static unsigned g_compiles;

/* A source that says it does not compile, does not. */
uint32_t *r300_glsl_to_spirv(const char *glsl, R300Stage stage, size_t *nwords, char **err)
{
    g_compiles++;
    if (strstr(glsl, "does not compile")) {
        *err = strdup("the stand-in compiler takes it at its word");
        return NULL;
    }
    *nwords = 8;
    return calloc(8, 4);
}

/* The two that read a texture's bytes, which may be VRAM, say so to the
 * driver: the GPU must have finished writing them. */
uint64_t r300_hash_bytes(const uint8_t *p, size_t n)
{
    uint64_t h = 0xcbf29ce484222325ull;

    if (!mock_host_access(p, n, false, "hashing a texture")) {
        return 0;
    }
    for (size_t i = 0; i < n; i++) {
        h = (h ^ p[i]) * 0x100000001b3ull;
    }
    return h;
}

void r300_tex_level_dims(const R300TexDesc *td, uint32_t l, uint32_t *w, uint32_t *h,
                         uint32_t *d)
{
    *w = td->width >> l ? td->width >> l : 1;
    *h = td->height >> l ? td->height >> l : 1;
    if (td->dim == R300_TEXDIM_CUBE) {
        *d = 6;
    } else if (td->dim == R300_TEXDIM_3D) {
        *d = td->depth >> l ? td->depth >> l : 1;
    } else {
        *d = 1;
    }
}

static uint32_t kind_bytes(uint32_t kind)
{
    return kind == R300_TEXK_R8 ? 1 : kind == R300_TEXK_RG8 ? 2 : 4;
}

uint8_t *r300_tex_level_bytes(const R300TexDesc *td, const uint8_t *src, uint32_t l,
                              uint32_t w, uint32_t h, uint32_t *bpr)
{
    /* what the real one reads: a face or slice of the level */
    mock_host_access(src, (size_t)td->lvl_pitch[l] * td->lvl_rows[l], false,
                     "rebuilding a texture");
    if (td->kind >= R300_TEXK_DXT1) {
        uint32_t bs = td->kind == R300_TEXK_DXT1 ? 8 : 16;
        *bpr = (w + 3) / 4 * bs;
        return calloc((size_t)*bpr * ((h + 3) / 4), 1);
    }
    *bpr = w * kind_bytes(td->kind);
    return calloc((size_t)*bpr * h, 1);
}

/* ---- the renderer under test ------------------------------------------ */

#define VRAM_SIZE   (4u << 20)

static PPCMacGPURenderer *R;
static void *OPQ;
static uint8_t *VRAM;
static uint64_t g_dirty[16];    /* VRAM the CPU wrote: the next dirty-log read reports it */
static unsigned g_ndirty;
static int g_pass;              /* which run of the scenario */

static void dirty_fn(void *arg, unsigned long *bitmap, uint64_t npages)
{
    for (unsigned i = 0; i < g_ndirty; i++) {
        uint64_t pg = g_dirty[i] / 4096;
        bitmap[pg / BITS_PER_LONG] |= 1ul << (pg % BITS_PER_LONG);
    }
    g_ndirty = 0;
}

static void cpu_wrote(uint64_t addr)
{
    if (g_ndirty < ARRAY_SIZE(g_dirty)) {
        g_dirty[g_ndirty++] = addr;
    }
}

/*
 * Completions.  A batch committed with a callback (submit) is reported
 * complete by the renderer's thread, once, in the order of the commits,
 * and not before the driver has finished everything submitted up to it:
 * the callback notes how far the driver had got, and check_done() compares
 * that with the submissions made by the time of the commit.
 */
#define MAX_SEQ 4096
static unsigned g_seq_sub[MAX_SEQ];     /* by batch number */
static struct {
    uint32_t seq;
    unsigned upto;
} g_done[1024];
static unsigned g_ndone, g_ncallback;   /* completions reported, and asked for */

static void done_fn(void *arg, uint32_t seq)
{
    unsigned i = g_ndone;               /* (only the renderer's thread writes it) */

    if (i < ARRAY_SIZE(g_done)) {
        g_done[i].upto = mock_finished_upto();
        g_done[i].seq = seq;
    }
    __atomic_store_n(&g_ndone, i + 1, __ATOMIC_SEQ_CST);
}

static unsigned ndone(void)
{
    return __atomic_load_n(&g_ndone, __ATOMIC_SEQ_CST);
}

static void check_done(void)
{
    static unsigned seen;
    unsigned n = ndone();

    for (unsigned i = seen; i < n && i < ARRAY_SIZE(g_done); i++) {
        unsigned need = g_seq_sub[g_done[i].seq % MAX_SEQ];

        if (g_done[i].upto < need) {
            mock_violation_add("batch %u was reported complete when the driver had finished "
                               "%u of the %u submissions made by its commit", g_done[i].seq,
                               g_done[i].upto, need);
        }
        if (i && (int32_t)(g_done[i].seq - g_done[i - 1].seq) <= 0) {
            mock_violation_add("completions out of order: batch %u reported after batch %u",
                               g_done[i].seq, g_done[i - 1].seq);
        }
    }
    seen = n;
}

/*
 * What becomes of the draws the renderer accepts.  A batch whose recording
 * or submission fails is cancelled, and every draw already accepted into
 * it is lost with it: that is counted apart from the draws refused.
 * before() and after() go around each call that may commit the open batch.
 */
static unsigned g_open_draws, g_marks;
static uint32_t g_open_seq;
static bool g_was_stopped;
static unsigned g_cancelled[2], g_lost_draws[2];

static unsigned cancel_marks(void)
{
    return mock_failed(MF_EndCommandBuffer) + mock_failed(MF_QueueSubmit);
}

static void before(void)
{
    g_open_seq = V.cur ? V.cur->seq : 0;
    g_marks = cancel_marks();
    g_was_stopped = qatomic_read(&V.lost);
}

static void after(void)
{
    if (g_open_seq && (!V.cur || V.cur->seq != g_open_seq)) {
        if (cancel_marks() != g_marks || g_was_stopped) {
            g_cancelled[g_pass]++;
            g_lost_draws[g_pass] += g_open_draws;
        }
        g_open_draws = 0;
    }
}

static bool renderer_start(void)
{
    VRAM = ppc_mac_gpu_vulkan_alloc_vram(VRAM_SIZE, &OPQ);
    if (!VRAM) {
        return false;
    }
    /* the host and the GPU share it piecemeal (mockvk.h) */
    mock_shared_buffer(V.vram_buf);
    R = ppc_mac_gpu_renderer_vulkan();
    if (!R->init(VRAM, VRAM_SIZE)) {
        return false;
    }
    R->set_dirty_source(OPQ, dirty_fn, NULL);
    return true;
}

/* Wait until the renderer's thread has finished with everything committed. */
static void quiesce(void)
{
    for (;;) {
        bool empty;
        qemu_mutex_lock(&V.lock);
        empty = g_queue_is_empty(V.waitq);
        qemu_mutex_unlock(&V.lock);
        if (empty) {
            return;
        }
        usleep(20);
    }
}

/* flush_r200: when it returns the device takes all work for finished. */
static void flush_now(void)
{
    unsigned n;

    before();
    R->flush_r200(OPQ);
    after();
    n = mock_running();
    if (n && !mock_device_lost()) {
        mock_violation_add("flush_r200 returned with %u submissions not known to have "
                           "finished", n);
    }
}

/*
 * With a batch in flight: is all the VRAM it uses reported busy?  The
 * device's question for every page, and what it would do on the answer --
 * write where it is told the page is free, read where it is told that the
 * batch only reads it.
 */
static void probe_pages(void)
{
    for (uint64_t a = 0; a < VRAM_SIZE; a += 4096) {
        if (!R->range_busy_r200(OPQ, a, a + 4096, true)) {
            mock_host_access(VRAM + a, 4096, true, "the device writing VRAM");
        } else if (!R->range_busy_r200(OPQ, a, a + 4096, false)) {
            mock_host_access(VRAM + a, 4096, false, "the device reading VRAM");
        }
    }
}

static bool g_probe;            /* flush(): every batch is probed in flight first */
static uint32_t submit(void);

static void flush(void)
{
    if (g_probe && V.cur) {
        mock_hold(true);                /* the GPU does not finish */
        submit();
        probe_pages();
        mock_hold(false);
    }
    flush_now();
    quiesce();
    check_done();
}

/* Commit the open batch and have its completion reported. */
static uint32_t submit(void)
{
    uint32_t seq;

    before();
    seq = R->submit_r200(OPQ, done_fn, NULL);
    after();
    if (seq) {
        g_seq_sub[seq % MAX_SEQ] = mock_submitted();
        g_ncallback++;
    }
    return seq;
}

/*
 * The device itself reads or writes VRAM [lo, hi), the way it goes about
 * it (r200_vram_access): if work batched or in flight uses the range, that
 * is finished first.
 */
static void device_access(uint64_t lo, uint64_t hi, bool write)
{
    if (R->range_busy_r200(OPQ, lo, hi, write)) {
        flush_now();
    }
    if (!mock_host_access(VRAM + lo, hi - lo, write,
                          write ? "the device writing VRAM" : "the device reading VRAM")) {
        return;
    }
    if (write) {
        VRAM[lo]++;
        cpu_wrote(lo);
    } else {
        volatile uint8_t v = VRAM[lo];
        (void)v;
    }
}

static uint32_t zpass(bool reset)
{
    uint32_t v;

    before();
    v = R->zpass_r300(OPQ, reset, 0);
    after();
    return v;
}

/* ---- packets ----------------------------------------------------------- */

static R300Vertex g_verts[8];
static float g_vs_in[3][4];
static uint32_t g_vs_idx[3] = { 0, 1, 2 };
static R300VSUniforms g_vsu;
static uint8_t g_host_tex[256];
#define BIG_VECS 460000         /* 7.36 MB of an 8 MB arena chunk */
static float g_big_vs_in[BIG_VECS][4];
#define HUGE_VECS 600000        /* 9.6 MB: more than an arena chunk */
static float g_huge_vs_in[HUGE_VECS][4];
static uint8_t g_big_tex[512 * 512 * 4 + 256 * 256 * 4];
static char g_glsl[4][32] = { "fragment program A", "fragment program B",
                              "fragment program C", "fragment program D" };
static char g_vs_glsl[] = "vertex program";

static void pkt_rt(R300DrawPacket *p, uint32_t addr, uint32_t w, uint32_t h)
{
    memset(p, 0, sizeof(*p));
    p->rt_gpu_addr = addr;
    p->rt_pitch = w;
    p->rt_width = w;
    p->rt_height = h;
    p->rt_bpp = 4;
    p->rt_view = R300_RTV_RGBA8;
    p->scissor[2] = w;
    p->scissor[3] = h;
    p->num_cb = 1;
    p->aa_samples = 1;
    p->glsl = g_glsl[0];
    p->glsl_id = 1;
    p->verts = g_verts;
    p->num_verts = 3;
}

static void pkt_z(R300DrawPacket *p, uint32_t addr, uint32_t bpp)
{
    p->depth.attach = true;
    p->depth.gpu_addr = addr;
    p->depth.pitch = p->rt_pitch;
    p->depth.bpp = bpp;
}

/* A texture of `levels` mip levels laid out one after another. */
static void tex(R300TexDesc *t, uint32_t addr, uint32_t w, uint32_t h, uint32_t kind,
                uint32_t dim, uint32_t depth, uint32_t levels)
{
    bool dxt = kind >= R300_TEXK_DXT1;
    uint32_t bs = kind == R300_TEXK_DXT1 ? 8 : 16;
    uint32_t bpp = kind == R300_TEXK_CONVERT16 ? 2 : kind_bytes(kind);
    uint32_t off = 0;

    memset(t, 0, sizeof(*t));
    t->bound = true;
    t->gpu_addr = addr;
    t->width = w;
    t->height = h;
    t->kind = kind;
    t->dim = dim;
    t->depth = depth;
    t->levels = levels;
    t->filter0 = 0x1200 + kind;
    for (uint32_t l = 0; l < levels; l++) {
        uint32_t lw, lh, n;
        r300_tex_level_dims(t, l, &lw, &lh, &n);
        t->lvl_off[l] = off;
        t->lvl_pitch[l] = dxt ? (lw + 3) / 4 * bs : lw * bpp;
        t->lvl_rows[l] = dxt ? (lh + 3) / 4 : lh;
        off += t->lvl_pitch[l] * t->lvl_rows[l] * n;
    }
    t->pitch_bytes = t->lvl_pitch[0];
    t->size_bytes = off;
}

/* Texels the shader decodes itself: copied out of the GART (addr 0), or in VRAM. */
static void tex_raw(R300TexDesc *t, uint32_t addr)
{
    memset(t, 0, sizeof(*t));
    t->bound = true;
    t->kind = R300_TEXK_RAW;
    t->width = t->height = 8;
    t->view_bpp = 4;
    t->levels = 1;
    t->gpu_addr = addr;
    t->host_data = addr ? NULL : g_host_tex;
    t->size_bytes = sizeof(g_host_tex);
    t->pitch_bytes = 32;
}

/* ---- the scenario ------------------------------------------------------ */

#define MAX_STEPS 2048
static int g_res[2][MAX_STEPS];
static unsigned g_nres[2];

static int draw_one(const R300DrawPacket *p)
{
    int r;

    /* what its shader reads of VRAM through the storage buffer */
    for (int t = 0; t < R300_NUM_TEX_UNITS; t++) {
        const R300TexDesc *td = &p->tex[t];
        if (td->bound && td->kind == R300_TEXK_RAW && !td->host_data) {
            mock_shader_reads(V.vram_buf, td->gpu_addr, td->size_bytes);
        }
    }
    before();
    r = R->draw_r300(OPQ, VRAM, VRAM_SIZE, p);
    after();
    mock_shader_reads_clear();
    if (r == 0 && V.cur && !qatomic_read(&V.lost)) {
        g_open_draws++;
    }
    return r;
}

static void draw(const R300DrawPacket *p)
{
    int r = draw_one(p);

    if (g_nres[g_pass] < MAX_STEPS) {
        g_res[g_pass][g_nres[g_pass]++] = r;
    }
}

static void scenario(void)
{
    R300DrawPacket p;

    /* a plain triangle, with a depth buffer; then one without, in the same
     * render pass */
    pkt_rt(&p, 0x000000, 64, 64);
    pkt_z(&p, 0x010000, 4);
    draw(&p);
    pkt_rt(&p, 0x000000, 64, 64);
    draw(&p);
    /* the device reads what they drew (a scanout, a 2D copy) */
    device_access(0x000000, 0x004000, false);

    /* every way a texture reaches the GPU */
    pkt_rt(&p, 0x000000, 64, 64);
    pkt_z(&p, 0x010000, 4);
    p.glsl = g_glsl[1];
    p.glsl_id = 2;
    tex(&p.tex[0], 0x030000, 32, 32, R300_TEXK_RGBA8, R300_TEXDIM_2D, 1, 1);
    tex(&p.tex[1], 0x040000, 16, 16, R300_TEXK_RGBA8, R300_TEXDIM_2D, 1, 3);
    tex(&p.tex[2], 0x050000, 8, 8, R300_TEXK_RGBA8, R300_TEXDIM_CUBE, 1, 2);
    tex(&p.tex[3], 0x060000, 8, 8, R300_TEXK_RGBA8, R300_TEXDIM_3D, 4, 2);
    tex_raw(&p.tex[4], 0);
    tex(&p.tex[5], 0x070000, 16, 16, R300_TEXK_DXT1, R300_TEXDIM_2D, 1, 2);
    tex(&p.tex[6], 0x080000, 16, 16, R300_TEXK_CONVERT16, R300_TEXDIM_2D, 1, 1);
    tex(&p.tex[7], 0x088000, 16, 16, R300_TEXK_R8, R300_TEXDIM_2D, 1, 1);
    tex_raw(&p.tex[8], 0x0c0000);
    tex(&p.tex[9], 0x08c000, 16, 16, R300_TEXK_RG8, R300_TEXDIM_2D, 1, 1);
    tex(&p.tex[10], 0x074000, 16, 16, R300_TEXK_DXT3, R300_TEXDIM_2D, 1, 1);
    tex(&p.tex[11], 0x078000, 16, 16, R300_TEXK_DXT5, R300_TEXDIM_2D, 1, 2);
    p.tex[11].filter0 = 0x1200 | 6 | 1 << 3 | 3 << 9 | 4 << 21;   /* border, mirror, anisotropic */
    draw(&p);
    /* the device writes the first texture, which the batch still has to
     * load: it is loaded again */
    device_access(0x030000, 0x030400, true);
    draw(&p);
    flush();

    /*
     * A rebuilt texture when the arena chunk is nearly used up (by a draw's
     * vertex data): its first level needs a new chunk, its second would
     * still fit the old one.
     */
    pkt_rt(&p, 0x000000, 64, 64);
    p.verts = NULL;
    p.vs_glsl = g_vs_glsl;
    p.vs_id = 9;
    p.vs_in = g_big_vs_in;
    p.vs_in_vecs = BIG_VECS;
    p.vs_idx = g_vs_idx;
    p.vs_u = &g_vsu;
    draw(&p);
    pkt_rt(&p, 0x000000, 64, 64);
    p.glsl = g_glsl[1];
    p.glsl_id = 2;
    tex(&p.tex[0], 0, 512, 512, R300_TEXK_RGBA8, R300_TEXDIM_2D, 1, 2);
    p.tex[0].host_data = g_big_tex;
    draw(&p);
    draw(&p);
    flush();

    /* more vertex data than an arena chunk holds */
    pkt_rt(&p, 0x000000, 64, 64);
    p.verts = NULL;
    p.vs_glsl = g_vs_glsl;
    p.vs_id = 9;
    p.vs_in = g_huge_vs_in;
    p.vs_in_vecs = HUGE_VECS;
    p.vs_idx = g_vs_idx;
    p.vs_u = &g_vsu;
    draw(&p);

    /* rendering into a buffer that the next draw samples as a rebuilt
     * texture: the batch has to be finished in the middle of that draw */
    pkt_rt(&p, 0x0b0000, 16, 16);
    draw(&p);
    pkt_rt(&p, 0x000000, 64, 64);
    p.glsl = g_glsl[1];
    p.glsl_id = 2;
    tex(&p.tex[0], 0x0b0000, 16, 16, R300_TEXK_CONVERT16, R300_TEXDIM_2D, 1, 1);
    draw(&p);
    flush();

    /* filled polygons with their edges as lines; other culling */
    pkt_rt(&p, 0x000000, 64, 64);
    p.num_line_verts = 2;
    p.cull = R300_CULL_BACK;
    draw(&p);
    pkt_rt(&p, 0x000000, 64, 64);
    p.prim_class = 1;
    p.num_verts = 2;
    draw(&p);

    /* vertex shading on the GPU */
    pkt_rt(&p, 0x000000, 64, 64);
    pkt_z(&p, 0x010000, 4);
    p.verts = NULL;
    p.vs_glsl = g_vs_glsl;
    p.vs_id = 9;
    p.vs_in = g_vs_in;
    p.vs_in_vecs = 3;
    p.vs_idx = g_vs_idx;
    p.vs_u = &g_vsu;
    draw(&p);

    /* two colour buffers and a 16-bit depth buffer */
    pkt_rt(&p, 0x000000, 64, 64);
    pkt_z(&p, 0x018000, 2);
    p.glsl = g_glsl[2];
    p.glsl_id = 3;
    p.num_cb = 2;
    p.cb[1].gpu_addr = 0x020000;
    p.cb[1].pitch = 64;
    p.cb[1].bpp = 4;
    p.cb[1].view = R300_RTV_R32U;
    draw(&p);

    /* two samples per pixel */
    pkt_rt(&p, 0x090000, 32, 32);
    pkt_z(&p, 0x0a0000, 4);
    p.aa_samples = 2;
    draw(&p);

    /* the Z-pass counter read with a batch open */
    zpass(false);

    /* completion reported through the callback */
    pkt_rt(&p, 0x000000, 64, 64);
    draw(&p);
    submit();
    quiesce();
    check_done();

    /* several batches in flight at once; the device writes VRAM that none
     * of them uses, then VRAM that they render to */
    mock_gate(true);
    for (int i = 0; i < 4; i++) {
        pkt_rt(&p, 0x000000, 64, 64);
        pkt_z(&p, 0x010000, 4);
        draw(&p);
        submit();
    }
    device_access(0x3f0000, 0x3f1000, true);
    device_access(0x000000, 0x001000, true);
    mock_gate(false);
    quiesce();
    check_done();

    /* more draws in a batch than one descriptor pool holds */
    for (int i = 0; i < VK_SETS_PER_POOL + 40; i++) {
        pkt_rt(&p, 0x000000, 64, 64);
        draw(&p);
    }
    flush();

    /* more buffers than the image cache and the framebuffer cache hold, all
     * rendered to in one batch: the ones dropped have not been written back */
    for (int i = 0; i < VK_MAX_IMG + 12; i++) {
        pkt_rt(&p, 0x100000 + i * 0x1000, 8, 8);
        draw(&p);
    }
    flush();

    /* more rebuilt textures than their cache holds */
    for (int i = 0; i < VK_MAX_TEXFULL + 12; i++) {
        pkt_rt(&p, 0x000000, 64, 64);
        p.glsl = g_glsl[3];
        p.glsl_id = 4;
        tex(&p.tex[0], 0x200000 + i * 0x1000, 8, 8, R300_TEXK_RGBA8, R300_TEXDIM_2D, 1, 2);
        VRAM[0x200000 + i * 0x1000] = i + 1;        /* each its own contents */
        draw(&p);
        if (i % 20 == 19) {
            flush();
        }
    }
    zpass(true);
    R->gpu_failures(OPQ);
    R->get_caps(OPQ);
    flush();
}

/* ---- what the renderer holds ------------------------------------------- */

static void mark_trash(const VkTrash *t)
{
    static const int type[] = { MT_IMAGE, MT_VIEW, MT_MEMORY, MT_FRAMEBUFFER, MT_BUFFER };

    mock_mark((void *)(uintptr_t)t->h, type[t->type], "a deferred destruction");
}

static void mark_chunk(const VkChunk *c)
{
    mock_mark(c->buf, MT_BUFFER, "an arena chunk's buffer");
    mock_mark(c->mem, MT_MEMORY, "an arena chunk's memory");
}

/* Mark every handle the renderer's state refers to. */
static void mark_held(void)
{
    GHashTableIter it;
    gpointer k, v;

    mock_mark_begin();
    mock_mark(V.inst, MT_INSTANCE, "V.inst");
    mock_mark(V.pdev, MT_PHYSDEV, "V.pdev");
    mock_mark(V.dev, MT_DEVICE, "V.dev");
    mock_mark(V.queue, MT_QUEUE, "V.queue");
    mock_mark(V.cmdpool, MT_CMDPOOL, "V.cmdpool");
    mock_mark(V.dsl, MT_DSL, "V.dsl");
    mock_mark(V.layout, MT_PLAYOUT, "V.layout");
    mock_mark(V.vram_buf, MT_BUFFER, "V.vram_buf");
    mock_mark(V.vram_mem, MT_MEMORY, "V.vram_mem");
    mock_mark(V.zpass_buf, MT_BUFFER, "V.zpass_buf");
    mock_mark(V.zpass_mem, MT_MEMORY, "V.zpass_mem");
    for (int i = 0; i < VK_NBATCH; i++) {
        VkBatch *b = &V.batch[i];
        mock_mark(b->cb, MT_CMDBUF, "a batch's command buffer");
        mock_mark(b->fence, MT_FENCE, "a batch's fence");
        for (guint j = 0; b->pools && j < b->pools->len; j++) {
            mock_mark(g_array_index(b->pools, VkDescriptorPool, j), MT_DPOOL,
                      "a batch's descriptor pool");
        }
        for (guint j = 0; b->chunks && j < b->chunks->len; j++) {
            mark_chunk(g_ptr_array_index(b->chunks, j));
        }
        for (guint j = 0; b->trash && j < b->trash->len; j++) {
            mark_trash(&g_array_index(b->trash, VkTrash, j));
        }
    }
    for (guint j = 0; V.free_chunks && j < V.free_chunks->len; j++) {
        mark_chunk(g_ptr_array_index(V.free_chunks, j));
    }
    for (int i = 0; i < VK_MAX_IMG; i++) {
        if (V.img[i].live) {
            mock_mark(V.img[i].img, MT_IMAGE, "a cached image");
            mock_mark(V.img[i].mem, MT_MEMORY, "a cached image's memory");
            mock_mark(V.img[i].view, MT_VIEW, "a cached image's view");
        }
    }
    for (int i = 0; i < VK_MAX_TEXFULL; i++) {
        if (V.texfull[i].live) {
            mock_mark(V.texfull[i].img, MT_IMAGE, "a rebuilt texture");
            mock_mark(V.texfull[i].mem, MT_MEMORY, "a rebuilt texture's memory");
            mock_mark(V.texfull[i].view, MT_VIEW, "a rebuilt texture's view");
        }
    }
    for (unsigned i = 0; i < V.nrp; i++) {
        mock_mark(V.rps[i].rp, MT_RENDERPASS, "a cached render pass");
    }
    for (int i = 0; i < VK_MAX_FB; i++) {
        mock_mark(V.fbs[i].fb, MT_FRAMEBUFFER, "a cached framebuffer");
    }
    if (V.progs) {
        g_hash_table_iter_init(&it, V.progs);
        while (g_hash_table_iter_next(&it, &k, &v)) {
            VkProg *pg = v;
            for (int s = 0; s < 3; s++) {
                mock_mark(pg->mod[s], MT_SHADER, "a program's shader module");
            }
            for (guint j = 0; j < pg->pipes->len; j++) {
                mock_mark(g_array_index(pg->pipes, VkPipeVar, j).pipe, MT_PIPELINE,
                          "a cached pipeline");
            }
        }
    }
    if (V.vs_mods) {
        g_hash_table_iter_init(&it, V.vs_mods);
        while (g_hash_table_iter_next(&it, &k, &v)) {
            mock_mark(v, MT_SHADER, "a vertex program's shader module");
        }
    }
    if (V.samplers) {
        g_hash_table_iter_init(&it, V.samplers);
        while (g_hash_table_iter_next(&it, &k, &v)) {
            mock_mark(v, MT_SAMPLER, "a cached sampler");
        }
    }
    for (int i = 0; i < 3; i++) {
        mock_mark(V.dummy_img[i], MT_IMAGE, "a stand-in texture");
        mock_mark(V.dummy_mem[i], MT_MEMORY, "a stand-in texture's memory");
        mock_mark(V.dummy_view[i], MT_VIEW, "a stand-in texture's view");
    }
}

static const int g_leak_types[] = {
    MT_CMDBUF, MT_FENCE, MT_BUFFER, MT_MEMORY, MT_IMAGE, MT_VIEW, MT_RENDERPASS,
    MT_FRAMEBUFFER, MT_SHADER, MT_PIPELINE, MT_SAMPLER, MT_DPOOL,
};

/* Objects alive that the renderer no longer refers to. */
static unsigned count_leaks(char *desc, size_t n)
{
    unsigned total = 0;
    size_t len = 0;

    mark_held();
    desc[0] = 0;
    for (unsigned i = 0; i < ARRAY_SIZE(g_leak_types); i++) {
        unsigned k = mock_unmarked(g_leak_types[i]);
        if (k && len < n) {
            len += snprintf(desc + len, n - len, "%s%u %s", total ? ", " : "", k,
                            mock_type_name[g_leak_types[i]]);
        }
        total += k;
    }
    return total;
}

/* ---- one run in its own process ---------------------------------------- */

typedef struct Result {
    int done, init_ok, lost;
    int crashed;                        /* the signal that ended the run, if one did */
    unsigned fired, nviol, leaks, rejected[2], ndraw[2];
    unsigned cancelled[2], lost_draws[2];   /* batches cancelled, accepted draws in them */
    unsigned pauses;                    /* the renderer's, waiting for an answer */
    unsigned calls[MF_COUNT][3];        /* in the first run of the scenario */
    unsigned failable;                  /* ... failable calls on the main thread */
    char leak_desc[160];
    char viol[3][320];
} Result;

static int g_result_fd = -1;
static Result g_result;

static void result_send(void)
{
    g_result.fired = mock_fired();
    g_result.nviol = mock_violations();
    for (unsigned i = 0; i < 3; i++) {
        snprintf(g_result.viol[i], sizeof(g_result.viol[i]), "%s", mock_violation(i));
    }
    if (g_result_fd >= 0) {
        ssize_t w = write(g_result_fd, &g_result, sizeof(g_result));
        (void)w;
    }
    COVERAGE_WRITE();
}

static void budget_exceeded(void)
{
    result_send();
    _exit(0);
}

/* Report a crash ourselves, rather than leave it to the system's crash
 * reporter (slow, and a report file for every case of a broken renderer). */
static void crashed(int sig)
{
    g_result.crashed = sig;
    result_send();
    _exit(0);
}

/*
 * A fault may be the host touching memory the driver has guarded: it has
 * reported that, and the access goes on.  Anything else is a crash, which
 * `crash` reports.
 */
static void (*g_crash)(int);

static void on_signal(int sig, siginfo_t *si, void *uc)
{
    if ((sig == SIGSEGV || sig == SIGBUS) && mock_trap(si->si_addr)) {
        return;
    }
    g_crash(sig);
}

static void catch_signals(void (*crash)(int))
{
    static const int sigs[] = { SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT, SIGALRM };
    struct sigaction sa = { .sa_sigaction = on_signal, .sa_flags = SA_SIGINFO };

    g_crash = crash;
    sigemptyset(&sa.sa_mask);
    for (unsigned i = 0; i < ARRAY_SIZE(sigs); i++) {
        sigaction(sigs[i], &sa, NULL);
    }
}

typedef struct Case {
    int fn;                     /* -1: nothing fails; MF_COUNT: every call */
    unsigned nth;
    bool persistent;
    int thread;
    VkResult res;
    /* or calls failing at random (fn -1): */
    unsigned permille, seed;
    bool gentle;                /* ... but not the ones that end rendering for good */
} Case;

/* The one failure that ends rendering without the device being lost. */
#define ENDS_RENDERING (1ul << MF_ResetFences)

static void case_body(const Case *c)
{
    mock_set_budget(4000000, budget_exceeded);
    /* a wait that keeps failing is let through the fourth time, or a run
     * in which every wait fails would not end: the renderer waits it out */
    mock_wait_patience(3);
    if (c->fn >= 0) {
        mock_plan(c->fn, c->nth, c->persistent, c->res, c->thread);
    }
    g_result.init_ok = renderer_start();
    if (g_result.init_ok) {
        if (c->permille) {
            /* once it runs: start-up mostly would not get through these, and
             * each of its calls has its own cases */
            mock_plan_random(c->seed, c->permille, c->gentle ? ENDS_RENDERING : 0);
        }
        g_pass = 0;
        scenario();
        for (int f = 0; f < MF_COUNT; f++) {
            for (int t = 0; t < 3; t++) {
                g_result.calls[f][t] = mock_calls(f, t);
            }
        }
        g_result.failable = mock_failable_calls();
        mock_plan_clear();
        g_pass = 1;
        scenario();
        quiesce();
        check_done();
        if (ndone() != g_ncallback) {
            mock_violation_add("%u completions reported for %u batches committed with a "
                               "callback", ndone(), g_ncallback);
        }
        g_result.pauses = g_pauses;
        g_result.lost = qatomic_read(&V.lost);
        if (!g_result.lost) {
            /* what the last batches left to destroy (the renderer does this
             * when it starts the next batch, which after a loss it never does) */
            vk_reap();
        }
        g_result.leaks = count_leaks(g_result.leak_desc, sizeof(g_result.leak_desc));
        for (int p = 0; p < 2; p++) {
            g_result.ndraw[p] = g_nres[p];
            g_result.cancelled[p] = g_cancelled[p];
            g_result.lost_draws[p] = g_lost_draws[p];
            for (unsigned i = 0; i < g_nres[p]; i++) {
                g_result.rejected[p] += g_res[p][i] != 0;
            }
        }
    }
    g_result.done = 1;
    result_send();
}

/* Returns 0, or the signal that ended the run. */
static int run_case(const Case *c, Result *r)
{
    int fd[2], status = 0;
    pid_t pid;

    memset(r, 0, sizeof(*r));
    if (pipe(fd)) {
        perror("pipe");
        exit(2);
    }
    fflush(NULL);
    pid = fork();
    if (pid < 0) {
        perror("fork");
        exit(2);
    }
    if (pid == 0) {
        close(fd[0]);
        g_result_fd = fd[1];
        catch_signals(crashed);
        alarm(60);
        case_body(c);
        _exit(0);
    }
    close(fd[1]);
    for (size_t got = 0; got < sizeof(*r);) {
        ssize_t n = read(fd[0], (char *)r + got, sizeof(*r) - got);
        if (n <= 0) {
            break;
        }
        got += n;
    }
    close(fd[0]);
    waitpid(pid, &status, 0);
    return WIFSIGNALED(status) ? WTERMSIG(status) : r->crashed;
}

/* May the run end with rendering stopped for good?  Only for a lost device,
 * or a fence that could not be reset: not for a wait that gave no answer. */
static bool lost_expected(const Case *c)
{
    if (c->permille) {
        return !c->gentle;
    }
    return c->fn == MF_COUNT || c->fn == MF_ResetFences ||
           ((c->fn == MF_QueueSubmit || c->fn == MF_WaitForFences) &&
            c->res == VK_ERROR_DEVICE_LOST);
}

/* Why a run fails, or NULL. */
static const char *judge(const Case *c, const Result *r, int sig, char *buf, size_t n)
{
    if (sig == SIGALRM) {
        return "the renderer hung";
    }
    if (sig) {
        snprintf(buf, n, "the renderer crashed (signal %d)", sig);
        return buf;
    }
    if (r->nviol) {
        snprintf(buf, n, "%s", r->viol[0]);
        return buf;
    }
    if (!r->done) {
        return "the run did not finish";
    }
    if (!r->init_ok) {
        /* start-up failed, cleanly */
        return c->fn < 0 && !c->permille ? "the renderer did not start" : NULL;
    }
    if (r->leaks) {
        snprintf(buf, n, "lost objects: %s", r->leak_desc);
        return buf;
    }
    if (r->lost && !lost_expected(c)) {
        return "rendering stopped for good, though the device was not lost";
    }
    if (!r->lost && r->fired && c->res == VK_ERROR_DEVICE_LOST &&
        (c->fn == MF_QueueSubmit || c->fn == MF_WaitForFences)) {
        return "the device was reported lost and rendering goes on";
    }
    if (c->fn >= 0 && c->fn < MF_COUNT && !c->persistent && !r->lost) {
        /* one call failed once: at most one draw is refused for it, or (its
         * recording or its submission failing) one batch cancelled */
        if (r->rejected[0] > 1) {
            snprintf(buf, n, "%u draws rejected after one call failed once", r->rejected[0]);
            return buf;
        }
        if (r->cancelled[0] > 1) {
            snprintf(buf, n, "%u batches cancelled after one call failed once", r->cancelled[0]);
            return buf;
        }
        if (r->cancelled[0] && c->fn != MF_EndCommandBuffer && c->fn != MF_QueueSubmit) {
            return "a batch was cancelled, though neither its recording nor its submission failed";
        }
    }
    if (!r->lost && r->rejected[1]) {
        snprintf(buf, n, "%u of %u draws rejected in the run with nothing failing",
                 r->rejected[1], r->ndraw[1]);
        return buf;
    }
    if (!r->lost && r->cancelled[1]) {
        return "a batch cancelled in the run with nothing failing";
    }
    if (c->fn < 0 && !c->permille && r->rejected[0]) {
        return "draws rejected with nothing failing";
    }
    return NULL;
}

/* Runs of digits to '#', so that messages differing in numbers count as one. */
static void blur(char *s)
{
    char *d = s;
    bool run = false;

    for (; *s; s++) {
        bool digit = *s >= '0' && *s <= '9';
        if (!digit || !run) {
            *d++ = digit ? '#' : *s;
        }
        run = digit;
    }
    *d = 0;
}

static struct {
    char msg[320], first[64];
    unsigned n;
} g_why[96];
static unsigned g_nwhy;

/*
 * What one call failing once costs, over the cases in which it did and
 * rendering went on: draws refused (the draw call says so), and draws that
 * were accepted into a batch which was then cancelled (it does not).
 */
static struct {
    unsigned cases, refused_max, cancel_cases, lost_max;
    unsigned long lost_total;
} g_cost;

/* Run one case; false, with the reason noted under `label`, if it fails. */
static bool sweep_case(const char *label, const Case *c, unsigned *unreached)
{
    Result r;
    char buf[320], m[320];
    int sig = run_case(c, &r);
    const char *bad = judge(c, &r, sig, buf, sizeof(buf));
    unsigned w;

    if (!bad) {
        *unreached += !sig && !r.fired;
        if (c->fn >= 0 && c->fn < MF_COUNT && !c->persistent && r.fired && !r.lost) {
            g_cost.cases++;
            g_cost.refused_max = MAX(g_cost.refused_max, r.rejected[0]);
            if (r.cancelled[0]) {
                g_cost.cancel_cases++;
                g_cost.lost_total += r.lost_draws[0];
                g_cost.lost_max = MAX(g_cost.lost_max, r.lost_draws[0]);
            }
        }
        return true;
    }
    snprintf(m, sizeof(m), "%s: %s", label, bad);
    blur(m);
    for (w = 0; w < g_nwhy && strcmp(g_why[w].msg, m); w++) {
    }
    if (w == g_nwhy && g_nwhy < ARRAY_SIZE(g_why)) {
        snprintf(g_why[w].msg, sizeof(g_why[w].msg), "%s", m);
        if (c->permille) {
            snprintf(g_why[w].first, sizeof(g_why[w].first), "random %u %u %d", c->seed,
                     c->permille, c->gentle);
        } else {
            snprintf(g_why[w].first, sizeof(g_why[w].first), "case %d %u %d %d %d", c->fn,
                     c->nth, c->persistent, c->thread, (int)c->res);
        }
        g_nwhy++;
    }
    if (w < g_nwhy) {
        g_why[w].n++;
    }
    return false;
}

static void sweep_row(const char *label, unsigned ncase, unsigned nfail, unsigned *total,
                      unsigned *failed)
{
    if (ncase) {
        printf("%-40s %6u %6u %6u\n", label, ncase, ncase - nfail, nfail);
    }
    *total += ncase;
    *failed += nfail;
}

/* One call failing, once and from then on, for each time the scenario makes it. */
static void sweep_call(const char *label, int fn, VkResult res, const Result *base,
                       unsigned *ncase, unsigned *nfail, unsigned *unreached)
{
    for (int t = 0; t < 2; t++) {
        for (unsigned k = 1; k <= base->calls[fn][t]; k++) {
            for (int persistent = 0; persistent < 2; persistent++) {
                Case c = { fn, k, persistent, t, res };
                ++*ncase;
                *nfail += !sweep_case(label, &c, unreached);
            }
        }
    }
}

static int sweep(void)
{
    /*
     * Results the specification lists for a call beyond the one every case
     * of it gets (mock_listed_error), where the renderer acts on which it
     * is.  VK_TIMEOUT is no error: a wait that came back early.
     */
    static const struct { int fn; VkResult res; } more[] = {
        { MF_QueueSubmit, VK_ERROR_OUT_OF_DEVICE_MEMORY },
        { MF_QueueSubmit, VK_ERROR_DEVICE_LOST },
        { MF_WaitForFences, VK_ERROR_OUT_OF_DEVICE_MEMORY },
        { MF_WaitForFences, VK_ERROR_DEVICE_LOST },
        { MF_WaitForFences, VK_ERROR_UNKNOWN },
        { MF_WaitForFences, VK_TIMEOUT },
        { MF_AllocateDescriptorSets, VK_ERROR_OUT_OF_DEVICE_MEMORY },
    };
    /* And errors it does not list for the call: beyond what a driver may do. */
    static const struct { int fn; VkResult res; } unlisted[] = {
        { MF_ResetFences, VK_ERROR_OUT_OF_HOST_MEMORY },
        { MF_ResetCommandBuffer, VK_ERROR_OUT_OF_HOST_MEMORY },
    };
    Case none = { -1 };
    Result base;
    char buf[320];
    unsigned total = 0, failed = 0, unreached = 0, ncase, nfail;
    int sig = run_case(&none, &base);
    const char *bad = judge(&none, &base, sig, buf, sizeof(buf));

    printf("with nothing failing: %u + %u draws, %s\n", base.ndraw[0], base.ndraw[1],
           bad ? bad : "ok");
    if (bad) {
        return 1;
    }
    printf("%-40s %6s %6s %6s\n", "what fails", "cases", "passed", "failed");

    /* One call, once; and that call from then on: with results the
     * specification lists for it. */
    for (int fn = 0; fn < MF_COUNT; fn++) {
        ncase = nfail = 0;
        sweep_call(mock_fn_name[fn], fn, mock_listed_error(fn), &base, &ncase, &nfail,
                   &unreached);
        for (unsigned i = 0; i < ARRAY_SIZE(more); i++) {
            if (more[i].fn == fn) {
                sweep_call(mock_fn_name[fn], fn, more[i].res, &base, &ncase, &nfail,
                           &unreached);
            }
        }
        sweep_row(mock_fn_name[fn], ncase, nfail, &total, &failed);
    }
    for (unsigned i = 0; i < ARRAY_SIZE(unlisted); i++) {
        char label[64];
        snprintf(label, sizeof(label), "%s, error not listed", mock_fn_name[unlisted[i].fn]);
        ncase = nfail = 0;
        sweep_call(label, unlisted[i].fn, unlisted[i].res, &base, &ncase, &nfail, &unreached);
        sweep_row(label, ncase, nfail, &total, &failed);
    }

    /* Every call from some point on, as when memory runs out: from each of
     * the first 300 failable calls, then from every 7th.  Each fails with
     * the error listed for it. */
    ncase = nfail = 0;
    for (unsigned k = 1; k <= base.failable; k += k < 300 ? 1 : 7) {
        Case c = { MF_COUNT, k, true, 0, VK_SUCCESS };
        ncase++;
        nfail += !sweep_case("every call from some point on", &c, &unreached);
    }
    sweep_row("every call from some point on", ncase, nfail, &total, &failed);

    /* Calls failing at random: 0.5%, 2% and 10% of them, 300 seeds each,
     * with and without the failure that ends rendering for good (a fence
     * that cannot be reset). */
    for (int gentle = 1; gentle >= 0; gentle--) {
        static const unsigned permille[] = { 5, 20, 100 };
        ncase = nfail = 0;
        for (unsigned pi = 0; pi < ARRAY_SIZE(permille); pi++) {
            for (unsigned seed = 1; seed <= 300; seed++) {
                Case c = { -1, .permille = permille[pi], .seed = seed, .gentle = gentle };
                ncase++;
                nfail += !sweep_case(gentle ? "calls at random, rendering can go on"
                                            : "calls at random, any", &c, &unreached);
            }
        }
        sweep_row(gentle ? "calls at random, rendering can go on" : "calls at random, any",
                  ncase, nfail, &total, &failed);
    }

    sweep_row("total", total, failed, &ncase, &nfail);
    if (unreached) {
        printf("%u cases in which no call was made to fail\n", unreached);
    }
    printf("one call failing once, rendering going on (%u cases): at most %u draw refused;\n"
           "  in %u of them the batch being recorded or submitted was cancelled, and with it\n"
           "  draws already accepted: up to %u, %lu in all\n", g_cost.cases,
           g_cost.refused_max, g_cost.cancel_cases, g_cost.lost_max, g_cost.lost_total);
    for (unsigned w = 0; w < g_nwhy; w++) {
        printf("  %5u x %s\n          (first: %s)\n", g_why[w].n, g_why[w].msg, g_why[w].first);
    }
    return failed != 0;
}

/* A value no call returns: outputs are set to it to see that they are written. */
static const void *const SENTINEL = (void *)(uintptr_t)0x5151515151515150ull;

/* ---- the stand-in driver's own test ------------------------------------- */

/*
 * The driver is what judges the renderer, so each thing it is relied on to
 * notice is done to it once, directly, and must be reported; a correct
 * sequence must not be.
 */
static struct {
    VkDevice dev;
    VkQueue queue;
    VkCommandPool cmdpool;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkRenderPass rp;
    VkPipeline pipe;
    VkSampler sampler;
} S;

typedef struct StImage {
    VkImage img;
    VkDeviceMemory mem;
    VkImageView view;
} StImage;

static StImage st_image(void)
{
    VkImageCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                              .imageType = VK_IMAGE_TYPE_2D, .extent = { 4, 4, 1 },
                              .format = VK_FORMAT_R8G8B8A8_UNORM,
                              .mipLevels = 1, .arrayLayers = 1 };
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = 4096 };
    VkImageViewCreateInfo vci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    StImage i;

    vkCreateImage(S.dev, &ici, NULL, &i.img);
    vkAllocateMemory(S.dev, &mai, NULL, &i.mem);
    vkBindImageMemory(S.dev, i.img, i.mem, 0);
    vci.image = i.img;
    vkCreateImageView(S.dev, &vci, NULL, &i.view);
    return i;
}

static VkBuffer st_buffer(void)
{
    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4096 };
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = 4096 };
    VkDeviceMemory mem;
    VkBuffer buf;

    vkCreateBuffer(S.dev, &bci, NULL, &buf);
    vkAllocateMemory(S.dev, &mai, NULL, &mem);
    vkBindBufferMemory(S.dev, buf, mem, 0);
    return buf;
}

/* A buffer of 4096 bytes in mapped memory. */
typedef struct StMapped {
    VkBuffer buf;
    uint8_t *map;
} StMapped;

static StMapped st_mapped(void)
{
    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4096 };
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = 4096 };
    VkDeviceMemory mem;
    StMapped m;
    void *map;

    vkCreateBuffer(S.dev, &bci, NULL, &m.buf);
    vkAllocateMemory(S.dev, &mai, NULL, &mem);
    vkBindBufferMemory(S.dev, m.buf, mem, 0);
    vkMapMemory(S.dev, mem, 0, VK_WHOLE_SIZE, 0, &map);
    m.map = map;
    return m;
}

static VkCommandBuffer st_cb(void)
{
    VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                        .commandPool = S.cmdpool, .commandBufferCount = 1 };
    VkCommandBuffer cb;

    vkAllocateCommandBuffers(S.dev, &cai, &cb);
    return cb;
}

static VkFence st_fence(void)
{
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence f;

    vkCreateFence(S.dev, &fci, NULL, &f);
    return f;
}

static void st_begin(VkCommandBuffer cb)
{
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };

    vkBeginCommandBuffer(cb, &bi);
}

/* A barrier that orders everything, as the renderer's vk_full_barrier. */
static void st_sync(VkCommandBuffer cb)
{
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                           .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                           .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT };

    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

/* UNDEFINED -> GENERAL naming no earlier access, as the renderer's vk_layout_general. */
static void st_general(VkCommandBuffer cb, VkImage img)
{
    VkImageMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                .newLayout = VK_IMAGE_LAYOUT_GENERAL, .image = img,
                                .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };

    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
}

static void st_clear(VkCommandBuffer cb, VkImage img)
{
    VkClearColorValue zero = { { 0 } };
    VkImageSubresourceRange sr = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    vkCmdClearColorImage(cb, img, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &sr);
}

static void st_copy_in(VkCommandBuffer cb, VkBuffer buf, VkImage img)
{
    VkBufferImageCopy r = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                            .imageExtent = { 4, 4, 1 } };

    vkCmdCopyBufferToImage(cb, buf, img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
}

/* An image made ready to be read: in GENERAL, written, ordered. */
static void st_ready(VkCommandBuffer cb, VkImage img)
{
    st_general(cb, img);
    st_clear(cb, img);
    st_sync(cb);
}

static VkDescriptorPool st_pool(void)
{
    VkDescriptorPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                       .maxSets = 4 };
    VkDescriptorPool pool;

    vkCreateDescriptorPool(S.dev, &pci, NULL, &pool);
    return pool;
}

static VkDescriptorSet st_set(VkDescriptorPool pool, VkImageView view, VkSampler sampler)
{
    VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                        .descriptorPool = pool, .descriptorSetCount = 1,
                                        .pSetLayouts = &S.dsl };
    VkDescriptorImageInfo ii = { sampler, view, VK_IMAGE_LAYOUT_GENERAL };
    VkWriteDescriptorSet w = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                               .descriptorCount = 1,
                               .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                               .pImageInfo = &ii };
    VkDescriptorSet set;

    vkAllocateDescriptorSets(S.dev, &dai, &set);
    w.dstSet = set;
    vkUpdateDescriptorSets(S.dev, 1, &w, 0, NULL);
    return set;
}

/* The same, with a uniform buffer and a storage buffer besides. */
static VkDescriptorSet st_set_buffers(VkDescriptorPool pool, VkImageView view, VkBuffer uniform,
                                      VkBuffer storage)
{
    VkDescriptorSet set = st_set(pool, view, S.sampler);
    VkDescriptorBufferInfo bu = { uniform, 0, 64 }, bs = { storage, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet w[2] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 1,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          .pBufferInfo = &bu },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 2,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &bs },
    };

    vkUpdateDescriptorSets(S.dev, 2, w, 0, NULL);
    return set;
}

static VkFramebuffer st_fb(VkImageView view)
{
    VkFramebufferCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                    .renderPass = S.rp, .attachmentCount = 1,
                                    .pAttachments = &view, .width = 4, .height = 4,
                                    .layers = 1 };
    VkFramebuffer fb;

    vkCreateFramebuffer(S.dev, &fci, NULL, &fb);
    return fb;
}

static void st_pass_begin(VkCommandBuffer cb, VkFramebuffer fb)
{
    VkRenderPassBeginInfo rbi = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                  .renderPass = S.rp, .framebuffer = fb };

    vkCmdBeginRenderPass(cb, &rbi, VK_SUBPASS_CONTENTS_INLINE);
}

static void st_draw(VkCommandBuffer cb, VkFramebuffer fb, VkDescriptorSet set)
{
    st_pass_begin(cb, fb);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, S.pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, S.layout, 0, 1, &set, 0, NULL);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRenderPass(cb);
}

static void st_submit(VkCommandBuffer cb, VkFence f)
{
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                        .pCommandBuffers = &cb };

    vkQueueSubmit(S.queue, 1, &si, f);
}

static void st_wait(VkFence f)
{
    vkWaitForFences(S.dev, 1, &f, VK_TRUE, UINT64_MAX);
}

static void st_setup(void)
{
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    VkCommandPoolCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    VkDescriptorSetLayoutCreateInfo dli = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    VkPipelineLayoutCreateInfo pli = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    VkSubpassDependency dep[2] = {
        { VK_SUBPASS_EXTERNAL, 0, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
          VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT },
        { 0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
          VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT },
    };
    VkRenderPassCreateInfo rci = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                                   .attachmentCount = 1, .dependencyCount = 2,
                                   .pDependencies = dep };
    VkShaderModuleCreateInfo mci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    VkSamplerCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    VkPipelineShaderStageCreateInfo st = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    VkGraphicsPipelineCreateInfo gci = { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                                         .stageCount = 1, .pStages = &st };
    VkPhysicalDevice pd;
    VkInstance inst;
    uint32_t n = 1;

    vkCreateInstance(&ici, NULL, &inst);
    vkEnumeratePhysicalDevices(inst, &n, &pd);
    vkCreateDevice(pd, &dci, NULL, &S.dev);
    vkGetDeviceQueue(S.dev, 0, 0, &S.queue);
    vkCreateCommandPool(S.dev, &cpi, NULL, &S.cmdpool);
    vkCreateDescriptorSetLayout(S.dev, &dli, NULL, &S.dsl);
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &S.dsl;
    vkCreatePipelineLayout(S.dev, &pli, NULL, &S.layout);
    vkCreateRenderPass(S.dev, &rci, NULL, &S.rp);
    vkCreateShaderModule(S.dev, &mci, NULL, &st.module);
    gci.layout = S.layout;
    gci.renderPass = S.rp;
    vkCreateGraphicsPipelines(S.dev, VK_NULL_HANDLE, 1, &gci, NULL, &S.pipe);
    vkCreateSampler(S.dev, &sci, NULL, &S.sampler);
}

static unsigned g_st_from, g_st_fails;

/* What was just done is correct: nothing must have been reported. */
static void st_accept(const char *what)
{
    bool quiet = mock_violations() == g_st_from;

    printf("  %-64s %s\n", what, quiet ? "accepted" : "REPORTED AS WRONG");
    g_st_fails += !quiet;
    g_st_from = mock_violations();
}

static void st_true(const char *what, bool ok)
{
    printf("  %-64s %s\n", what, ok ? "yes" : "NO");
    g_st_fails += !ok;
}

/* The misuse just committed must have been reported, in words containing `needle`. */
static void st_expect(const char *what, const char *needle)
{
    bool hit = false;

    for (unsigned i = g_st_from; i < mock_violations(); i++) {
        hit |= strstr(mock_violation(i), needle) != NULL;
    }
    printf("  %-64s %s\n", what, hit ? "reported" : "NOT REPORTED");
    g_st_fails += !hit;
    g_st_from = mock_violations();
}

static int selftest(void)
{
    VkCommandBuffer cb;
    VkDescriptorPool pool;
    VkDescriptorSet set;
    VkFramebuffer fb;
    VkBuffer buf;
    VkFence f;
    StImage tex, rt;

    st_setup();

    /* a correct batch: two images made ready, one sampled, one rendered to */
    tex = st_image();
    rt = st_image();
    buf = st_buffer();
    pool = st_pool();
    cb = st_cb();
    f = st_fence();
    st_begin(cb);
    st_general(cb, tex.img);
    st_copy_in(cb, buf, tex.img);
    st_ready(cb, rt.img);
    set = st_set(pool, tex.view, S.sampler);
    fb = st_fb(rt.view);
    st_draw(cb, fb, set);
    vkEndCommandBuffer(cb);
    st_submit(cb, f);
    st_wait(f);
    vkResetFences(S.dev, 1, &f);
    vkResetDescriptorPool(S.dev, pool, 0);
    vkDestroyFramebuffer(S.dev, fb, NULL);
    vkDestroyBuffer(S.dev, buf, NULL);
    printf("  %-64s %s\n", "a correct batch", mock_violations() ? "REPORTED AS WRONG" : "accepted");
    g_st_fails += mock_violations() != 0;
    g_st_from = mock_violations();

    {   /* handles */
        StImage i = st_image();
        VkImageCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                  .extent = { 4, 4, 1 }, .mipLevels = 1, .arrayLayers = 1 };
        VkImage bad;

        vkDestroyImageView(S.dev, i.view, NULL);
        vkDestroyImageView(S.dev, i.view, NULL);
        st_expect("an object destroyed twice", "used after it was destroyed");
        mock_plan(MF_CreateImage, mock_calls(MF_CreateImage, 0) + 1, false,
                  VK_ERROR_OUT_OF_HOST_MEMORY, 0);
        vkCreateImage(S.dev, &ici, NULL, &bad);
        mock_plan_clear();
        vkDestroyImage(S.dev, bad, NULL);
        st_expect("the output of a failed call destroyed", "undefined output of a call that failed");
        vkBindImageMemory(S.dev, i.img, VK_NULL_HANDLE, 0);
        st_expect("VK_NULL_HANDLE where an object is needed", "is VK_NULL_HANDLE");
        vkBindImageMemory(S.dev, (VkImage)i.mem, i.mem, 0);
        st_expect("an object of the wrong type", "not a image");
    }
    {   /* command buffer states */
        VkCommandBuffer c = st_cb();
        VkFence g = st_fence();
        VkViewport vp = { 0 };

        vkCmdSetViewport(c, 0, 1, &vp);
        st_expect("a command recorded into a buffer not begun", "is not recording");
        st_submit(c, g);
        st_expect("a command buffer submitted that was never ended", "is not executable");
        st_wait(g);
        st_expect("a wait for a fence nothing was submitted with", "would never return");
        st_begin(c);
        vkEndCommandBuffer(c);
        st_submit(c, g);
        st_begin(c);
        st_expect("a command buffer begun while its work is unfinished", "is pending");
        vkResetFences(S.dev, 1, &g);
        st_expect("a fence reset while its work is unfinished", "belongs to unfinished work");
        st_wait(g);
        st_begin(c);
        vkEndCommandBuffer(c);
        st_submit(c, g);
        st_expect("a submission with a fence still signaled", "is signaled");
    }
    {   /* lifetimes */
        VkCommandBuffer c = st_cb();
        VkDescriptorPool p = st_pool();
        VkFence g = st_fence();
        VkBuffer b = st_buffer();
        StImage t = st_image(), r = st_image();
        VkDescriptorSet s2;
        VkFramebuffer fb2;

        st_begin(c);
        st_general(c, t.img);
        st_copy_in(c, b, t.img);
        st_ready(c, r.img);
        s2 = st_set(p, t.view, S.sampler);
        fb2 = st_fb(r.view);
        st_draw(c, fb2, s2);
        vkEndCommandBuffer(c);
        st_submit(c, g);
        vkDestroyBuffer(S.dev, b, NULL);
        st_expect("a buffer destroyed while submitted work uses it", "still uses it");
        vkResetDescriptorPool(S.dev, p, 0);
        st_expect("a descriptor pool reset while submitted work uses it", "is reset while");
        st_wait(g);
        vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, S.layout, 0, 1, &s2, 0, NULL);
        st_expect("a descriptor set used after its pool was reset", "after its pool was reset");
        g_st_from = mock_violations();
    }
    {   /* images as commands execute */
        VkCommandBuffer c = st_cb();
        VkDescriptorPool p = st_pool();
        VkFence g = st_fence();
        VkBuffer b = st_buffer();
        StImage t = st_image(), r = st_image();
        VkFramebuffer fb2 = st_fb(r.view);

        st_begin(c);
        st_ready(c, r.img);
        st_draw(c, fb2, st_set(p, t.view, S.sampler));
        vkEndCommandBuffer(c);
        st_submit(c, g);
        st_wait(g);
        st_expect("an image sampled that was never given a layout", "it is in layout UNDEFINED");
        vkResetFences(S.dev, 1, &g);

        st_begin(c);
        st_general(c, t.img);
        st_sync(c);
        st_draw(c, fb2, st_set(p, t.view, S.sampler));
        vkEndCommandBuffer(c);
        st_submit(c, g);
        st_wait(g);
        st_expect("an image sampled that was never written", "never written");
        vkResetFences(S.dev, 1, &g);

        st_begin(c);
        st_general(c, t.img);
        st_clear(c, t.img);
        st_general(c, t.img);
        st_clear(c, t.img);
        vkEndCommandBuffer(c);
        st_submit(c, g);
        st_wait(g);
        st_expect("a layout transition right after a write, nothing between",
                  "nothing orders after an earlier write");
        vkResetFences(S.dev, 1, &g);

        st_begin(c);
        st_pass_begin(c, fb2);
        st_copy_in(c, b, t.img);
        st_expect("a copy inside a render pass", "inside a render pass");
        st_general(c, t.img);
        st_expect("a layout transition inside a render pass", "inside a render pass");
        vkEndCommandBuffer(c);
        st_expect("a command buffer ended with its render pass open", "render pass open");
        g_st_from = mock_violations();
    }
    {   /* descriptors */
        VkDescriptorPool p = st_pool();
        StImage t = st_image();
        VkResult r = VK_SUCCESS;
        VkDescriptorSet set5 = (VkDescriptorSet)SENTINEL;

        st_set(p, t.view, VK_NULL_HANDLE);
        st_expect("a combined image sampler without a sampler", "sampler is VK_NULL_HANDLE");
        for (int i = 1; i < 5; i++) {
            VkDescriptorSetAllocateInfo dai = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = p,
                .descriptorSetCount = 1, .pSetLayouts = &S.dsl };
            r = vkAllocateDescriptorSets(S.dev, &dai, &set5);
        }
        printf("  %-64s %s\n", "a pool of 4 sets asked for a 5th",
               r == VK_ERROR_OUT_OF_POOL_MEMORY && !set5 ? "refused" : "NOT REFUSED");
        g_st_fails += !(r == VK_ERROR_OUT_OF_POOL_MEMORY && !set5);
    }
    {   /* what nothing holds */
        StImage a = st_image(), b = st_image();
        unsigned before, after;

        mock_mark_begin();
        before = mock_unmarked(MT_IMAGE);
        mock_mark(a.img, MT_IMAGE, "one image");
        after = mock_unmarked(MT_IMAGE);
        printf("  %-64s %s\n", "objects that nothing holds are counted",
               before >= 2 && after == before - 1 ? "yes" : "NO");
        g_st_fails += !(before >= 2 && after == before - 1);
        (void)b;
        mock_mark(a.view, MT_IMAGE, "a view held as an image");
        st_expect("a held handle that is not what it is held as", "not a image");
    }
    {   /* memory that running work uses */
        VkCommandBuffer c = st_cb(), last = st_cb();
        VkDescriptorPool p = st_pool();
        VkFence g = st_fence(), glast = st_fence();
        StImage t = st_image(), r = st_image();
        VkFramebuffer fb2 = st_fb(r.view);
        StMapped u = st_mapped(), st = st_mapped();
        volatile uint8_t sink;

        /* what the misuse above left running is finished first: by a wait
         * for work that a barrier orders after it (which is tested below) */
        st_begin(last);
        st_sync(last);
        vkEndCommandBuffer(last);
        st_submit(last, glast);
        st_wait(glast);
        st_true("nothing is left running", mock_running() == 0);
        g_st_from = mock_violations();
        st_begin(c);
        st_ready(c, t.img);
        st_ready(c, r.img);
        st_draw(c, fb2, st_set_buffers(p, t.view, u.buf, st.buf));
        vkEndCommandBuffer(c);
        u.map[0] = 1;
        st.map[0] = 1;
        st_accept("the host writes buffers of a batch not yet submitted");
        st_submit(c, g);
        sink = u.map[0];
        st_accept("the host reads memory that running work reads");
        u.map[1] = 1;
        st_expect("the host writes memory that running work reads", "the host writes memory");
        sink = st.map[0];
        st_expect("the host reads memory that running work may write", "the host touches memory");
        mock_plan(MF_WaitForFences, mock_calls(MF_WaitForFences, 0) + 1, false,
                  VK_ERROR_OUT_OF_DEVICE_MEMORY, 0);
        st_wait(g);
        mock_plan_clear();
        st_true("a wait that fails leaves the work running", mock_running() == 1);
        mock_host_access(st.map, 16, false, "selftest");
        st_expect("... and its memory in use", "not known to have finished may write");
        mock_plan(MF_WaitForFences, mock_calls(MF_WaitForFences, 0) + 1, false, VK_TIMEOUT, 0);
        st_true("a wait that comes back early (VK_TIMEOUT) leaves it running",
                vkWaitForFences(S.dev, 1, &g, VK_TRUE, UINT64_MAX) == VK_TIMEOUT &&
                mock_running() == 1);
        mock_plan_clear();
        mock_hold(true);
        st_true("while the work is held, a wait with a time limit times out",
                vkWaitForFences(S.dev, 1, &g, VK_TRUE, 1000) == VK_TIMEOUT &&
                mock_running() == 1);
        mock_hold(false);
        st_wait(g);
        st_true("a wait that succeeds finishes it", mock_running() == 0);
        st.map[1] = 2;
        u.map[2] = 2;
        sink = st.map[1];
        st_accept("the host touches the memory once the work has finished");
        (void)sink;
    }
    {   /* memory shared piecemeal: uses by the byte, accesses declared */
        VkCommandBuffer c = st_cb();
        VkFence g = st_fence();
        StImage t = st_image();
        StMapped sh = st_mapped();
        VkBufferImageCopy in = { .bufferOffset = 1024,
                                 .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                                 .imageExtent = { 4, 4, 1 } };
        VkBufferImageCopy out = in;

        out.bufferOffset = 256;
        out.bufferRowLength = 8;        /* rows 32 bytes apart: 3 * 32 + 16 = 112 bytes */
        mock_shared_buffer(sh.buf);
        g_st_from = mock_violations();
        st_begin(c);
        st_general(c, t.img);
        vkCmdCopyBufferToImage(c, sh.buf, t.img, VK_IMAGE_LAYOUT_GENERAL, 1, &in);
        st_sync(c);
        vkCmdCopyImageToBuffer(c, t.img, VK_IMAGE_LAYOUT_GENERAL, sh.buf, 1, &out);
        vkEndCommandBuffer(c);
        st_submit(c, g);
        mock_host_access(sh.map, 256, true, "selftest");
        mock_host_access(sh.map + 368, 656, true, "selftest");
        mock_host_access(sh.map + 1088, 3008, true, "selftest");
        mock_host_access(sh.map + 1024, 64, false, "selftest");
        st_accept("shared memory: the host writes around the regions, reads what is read");
        mock_host_access(sh.map + 367, 1, false, "selftest");
        st_expect("shared memory: the host reads a byte that a copy writes", "work not known "
                  "to have finished writes");
        mock_host_access(sh.map + 1087, 2, true, "selftest");
        st_expect("shared memory: the host writes a byte that a copy reads", "work not known "
                  "to have finished reads");
        st_wait(g);
        mock_host_access(sh.map, 4096, true, "selftest");
        st_accept("shared memory: the host writes all of it once the work has finished");
    }
    {   /* what a wait says about earlier work */
        VkCommandBuffer a = st_cb(), b = st_cb(), c = st_cb(), d = st_cb();
        VkFence fa = st_fence(), fb = st_fence(), fc = st_fence(), fd = st_fence();

        g_st_from = mock_violations();
        st_begin(a);
        vkEndCommandBuffer(a);
        st_submit(a, fa);
        st_begin(b);                    /* no barrier: not ordered after a */
        vkEndCommandBuffer(b);
        st_submit(b, fb);
        st_wait(fb);
        st_true("work is not finished by a wait for later, unordered work",
                mock_running() == 1 && mock_finished_upto() == mock_submitted() - 2);
        st_begin(c);
        st_sync(c);                     /* a barrier first: after everything before */
        vkEndCommandBuffer(c);
        st_submit(c, fc);
        st_wait(fc);
        st_true("... and is by a wait for later work that a barrier orders after it",
                mock_running() == 0 && mock_finished_upto() == mock_submitted());
        vkResetFences(S.dev, 1, &fa);
        st_expect("its fence reset, which no wait has seen signaled", "no wait has seen it");
        st_begin(a);
        st_sync(a);
        vkEndCommandBuffer(a);
        st_submit(a, fa);
        st_begin(c);
        st_sync(c);
        vkEndCommandBuffer(c);
        vkResetFences(S.dev, 1, &fc);
        st_submit(c, fc);
        st_wait(fc);
        st_wait(fa);
        vkResetFences(S.dev, 1, &fa);
        st_accept("... and reset after a wait for it");

        /* last: the device stays lost */
        st_begin(d);
        vkEndCommandBuffer(d);
        mock_plan(MF_QueueSubmit, mock_calls(MF_QueueSubmit, 0) + 1, false,
                  VK_ERROR_DEVICE_LOST, 0);
        st_submit(d, fd);
        mock_plan_clear();
        st_true("a submission that reports the device lost counts as made",
                mock_device_lost() && mock_running() == 1);
        st_begin(d);
        st_expect("... its command buffer begun again", "is pending");
        st_true("a lost device: everything counts as finished",
                mock_finished_upto() == UINT_MAX);
    }
    {
        VkCommandBuffer c = st_cb();
        VkDescriptorPool p = st_pool();
        VkFence g = st_fence();
        StImage t = st_image(), r = st_image();
        VkFramebuffer fb2 = st_fb(r.view);
        StMapped u = st_mapped(), st = st_mapped();

        g_st_from = mock_violations();
        st_begin(c);
        st_ready(c, t.img);
        st_ready(c, r.img);
        st_draw(c, fb2, st_set_buffers(p, t.view, u.buf, st.buf));
        vkEndCommandBuffer(c);
        st_submit(c, g);
        st.map[0] = 3;
        mock_host_access(st.map, 16, true, "selftest");
        st_accept("a lost device: the host may touch memory its work was using");
    }
    printf("stand-in driver: %u of its checks failed\n", g_st_fails);
    return g_st_fails != 0;
}

/* ---- the helpers' contract --------------------------------------------- */

static unsigned g_checks, g_check_fails;

static void check(bool ok, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void check(bool ok, const char *fmt, ...)
{
    g_checks++;
    if (!ok) {
        va_list ap;
        g_check_fails++;
        printf("    FAIL: ");
        va_start(ap, fmt);
        vprintf(fmt, ap);
        va_end(ap);
        printf("\n");
    }
}

/* Violations since the last call, printed. */
static unsigned new_violations(void)
{
    static unsigned seen;
    unsigned n = mock_violations() - seen;

    for (unsigned i = seen; i < mock_violations() && i < seen + 4; i++) {
        printf("    FAIL: the driver saw: %s\n", mock_violation(i));
    }
    seen = mock_violations();
    return n;
}

typedef struct Counts {
    unsigned alive[MT_COUNT], destroyed[MT_COUNT];
} Counts;

static Counts counts(void)
{
    Counts c;

    for (int t = 0; t < MT_COUNT; t++) {
        c.alive[t] = mock_alive(t);
        c.destroyed[t] = mock_destroyed(t);
    }
    return c;
}

static void test_vk_image(void)
{
    static const struct { int fn; unsigned img, mem; } at[] = {
        { MF_CreateImage, 0, 0 }, { MF_AllocateMemory, 1, 0 },
        { MF_BindImageMemory, 1, 1 }, { MF_CreateImageView, 1, 1 },
        { -1, 1, 0 },                   /* no memory type fits */
    };

    for (unsigned i = 0; i < ARRAY_SIZE(at); i++) {
        VkImage img = (VkImage)SENTINEL;
        VkDeviceMemory mem = (VkDeviceMemory)SENTINEL;
        VkImageView view = (VkImageView)SENTINEL;
        Counts a = counts(), b;
        bool ok;

        if (at[i].fn < 0) {
            printf("  vk_image, no memory type fits\n");
            mock_no_memory_type(true);
        } else {
            printf("  vk_image, %s fails\n", mock_fn_name[at[i].fn]);
            mock_plan(at[i].fn, mock_calls(at[i].fn, 0) + 1, false,
                      VK_ERROR_OUT_OF_HOST_MEMORY, 0);
        }
        ok = vk_image(VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, 4, 4,
                      1, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT, &img, &mem, &view);
        mock_plan_clear();
        mock_no_memory_type(false);
        b = counts();
        check(!ok, "it reports success");
        check(img == VK_NULL_HANDLE, "*img is %p, not VK_NULL_HANDLE%s", (void *)img,
              mock_is_garbage(img) ? " (the failed call's output)" : "");
        check(mem == VK_NULL_HANDLE, "*mem is %p, not VK_NULL_HANDLE%s", (void *)mem,
              mock_is_garbage(mem) ? " (the failed call's output)" : "");
        check(view == VK_NULL_HANDLE, "*view is %p, not VK_NULL_HANDLE%s", (void *)view,
              mock_is_garbage(view) ? " (the failed call's output)" : "");
        check(b.destroyed[MT_IMAGE] - a.destroyed[MT_IMAGE] == at[i].img,
              "%u images destroyed, should be %u",
              b.destroyed[MT_IMAGE] - a.destroyed[MT_IMAGE], at[i].img);
        check(b.destroyed[MT_MEMORY] - a.destroyed[MT_MEMORY] == at[i].mem,
              "%u memory objects freed, should be %u",
              b.destroyed[MT_MEMORY] - a.destroyed[MT_MEMORY], at[i].mem);
        check(b.alive[MT_IMAGE] == a.alive[MT_IMAGE] && b.alive[MT_MEMORY] == a.alive[MT_MEMORY] &&
              b.alive[MT_VIEW] == a.alive[MT_VIEW], "objects left behind");
        check(!new_violations(), "the driver was misused");
    }
}

static void test_vk_buffer(void)
{
    static const struct { int fn; unsigned buf, mem; } at[] = {
        { MF_CreateBuffer, 0, 0 }, { MF_AllocateMemory, 1, 0 },
        { MF_BindBufferMemory, 1, 1 }, { MF_MapMemory, 1, 1 },
        { -1, 1, 0 },                   /* no memory type fits */
    };

    for (unsigned i = 0; i < ARRAY_SIZE(at); i++) {
        VkBuffer buf = (VkBuffer)SENTINEL;
        VkDeviceMemory mem = (VkDeviceMemory)SENTINEL;
        void *map = (void *)SENTINEL;
        Counts a = counts(), b;
        bool ok;

        if (at[i].fn < 0) {
            printf("  vk_buffer, no memory type fits\n");
            mock_no_memory_type(true);
        } else {
            printf("  vk_buffer, %s fails\n", mock_fn_name[at[i].fn]);
            mock_plan(at[i].fn, mock_calls(at[i].fn, 0) + 1, false,
                      VK_ERROR_OUT_OF_HOST_MEMORY, 0);
        }
        ok = vk_buffer(4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0, &buf, &mem, &map);
        mock_plan_clear();
        mock_no_memory_type(false);
        b = counts();
        check(!ok, "it reports success");
        check(buf == VK_NULL_HANDLE, "*buf is %p, not VK_NULL_HANDLE%s", (void *)buf,
              mock_is_garbage(buf) ? " (the failed call's output)" : "");
        check(mem == VK_NULL_HANDLE, "*mem is %p, not VK_NULL_HANDLE%s", (void *)mem,
              mock_is_garbage(mem) ? " (the failed call's output)" : "");
        check(map == NULL, "*map is %p, not NULL%s", map,
              mock_is_garbage(map) ? " (the failed call's output)" : "");
        check(b.destroyed[MT_BUFFER] - a.destroyed[MT_BUFFER] == at[i].buf,
              "%u buffers destroyed, should be %u",
              b.destroyed[MT_BUFFER] - a.destroyed[MT_BUFFER], at[i].buf);
        check(b.destroyed[MT_MEMORY] - a.destroyed[MT_MEMORY] == at[i].mem,
              "%u memory objects freed, should be %u",
              b.destroyed[MT_MEMORY] - a.destroyed[MT_MEMORY], at[i].mem);
        check(b.alive[MT_BUFFER] == a.alive[MT_BUFFER] && b.alive[MT_MEMORY] == a.alive[MT_MEMORY],
              "objects left behind");
        check(!new_violations(), "the driver was misused");
    }
}

/*
 * vk_dummies with one creation call of stand-in `which` failing; then again
 * with nothing failing, in the same batch (same_batch) or after the first
 * attempt's batch, if any, was committed.
 */
static void test_vk_dummies(unsigned which, int fn, bool same_batch)
{
    Counts a, b;
    R300DrawPacket p;
    VkBatch *open;
    bool ok;

    printf("  vk_dummies, stand-in %u: %s fails; retry in %s\n", which, mock_fn_name[fn],
           same_batch ? "the same open batch" : "a later batch");
    check(!V.dummy_ready && !V.dummy_view[0] && !V.dummy_view[1] && !V.dummy_view[2],
          "the stand-ins exist before the test");
    open = same_batch ? vk_batch() : NULL;
    a = counts();
    mock_plan(fn, mock_calls(fn, 0) + which + 1, false, VK_ERROR_OUT_OF_HOST_MEMORY, 0);
    ok = vk_dummies();
    mock_plan_clear();
    b = counts();
    check(!ok, "the failed attempt reports success");
    check(!V.dummy_ready, "the stand-ins count as ready after the failed attempt");
    for (unsigned i = 0; i < 3; i++) {
        bool all = V.dummy_img[i] && V.dummy_mem[i] && V.dummy_view[i];
        bool none = !V.dummy_img[i] && !V.dummy_mem[i] && !V.dummy_view[i];
        check(i < which ? all : none,
              "stand-in %u after the failed attempt: image %p memory %p view %p "
              "(should be %s)", i, (void *)V.dummy_img[i], (void *)V.dummy_mem[i],
              (void *)V.dummy_view[i], i < which ? "all there" : "all VK_NULL_HANDLE");
    }
    check(b.alive[MT_IMAGE] - a.alive[MT_IMAGE] == which &&
          b.alive[MT_MEMORY] - a.alive[MT_MEMORY] == which &&
          b.alive[MT_VIEW] - a.alive[MT_VIEW] == which,
          "%u images, %u memory objects, %u views kept; should be %u of each",
          b.alive[MT_IMAGE] - a.alive[MT_IMAGE], b.alive[MT_MEMORY] - a.alive[MT_MEMORY],
          b.alive[MT_VIEW] - a.alive[MT_VIEW], which);
    check(V.cur == open, "the failed attempt %s a batch", open ? "closed" : "opened");
    if (!same_batch && V.cur) {
        flush();
    }
    a = counts();
    ok = vk_dummies();
    b = counts();
    check(ok, "the second attempt fails");
    check(V.dummy_ready, "the stand-ins are not ready after the second attempt");
    check(b.alive[MT_IMAGE] - a.alive[MT_IMAGE] == 3 - which &&
          b.alive[MT_VIEW] - a.alive[MT_VIEW] == 3 - which,
          "the second attempt created %u images and %u views, should be %u of each",
          b.alive[MT_IMAGE] - a.alive[MT_IMAGE], b.alive[MT_VIEW] - a.alive[MT_VIEW], 3 - which);
    check(!same_batch || V.cur == open, "the second attempt is not in the same batch");
    /* use them: a draw binds a stand-in at every unit, and the driver checks
     * their layout and contents when the batch runs */
    pkt_rt(&p, 0x000000, 64, 64);
    check(draw_one(&p) == 0, "a draw after it is rejected");
    flush();
    check(!new_violations(), "the driver was misused");
}

/*
 * More attachment layouts than the renderer keeps render passes for: the
 * first VK_MAX_RP are drawn, the others rejected, and a rejected draw
 * leaves no render pass behind.
 */
static void test_render_passes(void)
{
    static const struct { uint32_t view, bpp; } v[6] = {
        { R300_RTV_RGBA8, 4 }, { R300_RTV_R8U, 1 }, { R300_RTV_R16U, 2 },
        { R300_RTV_R32U, 4 }, { R300_RTV_RG32U, 8 }, { R300_RTV_RGBA32U, 16 },
    };
    unsigned ok = 0, rejected = 0, again = 0;
    R300DrawPacket p;

    printf("  %u attachment layouts, %u render passes kept\n", 36, VK_MAX_RP);
    for (int round = 0; round < 2; round++) {
        for (unsigned a = 0; a < 6; a++) {
            for (unsigned b = 0; b < 6; b++) {
                int r;
                pkt_rt(&p, 0x000000, 16, 16);
                p.rt_view = v[a].view;
                p.rt_bpp = v[a].bpp;
                p.num_cb = 2;
                p.cb[1].gpu_addr = 0x008000;
                p.cb[1].pitch = 16;
                p.cb[1].bpp = v[b].bpp;
                p.cb[1].view = v[b].view;
                r = R->draw_r300(OPQ, VRAM, VRAM_SIZE, &p);
                if (round == 0) {
                    ok += r == 0;
                    rejected += r != 0;
                    check(r == (a * 6 + b < VK_MAX_RP ? 0 : -1),
                          "layout %u: the draw returns %d", a * 6 + b, r);
                } else {
                    again += r == 0;
                }
            }
        }
        flush();
    }
    check(ok == VK_MAX_RP && rejected == 36 - VK_MAX_RP, "%u drawn, %u rejected", ok, rejected);
    check(again == VK_MAX_RP, "%u drawn the second time round, should be %u", again, VK_MAX_RP);
    check(mock_alive(MT_RENDERPASS) == VK_MAX_RP,
          "%u render passes exist, the renderer keeps %u", mock_alive(MT_RENDERPASS), V.nrp);
    vk_reap();
    {
        char desc[160];
        unsigned leaks = count_leaks(desc, sizeof(desc));
        check(!leaks, "objects nothing refers to: %s", desc);
    }
    check(!new_violations(), "the driver was misused");
}

static void run_render_passes(void *arg)
{
    test_render_passes();
}

/*
 * Every batch is in flight and a draw has to wait for one to come free.
 * Then again with a wait failing meanwhile, and not because the device is
 * lost: the batch whose wait failed may still be running and must keep its
 * fence, its command buffer and what it uses until the wait is answered.
 * Rendering goes on.
 */
static void *open_gate_later(void *arg)
{
    usleep(100000);
    mock_gate(false);
    return NULL;
}

static void test_all_in_flight(bool wait_fails)
{
    R300DrawPacket p;
    pthread_t th;
    int r;

    printf("  every batch in flight and a draw waiting for one%s\n",
           wait_fails ? "; a wait fails for lack of memory" : "");
    mock_gate(true);                    /* the renderer's thread stops in its first wait */
    for (int i = 0; i < VK_NBATCH; i++) {
        pkt_rt(&p, 0x000000, 64, 64);
        check(draw_one(&p) == 0, "draw %d is rejected", i);
        submit();
    }
    if (wait_fails) {
        mock_plan(MF_WaitForFences, mock_calls(MF_WaitForFences, 1) + 1, false,
                  VK_ERROR_OUT_OF_HOST_MEMORY, 1);
    }
    pthread_create(&th, NULL, open_gate_later, NULL);
    pkt_rt(&p, 0x000000, 64, 64);
    r = draw_one(&p);                   /* no batch is free: it waits */
    pthread_join(th, NULL);
    mock_plan_clear();
    quiesce();
    check_done();
    if (wait_fails) {
        check(mock_fired() == 1, "the wait was not made to fail");
        check(g_pauses >= 1, "the renderer did not pause before it asked again");
    }
    check(!qatomic_read(&V.lost), "rendering has stopped");
    check(r == 0, "the draw that waited for a batch returns %d", r);
    check(ndone() == VK_NBATCH, "%u completions reported for %u batches", ndone(), VK_NBATCH);
    pkt_rt(&p, 0x000000, 64, 64);
    check(draw_one(&p) == 0, "a later draw is rejected");
    flush();
    check(!new_violations(), "the driver was misused");
}

/* ---- the GPU gives no answer -------------------------------------------- */

/* Wait, not for ever, for something another thread brings about. */
#define WAIT_FOR(cond)                                          \
    ({                                                          \
        int left_ = 8000;       /* 4 s */                       \
        while (!(cond) && --left_) {                            \
            usleep(500);                                        \
        }                                                       \
        left_ != 0;                                             \
    })

/* The renderer is entered by one thread at a time (the device holds a lock
 * around it): a call that is to be seen waiting is made on a thread of its
 * own, and nothing else calls the renderer until it is back. */
typedef struct Blocked {
    pthread_t th;
    int which;                  /* 0: flush_r200, 1: zpass_r300 */
    int back;
} Blocked;

static void *blocked_fn(void *arg)
{
    Blocked *b = arg;

    if (b->which) {
        zpass(false);
    } else {
        flush_now();
    }
    __atomic_store_n(&b->back, 1, __ATOMIC_SEQ_CST);
    return NULL;
}

static bool is_back(Blocked *b)
{
    return __atomic_load_n(&b->back, __ATOMIC_SEQ_CST);
}

/*
 * A batch is in flight and no wait for it is answered: each returns `res`,
 * an error other than a lost device, or VK_TIMEOUT.  The GPU may still be
 * working, so the batch is not complete: its pages stay busy, no completion
 * is reported, and a flush (or a read of the Z-pass counter, `which`) does
 * not return.  That holds while the waits fail, and after they stop failing
 * for as long as the GPU has not finished.  Then all of it goes ahead, and
 * rendering with it.
 */
static void test_no_answer(VkResult res, int which)
{
    Blocked bl = { .which = which };
    R300DrawPacket p;
    uint32_t seq;
    unsigned w0;

    printf("  no answer about a batch (the wait returns %d); %s meanwhile\n", (int)res,
           which ? "the Z-pass counter is read" : "a flush");
    g_pause_sleeps = true;
    mock_hold(true);                    /* the GPU does not finish */
    mock_plan(MF_WaitForFences, mock_calls(MF_WaitForFences, 2) + 1, true, res, -1);
    pkt_rt(&p, 0x000000, 64, 64);
    pkt_z(&p, 0x010000, 4);
    tex(&p.tex[0], 0x030000, 32, 32, R300_TEXK_RGBA8, R300_TEXDIM_2D, 1, 1);
    check(draw_one(&p) == 0, "the draw is rejected");
    seq = submit();
    check(seq != 0, "nothing was committed");

    /* the renderer's thread asks, and asks again */
    check(WAIT_FOR(mock_calls(MF_WaitForFences, 1) >= 20), "its thread does not ask again");
    check(ndone() == 0, "the batch was reported complete");
    check(mock_running() == 1, "%u submissions are running, not 1", mock_running());
    check(R->range_busy_r200(OPQ, 0x000000, 0x004000, false),
          "the colour buffer it renders to counts as free to read");
    check(R->range_busy_r200(OPQ, 0x010000, 0x014000, true),
          "the depth buffer it renders to counts as free to write");
    check(R->range_busy_r200(OPQ, 0x030000, 0x031000, true),
          "the texture it loads counts as free to write");
    check(!R->range_busy_r200(OPQ, 0x030000, 0x031000, false),
          "the texture it loads counts as busy to read");
    check(!R->range_busy_r200(OPQ, 0x3f0000, 0x3f1000, true),
          "VRAM it does not use counts as busy");
    device_access(0x3f0000, 0x3f1000, true);        /* ... and may be written */
    check(!qatomic_read(&V.lost), "rendering has stopped");
    check(g_pauses > 0, "the renderer does not pause between its attempts");

    /* a flush cannot return, nor the Z-pass counter be read */
    w0 = mock_calls(MF_WaitForFences, 0);
    pthread_create(&bl.th, NULL, blocked_fn, &bl);
    check(WAIT_FOR(mock_calls(MF_WaitForFences, 0) >= w0 + 20), "it does not ask again");
    check(!is_back(&bl), "it returned while no wait was answered");
    check(ndone() == 0, "the batch was reported complete");

    /* the waits no longer fail, but the GPU has not finished */
    mock_plan_clear();
    usleep(100000);
    check(!is_back(&bl), "it returned before the GPU had finished");
    check(ndone() == 0, "the batch was reported complete before the GPU had finished");
    check(mock_running() == 1, "%u submissions are running, not 1", mock_running());

    /* it finishes */
    mock_hold(false);
    check(WAIT_FOR(is_back(&bl)), "it does not return once the GPU has finished");
    pthread_join(bl.th, NULL);
    quiesce();
    check_done();
    check(ndone() == 1 && g_done[0].seq == seq, "%u completions reported, the first for "
          "batch %u (committed: one, batch %u)", ndone(), g_done[0].seq, seq);
    check(mock_running() == 0, "%u submissions are still running", mock_running());
    check(!R->range_busy_r200(OPQ, 0x000000, 0x004000, true), "the colour buffer stays busy");
    device_access(0x000000, 0x001000, true);
    device_access(0x030000, 0x030400, true);
    check(!qatomic_read(&V.lost), "rendering has stopped");
    pkt_rt(&p, 0x000000, 64, 64);
    check(draw_one(&p) == 0, "a later draw is rejected");
    flush();
    vk_reap();
    {
        char desc[160];
        check(!count_leaks(desc, sizeof(desc)), "objects nothing refers to: %s", desc);
    }
    check(!new_violations(), "the driver was misused");
}

struct no_answer_arg {
    VkResult res;
    int which;
};

static void run_no_answer(void *arg)
{
    struct no_answer_arg *a = arg;

    test_no_answer(a->res, a->which);
}

/*
 * Two wait for a batch -- the renderer's thread and a flush -- and only one
 * of them is answered.  The thread's answer settles it for the flush too,
 * which stops asking.  The flush's answer frees the memory, and the flush
 * returns; the thread goes on asking, for it is on its own answer that the
 * batch's fence is used again and the completion reported.
 */
static void test_one_answered(bool thread_answered)
{
    Blocked bl = { .which = 0 };
    R300DrawPacket p;
    uint32_t seq;
    unsigned w0;

    printf("  two wait for a batch; only %s is answered\n",
           thread_answered ? "the renderer's thread" : "the flush");
    g_pause_sleeps = true;
    pkt_rt(&p, 0x000000, 64, 64);
    check(draw_one(&p) == 0, "the draw is rejected");
    if (thread_answered) {
        /* the thread is kept from asking until the flush has asked in vain */
        mock_gate(true);
        mock_plan(MF_WaitForFences, mock_calls(MF_WaitForFences, 0) + 1, true,
                  VK_ERROR_OUT_OF_DEVICE_MEMORY, 0);
        seq = submit();
        w0 = mock_calls(MF_WaitForFences, 0);
        pthread_create(&bl.th, NULL, blocked_fn, &bl);
        check(WAIT_FOR(mock_calls(MF_WaitForFences, 0) >= w0 + 20), "the flush does not ask again");
        check(!is_back(&bl), "the flush returned unanswered");
        check(ndone() == 0, "the batch was reported complete");
        mock_gate(false);
        check(WAIT_FOR(ndone() == 1), "the thread's answer is not reported");
        check(WAIT_FOR(is_back(&bl)), "the flush goes on asking though the batch is complete");
        pthread_join(bl.th, NULL);
        check(mock_failed(MF_WaitForFences) >= 20, "the flush's waits were not made to fail");
    } else {
        unsigned resets = mock_calls(MF_ResetFences, 2);

        mock_plan(MF_WaitForFences, mock_calls(MF_WaitForFences, 1) + 1, true,
                  VK_ERROR_OUT_OF_DEVICE_MEMORY, 1);
        seq = submit();
        check(WAIT_FOR(mock_calls(MF_WaitForFences, 1) >= 20), "the thread does not ask again");
        check(ndone() == 0, "the batch was reported complete");
        check(R->range_busy_r200(OPQ, 0x000000, 0x004000, false), "its colour buffer is free");
        flush_now();                    /* answered: the batch has finished */
        check(mock_running() == 0, "%u submissions are still running", mock_running());
        check(!R->range_busy_r200(OPQ, 0x000000, 0x004000, true), "its colour buffer stays busy");
        device_access(0x000000, 0x001000, true);
        /* the thread has no answer of its own: the batch is not its to use
         * again, and it is the thread that reports completions */
        pkt_rt(&p, 0x000000, 64, 64);
        check(draw_one(&p) == 0, "a draw meanwhile is rejected");
        check(mock_calls(MF_ResetFences, 2) == resets, "the batch was recycled on the "
              "flush's answer, its own fence not seen signaled");
        check(ndone() == 0, "the completion was reported without the thread's own answer");
    }
    mock_plan_clear();
    quiesce();
    check_done();
    check(ndone() == 1 && g_done[0].seq == seq, "%u completions reported, the first for "
          "batch %u (committed: one, batch %u)", ndone(), g_done[0].seq, seq);
    check(!qatomic_read(&V.lost), "rendering has stopped");
    pkt_rt(&p, 0x000000, 64, 64);
    check(draw_one(&p) == 0, "a later draw is rejected");
    flush();
    check(!new_violations(), "the driver was misused");
}

static void run_thread_answered(void *arg)
{
    test_one_answered(true);
}

static void run_flush_answered(void *arg)
{
    test_one_answered(false);
}

/*
 * The device is lost while batches are in flight: the wait of the
 * renderer's thread says so, or a flush's.  That is an answer.  Completion
 * is reported for each batch, in order, VRAM may be touched (what it holds
 * is undefined), nothing more is submitted and nothing is recycled.
 */
static void test_device_lost(bool in_flush)
{
    R300DrawPacket p;
    unsigned subs, resets;

    printf("  the device is lost with batches in flight: %s finds out\n",
           in_flush ? "a flush" : "the renderer's thread");
    mock_gate(true);
    for (int i = 0; i < 3; i++) {
        pkt_rt(&p, 0x000000, 64, 64);
        pkt_z(&p, 0x010000, 4);
        check(draw_one(&p) == 0, "draw %d is rejected", i);
        submit();
    }
    resets = mock_calls(MF_ResetFences, 2);
    mock_plan(MF_WaitForFences, mock_calls(MF_WaitForFences, in_flush ? 0 : 1) + 1, false,
              VK_ERROR_DEVICE_LOST, in_flush ? 0 : 1);
    if (in_flush) {
        flush_now();
        check(qatomic_read(&V.lost), "rendering goes on after the flush was told");
        check(!R->range_busy_r200(OPQ, 0x000000, 0x004000, true), "the colour buffer stays busy");
        device_access(0x000000, 0x001000, true);
    }
    mock_gate(false);
    quiesce();
    check_done();
    mock_plan_clear();
    check(mock_fired() == 1, "no wait was made to report the device lost");
    check(mock_device_lost(), "the driver does not count the device as lost");
    check(qatomic_read(&V.lost), "rendering goes on");
    check(ndone() == 3, "%u completions reported for 3 batches", ndone());
    check(!R->range_busy_r200(OPQ, 0x000000, 0x004000, true), "the colour buffer stays busy");
    device_access(0x000000, 0x001000, true);
    check(R->gpu_failures(OPQ) != 0, "gpu_failures does not tell");
    subs = mock_submitted();
    pkt_rt(&p, 0x000000, 64, 64);
    check(draw_one(&p) == 0, "a draw after the loss returns an error");
    flush();
    check(mock_submitted() == subs, "work was submitted after the loss");
    check(mock_calls(MF_ResetFences, 2) == resets, "a batch was recycled after the loss");
    check(!new_violations(), "the driver was misused");
}

static void run_lost_thread(void *arg)
{
    test_device_lost(false);
}

static void run_lost_flush(void *arg)
{
    test_device_lost(true);
}

/*
 * Several batches in flight, and each wait fails three times before it is
 * answered, on both threads: every completion is reported, once, in the
 * order of the commits, and none before its batch has finished.
 */
static void test_completions_in_order(void)
{
    R300DrawPacket p;
    uint32_t seq[5];

    printf("  five batches in flight, every wait unanswered three times\n");
    mock_wait_patience(3);
    mock_plan(MF_WaitForFences, mock_calls(MF_WaitForFences, 2) + 1, true,
              VK_ERROR_OUT_OF_HOST_MEMORY, -1);
    for (int i = 0; i < 5; i++) {
        pkt_rt(&p, 0x000000, 64, 64);
        check(draw_one(&p) == 0, "draw %d is rejected", i);
        seq[i] = submit();
        if (i == 2) {
            device_access(0x000000, 0x001000, false);   /* a flush in between */
        }
    }
    quiesce();
    check_done();
    mock_plan_clear();
    check(ndone() == 5, "%u completions reported for 5 batches", ndone());
    for (unsigned i = 0; i < 5 && i < ndone(); i++) {
        check(g_done[i].seq == seq[i], "completion %u is for batch %u, not %u", i,
              g_done[i].seq, seq[i]);
    }
    check(mock_fired() >= 9, "only %u waits were made to fail", mock_fired());
    check(!qatomic_read(&V.lost), "rendering has stopped");
    flush();
    check(!new_violations(), "the driver was misused");
}

static void run_in_order(void *arg)
{
    test_completions_in_order();
}

/*
 * What a batch writes back to VRAM can be more than its draws touched, and
 * every page of it has to count as busy while the batch is in flight, or
 * the device writes VRAM that the GPU is about to write, too.
 *  - More scissor rectangles drawn into a buffer than the renderer keeps
 *    apart (VK_MAX_DR): the ninth is merged with the nearest into one that
 *    spans rows neither was drawn to.
 *  - A buffer whose pitch is not a multiple of 4: buffer offsets of a copy
 *    have to be, so its rows are written back from the first on.
 */
static void test_written_back(bool odd_pitch)
{
    const uint32_t base = 0x100000, w = odd_pitch ? 30 : 64, h = odd_pitch ? 600 : 256;
    const uint32_t bpp = odd_pitch ? 1 : 4, pitch = w * bpp;
    R300DrawPacket p;

    printf("  %s\n", odd_pitch ? "a draw low in a buffer whose pitch is not a multiple of 4"
                              : "nine scissor rectangles in one buffer, eight kept apart");
    for (unsigned i = 0; i < (odd_pitch ? 2 : 10); i++) {
        /* the first loads the buffer, in a batch that is over; then far-apart
         * scissors: a page of its own each, the last one far off */
        unsigned y = odd_pitch ? 500 : i < 9 ? (i - 1) * 16 : 240;
        pkt_rt(&p, base, w, h);
        p.rt_bpp = bpp;
        p.rt_view = odd_pitch ? R300_RTV_R8U : R300_RTV_RGBA8;
        if (i) {
            p.scissor[0] = 0;
            p.scissor[1] = y;
            p.scissor[2] = 8;
            p.scissor[3] = y + 2;
        }
        check(draw_one(&p) == 0, "draw %u is rejected", i);
        if (!i) {
            flush();
        }
    }
    mock_hold(true);                    /* the GPU does not finish */
    submit();
    for (uint64_t pg = base / 4096; pg <= (base + (uint64_t)pitch * h - 1) / 4096; pg++) {
        bool busy = R->range_busy_r200(OPQ, pg * 4096 + 2048, pg * 4096 + 2112, true);
        /* what the device then does: write, if it is told the page is free */
        bool clash = false;
        for (uint64_t at = pg * 4096; !busy && !clash && at < (pg + 1) * 4096; at += 16) {
            clash = !mock_host_access(VRAM + at, 16, true, "the device writing VRAM");
        }
        check(!clash, "VRAM %#llx-%#llx counts as free while the batch writes it back",
              (unsigned long long)pg * 4096, (unsigned long long)(pg + 1) * 4096);
    }
    mock_hold(false);
    quiesce();
    check_done();
    flush();
    check(!new_violations(), "the driver was misused");
}

/*
 * The whole scenario, nothing failing, with every batch it flushes held in
 * flight first and every page of VRAM asked about.
 */
static void test_scenario_probed(void *arg)
{
    printf("  the scenario with every batch probed in flight\n");
    g_probe = true;
    scenario();
    g_probe = false;
    quiesce();
    check_done();
    check(ndone() == g_ncallback, "%u completions reported for %u batches", ndone(),
          g_ncallback);
    for (unsigned i = 0; i < g_nres[0]; i++) {
        check(g_res[0][i] == 0, "draw %u is rejected", i);
    }
    check(!new_violations(), "the driver was misused");
}

static void run_merged_rectangles(void *arg)
{
    test_written_back(false);
}

static void run_odd_pitch(void *arg)
{
    test_written_back(true);
}

static void run_in_flight(void *arg)
{
    test_all_in_flight(false);
}

static void run_wait_fails(void *arg)
{
    test_all_in_flight(true);
}

/*
 * A program that does not compile is rejected, compiled once and remembered
 * as such: unlike one whose module could not be created for lack of memory
 * (the sweep), it is not tried again.
 */
static void test_broken_shaders(void)
{
    static char bad_fs[] = "fragment program that does not compile";
    static char bad_vs[] = "vertex program that does not compile";
    static char good_fs[] = "fragment program for the vertex program that does not compile";
    R300DrawPacket p;
    unsigned c0;

    printf("  programs that do not compile\n");
    pkt_rt(&p, 0x000000, 64, 64);
    p.glsl = bad_fs;
    p.glsl_id = 77;
    c0 = g_compiles;
    for (int i = 0; i < 3; i++) {
        check(R->draw_r300(OPQ, VRAM, VRAM_SIZE, &p) == -1, "draw %d with it is accepted", i);
    }
    check(g_compiles - c0 == 2, "its two stages were compiled %u times in all, not once each",
          g_compiles - c0);

    pkt_rt(&p, 0x000000, 64, 64);
    p.glsl = good_fs;
    p.glsl_id = 78;
    p.verts = NULL;
    p.vs_glsl = bad_vs;
    p.vs_id = 55;
    p.vs_in = g_vs_in;
    p.vs_in_vecs = 3;
    p.vs_idx = g_vs_idx;
    p.vs_u = &g_vsu;
    c0 = g_compiles;
    for (int i = 0; i < 3; i++) {
        check(R->draw_r300(OPQ, VRAM, VRAM_SIZE, &p) == -1,
              "draw %d with the vertex program is accepted", i);
    }
    check(g_compiles - c0 == 2, "%u compilations, not one of the vertex program and one of "
          "the fragment program", g_compiles - c0);

    pkt_rt(&p, 0x000000, 64, 64);
    check(R->draw_r300(OPQ, VRAM, VRAM_SIZE, &p) == 0, "a draw with a program that compiles "
          "is rejected");
    flush();
    check(!new_violations(), "the driver was misused");
}

static void run_broken_shaders(void *arg)
{
    test_broken_shaders();
}

/*
 * Draws the renderer refuses, or draws less of, for what they are -- no call
 * fails.  Each is followed by an ordinary draw in the same batch: what the
 * refusal left behind must be something to go on from.
 */
static void test_refused_draws(void)
{
    static const struct { const char *what; int want; } c[] = {
        { "not this renderer's VRAM", -1 },
        { "an empty scissor", 0 },
        { "everything culled", 0 },
        { "multisampled buffers of different pitches", -1 },
        { "a colour buffer past the end of VRAM", -1 },
        { "a colour buffer at an odd address", -1 },
        { "a second colour buffer past the end of VRAM", -1 },
        { "a depth buffer at an odd address (drawn without)", 0 },
        { "a texture past the end of VRAM (drawn without)", 0 },
        { "a cube map that is not square (drawn without)", 0 },
        { "a texture wider than its pitch (drawn narrower)", 0 },
    };
    R300DrawPacket p, good;
    char desc[160];

    printf("  draws refused for what they are\n");
    pkt_rt(&good, 0x000000, 64, 64);
    pkt_z(&good, 0x010000, 4);
    for (unsigned i = 0; i < ARRAY_SIZE(c); i++) {
        uint8_t *vram = VRAM;
        int r;

        pkt_rt(&p, 0x000000, 64, 64);
        switch (i) {
        case 0: vram = VRAM + 4096; break;
        case 1: p.scissor[2] = 0; break;
        case 2: p.cull = R300_CULL_FRONT | R300_CULL_BACK; break;
        case 3: p.aa_samples = 2; pkt_z(&p, 0x0a0000, 4); p.depth.pitch = 32; break;
        case 4: p.rt_gpu_addr = VRAM_SIZE - 64; break;
        case 5: p.rt_gpu_addr = 0x000002; break;
        case 6:
            p.num_cb = 2;
            p.cb[1].gpu_addr = VRAM_SIZE - 64;
            p.cb[1].pitch = 64;
            p.cb[1].bpp = 4;
            p.cb[1].view = R300_RTV_R32U;
            break;
        case 7: pkt_z(&p, 0x010002, 4); break;
        case 8: tex(&p.tex[0], VRAM_SIZE - 64, 16, 16, R300_TEXK_RGBA8, R300_TEXDIM_2D, 1, 1); break;
        case 9: tex(&p.tex[0], 0x050000, 8, 4, R300_TEXK_RGBA8, R300_TEXDIM_CUBE, 1, 1); break;
        case 10:
            tex(&p.tex[0], 0x030000, 32, 32, R300_TEXK_RGBA8, R300_TEXDIM_2D, 1, 1);
            p.tex[0].pitch_bytes = 64;
            break;
        }
        r = R->draw_r300(OPQ, vram, VRAM_SIZE, &p);
        check(r == c[i].want, "%s: the draw returns %d, not %d", c[i].what, r, c[i].want);
        check(R->draw_r300(OPQ, VRAM, VRAM_SIZE, &good) == 0,
              "%s: the ordinary draw after it is rejected", c[i].what);
    }
    flush();
    vk_reap();
    check(!count_leaks(desc, sizeof(desc)), "objects nothing refers to: %s", desc);
    check(!new_violations(), "the driver was misused");
}

static void run_refused_draws(void *arg)
{
    test_refused_draws();
}

/* Start-up: the driver maps VRAM at an address that is not page-aligned. */
static void test_vram_misaligned(void *arg)
{
    void *opaque = NULL;
    uint8_t *vram;

    printf("  start-up with VRAM mapped off a page boundary\n");
    mock_misalign_maps(true);
    vram = ppc_mac_gpu_vulkan_alloc_vram(VRAM_SIZE, &opaque);
    mock_misalign_maps(false);
    check(!vram, "it is accepted");
    check(!V.vram_buf && !V.vram_mem, "the buffer %p and its memory %p are kept",
          (void *)V.vram_buf, (void *)V.vram_mem);
    check(!mock_alive(MT_BUFFER) && !mock_alive(MT_MEMORY),
          "%u buffers and %u memory objects are left", mock_alive(MT_BUFFER),
          mock_alive(MT_MEMORY));
    check(!new_violations(), "the driver was misused");
}

/* Each helper test in a process of its own, as the renderer's state is. */
static int g_helper_fd = -1;

static void helper_done(void)
{
    unsigned res[2] = { g_checks, g_check_fails };
    ssize_t w;

    fflush(NULL);
    w = write(g_helper_fd, res, sizeof(res));
    (void)w;
    COVERAGE_WRITE();
    _exit(0);
}

static void helper_crashed(int sig)
{
    printf("    FAIL: %s\n", sig == SIGALRM ? "it hung" : "it crashed");
    g_check_fails++;
    helper_done();
}

static int helper_child(void (*fn)(void *), void *arg, bool start)
{
    int status = 0, fd[2];
    unsigned res[2] = { 0, 1 };
    pid_t pid;

    fflush(NULL);
    if (pipe(fd)) {
        exit(2);
    }
    pid = fork();
    if (pid == 0) {
        close(fd[0]);
        g_helper_fd = fd[1];
        catch_signals(helper_crashed);
        alarm(60);
        g_checks = g_check_fails = 0;
        if (start && !renderer_start()) {
            printf("    FAIL: the renderer did not start\n");
            g_check_fails++;
        } else {
            new_violations();
            fn(arg);
        }
        helper_done();
    }
    close(fd[1]);
    if (read(fd[0], res, sizeof(res)) != sizeof(res)) {
        res[0] = 0;
        res[1] = 1;
    }
    close(fd[0]);
    waitpid(pid, &status, 0);
    if (WIFSIGNALED(status)) {
        printf("    FAIL: ended by signal %d\n", WTERMSIG(status));
        res[1] += 1;
    }
    g_checks += res[0];
    g_check_fails += res[1];
    return res[1];
}

static void run_image(void *arg)
{
    test_vk_image();
}

static void run_buffer(void *arg)
{
    test_vk_buffer();
}

struct dummy_arg {
    unsigned which;
    int fn;
    bool same_batch;
};

static void run_dummies(void *arg)
{
    struct dummy_arg *d = arg;

    test_vk_dummies(d->which, d->fn, d->same_batch);
}

static int helpers(void)
{
    static const int per_image[] = { MF_CreateImage, MF_AllocateMemory, MF_BindImageMemory,
                                     MF_CreateImageView };

    helper_child(test_vram_misaligned, NULL, false);
    helper_child(run_image, NULL, true);
    helper_child(run_buffer, NULL, true);
    helper_child(run_render_passes, NULL, true);
    helper_child(run_broken_shaders, NULL, true);
    helper_child(run_refused_draws, NULL, true);
    helper_child(run_in_flight, NULL, true);
    helper_child(run_wait_fails, NULL, true);
    for (int which = 0; which < 2; which++) {
        static const VkResult res[] = { VK_ERROR_OUT_OF_HOST_MEMORY,
                                        VK_ERROR_OUT_OF_DEVICE_MEMORY, VK_ERROR_UNKNOWN,
                                        VK_TIMEOUT };
        for (unsigned i = 0; i < ARRAY_SIZE(res); i++) {
            struct no_answer_arg a = { res[i], which };
            helper_child(run_no_answer, &a, true);
        }
    }
    helper_child(run_thread_answered, NULL, true);
    helper_child(run_flush_answered, NULL, true);
    helper_child(run_lost_thread, NULL, true);
    helper_child(run_lost_flush, NULL, true);
    helper_child(run_in_order, NULL, true);
    helper_child(test_scenario_probed, NULL, true);
    helper_child(run_merged_rectangles, NULL, true);
    helper_child(run_odd_pitch, NULL, true);
    for (int same = 0; same < 2; same++) {
        for (unsigned which = 0; which < 3; which++) {
            for (unsigned k = 0; k < ARRAY_SIZE(per_image); k++) {
                struct dummy_arg d = { which, per_image[k], same };
                helper_child(run_dummies, &d, true);
            }
        }
    }
    printf("helper contract: %u checks, %u failed\n", g_checks, g_check_fails);
    return g_check_fails != 0;
}

static void selftest_crashed(int sig)
{
    printf("stand-in driver: its test ended by signal %d\n", sig);
    fflush(NULL);
    _exit(1);
}

int main(int argc, char **argv)
{
    int i = 1, rc = 0;

    for (; i < argc && argv[i][0] == '-'; i++) {
        g_verbose |= !strcmp(argv[i], "-v");
        mock_interlock |= !strcmp(argv[i], "-i");
    }
    if (g_verbose) {
        setenv("MOCK_TRACE", "1", 1);
    }
    if (i < argc && ((!strcmp(argv[i], "case") && i + 4 < argc) ||
                     (!strcmp(argv[i], "random") && i + 3 < argc))) {
        Case c = { atoi(argv[i + 1]), atoi(argv[i + 2]), atoi(argv[i + 3]),
                   i + 4 < argc ? atoi(argv[i + 4]) : 0, i + 5 < argc ? atoi(argv[i + 5]) : 0 };
        Result r;
        char buf[320];
        int sig;
        const char *bad;

        if (!strcmp(argv[i], "random")) {
            c = (Case){ -1, .seed = atoi(argv[i + 1]), .permille = atoi(argv[i + 2]),
                        .gentle = atoi(argv[i + 3]) };
        } else if (!c.res && c.fn >= 0 && c.fn < MF_COUNT) {
            c.res = mock_listed_error(c.fn);
        }
        sig = run_case(&c, &r);
        bad = judge(&c, &r, sig, buf, sizeof(buf));
        if (c.permille) {
            printf("random seed %u, %u per mille%s: %s\n", c.seed, c.permille,
                   c.gentle ? ", rendering can go on" : "", bad ? bad : "ok");
        } else {
            printf("%s call %u%s on thread %d -> %d: %s\n",
                   c.fn == MF_COUNT ? "every" : c.fn >= 0 ? mock_fn_name[c.fn] : "nothing",
                   c.nth, c.persistent ? " and after" : "", c.thread, (int)c.res,
                   bad ? bad : "ok");
        }
        printf("  started %d, fired %u, violations %u, lost %d, rejected %u/%u then %u/%u, "
               "leaks %u (%s)\n", r.init_ok, r.fired, r.nviol, r.lost, r.rejected[0],
               r.ndraw[0], r.rejected[1], r.ndraw[1], r.leaks, r.leak_desc);
        printf("  batches cancelled %u then %u, accepted draws lost with them %u then %u; "
               "%u pauses for an answer\n", r.cancelled[0], r.cancelled[1], r.lost_draws[0],
               r.lost_draws[1], r.pauses);
        for (unsigned k = 0; k < 3 && k < r.nviol; k++) {
            printf("  %s\n", r.viol[k]);
        }
        return bad != NULL;
    }
    if (i < argc && (i + 1 < argc || (strcmp(argv[i], "helpers") && strcmp(argv[i], "sweep") &&
                                      strcmp(argv[i], "selftest")))) {
        fprintf(stderr, "usage: %s [-v] [-i] [selftest | helpers | sweep | "
                "case FN NTH PERSISTENT THREAD [RESULT] | random SEED PERMILLE GENTLE]\n",
                argv[0]);
        return 2;
    }
    if (i == argc || !strcmp(argv[i], "selftest")) {
        /* in a process of its own: it leaves the driver full of violations */
        int status = 0;
        pid_t pid;
        fflush(NULL);
        pid = fork();
        if (pid == 0) {
            int st;
            catch_signals(selftest_crashed);
            st = selftest();
            fflush(NULL);
            _exit(st);
        }
        waitpid(pid, &status, 0);
        rc |= !WIFEXITED(status) || WEXITSTATUS(status);
    }
    if (i == argc || !strcmp(argv[i], "helpers")) {
        rc |= helpers();
    }
    if (i == argc || !strcmp(argv[i], "sweep")) {
        rc |= sweep();
    }
    printf("%s\n", rc ? "RESULT: FAILED" : "RESULT: all passed");
    return rc;
}
