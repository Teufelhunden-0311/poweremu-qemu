/*
 * PPC_TBINV_DIAG=1: see include/exec/tb-diag.h.  Diagnostic, not for
 * merging.  The hooks can run on the vCPU and on device threads (DMA into a
 * code page), so one mutex covers all of it.  AI-generated (Claude).
 */
#include "qemu/osdep.h"
#include "qemu/thread.h"
#include "exec/exec-all.h"
#include "exec/translation-block.h"
#include "tcg/tcg.h"
#include "exec/tb-diag.h"

bool tbdiag_on;
__thread int tbdiag_cause;

#define TBD_EV_MAX   64      /* identities remembered per event (dedupe) */
#define TBD_CS_MAX   8       /* cs_base values tried for a cross-slot copy */
#define TBD_TOP      24      /* pages in the exit table */

typedef struct {
    uint64_t pc, cs_base;
    uint32_t flags, cflags;
} TbdKey;

enum { ST_LIVE = 1, ST_INVAL = 2 };

typedef struct {
    uint64_t phys;
    uint32_t epoch;          /* whole flushes before it was made */
    uint8_t state, cause;    /* ST_*, and the TBD_* that removed it */
} TbdVal;

typedef struct {
    uint64_t ev[TBD_NCAUSE], ev_none[TBD_NCAUSE], tbs[TBD_NCAUSE];
    uint64_t writer_same, writer_other;   /* writer TB on this page or not */
    uint64_t sample_pc, writer_pc;
    uint32_t cs_mask;
} TbdPage;

static struct {
    uint64_t gen, gen_new, gen_reinval[TBD_NCAUSE], gen_reflush, gen_remap;
    uint64_t gen_other, gen_oneshot, gen_dup, gen_reinval_dup;
    uint64_t inval, inval_noev, inval_cs[TBD_CS_MAX + 1];
    uint64_t ev[TBD_NCAUSE], ev_none[TBD_NCAUSE], ev_tbs[TBD_NCAUSE];
    uint64_t ev_uniq[TBD_NCAUSE], ev_multi_cs[TBD_NCAUSE];
    uint64_t ev_writer_same, ev_writer_other, ev_writer_unknown;
    uint64_t flush, flush_full, pages_all;
} td;

static QemuMutex td_lock;
static GHashTable *td_keys;          /* TbdKey -> TbdVal */
static GHashTable *td_pages;         /* phys page -> TbdPage */
static GHashTable *td_win;           /* phys pages with a removal this second */
static uint32_t td_epoch;
static int64_t td_next;
static bool td_flush_full_req;

/* the event in progress (one at a time: callers hold the page locks) */
static __thread struct {
    bool on;
    uint64_t page;
    uintptr_t retaddr;
    unsigned n, nu;
    uint32_t cs_mask;
    TbdKey seen[TBD_EV_MAX];
} ev;

static guint key_hash(gconstpointer p)
{
    const TbdKey *k = p;
    uint64_t h = k->pc * 0x9E3779B97F4A7C15ull ^ k->cs_base * 0xC2B2AE3D27D4EB4Full ^
                 ((uint64_t)k->flags << 32 | k->cflags) * 0x165667B19E3779F9ull;
    return h ^ (h >> 29);
}

static gboolean key_equal(gconstpointer a, gconstpointer b)
{
    return !memcmp(a, b, sizeof(TbdKey));
}

static void td_report(const char *tag)
{
    double t = g_get_real_time() / 1e6;

    fprintf(stderr, "tbdiag%s T=%.3f gen=%" PRIu64 " new=%" PRIu64
            " reinval=%" PRIu64 "/%" PRIu64 "/%" PRIu64 " reflush=%" PRIu64
            " remap=%" PRIu64 " other=%" PRIu64 " oneshot=%" PRIu64
            " dup=%" PRIu64 " reinval_dup=%" PRIu64 " flush=%" PRIu64
            " flush_full=%" PRIu64 " nb_tbs=%zu code_kb=%zu/%zu\n", tag, t,
            td.gen, td.gen_new, td.gen_reinval[TBD_OTHER],
            td.gen_reinval[TBD_STORE], td.gen_reinval[TBD_DMA], td.gen_reflush,
            td.gen_remap, td.gen_other, td.gen_oneshot, td.gen_dup,
            td.gen_reinval_dup, td.flush, td.flush_full, tcg_nb_tbs(),
            tcg_code_size() >> 10, tcg_code_capacity() >> 10);
    fprintf(stderr, "tbdiag-inv%s T=%.3f inval=%" PRIu64 " noev=%" PRIu64
            " cs=", tag, t, td.inval, td.inval_noev);
    for (int i = 0; i <= TBD_CS_MAX; i++) {
        fprintf(stderr, "%s%" PRIu64, i ? "/" : "", td.inval_cs[i]);
    }
    fprintf(stderr, " pages_win=%u pages_all=%" PRIu64 " writer=%" PRIu64 "/%"
            PRIu64 "/%" PRIu64 "\n", g_hash_table_size(td_win), td.pages_all,
            td.ev_writer_same, td.ev_writer_other, td.ev_writer_unknown);
    for (int c = 0; c < TBD_NCAUSE; c++) {
        static const char *const name[] = { "other", "store", "dma" };
        fprintf(stderr, "tbdiag-ev%s T=%.3f cause=%s ev=%" PRIu64 " none=%"
                PRIu64 " tbs=%" PRIu64 " uniq=%" PRIu64 " multi_cs=%" PRIu64
                "\n", tag, t, name[c], td.ev[c], td.ev_none[c], td.ev_tbs[c],
                td.ev_uniq[c], td.ev_multi_cs[c]);
    }
    g_hash_table_remove_all(td_win);
}

static void td_tick(void)
{
    int64_t now = g_get_monotonic_time();

    if (now >= td_next) {
        td_next = now + 1000000;
        td_report("");
    }
}

static gint page_cmp(gconstpointer a, gconstpointer b, gpointer u)
{
    const TbdPage *pa = g_hash_table_lookup(td_pages, a);
    const TbdPage *pb = g_hash_table_lookup(td_pages, b);
    uint64_t ta = pa->tbs[0] + pa->tbs[1] + pa->tbs[2];
    uint64_t tb = pb->tbs[0] + pb->tbs[1] + pb->tbs[2];

    return ta < tb ? 1 : ta > tb ? -1 : 0;
}

static void td_final(void)
{
    GList *pages, *l;
    int n = 0;

    qemu_mutex_lock(&td_lock);
    td_report("-final");
    pages = g_list_sort_with_data(g_hash_table_get_keys(td_pages), page_cmp,
                                  NULL);
    for (l = pages; l && n < TBD_TOP; l = l->next, n++) {
        const TbdPage *p = g_hash_table_lookup(td_pages, l->data);
        fprintf(stderr, "tbdiag-page phys=%08" PRIx64 " tbs=%" PRIu64 "/%"
                PRIu64 "/%" PRIu64 " ev=%" PRIu64 "/%" PRIu64 "/%" PRIu64
                " none=%" PRIu64 "/%" PRIu64 "/%" PRIu64 " cs_mask=%x"
                " writer_same=%" PRIu64 " writer_other=%" PRIu64
                " sample_pc=%08" PRIx64 " writer_pc=%08" PRIx64 "\n",
                *(uint64_t *)l->data, p->tbs[0], p->tbs[1], p->tbs[2],
                p->ev[0], p->ev[1], p->ev[2], p->ev_none[0], p->ev_none[1],
                p->ev_none[2], p->cs_mask, p->writer_same, p->writer_other,
                p->sample_pc, p->writer_pc);
    }
    g_list_free(pages);
    qemu_mutex_unlock(&td_lock);
}

static void __attribute__((constructor)) td_init(void)
{
    const char *e = getenv("PPC_TBINV_DIAG");

    tbdiag_on = e && e[0] == '1';
    if (tbdiag_on) {
        qemu_mutex_init(&td_lock);
        td_keys = g_hash_table_new_full(key_hash, key_equal, g_free, g_free);
        td_pages = g_hash_table_new_full(g_int64_hash, g_int64_equal, g_free,
                                         g_free);
        td_win = g_hash_table_new_full(g_int64_hash, g_int64_equal, g_free,
                                       NULL);
        atexit(td_final);
    }
}

static TbdPage *td_page(uint64_t page)
{
    TbdPage *p = g_hash_table_lookup(td_pages, &page);

    if (!p) {
        uint64_t *k = g_new(uint64_t, 1);
        *k = page;
        p = g_new0(TbdPage, 1);
        g_hash_table_insert(td_pages, k, p);
        td.pages_all++;
    }
    return p;
}

void tbdiag_gen(uint64_t pc, uint64_t cs_base, uint32_t flags,
                uint32_t cflags, uint64_t phys_pc)
{
    TbdKey k = { pc, cs_base, flags, cflags & ~CF_INVALID };
    TbdVal *v;
    bool dup = false;

    qemu_mutex_lock(&td_lock);
    td.gen++;
    if ((cflags & CF_COUNT_MASK) == 1) {
        td.gen_oneshot++;
    }
    for (uint64_t cs = 0; cs < TBD_CS_MAX && !dup; cs++) {
        if (cs != cs_base) {
            TbdKey o = k;
            TbdVal *ov;
            o.cs_base = cs;
            ov = g_hash_table_lookup(td_keys, &o);
            dup = ov && ov->state == ST_LIVE && ov->epoch == td_epoch;
        }
    }
    td.gen_dup += dup;
    v = g_hash_table_lookup(td_keys, &k);
    if (!v) {
        td.gen_new++;
        v = g_new0(TbdVal, 1);
        g_hash_table_insert(td_keys, g_memdup2(&k, sizeof(k)), v);
    } else if (v->state == ST_INVAL) {
        td.gen_reinval[v->cause]++;
        td.gen_reinval_dup += dup;
    } else if (v->epoch != td_epoch) {
        td.gen_reflush++;
    } else if (v->phys != phys_pc) {
        td.gen_remap++;
    } else {
        td.gen_other++;
    }
    v->state = ST_LIVE;
    v->epoch = td_epoch;
    v->phys = phys_pc;
    td_tick();
    qemu_mutex_unlock(&td_lock);
}

void tbdiag_inval(uint64_t pc, uint64_t cs_base, uint32_t flags,
                  uint32_t cflags)
{
    TbdKey k = { pc, cs_base, flags, cflags & ~CF_INVALID };
    TbdVal *v;

    qemu_mutex_lock(&td_lock);
    td.inval++;
    td.inval_cs[MIN(cs_base, TBD_CS_MAX)]++;
    v = g_hash_table_lookup(td_keys, &k);
    if (v) {
        v->state = ST_INVAL;
        v->cause = ev.on ? tbdiag_cause : TBD_OTHER;
    }
    if (ev.on) {
        TbdKey u = k;
        unsigned i;

        u.cs_base = 0;
        ev.n++;
        ev.cs_mask |= 1u << MIN(cs_base, 31);
        for (i = 0; i < MIN(ev.nu, TBD_EV_MAX); i++) {
            if (key_equal(&ev.seen[i], &u)) {
                break;
            }
        }
        if (i == MIN(ev.nu, TBD_EV_MAX)) {
            if (i < TBD_EV_MAX) {
                ev.seen[i] = u;
            }
            ev.nu++;
        }
        if (ev.n == 1) {
            td_page(ev.page)->sample_pc = pc;
        }
    } else {
        td.inval_noev++;
    }
    qemu_mutex_unlock(&td_lock);
}

void tbdiag_ev_begin(uint64_t start, uintptr_t retaddr)
{
    ev.on = true;
    ev.page = start & TARGET_PAGE_MASK;
    ev.retaddr = retaddr;
    ev.n = ev.nu = 0;
    ev.cs_mask = 0;
}

void tbdiag_ev_end(void)
{
    int c = tbdiag_cause;
    TbdPage *p;

    if (!ev.on) {
        return;
    }
    ev.on = false;
    qemu_mutex_lock(&td_lock);
    p = td_page(ev.page);
    if (!ev.n) {
        td.ev_none[c]++;
        p->ev_none[c]++;
    } else {
        TranslationBlock *w = ev.retaddr ? tcg_tb_lookup(ev.retaddr) : NULL;

        td.ev[c]++;
        td.ev_tbs[c] += ev.n;
        td.ev_uniq[c] += ev.nu;
        td.ev_multi_cs[c] += !!(ev.cs_mask & (ev.cs_mask - 1));
        p->ev[c]++;
        p->tbs[c] += ev.n;
        p->cs_mask |= ev.cs_mask;
        if (w) {
            bool same = (tb_page_addr0(w) & TARGET_PAGE_MASK) == ev.page;
            td.ev_writer_same += same;
            td.ev_writer_other += !same;
            p->writer_same += same;
            p->writer_other += !same;
            p->writer_pc = w->pc;
        } else {
            td.ev_writer_unknown++;
        }
        g_hash_table_add(td_win, g_memdup2(&ev.page, sizeof(ev.page)));
    }
    td_tick();
    qemu_mutex_unlock(&td_lock);
}

void tbdiag_flush_full(void)
{
    qatomic_set(&td_flush_full_req, true);
}

void tbdiag_flush(void)
{
    qemu_mutex_lock(&td_lock);
    td.flush++;
    if (qatomic_xchg(&td_flush_full_req, false)) {
        td.flush_full++;
    }
    td_epoch++;
    qemu_mutex_unlock(&td_lock);
}
