/*
 * PPC_VCPU_DIAG=1: see include/exec/vcpu-diag.h.  Diagnostic, not for
 * merging.  Everything but the BQL table runs on the (single) vCPU thread;
 * the BQL table is updated with the BQL held.  AI-generated (Claude).
 */
#include "qemu/osdep.h"
#include "qemu/timer.h"
#include "qemu/log.h"
#include "exec/vcpu-diag.h"
#include "exec/memory.h"
#include <pthread.h>

bool vdiag_on;
uint64_t vdiag_ij_exec[VDIAG_IJ_N];
uint32_t vdiag_ij_pending;
int vdiag_ij_kind_next;
uint64_t vdiag_r300_draws;

static const char *const ijname[VDIAG_IJ_N] = { "other", "bclr", "bcctr", "direct" };
static const char *const jcname[VDIAG_JC_N] = { "hit", "empty", "gen", "pc", "state" };
static const char *const xname[VDIAG_X_N] = { "real", "bat", "pteg1", "pteg2", "fault",
                                              "other" };

/* softmmu slow path: [access type][outcome], primary-miss classes, flag classes */
static uint64_t slow[3][3], slow_miss[3][4], slow_flags[3][5];
/* refills: [access type][from TB lookup] */
static uint64_t fills[3][2];
static bool in_lookup;
/* shadow caches after the victim TLB: [shadow][code fetch] */
/* after the victim TLB (refills only), keyed by mmu_idx, page and access type */
enum { SH_SA4, SH_SA8, SH_INF, SH_N };
static const char *const shname[SH_N] = { "sa4x64", "sa8x64", "inf" };
static uint64_t sh_hit[SH_N][2], sh_inf_overflow;
static uint64_t flush_all, flush_page;
static uint64_t xlate[VDIAG_X_N];
/* tb_lookup: [from helper][kind][jc outcome], table hit/miss [kind] */
static uint64_t look[2][VDIAG_IJ_N][VDIAG_JC_N], table_hit[VDIAG_IJ_N], table_miss[VDIAG_IJ_N];
static bool in_helper;
static uint32_t helper_kind;

typedef struct Sh {
    uint64_t key, lru;
    uint32_t gen;
} Sh;

#define SH_TOMB 0xFFFFFFFFu
#define INF_BITS 20
static Sh sa4[64][4], sa8[64][8];
/* victim-TLB models: sizes 8/16/32/64 per mmu_idx, round robin, swap on hit */
#define VM_N 4
#define VM_IDX 16
static const int vm_size[VM_N] = { 8, 16, 32, 64 };
static Sh vm[VM_N][VM_IDX][64];
static unsigned vm_next[VM_N][VM_IDX];
static uint64_t vm_hit[VM_N], vm_miss, real_victim_hit;
static Sh *inf;
static uint32_t sh_gen = 1;
static uint64_t sh_tick;

static bool fa_access(Sh *a, int n, uint64_t key)
{
    int victim = 0;

    for (int i = 0; i < n; i++) {
        if (a[i].gen == sh_gen && a[i].key == key) {
            a[i].lru = sh_tick;
            return true;
        }
        if (a[victim].gen == sh_gen && (a[i].gen != sh_gen || a[i].lru < a[victim].lru)) {
            victim = i;
        }
    }
    a[victim] = (Sh){ key, sh_tick, sh_gen };
    return false;
}

static void fa_remove(Sh *a, int n, uint64_t key)
{
    for (int i = 0; i < n; i++) {
        if (a[i].gen == sh_gen && a[i].key == key) {
            a[i].gen = 0;
        }
    }
}

static unsigned set_of(uint64_t key)
{
    return (key ^ (key >> 6) ^ (key >> 12)) & 63;
}

static bool inf_access(uint64_t key, bool insert)
{
    uint64_t h = (key * 0x9E3779B97F4A7C15ull) >> (64 - INF_BITS);
    Sh *tomb = NULL;

    for (int probe = 0; probe < 64; probe++) {
        Sh *e = &inf[(h + probe) & ((1u << INF_BITS) - 1)];
        if (e->gen == sh_gen && e->key == key) {
            if (!insert) {
                e->gen = SH_TOMB;           /* removal */
            }
            return true;
        }
        if (e->gen == SH_TOMB) {
            if (!tomb) {
                tomb = e;
            }
            continue;
        }
        if (e->gen != sh_gen) {             /* empty */
            if (insert) {
                *(tomb ? tomb : e) = (Sh){ key, 0, sh_gen };
            }
            return false;
        }
    }
    if (insert) {
        if (tomb) {
            *tomb = (Sh){ key, 0, sh_gen };
        } else {
            sh_inf_overflow++;
        }
    }
    return false;
}

static void report(double t);

static void vdiag_final(void)
{
    qemu_log("vdiag final\n");
    report(get_clock() / 1e9);
}

static void __attribute__((constructor)) vdiag_init(void)
{
    const char *e = getenv("PPC_VCPU_DIAG");

    vdiag_on = e && e[0] == '1';
    if (vdiag_on) {
        inf = g_new0(Sh, 1u << INF_BITS);
        atexit(vdiag_final);
    }
}

void vdiag_slow(int type, int outcome, int miss_class, int flags)
{
    slow[type][outcome]++;
    slow_miss[type][miss_class]++;
    for (int i = 0; i < 5; i++) {
        if (flags & (1 << i)) {
            slow_flags[type][i]++;
        }
    }
}

/* MMIO accesses by region (vCPU thread). */
typedef struct MmioStat {
    const MemoryRegion *mr;
    uint64_t n[2];
} MmioStat;
static MmioStat mmio[48];

void vdiag_mmio(const void *mr, bool store)
{
    for (int i = 0; i < 48; i++) {
        if (mmio[i].mr == mr || !mmio[i].mr) {
            mmio[i].mr = mr;
            mmio[i].n[store]++;
            return;
        }
    }
}

void vdiag_primary_miss(int mmu_idx, uint64_t page, uint64_t occupant, bool victim_hit)
{
    if (mmu_idx < 0 || mmu_idx >= VM_IDX) {
        return;
    }
    vm_miss++;
    real_victim_hit += victim_hit;
    for (int m = 0; m < VM_N; m++) {
        Sh *v = vm[m][mmu_idx];
        int n = vm_size[m], at = -1;
        for (int i = 0; i < n; i++) {
            if (v[i].gen == sh_gen && v[i].key == page) {
                at = i;
                break;
            }
        }
        if (at >= 0) {
            vm_hit[m]++;
            /* swap: the requested page goes to the primary, the occupant here */
            if (occupant != (uint64_t)-1) {
                v[at] = (Sh){ occupant, 0, sh_gen };
            } else {
                v[at].gen = 0;
            }
        } else if (occupant != (uint64_t)-1) {
            unsigned w = vm_next[m][mmu_idx]++ % n;
            v[w] = (Sh){ occupant, 0, sh_gen };
        }
    }
}

void vdiag_fill(int mmu_idx, uint64_t page, int type)
{
    uint64_t key = ((uint64_t)mmu_idx << 52) | ((uint64_t)type << 48) | (page >> 12);
    bool code = type == 2;
    unsigned s = set_of(key);

    fills[type][in_lookup]++;
    sh_tick++;
    sh_hit[SH_SA4][code] += fa_access(sa4[s], 4, key);
    sh_hit[SH_SA8][code] += fa_access(sa8[s], 8, key);
    sh_hit[SH_INF][code] += inf_access(key, true);
}

void vdiag_flush_all(void)
{
    flush_all++;
    if (++sh_gen == SH_TOMB) {
        sh_gen = 1;
    }
}

void vdiag_flush_page(int mmu_idx, uint64_t page)
{
    flush_page++;
    for (int t = 0; t < 3; t++) {
        uint64_t key = ((uint64_t)mmu_idx << 52) | ((uint64_t)t << 48) | (page >> 12);
        unsigned s = set_of(key);
        fa_remove(sa4[s], 4, key);
        fa_remove(sa8[s], 8, key);
        inf_access(key, false);
    }
    if (mmu_idx >= 0 && mmu_idx < VM_IDX) {
        for (int m = 0; m < VM_N; m++) {
            fa_remove(vm[m][mmu_idx], vm_size[m], page);
        }
    }
}

void vdiag_xlate(int kind)
{
    xlate[kind]++;
}

void vdiag_helper_enter(void)
{
    in_helper = true;
    helper_kind = vdiag_ij_pending;
    vdiag_ij_pending = VDIAG_IJ_OTHER;
}

void vdiag_helper_exit(void)
{
    in_helper = false;
}

/* jc: VDIAG_JC_*; table_hit: -1 not looked up, 0 miss, 1 hit.  in_lookup
 * brackets the table search (refills there are from TB lookup). */
void vdiag_lookup(bool unused, int jc, int table)
{
    unsigned k = in_helper ? helper_kind : VDIAG_IJ_OTHER;

    look[in_helper][k][jc]++;
    if (table == 1) {
        table_hit[k]++;
    } else if (table == 0) {
        table_miss[k]++;
    }
}

void vdiag_lookup_table(bool begin)
{
    in_lookup = begin;
}

/* BQL waits, keyed by waiter thread + call site and owner thread + call site. */
typedef struct BqlStat {
    char waiter[16], owner[16];
    const char *wfile, *ofile;
    int wline, oline;
    uint64_t n, ns;
} BqlStat;
static BqlStat bql[256];
static int nbql;
static char owner_name[16];
static const char *owner_file;
static int owner_line;
static __thread char my_name[16];
static __thread char snap_name[16];
static __thread const char *snap_file;
static __thread int snap_line;

static const char *thread_name(void)
{
    if (!my_name[0]) {
        pthread_getname_np(pthread_self(), my_name, sizeof(my_name));
        if (!my_name[0]) {
            snprintf(my_name, sizeof(my_name), "main");
        }
    }
    return my_name;
}

void vdiag_bql_wait_begin(void)
{
    memcpy(snap_name, owner_name, sizeof(snap_name));
    snap_name[15] = 0;
    snap_file = owner_file;
    snap_line = owner_line;
}

void vdiag_bql_wait(const char *file, int line, int64_t ns)
{
    const char *me = thread_name();
    BqlStat *b = NULL;

    for (int i = 0; i < nbql; i++) {
        if (bql[i].wfile == file && bql[i].wline == line && bql[i].ofile == snap_file &&
            bql[i].oline == snap_line && !strcmp(bql[i].waiter, me) &&
            !strcmp(bql[i].owner, snap_name)) {
            b = &bql[i];
            break;
        }
    }
    if (!b) {
        if (nbql == 256) {
            b = &bql[255];          /* overflow bucket */
        } else {
            b = &bql[nbql++];
            snprintf(b->waiter, sizeof(b->waiter), "%s", me);
            snprintf(b->owner, sizeof(b->owner), "%s", snap_name);
            b->wfile = file;
            b->wline = line;
            b->ofile = snap_file;
            b->oline = snap_line;
        }
    }
    b->n++;
    b->ns += ns;
}

void vdiag_bql_acquired(const char *file, int line)
{
    memcpy(owner_name, thread_name(), sizeof(owner_name));
    owner_file = file;
    owner_line = line;
}

static const char *base(const char *f)
{
    const char *s = f ? strrchr(f, '/') : NULL;
    return s ? s + 1 : (f ? f : "?");
}

static void report(double t)
{
    qemu_log("vdiag t %.3f draws %" PRIu64 "\n", t, qatomic_read(&vdiag_r300_draws));
    for (int a = 0; a < 3; a++) {
        static const char *an[3] = { "load", "store", "fetch" };
        qemu_log("vdiag slow %s hit %" PRIu64 " victim %" PRIu64 " fill %" PRIu64
                 " miss_empty %" PRIu64 " miss_conflict %" PRIu64 " miss_perm %" PRIu64
                 " mmio %" PRIu64 " notdirty %" PRIu64
                 " watch %" PRIu64 " discard %" PRIu64 " otherflag %" PRIu64
                 " fills_normal %" PRIu64 " fills_tblookup %" PRIu64 "\n", an[a],
                 slow[a][0], slow[a][1], slow[a][2], slow_miss[a][1], slow_miss[a][2],
                 slow_miss[a][3], slow_flags[a][0],
                 slow_flags[a][1], slow_flags[a][2], slow_flags[a][3], slow_flags[a][4],
                 fills[a][0], fills[a][1]);
    }
    for (int s = 0; s < SH_N; s++) {
        qemu_log("vdiag shadow %s data %" PRIu64 " code %" PRIu64 "\n", shname[s],
                 sh_hit[s][0], sh_hit[s][1]);
    }
    qemu_log("vdiag victim primary_misses %" PRIu64 " real_hits %" PRIu64
             " model8 %" PRIu64 " model16 %" PRIu64 " model32 %" PRIu64 " model64 %" PRIu64 "\n",
             vm_miss, real_victim_hit, vm_hit[0], vm_hit[1], vm_hit[2], vm_hit[3]);
    qemu_log("vdiag flush all %" PRIu64 " page %" PRIu64 " inf_overflow %" PRIu64 "\n",
             flush_all, flush_page, sh_inf_overflow);
    qemu_log("vdiag xlate");
    for (int k = 0; k < VDIAG_X_N; k++) {
        qemu_log(" %s %" PRIu64, xname[k], xlate[k]);
    }
    qemu_log("\n");
    for (int k = 0; k < VDIAG_IJ_N; k++) {
        qemu_log("vdiag ij %s exec %" PRIu64 " table_hit %" PRIu64 " table_miss %" PRIu64,
                 ijname[k], qatomic_read(&vdiag_ij_exec[k]), table_hit[k], table_miss[k]);
        for (int h = 0; h < 2; h++) {
            qemu_log(" %s", h ? "helper" : "loop");
            for (int j = 0; j < VDIAG_JC_N; j++) {
                qemu_log(" %s %" PRIu64, jcname[j], look[h][k][j]);
            }
        }
        qemu_log("\n");
    }
    for (int i = 0; i < 48 && mmio[i].mr; i++) {
        qemu_log("vdiag mmio %s load %" PRIu64 " store %" PRIu64 "\n",
                 memory_region_name(mmio[i].mr), mmio[i].n[0], mmio[i].n[1]);
    }
    for (int i = 0; i < nbql; i++) {
        qemu_log("vdiag bql waiter %s %s:%d owner %s %s:%d n %" PRIu64 " ms %.1f\n",
                 bql[i].waiter, base(bql[i].wfile), bql[i].wline, bql[i].owner,
                 base(bql[i].ofile), bql[i].oline, bql[i].n, bql[i].ns / 1e6);
    }
}

void vdiag_maybe_report(void)
{
    static int64_t next;
    int64_t now = get_clock();

    if (now < next) {
        return;
    }
    next = now + 10 * NANOSECONDS_PER_SECOND;
    report(now / 1e9);
}
