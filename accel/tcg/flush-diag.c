/*
 * PPC_FLUSH_DIAG=1: see include/exec/flush-diag.h.  Diagnostic, not for
 * merging.  All hooks run on the single vCPU thread.  AI-generated (Claude).
 */
#include "qemu/osdep.h"
#include "qemu/host-utils.h"
#include "exec/flush-diag.h"

bool fdiag_on;

#define FD_RECENT 32

static struct {
    uint64_t sync, mtsr, mtsr_chg;
    uint64_t srchg_ev, dist[8], distinct, srchg_bat, srchg_sdr1, bat_only_ev;
    uint64_t sdr1;
    uint64_t bat_store[2], bat_skip[2], bat_pages[2], bat_whole[2];
    uint64_t req[FD_NREQ][FD_NCAUSE], req_pages;
    uint64_t exec[FD_NCAUSE], asked_bits, cleaned_bits;
    uint64_t jc_tlb, jc_all, jc_page, lp_esc, slot_hit, slot_evict;
    uint64_t tlbie, tlbie_seg[16], tlbie_per_req[6];
} fd;

static unsigned fd_pending, fd_inflight, fd_tlbie_since;
static bool fd_bat_since, fd_sdr1_since;
static uint32_t fd_last_sr[16];
static uint64_t fd_recent[FD_RECENT];
static unsigned fd_nrecent;
static GHashTable *fd_seen;
static int64_t fd_next;
static unsigned fd_calls;

static void fd_report(const char *tag)
{
    double t = g_get_real_time() / 1e6;

    fprintf(stderr, "fdiag%s T=%.3f sync=%" PRIu64 " mtsr=%" PRIu64
            " mtsr_chg=%" PRIu64 " srchg_ev=%" PRIu64
            " dist=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64
            "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 " distinct=%" PRIu64
            " srchg_bat=%" PRIu64 " srchg_sdr1=%" PRIu64 " bat_only_ev=%" PRIu64
            " sdr1=%" PRIu64 "\n", tag, t, fd.sync, fd.mtsr, fd.mtsr_chg,
            fd.srchg_ev, fd.dist[0], fd.dist[1], fd.dist[2], fd.dist[3],
            fd.dist[4], fd.dist[5], fd.dist[6], fd.dist[7], fd.distinct,
            fd.srchg_bat, fd.srchg_sdr1, fd.bat_only_ev, fd.sdr1);
    fprintf(stderr, "fdiag-bat T=%.3f i_store=%" PRIu64 " i_skip=%" PRIu64
            " i_pages=%" PRIu64 " i_whole=%" PRIu64 " d_store=%" PRIu64
            " d_skip=%" PRIu64 " d_pages=%" PRIu64 " d_whole=%" PRIu64 "\n", t,
            fd.bat_store[0], fd.bat_skip[0], fd.bat_pages[0], fd.bat_whole[0],
            fd.bat_store[1], fd.bat_skip[1], fd.bat_pages[1], fd.bat_whole[1]);
    fprintf(stderr, "fdiag-exec T=%.3f asked_bits=%" PRIu64 " cleaned_bits=%"
            PRIu64 " jc_tlb=%" PRIu64 " jc_all=%" PRIu64 " jc_page=%" PRIu64
            " lp_esc=%" PRIu64 " slot_hit=%" PRIu64 " slot_evict=%" PRIu64
            " req_pages=%" PRIu64 "\n", t, fd.asked_bits, fd.cleaned_bits,
            fd.jc_tlb, fd.jc_all, fd.jc_page, fd.lp_esc, fd.slot_hit,
            fd.slot_evict, fd.req_pages);
    fprintf(stderr, "fdiag-tlbie T=%.3f tlbie=%" PRIu64 " per_req=%" PRIu64 "/%"
            PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 " seg=", t,
            fd.tlbie, fd.tlbie_per_req[0], fd.tlbie_per_req[1],
            fd.tlbie_per_req[2], fd.tlbie_per_req[3], fd.tlbie_per_req[4],
            fd.tlbie_per_req[5]);
    for (int i = 0; i < 16; i++) {
        fprintf(stderr, "%s%" PRIu64, i ? "/" : "", fd.tlbie_seg[i]);
    }
    fprintf(stderr, "\n");
    for (int c = 0; c < FD_NCAUSE; c++) {
        if (fd.exec[c]) {
            fprintf(stderr, "fdiag-x T=%.3f causes=%02x n=%" PRIu64 "\n", t, c,
                    fd.exec[c]);
        }
        for (int r = 0; r < FD_NREQ; r++) {
            if (fd.req[r][c]) {
                fprintf(stderr, "fdiag-r T=%.3f req=%d causes=%02x n=%" PRIu64
                        "\n", t, r, c, fd.req[r][c]);
            }
        }
    }
}

static void fd_final(void)
{
    fd_report("-final");
}

static void __attribute__((constructor)) fd_init(void)
{
    const char *e = getenv("PPC_FLUSH_DIAG");

    fdiag_on = e && e[0] == '1';
    if (fdiag_on) {
        fd_seen = g_hash_table_new(g_int64_hash, g_int64_equal);
        atexit(fd_final);
    }
}

void fdiag_pending(int cause)
{
    fd_pending |= cause;
}

void fdiag_tlbie(uint32_t ea)
{
    fd.tlbie++;
    fd.tlbie_seg[ea >> 28]++;
    fd_tlbie_since++;
    fd_pending |= FD_TLBIE;
}

void fdiag_inflight(int cause)
{
    fd_inflight |= cause;
    if (cause & FD_SDR1) {
        fd.sdr1++;
        fd_sdr1_since = true;
    }
}

void fdiag_mtsr(bool changed)
{
    fd.mtsr++;
    if (changed) {
        fd.mtsr_chg++;
        fd_pending |= FD_SR;
    }
}

void fdiag_bat(bool data, bool skipped, unsigned pages, bool whole)
{
    fd.bat_store[data]++;
    fd.bat_skip[data] += skipped;
    fd.bat_pages[data] += pages;
    fd.bat_whole[data] += whole;
    fd_bat_since = true;
}

/* the SR set changed at this context-synchronizing event */
static void fd_sr_event(const uint32_t *sr)
{
    uint64_t h = 0xcbf29ce484222325ull;
    unsigned p;

    for (int i = 0; i < 16; i++) {
        h = (h ^ sr[i]) * 0x100000001b3ull;
    }
    fd.srchg_ev++;
    fd.srchg_bat += fd_bat_since;
    fd.srchg_sdr1 += fd_sdr1_since;
    for (p = 0; p < fd_nrecent && fd_recent[p] != h; p++) {
    }
    /* position in the recent list (0 = the set just left can't recur):
     * buckets 0, 1, 2, 3, 4-7, 8-15, 16-31, new */
    fd.dist[p >= fd_nrecent ? 7 : p == 0 ? 0 : p < 4 ? p :
            p < 8 ? 4 : p < 16 ? 5 : 6]++;
    if (p >= fd_nrecent) {
        p = fd_nrecent < FD_RECENT ? fd_nrecent++ : FD_RECENT - 1;
    }
    memmove(&fd_recent[1], &fd_recent[0], p * sizeof(fd_recent[0]));
    fd_recent[0] = h;
    if (!g_hash_table_contains(fd_seen, &h)) {
        uint64_t *k = g_new(uint64_t, 1);
        *k = h;
        g_hash_table_add(fd_seen, k);
        fd.distinct++;
    }
}

int fdiag_sync(unsigned need, bool global, const uint32_t *sr)
{
    fd.sync++;
    if (memcmp(sr, fd_last_sr, sizeof(fd_last_sr))) {
        fd_sr_event(sr);
        memcpy(fd_last_sr, sr, sizeof(fd_last_sr));
        fd_bat_since = fd_sdr1_since = false;
    } else if (fd_bat_since || fd_sdr1_since) {
        fd.bat_only_ev++;
        fd_bat_since = fd_sdr1_since = false;
    }
    if (!(++fd_calls & 4095)) {
        int64_t now = g_get_real_time();
        if (now >= fd_next) {
            fd_next = now + 10000000;
            fd_report("");
        }
    }
    return fd_pending;
}

void fdiag_request(int outcome, int causes, unsigned pages)
{
    fd.req[outcome][causes & (FD_NCAUSE - 1)]++;
    switch (outcome) {
    case FD_REQ_GLOBAL:
    case FD_REQ_LOCAL:
        if (fd_pending & FD_TLBIE) {
            /* tlbies consumed by this request: 1, 2-3, 4-7, 8-15, 16-63, 64+ */
            unsigned n = fd_tlbie_since;
            fd.tlbie_per_req[n < 2 ? 0 : n < 4 ? 1 : n < 8 ? 2 : n < 16 ? 3 :
                             n < 64 ? 4 : 5]++;
            fd_tlbie_since = 0;
        }
        fd_inflight |= fd_pending;
        fd_pending = 0;
        break;
    case FD_REQ_PAGE:
        fd.req_pages += pages;
        fd_pending &= ~FD_TLBIE;
        fd_tlbie_since = 0;
        break;
    case FD_REQ_SRSW:
        fd_pending &= ~FD_SR;
        break;
    }
}

void fdiag_slot(bool hit)
{
    if (hit) {
        fd.slot_hit++;
    } else {
        fd.slot_evict++;
    }
}

void fdiag_flush_exec(uint16_t asked, uint16_t cleaned)
{
    fd.exec[fd_inflight & (FD_NCAUSE - 1)]++;
    fd_inflight = 0;
    fd.asked_bits += ctpop16(asked);
    fd.cleaned_bits += ctpop16(cleaned);
}

void fdiag_jc_bump(bool from_tlb)
{
    if (from_tlb) {
        fd.jc_tlb++;            /* subset of jc_all */
    } else {
        fd.jc_all++;
    }
}

void fdiag_jc_page(void)
{
    fd.jc_page++;
}

void fdiag_lp_escalate(void)
{
    fd.lp_esc++;
}
