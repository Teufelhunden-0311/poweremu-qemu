/*
 * PPC_FLUSH_DIAG=1 (diagnostic, not for merging): why the QEMU TLB and the
 * TB jump cache get flushed on a 32-bit hash-MMU guest.
 *   - setters: segment register changes, tlbie, tlbia, BAT stores, SDR1;
 *     each leaves a cause bit; bits combine until consumed;
 *   - requests: what check_tlb_flush did (global / local / page / SR-slot
 *     switch) by the combined cause bits pending then;
 *   - executions: each whole-TLB flush on the vCPU with the causes behind
 *     it, the MMU indexes asked for and actually cleaned, and the jump-cache
 *     generation bumps it made; other jump-cache invalidations; per-index
 *     flushes forced by a page flush inside a large page;
 *   - SR-set recurrence at the context-synchronizing event (stack distance
 *     over recently seen sets, kept across flushes) and BAT/SDR1 changes
 *     since the previous event; SR-slot hits and evictions.
 * Cumulative lines every 10 s and at exit.  AI-generated (Claude).
 */
#ifndef EXEC_FLUSH_DIAG_H
#define EXEC_FLUSH_DIAG_H

extern bool fdiag_on;

/* cause bits */
enum {
    FD_SR = 1,          /* a segment register value changed */
    FD_TLBIE = 2,       /* tlbie (helper ran) */
    FD_TLBIA = 4,
    FD_IBAT = 8,        /* IBAT store that escalated to a whole flush */
    FD_DBAT = 16,       /* DBAT store that escalated to a whole flush */
    FD_SDR1 = 32,
    FD_NCAUSE = 64
};

/* check_tlb_flush outcomes */
enum { FD_REQ_NONE, FD_REQ_GLOBAL, FD_REQ_LOCAL, FD_REQ_PAGE, FD_REQ_SRSW,
       FD_REQ_CLASS,            /* GLOBAL let go: PPC_TLBIE=class did it */
       FD_NREQ };

/* a deferred cause (SR, tlbie): consumed by check_tlb_flush */
void fdiag_pending(int cause);
/* tlbie of effective address @ea (setter count; per-request histogram) */
void fdiag_tlbie(uint32_t ea);
/* PPC_TLBIE=class: a tlbie invalidated its class at once (no pending cause) */
void fdiag_tlbie_class(uint32_t ea);
/* an immediate whole flush is about to be requested for @cause */
void fdiag_inflight(int cause);
/* store to a segment register: @changed if the value differs */
void fdiag_mtsr(bool changed);
/* BAT store: @data, @skipped (old/new side invalid), @pages flushed,
 * @whole (escalated to a whole flush) */
void fdiag_bat(bool data, bool skipped, unsigned pages, bool whole);
/*
 * check_tlb_flush entry: @need = env->tlb_need_flush, @global, @sr = the
 * 16 segment registers; returns the causes pending (for fdiag_request).
 */
int fdiag_sync(unsigned need, bool global, const uint32_t *sr);
/* what check_tlb_flush did with the pending causes */
void fdiag_request(int outcome, int causes, unsigned pages);
/* SR-slot sync for the current SR set: @hit, or a miss that filled an
 * @unused slot or replaced another set; @slot = where the set now is */
void fdiag_slot(bool hit, bool unused, int slot);
/* a TB was translated */
void fdiag_tb_gen(void);
/* cpu_exec_loop is about to translate: @dup if the same pc/flags exist
 * under another cs_base (another SR slot's tag) */
void fdiag_tb_miss(bool dup);
/* cputlb: one whole-flush execution */
void fdiag_flush_exec(uint16_t asked, uint16_t cleaned);
/* tcg_flush_jmp_cache: every generation bump (!@from_tlb), and again
 * with @from_tlb when the whole-flush worker made it */
void fdiag_jc_bump(bool from_tlb);
/* cputlb: a page flush cleared jump-cache pages */
void fdiag_jc_page(void);
/* cputlb: a page flush inside a large page flushed a whole index */
void fdiag_lp_escalate(void);

#endif
