/*
 * PPC_TBINV_DIAG=1: why translated code goes away and comes back.
 * Diagnostic, not for merging.  AI-generated (Claude).
 *
 * Counts, per run, every translation by what became of the previous TB
 * with the same identity (pc, cs_base, flags, cflags): none ever, removed
 * by an individual invalidation (and that invalidation's cause), removed by
 * a whole code-cache flush, or still recorded (a remap or something else);
 * whether the same pc/flags/cflags is live under another cs_base (another
 * SR slot's tag); every individual invalidation event (one written range of
 * one physical code page) by cause, with the TBs it removed and how many
 * distinct pc/flags/cflags those were; and whole code-cache flushes.
 * One "tbdiag" line a second (cumulative counts) and a per-page table at
 * exit, on stderr.
 */
#ifndef EXEC_TB_DIAG_H
#define EXEC_TB_DIAG_H

extern bool tbdiag_on;

/* What is writing, for the invalidation in progress on this thread. */
enum { TBD_OTHER, TBD_STORE, TBD_DMA, TBD_NCAUSE };
extern __thread int tbdiag_cause;

/* tb_gen_code made a TB. */
void tbdiag_gen(uint64_t pc, uint64_t cs_base, uint32_t flags,
                uint32_t cflags, uint64_t phys_pc);
/* do_tb_phys_invalidate removed a TB from the hash table. */
void tbdiag_inval(uint64_t pc, uint64_t cs_base, uint32_t flags,
                  uint32_t cflags);
/* tb_invalidate_phys_page_range__locked: one written range of one page;
 * @retaddr is the writing store's host return address, or 0. */
void tbdiag_ev_begin(uint64_t start, uintptr_t retaddr);
void tbdiag_ev_end(void);
/* tb_gen_code ran out of code buffer and asked for a flush. */
void tbdiag_flush_full(void);
/* do_tb_flush flushed the whole code cache. */
void tbdiag_flush(void);

#endif
