/*
 * PPC_VCPU_DIAG=1 (diagnostic, not for merging): counters that classify
 * where the vCPU's slow paths go, without changing behaviour.
 *   - softmmu slow-path entries by access type: primary hit (flags only),
 *     victim-TLB hit, refill; on a miss, whether the primary slot held
 *     another page; the flags that forced the slow path; MMIO by region;
 *   - refills by access type and whether they came from TB lookup, with
 *     shadow caches after the victim TLB (bigger victim, set-associative,
 *     unbounded since the last flush) that count would-be hits;
 *   - the PPC hash32 walk: real mode, BAT, primary/secondary PTEG, fault;
 *   - indirect-branch lookups by kind (bclr, bcctr, direct cross-page):
 *     executions, inline-probe misses, and why the jump cache missed;
 *   - BQL waits: time by waiter/owner thread and call site.
 * Cumulative lines are logged every 10 s and at exit.  AI-generated
 * (Claude).
 */
#ifndef EXEC_VCPU_DIAG_H
#define EXEC_VCPU_DIAG_H

#include "qemu/compiler.h"

extern bool vdiag_on;

/* Indirect-branch kinds (vdiag_ij_exec / vdiag_ij_pending). */
enum { VDIAG_IJ_OTHER, VDIAG_IJ_LR, VDIAG_IJ_CTR, VDIAG_IJ_DIRECT, VDIAG_IJ_N };
extern uint64_t vdiag_ij_exec[VDIAG_IJ_N];   /* incremented by generated code */
extern uint32_t vdiag_ij_pending;            /* kind of the probe that missed */
extern int vdiag_ij_kind_next;               /* set by the target before
                                                translator_lookup_and_goto_ptr */

/* Jump-cache outcomes in tb_lookup. */
enum { VDIAG_JC_HIT, VDIAG_JC_EMPTY, VDIAG_JC_GEN, VDIAG_JC_PC, VDIAG_JC_STATE,
       VDIAG_JC_N };

void vdiag_slow(int access_type, int outcome, int miss_class, int flags);
/* A primary-TLB miss for @page (not a same-page permission miss): @occupant
 * is the page the primary slot held (or -1), @victim_hit what the real
 * victim TLB did.  Drives the victim-TLB size models. */
void vdiag_primary_miss(int mmu_idx, uint64_t page, uint64_t occupant, bool victim_hit);
void vdiag_lookup_table(bool begin);
void vdiag_bql_wait_begin(void);
void vdiag_mmio(const void *mr, bool store);
void vdiag_fill(int mmu_idx, uint64_t page, int access_type);
void vdiag_flush_all(void);
void vdiag_flush_page(int mmu_idx, uint64_t page);
void vdiag_xlate(int kind);
void vdiag_lookup(bool unused, int jc, int table_hit);
void vdiag_helper_enter(void);
void vdiag_helper_exit(void);
void vdiag_bql_wait(const char *file, int line, int64_t ns);
void vdiag_bql_acquired(const char *file, int line);
void vdiag_maybe_report(void);
extern uint64_t vdiag_r300_draws;

/* vdiag_slow outcomes and primary-miss classes */
enum { VDIAG_SLOW_HIT, VDIAG_SLOW_VICTIM, VDIAG_SLOW_FILL };
enum { VDIAG_MISS_NONE, VDIAG_MISS_EMPTY, VDIAG_MISS_CONFLICT, VDIAG_MISS_PERM };
/* vdiag_xlate kinds */
enum { VDIAG_X_REAL, VDIAG_X_BAT, VDIAG_X_PTEG1, VDIAG_X_PTEG2, VDIAG_X_FAULT,
       VDIAG_X_OTHER, VDIAG_X_N };

#endif
