/*
 * Diagnostic counters for TB lookup, code generation, the softmmu TLB and
 * the PowerPC segment-register TLB slots (QEMU_PERF_COUNTERS=1).  Off by
 * default; when on, one line a second goes to the QEMU log ("perf:"), each
 * a count since the previous line.  Plain increments: one vCPU thread
 * writes them and the report only reads, so a line can be off by a few.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#ifndef EXEC_PERF_COUNTERS_H
#define EXEC_PERF_COUNTERS_H

#include "qemu/compiler.h"

typedef struct QemuPerfCounters {
    uint64_t lookup;        /* tb_lookup calls (inline jump-cache probe misses) */
    uint64_t jc_hit;        /* ... found in the jump cache */
    uint64_t ht_hit;        /* ... found in the global TB table */
    uint64_t ht_miss;       /* ... not found: translated */
    uint64_t tb_gen;        /* blocks translated */
    uint64_t tb_bytes;      /* host code bytes generated */
    uint64_t code_flush;    /* whole code-cache flushes (tb_flush) */
    uint64_t tb_inval;      /* blocks invalidated (code page written, ...) */
    uint64_t tlb_fill;      /* softmmu refills (guest page-table walks) */
    uint64_t tlb_flush;     /* whole-TLB flushes */
    uint64_t tlb_grow;      /* dynamic TLB resizes up / down */
    uint64_t tlb_shrink;
    uint64_t slot_sync;     /* PPC SR-slot syncs: a set was made current */
    uint64_t slot_hit;      /* ... already cached */
    uint64_t slot_evict;    /* ... evicted the least recently used set */
} QemuPerfCounters;

extern QemuPerfCounters qemu_perf;
extern bool qemu_perf_on;

#define QEMU_PERF_INC(field) \
    do { if (unlikely(qemu_perf_on)) { qemu_perf.field++; } } while (0)
#define QEMU_PERF_ADD(field, n) \
    do { if (unlikely(qemu_perf_on)) { qemu_perf.field += (n); } } while (0)

/* Write the "perf:" line if a second has passed (call from the vCPU). */
void qemu_perf_maybe_report(void *cpu);

/* Entries allocated in the dynamic TLBs, over all MMU indexes. */
size_t tlb_dyn_entries_total(void *cpu);

#endif
