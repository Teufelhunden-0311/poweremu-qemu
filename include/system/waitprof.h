/*
 * Debug only (waitprof branch, not for merging): PPC_WAIT_PROF=1 attributes
 * vCPU idle episodes (what ended them) and BQL waits (who held the lock
 * meanwhile).  Aggregates go to stderr every 10 s.  See system/cpus.c.
 */
#ifndef SYSTEM_WAITPROF_H
#define SYSTEM_WAITPROF_H

#include "qemu/thread.h"

extern bool waitprof_on;

/*
 * Work labels for BQL holds: the holder's waiters are split by these.
 * waitprof_act returns the previous label so callers can nest:
 *   int wp = WAITPROF_ACT(WAITPROF_ACT_FENCE); ...; WAITPROF_ACT_END(wp);
 */
enum {
    WAITPROF_ACT_NONE,      /* unlabelled: PM4 dispatch etc. on the CP thread */
    WAITPROF_ACT_BUILD,     /* r300_draw_build: vertex fetch, shader memo */
    WAITPROF_ACT_SUBMIT,    /* renderer->draw_r300: enqueue to the renderer */
    WAITPROF_ACT_FENCE,     /* r200_scratch_write: fence submit + hybrid spin */
    WAITPROF_ACT_VRAMACC,   /* r200_vram_access: CPU access ordering */
    WAITPROF_ACT_2D,        /* 2D blits */
    WAITPROF_ACT_DISPLAY,   /* display refresh (gfx_update) */
    /* Inside r300_draw_build (r300_phase_hook); "build" is then the rest:
     * packet/state setup, vertex order, constants, layout/routing. */
    WAITPROF_ACT_B_FETCH,   /* GPU-VS path: per-distinct-vertex fetch/decode loop */
    WAITPROF_ACT_B_VSPACK,  /* GPU-VS path: the rest (VS memo, index list, uniforms) */
    WAITPROF_ACT_B_FSMEMO,  /* fragment shader memo (r300_us_glsl_cached) */
    WAITPROF_ACT_B_CPU,     /* CPU-VS path: transform loop incl. its fetches */
    WAITPROF_NACT
};
static const char *const waitprof_act_name[WAITPROF_NACT] __attribute__((unused)) = {
    "none", "build", "submit", "fence", "vramacc", "2d", "display",
    "b_fetch", "b_vspack", "b_fsmemo", "b_cpu" };
int waitprof_act(int act);
#define WAITPROF_ACT(a) (waitprof_on ? waitprof_act(a) : 0)
#define WAITPROF_ACT_END(prev) do { if (waitprof_on) { waitprof_act(prev); } } while (0)

/* Interrupt sources: 0..254 = OpenPIC source number, 255 = the GPU device. */
#define WAITPROF_SRC_GPU 255
void waitprof_irq(int src, int level);
void waitprof_kick(void);
/* do_run_on_cpu's wait: accounted as a BQL condition wait if mutex is the BQL. */
void waitprof_cond_wait_mutex(QemuCond *cond, QemuMutex *mutex);

/* RR vCPU thread, BQL held. */
void waitprof_idle_enter(void);
void waitprof_idle_wait(bool begin);
void waitprof_idle_exit(CPUState *cpu);
/* RR vCPU thread, just before guest execution (BQL not held). */
void waitprof_exec(void);

/* Target hook: the CPU's pending interrupt bits, labelled (D)ECR/(E)XT/(O)ther. */
extern const char *(*waitprof_pending)(CPUState *cpu);

#endif
