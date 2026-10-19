/*
 * Debug only (waitprof branch, not for merging): PPC_WAIT_PROF=1 attributes
 * vCPU idle episodes (what ended them) and BQL waits (who held the lock
 * meanwhile).  Aggregates go to stderr every 10 s.  See system/cpus.c.
 */
#ifndef SYSTEM_WAITPROF_H
#define SYSTEM_WAITPROF_H

extern bool waitprof_on;

/* Interrupt sources: 0..254 = OpenPIC source number, 255 = the GPU device. */
#define WAITPROF_SRC_GPU 255
void waitprof_irq(int src, int level);
void waitprof_kick(void);

/* RR vCPU thread, BQL held. */
void waitprof_idle_enter(void);
void waitprof_idle_wait(bool begin);
void waitprof_idle_exit(CPUState *cpu);
/* RR vCPU thread, just before guest execution (BQL not held). */
void waitprof_exec(void);

/* Target hook: the CPU's pending interrupt bits, labelled (D)ECR/(E)XT/(O)ther. */
extern const char *(*waitprof_pending)(CPUState *cpu);

#endif
