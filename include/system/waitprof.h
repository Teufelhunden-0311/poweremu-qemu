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
void waitprof_idle_enter(CPUState *cpu);
void waitprof_idle_wait(bool begin);
void waitprof_idle_exit(CPUState *cpu);
/* RR vCPU thread, just before guest execution (BQL not held). */
void waitprof_exec(void);

/* Target hook: the CPU's pending interrupt bits, labelled (D)ECR/(E)XT/(O)ther. */
extern const char *(*waitprof_pending)(CPUState *cpu);
/*
 * First return to user mode after idle: the target calls this from its
 * return-from-interrupt path (vCPU thread, BQL not necessarily held) when
 * waitprof_ret_pending is set and the new MSR is user mode.  asid is the
 * target's address-space id (PPC: VSID of SR0).
 */
extern bool waitprof_ret_pending;
void waitprof_user_return(CPUState *cpu, uint32_t nip, uint32_t lr, uint32_t r1,
                          uint32_t asid);

/*
 * Most recent user-to-kernel entry before idle.  The target calls
 * waitprof_kentry for every delivered exception (pr: the interrupted context
 * was user mode; r0 only for system calls, else 0) and waitprof_kexit for
 * every return to user mode, which ends that kernel interval.
 */
void waitprof_kentry(CPUState *cpu, int excp, bool pr, uint32_t nip, uint32_t lr,
                     uint32_t r0, uint32_t r1, uint32_t asid);
void waitprof_kexit(CPUState *cpu);

/* Target hook: the halted CPU's context at idle entry: NIP, LR, r1, MSR. */
extern void (*waitprof_ctx)(CPUState *cpu, uint32_t ctx[4]);

#endif
