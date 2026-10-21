/*
 * QEMU System Emulator
 *
 * Copyright (c) 2003-2008 Fabrice Bellard
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "qemu/osdep.h"
#include "monitor/monitor.h"
#include "qemu/coroutine-tls.h"
#include "qapi/error.h"
#include "qapi/qapi-commands-machine.h"
#include "qapi/qapi-commands-misc.h"
#include "qapi/qapi-events-run-state.h"
#include "qapi/qmp/qerror.h"
#include "exec/gdbstub.h"
#include "system/accel-ops.h"
#include "system/hw_accel.h"
#include "exec/cpu-common.h"
#include "qemu/thread.h"
#include "qemu/main-loop.h"
#include "qemu/plugin.h"
#include "system/cpus.h"
#include "system/waitprof.h"
#include "exec/memory.h"
#include "qemu/cutils.h"
#include "qemu/timer.h"
#include "qemu/guest-random.h"
#include "hw/nmi.h"
#include "system/replay.h"
#include "system/runstate.h"
#include "system/cpu-timers.h"
#include "system/whpx.h"
#include "hw/boards.h"
#include "hw/hw.h"
#include "trace.h"

#ifdef CONFIG_LINUX

#include <sys/prctl.h>

#ifndef PR_MCE_KILL
#define PR_MCE_KILL 33
#endif

#ifndef PR_MCE_KILL_SET
#define PR_MCE_KILL_SET 1
#endif

#ifndef PR_MCE_KILL_EARLY
#define PR_MCE_KILL_EARLY 1
#endif

#endif /* CONFIG_LINUX */

/* The Big QEMU Lock (BQL) */
static QemuMutex bql;

/*
 * The chosen accelerator is supposed to register this.
 */
static const AccelOpsClass *cpus_accel;

bool cpu_is_stopped(CPUState *cpu)
{
    return cpu->stopped || !runstate_is_running();
}

bool cpu_work_list_empty(CPUState *cpu)
{
    return QSIMPLEQ_EMPTY_ATOMIC(&cpu->work_list);
}

bool cpu_thread_is_idle(CPUState *cpu)
{
    if (cpu->stop || !cpu_work_list_empty(cpu)) {
        return false;
    }
    if (cpu_is_stopped(cpu)) {
        return true;
    }
    if (!cpu->halted || cpu_has_work(cpu)) {
        return false;
    }
    if (cpus_accel->cpu_thread_is_idle) {
        return cpus_accel->cpu_thread_is_idle(cpu);
    }
    return true;
}

bool all_cpu_threads_idle(void)
{
    CPUState *cpu;

    CPU_FOREACH(cpu) {
        if (!cpu_thread_is_idle(cpu)) {
            return false;
        }
    }
    return true;
}

/***********************************************************/
void hw_error(const char *fmt, ...)
{
    va_list ap;
    CPUState *cpu;

    va_start(ap, fmt);
    fprintf(stderr, "qemu: hardware error: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    CPU_FOREACH(cpu) {
        fprintf(stderr, "CPU #%d:\n", cpu->cpu_index);
        cpu_dump_state(cpu, stderr, CPU_DUMP_FPU);
    }
    va_end(ap);
    abort();
}

void cpu_synchronize_all_states(void)
{
    CPUState *cpu;

    CPU_FOREACH(cpu) {
        cpu_synchronize_state(cpu);
    }
}

void cpu_synchronize_all_post_reset(void)
{
    CPUState *cpu;

    CPU_FOREACH(cpu) {
        cpu_synchronize_post_reset(cpu);
    }
}

void cpu_synchronize_all_post_init(void)
{
    CPUState *cpu;

    CPU_FOREACH(cpu) {
        cpu_synchronize_post_init(cpu);
    }
}

void cpu_synchronize_all_pre_loadvm(void)
{
    CPUState *cpu;

    CPU_FOREACH(cpu) {
        cpu_synchronize_pre_loadvm(cpu);
    }
}

void cpu_synchronize_state(CPUState *cpu)
{
    if (cpus_accel->synchronize_state) {
        cpus_accel->synchronize_state(cpu);
    }
}

void cpu_synchronize_post_reset(CPUState *cpu)
{
    if (cpus_accel->synchronize_post_reset) {
        cpus_accel->synchronize_post_reset(cpu);
    }
}

void cpu_synchronize_post_init(CPUState *cpu)
{
    if (cpus_accel->synchronize_post_init) {
        cpus_accel->synchronize_post_init(cpu);
    }
}

void cpu_synchronize_pre_loadvm(CPUState *cpu)
{
    if (cpus_accel->synchronize_pre_loadvm) {
        cpus_accel->synchronize_pre_loadvm(cpu);
    }
}

bool cpus_are_resettable(void)
{
    if (cpus_accel->cpus_are_resettable) {
        return cpus_accel->cpus_are_resettable();
    }
    return true;
}

void cpu_exec_reset_hold(CPUState *cpu)
{
    if (cpus_accel->cpu_reset_hold) {
        cpus_accel->cpu_reset_hold(cpu);
    }
}

int64_t cpus_get_virtual_clock(void)
{
    /*
     * XXX
     *
     * need to check that cpus_accel is not NULL, because qcow2 calls
     * qemu_get_clock_ns(CLOCK_VIRTUAL) without any accel initialized and
     * with ticks disabled in some io-tests:
     * 030 040 041 060 099 120 127 140 156 161 172 181 191 192 195 203 229 249 256 267
     *
     * is this expected?
     *
     * XXX
     */
    if (cpus_accel && cpus_accel->get_virtual_clock) {
        return cpus_accel->get_virtual_clock();
    }
    return cpu_get_clock();
}

/*
 * Signal the new virtual time to the accelerator. This is only needed
 * by accelerators that need to track the changes as we warp time.
 */
void cpus_set_virtual_clock(int64_t new_time)
{
    if (cpus_accel && cpus_accel->set_virtual_clock) {
        cpus_accel->set_virtual_clock(new_time);
    }
}

/*
 * return the time elapsed in VM between vm_start and vm_stop.  Unless
 * icount is active, cpus_get_elapsed_ticks() uses units of the host CPU cycle
 * counter.
 */
int64_t cpus_get_elapsed_ticks(void)
{
    if (cpus_accel->get_elapsed_ticks) {
        return cpus_accel->get_elapsed_ticks();
    }
    return cpu_get_ticks();
}

static void generic_handle_interrupt(CPUState *cpu, int mask)
{
    cpu->interrupt_request |= mask;

    if (!qemu_cpu_is_self(cpu)) {
        qemu_cpu_kick(cpu);
    }
}

void cpu_interrupt(CPUState *cpu, int mask)
{
    if (cpus_accel->handle_interrupt) {
        cpus_accel->handle_interrupt(cpu, mask);
    } else {
        generic_handle_interrupt(cpu, mask);
    }
}

/*
 * True if the vm was previously suspended, and has not been woken or reset.
 */
static int vm_was_suspended;

void vm_set_suspended(bool suspended)
{
    vm_was_suspended = suspended;
}

bool vm_get_suspended(void)
{
    return vm_was_suspended;
}

static int do_vm_stop(RunState state, bool send_stop)
{
    int ret = 0;
    RunState oldstate = runstate_get();

    if (runstate_is_live(oldstate)) {
        vm_was_suspended = (oldstate == RUN_STATE_SUSPENDED);
        runstate_set(state);
        cpu_disable_ticks();
        if (oldstate == RUN_STATE_RUNNING) {
            pause_all_vcpus();
        }
        vm_state_notify(0, state);
        if (send_stop) {
            qapi_event_send_stop();
        }
    }

    bdrv_drain_all();
    ret = bdrv_flush_all();
    trace_vm_stop_flush_all(ret);

    return ret;
}

/* Special vm_stop() variant for terminating the process.  Historically clients
 * did not expect a QMP STOP event and so we need to retain compatibility.
 */
int vm_shutdown(void)
{
    return do_vm_stop(RUN_STATE_SHUTDOWN, false);
}

bool cpu_can_run(CPUState *cpu)
{
    if (cpu->stop) {
        return false;
    }
    if (cpu_is_stopped(cpu)) {
        return false;
    }
    return true;
}

void cpu_handle_guest_debug(CPUState *cpu)
{
    if (replay_running_debug()) {
        if (!cpu->singlestep_enabled) {
            /*
             * Report about the breakpoint and
             * make a single step to skip it
             */
            replay_breakpoint();
            cpu_single_step(cpu, SSTEP_ENABLE);
        } else {
            cpu_single_step(cpu, 0);
        }
    } else {
        gdb_set_stop_cpu(cpu);
        qemu_system_debug_request();
        cpu->stopped = true;
    }
}

#ifdef CONFIG_LINUX
static void sigbus_reraise(void)
{
    sigset_t set;
    struct sigaction action;

    memset(&action, 0, sizeof(action));
    action.sa_handler = SIG_DFL;
    if (!sigaction(SIGBUS, &action, NULL)) {
        raise(SIGBUS);
        sigemptyset(&set);
        sigaddset(&set, SIGBUS);
        pthread_sigmask(SIG_UNBLOCK, &set, NULL);
    }
    perror("Failed to re-raise SIGBUS!");
    abort();
}

static void sigbus_handler(int n, siginfo_t *siginfo, void *ctx)
{
    if (siginfo->si_code != BUS_MCEERR_AO && siginfo->si_code != BUS_MCEERR_AR) {
        sigbus_reraise();
    }

    if (current_cpu) {
        /* Called asynchronously in VCPU thread.  */
        if (kvm_on_sigbus_vcpu(current_cpu, siginfo->si_code, siginfo->si_addr)) {
            sigbus_reraise();
        }
    } else {
        /* Called synchronously (via signalfd) in main thread.  */
        if (kvm_on_sigbus(siginfo->si_code, siginfo->si_addr)) {
            sigbus_reraise();
        }
    }
}

static void qemu_init_sigbus(void)
{
    struct sigaction action;

    /*
     * ALERT: when modifying this, take care that SIGBUS forwarding in
     * qemu_prealloc_mem() will continue working as expected.
     */
    memset(&action, 0, sizeof(action));
    action.sa_flags = SA_SIGINFO;
    action.sa_sigaction = sigbus_handler;
    sigaction(SIGBUS, &action, NULL);

    prctl(PR_MCE_KILL, PR_MCE_KILL_SET, PR_MCE_KILL_EARLY, 0, 0);
}
#else /* !CONFIG_LINUX */
static void qemu_init_sigbus(void)
{
}
#endif /* !CONFIG_LINUX */

static QemuThread io_thread;

/* cpu creation */
static QemuCond qemu_cpu_cond;
/* system init */
static QemuCond qemu_pause_cond;

void qemu_init_cpu_loop(void)
{
    qemu_init_sigbus();
    qemu_cond_init(&qemu_cpu_cond);
    qemu_cond_init(&qemu_pause_cond);
    qemu_mutex_init(&bql);

    qemu_thread_get_self(&io_thread);
}

void run_on_cpu(CPUState *cpu, run_on_cpu_func func, run_on_cpu_data data)
{
    do_run_on_cpu(cpu, func, data, &bql);
}

static void qemu_cpu_stop(CPUState *cpu, bool exit)
{
    g_assert(qemu_cpu_is_self(cpu));
    cpu->stop = false;
    cpu->stopped = true;
    if (exit) {
        cpu_exit(cpu);
    }
    qemu_cond_broadcast(&qemu_pause_cond);
}

void qemu_wait_io_event_common(CPUState *cpu)
{
    qatomic_set_mb(&cpu->thread_kicked, false);
    if (cpu->stop) {
        qemu_cpu_stop(cpu, false);
    }
    process_queued_cpu_work(cpu);
}

void qemu_wait_io_event(CPUState *cpu)
{
    bool slept = false;

    while (cpu_thread_is_idle(cpu)) {
        if (!slept) {
            slept = true;
            qemu_plugin_vcpu_idle_cb(cpu);
        }
        qemu_cond_wait(cpu->halt_cond, &bql);
    }
    if (slept) {
        qemu_plugin_vcpu_resume_cb(cpu);
    }

    qemu_wait_io_event_common(cpu);
}

void cpus_kick_thread(CPUState *cpu)
{
    if (cpu->thread_kicked) {
        return;
    }
    cpu->thread_kicked = true;

#ifndef _WIN32
    int err = pthread_kill(cpu->thread->thread, SIG_IPI);
    if (err && err != ESRCH) {
        fprintf(stderr, "qemu:%s: %s", __func__, strerror(err));
        exit(1);
    }
#else
    qemu_sem_post(&cpu->sem);
#endif
}

void qemu_cpu_kick(CPUState *cpu)
{
    if (waitprof_on) {
        waitprof_kick();
    }
    qemu_cond_broadcast(cpu->halt_cond);
    if (cpus_accel->kick_vcpu_thread) {
        cpus_accel->kick_vcpu_thread(cpu);
    } else { /* default */
        cpus_kick_thread(cpu);
    }
}

void qemu_cpu_kick_self(void)
{
    assert(current_cpu);
    cpus_kick_thread(current_cpu);
}

bool qemu_cpu_is_self(CPUState *cpu)
{
    return qemu_thread_is_self(cpu->thread);
}

bool qemu_in_vcpu_thread(void)
{
    return current_cpu && qemu_cpu_is_self(current_cpu);
}

QEMU_DEFINE_STATIC_CO_TLS(bool, bql_locked)

static uint32_t bql_unlock_blocked;

void bql_block_unlock(bool increase)
{
    uint32_t new_value;

    assert(bql_locked());

    /* check for overflow! */
    new_value = bql_unlock_blocked + increase - !increase;
    assert((new_value > bql_unlock_blocked) == increase);
    bql_unlock_blocked = new_value;
}

bool bql_locked(void)
{
    return get_bql_locked();
}

bool qemu_in_main_thread(void)
{
    return bql_locked();
}

void rust_bql_mock_lock(void)
{
    error_report("This function should be used only from tests");
    abort();
}


/*
 * Debug only (waitprof branch, not for merging): PPC_WAIT_PROF=1.
 *
 * Idle: an episode starts when the RR vCPU thread finds every CPU idle and
 * ends when it leaves rr_wait_io_event.  It is classified by the CPU's
 * pending interrupt bits at the end and by the interrupt sources that rose
 * during it (OpenPIC source numbers; "gpu" = the GPU device's own line).
 * Condition waits that returned without ending the episode are counted
 * (spurious or not-yet-runnable wakes).  The condition wait includes the
 * BQL reacquisition; it is NOT also counted as a BQL wait.
 *
 * BQL: per acquisition, the wait from the request to the acquisition.
 * Every hold (lock..unlock, and the holds around condition waits) is
 * appended to a ring when it ends, so a waiter can split its wait among
 * all owners whose holds overlapped it; the rest is "(free)": lock free
 * but not yet acquired (hand-off / scheduling).
 *
 * All tables are updated with the BQL held, except the atomic IRQ/kick
 * marks.  Every 10 s the releasing thread copies and resets them under the
 * BQL and prints the copy after unlocking.
 */
bool waitprof_on;
const char *(*waitprof_pending)(CPUState *cpu);
void (*waitprof_ctx)(CPUState *cpu, uint32_t ctx[4]);
static uint32_t wp_stk_nip;      /* stack capture site, see wp_walk */
static QemuSpin wp_ret_lock;     /* user-return tables vs. the snapshot */

static void __attribute__((constructor)) waitprof_init(void)
{
    const char *e = getenv("PPC_WAIT_PROF");
    waitprof_on = e && e[0] == '1';
    qemu_spin_init(&wp_ret_lock);
    e = getenv("PPC_WP_STACK_NIP");
    wp_stk_nip = e ? strtoul(e, NULL, 16) : 0x900411f8u;
}

#define WP_NHIST 6
static const int64_t wp_lim[WP_NHIST - 1] = {
    1000, 10000, 100000, 1000000, 4000000 };     /* ns: 1us 10us 100us 1ms 4ms */
static int wp_bucket(int64_t ns)
{
    int i = 0;
    while (i < WP_NHIST - 1 && ns >= wp_lim[i]) {
        i++;
    }
    return i;
}

/* --- interrupt / kick marks (any thread) --- */
static uint64_t wp_raised[4];
static uint8_t wp_level[256];
static uint64_t wp_kicks;
static int64_t wp_last_kick;

void waitprof_irq(int src, int level)
{
    src &= 255;
    if (level && !wp_level[src]) {
        qatomic_or(&wp_raised[src >> 6], 1ull << (src & 63));
    }
    wp_level[src] = !!level;
}

void waitprof_kick(void)
{
    qatomic_inc(&wp_kicks);
    qatomic_set(&wp_last_kick, get_clock());
}

/* --- idle episodes (vCPU thread, BQL held) --- */
typedef struct {
    char key[48];
    uint64_t n, tot, hist[WP_NHIST], waits, multi, kick2ret, kick2ret_n, ret2exit;
} WPIdle;
#define WP_NIDLE 48
static WPIdle wp_idle[WP_NIDLE];
static int wp_nidle;
static int64_t wp_t0, wp_wait_t0, wp_last_ret, wp_resume;
static unsigned wp_nwaits;
static uint64_t wp_execlat, wp_execlat_n;

/*
 * Idle episodes by the halted CPU's context at entry (NIP, LR, privilege).
 * This is where the CPU went idle, not which guest thread is blocked.
 * cls splits the time by wake: 0 = only DECR pending and no source rose,
 * 1 = EXT pending or a source rose, 2 = anything else.
 */
typedef struct {
    uint32_t nip, lr, msr, r1min, r1max;
    bool used;
    uint64_t n, tot, hist[WP_NHIST], cls[3];
} WPCtx;
#define WP_NCTX 512
static WPCtx wp_ctx[WP_NCTX];
static int wp_nctx;
static uint64_t wp_ctx_drop, wp_ctx_drop_ns;
static uint32_t wp_ctx_cur[4];

static void wp_ctx_add(int64_t ns, int cls)
{
    uint32_t nip = wp_ctx_cur[0], lr = wp_ctx_cur[1], r1 = wp_ctx_cur[2];
    uint32_t msr = wp_ctx_cur[3], pr = (msr >> 14) & 1;
    unsigned h = ((nip >> 2) * 0x9e3779b1u ^ (lr >> 2) * 0x85ebca6bu ^ pr) % WP_NCTX;
    WPCtx *c = NULL;

    for (int i = 0; i < WP_NCTX; i++, h = (h + 1) % WP_NCTX) {
        WPCtx *t = &wp_ctx[h];
        if (!t->used) {
            if (wp_nctx >= WP_NCTX * 3 / 4) {
                break;
            }
            memset(t, 0, sizeof(*t));
            t->used = true;
            t->nip = nip;
            t->lr = lr;
            t->msr = msr;
            t->r1min = t->r1max = r1;
            wp_nctx++;
            c = t;
            break;
        }
        if (t->nip == nip && t->lr == lr && ((t->msr >> 14) & 1) == pr) {
            c = t;
            break;
        }
    }
    if (!c) {
        wp_ctx_drop++;
        wp_ctx_drop_ns += ns;
        return;
    }
    c->n++;
    c->tot += ns;
    c->hist[wp_bucket(ns)]++;
    c->cls[cls] += ns;
    c->msr = msr;
    c->r1min = MIN(c->r1min, r1);
    c->r1max = MAX(c->r1max, r1);
}

/*
 * First return to user mode after idle.  Idle episodes accumulate in a
 * pending batch (vCPU thread only) until the target reports an rfi into
 * user mode; the batch is then charged once to that return's bucket
 * (NIP, address space).  Episodes still pending at a snapshot stay pending.
 * This is the first user context resumed after idle, not necessarily the
 * thread that waited.
 */
bool waitprof_ret_pending;
static struct {
    uint64_t n, ns, cls[3];
    int64_t last_exit;
} wp_pend;
typedef struct {
    uint32_t nip, asid, lr, lr_other;    /* lr_other: returns whose LR != lr */
    bool used;
    uint64_t rets, eps, ns, cls[3], delay, dhist[WP_NHIST], batch[4];
} WPRet;
#define WP_NRET 1024
static WPRet wp_ret[WP_NRET];
static int wp_nret;
static uint64_t wp_ret_drop, wp_ret_drop_eps, wp_ret_drop_ns;
/* Address spaces seen: SR0 VSID -> main executable __TEXT vmsize, last string at the stack top. */
typedef struct {
    uint32_t asid, text;
    uint64_t hdr;                        /* FNV-1a of the 4 KB at 0x1000, 0 = unread */
    char path[96];
} WPAs;
#define WP_NAS 128
static WPAs wp_as[WP_NAS];
static int wp_nas;

/*
 * Debug read of guest virtual memory through the current translation, RAM
 * only: unlike cpu_memory_rw_debug, never dispatches to a device region.
 */
static bool wp_ram_read(CPUState *cpu, uint32_t va, void *buf, unsigned len)
{
    uint8_t *p = buf;

    RCU_READ_LOCK_GUARD();
    while (len) {
        MemTxAttrs attrs;
        hwaddr pa = cpu_get_phys_page_attrs_debug(cpu, va & ~0xfffu, &attrs);
        unsigned l = MIN(len, 0x1000 - (va & 0xfff));
        hwaddr xlat, ml = l;
        MemoryRegion *mr;

        if (pa == -1) {
            return false;
        }
        AddressSpace *as = cpu_get_address_space(cpu, cpu_asidx_from_attrs(cpu, attrs));

        mr = address_space_translate(as, pa + (va & 0xfff), &xlat, &ml, false, attrs);
        if (!memory_region_is_ram(mr) || memory_region_is_ram_device(mr) || ml < l) {
            return false;
        }
        memcpy(p, (uint8_t *)memory_region_get_ram_ptr(mr) + xlat, l);
        p += l;
        va += l;
        len -= l;
    }
    return true;
}

static void wp_as_ident(CPUState *cpu, WPAs *a)
{
    static uint8_t b[4096];

    a->text = 0;
    a->hdr = 0;
    pstrcpy(a->path, sizeof(a->path), "?");
    if (wp_ram_read(cpu, 0x1000, b, sizeof(b)) && ldl_be_p(b) == 0xfeedface) {
        uint64_t h = 0xcbf29ce484222325ull;
        for (unsigned i = 0; i < sizeof(b); i++) {
            h = (h ^ b[i]) * 0x100000001b3ull;
        }
        a->hdr = h;
        uint32_t ncmds = ldl_be_p(b + 16), o = 28;
        for (uint32_t i = 0; i < ncmds && o + 32 <= sizeof(b); i++) {
            uint32_t cmd = ldl_be_p(b + o), size = ldl_be_p(b + o + 4);
            if (cmd == 1 && !memcmp(b + o + 8, "__TEXT", 7)) {
                a->text = ldl_be_p(b + o + 28);
                break;
            }
            if (size < 8) {
                break;
            }
            o += size;
        }
    }
    /*
     * Exec strings sit at the top of the user stack, below 0xc0000000, with
     * the argv/envp/apple pointer arrays under them; the lowest pointer into
     * that area that starts a "/..." string is the executable path (Tiger:
 * apple[0]).  Frame back-chain pointers are skipped by that test.
     */
    static uint8_t t[8192];
    const uint32_t base = 0xc0000000u - sizeof(t);
    uint32_t lo = 0;
    if (!wp_ram_read(cpu, base, t, sizeof(t) / 2)) {
        memset(t, 0, sizeof(t) / 2);      /* lower page unmapped: top page only */
    }
    if (wp_ram_read(cpu, base + sizeof(t) / 2, t + sizeof(t) / 2, sizeof(t) / 2)) {
        for (uint32_t o = 0; o < sizeof(t); o += 4) {
            uint32_t v = ldl_be_p(t + o);
            if (v > base + o && v < 0xc0000000u && (!lo || v < lo) &&
                t[v - base] == '/' && !t[v - base - 1]) {     /* skip frame links */
                lo = v;
            }
        }
    }
    if (lo) {
        int n = 0;
        for (uint32_t i = lo - base; i < sizeof(t) && t[i] && n < (int)sizeof(a->path) - 1; i++) {
            a->path[n++] = t[i] > ' ' && t[i] < 127 ? t[i] : '_';
        }
        a->path[n] = 0;
    }
}

/*
 * At returns to one NIP (PPC_WP_STACK_NIP, default the mach_wait_until
 * return inside nanosleep): r1 and up to four back-chain frames, Darwin PPC
 * ABI (back chain at 0(sp), a callee saves its LR at 8 of its caller's sp).
 * f[0] is LR, f[k] the return address saved for the k-th frame's function.
 * why: 0 four frames read, 1 null back chain, 2 invalid link (alignment,
 * bounds, no forward progress, step over 1 MB), 3 read failed or not RAM.
 */
#define WP_NF 5
typedef struct {
    uint32_t asid, reg, f[WP_NF], r1min, r1max;
    uint8_t depth, why;
    bool used;
    uint64_t rets, eps, ns, cls[3];
} WPStk;
#define WP_NSTK 256
static WPStk wp_stk[WP_NSTK];
static int wp_nstk;
static uint64_t wp_stk_drop, wp_stk_drop_ns;
/* Same-run code bytes around each distinct frame address, printed once. */
typedef struct {
    uint32_t asid, addr;
    bool ok;
    uint8_t b[64];
} WPCode;
#define WP_NCODE 160
static WPCode wp_code[WP_NCODE];
static int wp_ncode, wp_ncode_out;
static uint64_t wp_code_drop;

static int wp_walk(CPUState *cpu, uint32_t r1, uint32_t f[WP_NF], int *why)
{
    uint32_t sp = r1;

    for (int k = 1; k < WP_NF; k++) {
        uint8_t w[4];
        uint32_t next;

        if ((sp & 3) || sp < 0x1000 || sp > 0xfffff000u) {
            *why = 2;
            return k - 1;
        }
        if (!wp_ram_read(cpu, sp, w, 4)) {
            *why = 3;
            return k - 1;
        }
        next = ldl_be_p(w);
        if (!next) {
            *why = 1;
            return k - 1;
        }
        if ((next & 3) || next <= sp || next - sp > 0x100000 || next > 0xfffff000u) {
            *why = 2;
            return k - 1;
        }
        if (!wp_ram_read(cpu, next + 8, w, 4)) {
            *why = 3;
            return k - 1;
        }
        f[k] = ldl_be_p(w);
        sp = next;
    }
    *why = 0;
    return WP_NF - 1;
}

/* vCPU thread only writes wp_code, so it may scan without the lock. */
static void wp_code_note(CPUState *cpu, uint32_t asid, uint32_t addr)
{
    WPCode c = { .asid = asid, .addr = addr };

    for (int i = 0; i < wp_ncode; i++) {
        if (wp_code[i].asid == asid && wp_code[i].addr == addr) {
            return;
        }
    }
    if (wp_ncode >= WP_NCODE) {
        wp_code_drop++;
        return;
    }
    c.ok = addr >= 0x1020 && addr < 0xffffffe0u &&
           wp_ram_read(cpu, (addr & ~3u) - 32, c.b, sizeof(c.b));
    qemu_spin_lock(&wp_ret_lock);
    wp_code[wp_ncode++] = c;
    qemu_spin_unlock(&wp_ret_lock);
}

static void wp_stk_add(uint32_t asid, uint32_t r1, uint32_t f[WP_NF], int depth, int why)
{
    uint32_t reg = r1 >> 19;
    unsigned h = asid * 0x85ebca6bu ^ reg * 0x27d4eb2fu ^ depth ^ why << 4;
    WPStk *s = NULL;

    for (int i = 0; i < WP_NF; i++) {
        h = h * 0x9e3779b1u ^ f[i];
    }
    h %= WP_NSTK;
    for (int i = 0; i < WP_NSTK; i++, h = (h + 1) % WP_NSTK) {
        WPStk *t = &wp_stk[h];
        if (!t->used) {
            if (wp_nstk >= WP_NSTK * 3 / 4) {
                break;
            }
            memset(t, 0, sizeof(*t));
            t->used = true;
            t->asid = asid;
            t->reg = reg;
            memcpy(t->f, f, sizeof(t->f));
            t->depth = depth;
            t->why = why;
            t->r1min = t->r1max = r1;
            wp_nstk++;
            s = t;
            break;
        }
        if (t->asid == asid && t->reg == reg && t->depth == depth && t->why == why &&
            !memcmp(t->f, f, sizeof(t->f))) {
            s = t;
            break;
        }
    }
    if (!s) {
        wp_stk_drop++;
        wp_stk_drop_ns += wp_pend.ns;
        return;
    }
    s->rets++;
    s->eps += wp_pend.n;
    s->ns += wp_pend.ns;
    for (int i = 0; i < 3; i++) {
        s->cls[i] += wp_pend.cls[i];
    }
    s->r1min = MIN(s->r1min, r1);
    s->r1max = MAX(s->r1max, r1);
}

void waitprof_user_return(CPUState *cpu, uint32_t nip, uint32_t lr, uint32_t r1,
                          uint32_t asid)
{
    uint32_t f[WP_NF] = { lr };
    int depth = -1, why = 0;
    int64_t now = get_clock();
    unsigned h = ((nip >> 2) * 0x9e3779b1u ^ asid * 0x85ebca6bu) % WP_NRET;
    uint64_t n = wp_pend.n;
    bool known = false;
    WPRet *r = NULL;

    for (int i = 0; i < wp_nas; i++) {
        if (wp_as[i].asid == asid) {
            known = true;
            break;
        }
    }
    if (!known && wp_nas < WP_NAS) {
        WPAs a = { .asid = asid };
        wp_as_ident(cpu, &a);
        qemu_spin_lock(&wp_ret_lock);
        wp_as[wp_nas++] = a;
        qemu_spin_unlock(&wp_ret_lock);
    }
    if (nip == wp_stk_nip) {
        depth = wp_walk(cpu, r1, f, &why);           /* reads outside the lock */
        for (int i = 0; i <= depth; i++) {
            wp_code_note(cpu, asid, f[i]);
        }
    }
    qemu_spin_lock(&wp_ret_lock);
    if (depth >= 0) {
        wp_stk_add(asid, r1, f, depth, why);
    }
    for (int i = 0; i < WP_NRET; i++, h = (h + 1) % WP_NRET) {
        WPRet *t = &wp_ret[h];
        if (!t->used) {
            if (wp_nret >= WP_NRET * 3 / 4) {
                break;
            }
            memset(t, 0, sizeof(*t));
            t->used = true;
            t->nip = nip;
            t->asid = asid;
            t->lr = lr;
            wp_nret++;
            r = t;
            break;
        }
        if (t->nip == nip && t->asid == asid) {
            r = t;
            break;
        }
    }
    if (r) {
        r->rets++;
        r->eps += n;
        r->ns += wp_pend.ns;
        for (int i = 0; i < 3; i++) {
            r->cls[i] += wp_pend.cls[i];
        }
        r->delay += now - wp_pend.last_exit;
        r->dhist[wp_bucket(now - wp_pend.last_exit)]++;
        r->batch[n <= 1 ? 0 : n == 2 ? 1 : n <= 4 ? 2 : 3]++;
        r->lr_other += lr != r->lr;
    } else {
        wp_ret_drop++;
        wp_ret_drop_eps += n;
        wp_ret_drop_ns += wp_pend.ns;
    }
    qemu_spin_unlock(&wp_ret_lock);
    memset(&wp_pend, 0, sizeof(wp_pend));
    waitprof_ret_pending = false;
}

void waitprof_idle_enter(CPUState *cpu)
{
    if (waitprof_ctx) {
        waitprof_ctx(cpu, wp_ctx_cur);
    }
    for (int i = 0; i < 4; i++) {
        qatomic_set(&wp_raised[i], 0);
    }
    wp_t0 = get_clock();
    wp_nwaits = 0;
    wp_last_ret = 0;
}

void waitprof_idle_wait(bool begin)
{
    if (begin) {
        wp_wait_t0 = get_clock();
    } else {
        wp_last_ret = get_clock();
        wp_nwaits++;
    }
}

void waitprof_idle_exit(CPUState *cpu)
{
    int64_t now = get_clock(), lk = qatomic_read(&wp_last_kick);
    char key[48];
    int n = 0, k = 0;
    const char *pend = waitprof_pending ? waitprof_pending(cpu) : "?";

    n = snprintf(key, sizeof(key), "%s |", pend);
    for (int src = 0; src < 256; src++) {
        if (qatomic_read(&wp_raised[src >> 6]) & (1ull << (src & 63))) {
            if (++k > 4) {
                n += snprintf(key + n, sizeof(key) - n, " +");
                break;
            }
            n += snprintf(key + n, sizeof(key) - n, src == WAITPROF_SRC_GPU ?
                          " gpu" : " %02x", src);
        }
    }
    if (!k) {
        snprintf(key + n, sizeof(key) - n, " none");
    }
    int cls = !k && !strcmp(pend, "D") ? 0 : (k || strchr(pend, 'E')) ? 1 : 2;
    wp_ctx_add(now - wp_t0, cls);
    wp_pend.n++;
    wp_pend.ns += now - wp_t0;
    wp_pend.cls[cls] += now - wp_t0;
    wp_pend.last_exit = now;
    waitprof_ret_pending = true;
    WPIdle *e = NULL;
    for (int i = 0; i < wp_nidle; i++) {
        if (!strcmp(wp_idle[i].key, key)) {
            e = &wp_idle[i];
            break;
        }
    }
    if (!e && wp_nidle < WP_NIDLE) {
        e = &wp_idle[wp_nidle++];
        memset(e, 0, sizeof(*e));
        pstrcpy(e->key, sizeof(e->key), key);
    }
    if (e) {
        e->n++;
        e->tot += now - wp_t0;
        e->hist[wp_bucket(now - wp_t0)]++;
        e->waits += wp_nwaits;
        e->multi += wp_nwaits > 1;
        if (wp_last_ret) {
            e->ret2exit += now - wp_last_ret;
            if (lk >= wp_wait_t0 && lk <= wp_last_ret) {
                e->kick2ret += wp_last_ret - lk;
                e->kick2ret_n++;
            }
        }
    }
    wp_resume = now;
}

void waitprof_exec(void)
{
    if (wp_resume) {
        wp_execlat += get_clock() - wp_resume;
        wp_execlat_n++;
        wp_resume = 0;
    }
}

/* --- BQL (tables touched with the BQL held) --- */
typedef struct {
    const char *file;
    int line;
    char thread[16];
    uint64_t n, wait, wait_max, hist[WP_NHIST], hold, hhist[WP_NHIST];
    uint64_t out, out_n;    /* this thread's time without the BQL before this acquire */
} WPSite;
#define WP_NSITES 256
static WPSite wp_site[WP_NSITES];
static int wp_nsites;
typedef struct { uint16_t w, o; uint64_t ns, n; } WPPair;
#define WP_NPAIRS 512
static WPPair wp_pair[WP_NPAIRS];
static int wp_npairs;
typedef struct { uint16_t site; int64_t t0, t1; } WPHold;
#define WP_RING 8192
static WPHold wp_ring[WP_RING];
static uint64_t wp_ring_head;
static __thread int wp_cur = -1;
static __thread int64_t wp_hold_t0, wp_last_rel;
/* Accounting checks: waits >= 1 us (total, attributed to holds, free), lost records. */
static uint64_t wp_w1, wp_w1_owned, wp_w1_free, wp_ring_trunc, wp_pair_drop, wp_site_drop;
static __thread char wp_tname[16];
static int64_t wp_last_dump;
#define WP_FREE 0xffff

static int wp_site_find(const char *file, int line)
{
    if (!wp_tname[0]) {
        pthread_getname_np(pthread_self(), wp_tname, sizeof(wp_tname));
        if (!wp_tname[0]) {
            snprintf(wp_tname, sizeof(wp_tname), "t%x",
                     (unsigned)pthread_mach_thread_np(pthread_self()));
        }
    }
    for (int i = 0; i < wp_nsites; i++) {
        WPSite *s = &wp_site[i];
        if (s->line == line && s->file == file && !strcmp(s->thread, wp_tname)) {
            return i;
        }
    }
    if (wp_nsites == WP_NSITES) {
        wp_site_drop++;
        return -1;
    }
    WPSite *s = &wp_site[wp_nsites];
    s->file = file;
    s->line = line;
    memcpy(s->thread, wp_tname, sizeof(wp_tname));
    return wp_nsites++;
}

static void wp_pair_add(int w, int o, int64_t ns)
{
    for (int i = 0; i < wp_npairs; i++) {
        if (wp_pair[i].w == w && wp_pair[i].o == o) {
            wp_pair[i].ns += ns;
            wp_pair[i].n++;
            return;
        }
    }
    if (wp_npairs < WP_NPAIRS) {
        wp_pair[wp_npairs++] = (WPPair){ w, o, ns, 1 };
    } else {
        wp_pair_drop++;
    }
}

static void wp_acquired(const char *file, int line, int64_t t_req)
{
    int64_t now = get_clock(), wait = now - t_req;
    int w = wp_site_find(file, line);

    wp_cur = w;
    wp_hold_t0 = now;
    if (w < 0) {
        return;
    }
    WPSite *s = &wp_site[w];
    s->n++;
    s->wait += wait;
    s->wait_max = MAX(s->wait_max, wait);
    s->hist[wp_bucket(wait)]++;
    if (wp_last_rel) {
        s->out += now - wp_last_rel;
        s->out_n++;
    }
    if (wait < 1000) {
        return;
    }
    /* Split the wait among the holds that overlapped it (newest first). */
    int64_t owned = 0;
    uint64_t k;
    for (k = 0; k < WP_RING && k < wp_ring_head; k++) {
        WPHold *h = &wp_ring[(wp_ring_head - 1 - k) % WP_RING];
        if (h->t1 <= t_req) {
            break;
        }
        int64_t ov = MIN(h->t1, now) - MAX(h->t0, t_req);
        if (ov > 0) {
            wp_pair_add(w, h->site, ov);
            owned += ov;
        }
    }
    if (k == WP_RING) {
        wp_ring_trunc++;        /* older holds overwritten: attribution incomplete */
    }
    wp_w1 += wait;
    wp_w1_owned += owned;
    if (wait > owned) {
        wp_pair_add(w, WP_FREE, wait - owned);
        wp_w1_free += wait - owned;
    }
}

/* About to release (still held): record the hold. */
static bool wp_release(void)
{
    int64_t now = get_clock();
    if (wp_cur >= 0) {
        wp_site[wp_cur].hold += now - wp_hold_t0;
        wp_site[wp_cur].hhist[wp_bucket(now - wp_hold_t0)]++;
        wp_ring[wp_ring_head++ % WP_RING] = (WPHold){ wp_cur, wp_hold_t0, now };
    }
    wp_last_rel = now;
    return wp_last_dump && now - wp_last_dump >= 10 * NANOSECONDS_PER_SECOND;
}

static struct {
    int64_t now, dt;
    WPSite site[WP_NSITES];
    int nsites;
    WPPair pair[WP_NPAIRS];
    int npairs;
    WPIdle idle[WP_NIDLE];
    int nidle;
    WPCtx ctx[WP_NCTX];
    int nctx;
    uint64_t ctx_drop, ctx_drop_ns;
    WPRet ret[WP_NRET];
    int nret;
    uint64_t ret_drop, ret_drop_eps, ret_drop_ns, pend_n, pend_ns;
    WPAs as[WP_NAS];
    int nas;
    WPStk stk[WP_NSTK];
    int nstk;
    uint64_t stk_drop, stk_drop_ns;
    WPCode code[WP_NCODE];
    int ncode;
    uint64_t code_drop;
    uint64_t execlat, execlat_n, kicks;
    uint64_t w1, w1_owned, w1_free, ring_trunc, pair_drop, site_drop;
} wp_snap;

/* Under the BQL: copy and reset. */
static void wp_snapshot(void)
{
    int64_t now = get_clock();
    wp_snap.now = now;
    wp_snap.dt = now - wp_last_dump;
    memcpy(wp_snap.site, wp_site, sizeof(wp_site));
    wp_snap.nsites = wp_nsites;
    memcpy(wp_snap.pair, wp_pair, sizeof(wp_pair));
    wp_snap.npairs = wp_npairs;
    memcpy(wp_snap.idle, wp_idle, sizeof(wp_idle));
    wp_snap.nidle = wp_nidle;
    memcpy(wp_snap.ctx, wp_ctx, sizeof(wp_ctx));
    wp_snap.nctx = wp_nctx;
    wp_snap.ctx_drop = wp_ctx_drop;
    wp_snap.ctx_drop_ns = wp_ctx_drop_ns;
    memset(wp_ctx, 0, sizeof(wp_ctx));
    wp_nctx = 0;
    wp_ctx_drop = wp_ctx_drop_ns = 0;
    qemu_spin_lock(&wp_ret_lock);
    memcpy(wp_snap.ret, wp_ret, sizeof(wp_ret));
    wp_snap.nret = wp_nret;
    wp_snap.ret_drop = wp_ret_drop;
    wp_snap.ret_drop_eps = wp_ret_drop_eps;
    wp_snap.ret_drop_ns = wp_ret_drop_ns;
    memcpy(wp_snap.as, wp_as, sizeof(wp_as));
    wp_snap.nas = wp_nas;
    memset(wp_ret, 0, sizeof(wp_ret));
    wp_nret = 0;
    memcpy(wp_snap.stk, wp_stk, sizeof(wp_stk));
    wp_snap.nstk = wp_nstk;
    wp_snap.stk_drop = wp_stk_drop;
    wp_snap.stk_drop_ns = wp_stk_drop_ns;
    memset(wp_stk, 0, sizeof(wp_stk));
    wp_nstk = 0;
    wp_stk_drop = wp_stk_drop_ns = 0;
    wp_snap.ncode = wp_ncode - wp_ncode_out;
    memcpy(wp_snap.code, wp_code + wp_ncode_out, wp_snap.ncode * sizeof(WPCode));
    wp_ncode_out = wp_ncode;
    wp_snap.code_drop = wp_code_drop;
    wp_ret_drop = wp_ret_drop_eps = wp_ret_drop_ns = 0;
    qemu_spin_unlock(&wp_ret_lock);
    wp_snap.pend_n = qatomic_read(&wp_pend.n);      /* vCPU-thread fields: approximate */
    wp_snap.pend_ns = qatomic_read(&wp_pend.ns);
    wp_snap.execlat = wp_execlat;
    wp_snap.execlat_n = wp_execlat_n;
    wp_snap.kicks = qatomic_xchg(&wp_kicks, 0);
    for (int i = 0; i < wp_nsites; i++) {
        WPSite *s = &wp_site[i];
        s->n = s->wait = s->wait_max = s->hold = s->out = s->out_n = 0;
        memset(s->hist, 0, sizeof(s->hist));
        memset(s->hhist, 0, sizeof(s->hhist));
    }
    wp_snap.w1 = wp_w1;
    wp_snap.w1_owned = wp_w1_owned;
    wp_snap.w1_free = wp_w1_free;
    wp_snap.ring_trunc = wp_ring_trunc;
    wp_snap.pair_drop = wp_pair_drop;
    wp_snap.site_drop = wp_site_drop;
    wp_w1 = wp_w1_owned = wp_w1_free = wp_ring_trunc = wp_pair_drop = wp_site_drop = 0;
    wp_npairs = 0;
    wp_nidle = 0;
    wp_execlat = wp_execlat_n = 0;
    wp_last_dump = now;
}

static const char *wp_sname(WPSite *s, char *buf, size_t n)
{
    const char *f = strrchr(s->file, '/');
    snprintf(buf, n, "%s:%d[%s]", f ? f + 1 : s->file, s->line, s->thread);
    return buf;
}

/* After unlocking: print the copy. */
static void wp_print(void)
{
    double ms = wp_snap.dt / 1e6;
    char a[96], b[96];
    fprintf(stderr, "waitprof T=%.3f dt_ms=%.0f kicks=%" PRIu64
            " exec_after_idle n=%" PRIu64 " avg_us=%.1f\n",
            g_get_real_time() / 1e6, ms, wp_snap.kicks, wp_snap.execlat_n,
            wp_snap.execlat_n ? wp_snap.execlat / 1e3 / wp_snap.execlat_n : 0);
    for (int i = 0; i < wp_snap.nidle; i++) {
        WPIdle *e = &wp_snap.idle[i];
        fprintf(stderr, "wp-idle %-40s n=%" PRIu64 " ms=%.1f pct=%.2f h=%" PRIu64
                "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64
                " waits=%" PRIu64 " multi=%" PRIu64 " kick2ret_us=%.1f(n=%" PRIu64
                ") ret2exit_us=%.1f\n", e->key, e->n, e->tot / 1e6,
                100.0 * e->tot / wp_snap.dt, e->hist[0], e->hist[1], e->hist[2],
                e->hist[3], e->hist[4], e->hist[5], e->waits, e->multi,
                e->kick2ret_n ? e->kick2ret / 1e3 / e->kick2ret_n : 0,
                e->kick2ret_n, e->n ? e->ret2exit / 1e3 / e->n : 0);
    }
    fprintf(stderr, "wp-ctxsum entries=%d drop_n=%" PRIu64 " drop_ms=%.2f\n",
            wp_snap.nctx, wp_snap.ctx_drop, wp_snap.ctx_drop_ns / 1e6);
    for (int i = 0; i < WP_NCTX; i++) {
        WPCtx *c = &wp_snap.ctx[i];
        if (!c->used) {
            continue;
        }
        fprintf(stderr, "wp-ctx nip=%08x lr=%08x pr=%u msr=%08x r1=%08x-%08x n=%" PRIu64
                " ms=%.3f h=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64
                "/%" PRIu64 " cls_ms=%.3f/%.3f/%.3f\n", c->nip, c->lr,
                (c->msr >> 14) & 1, c->msr, c->r1min, c->r1max, c->n, c->tot / 1e6,
                c->hist[0], c->hist[1], c->hist[2], c->hist[3], c->hist[4], c->hist[5],
                c->cls[0] / 1e6, c->cls[1] / 1e6, c->cls[2] / 1e6);
    }
    {
        uint64_t rets = 0, eps = 0, ns = 0;
        for (int i = 0; i < WP_NRET; i++) {
            rets += wp_snap.ret[i].rets;
            eps += wp_snap.ret[i].eps;
            ns += wp_snap.ret[i].ns;
        }
        fprintf(stderr, "wp-retsum entries=%d rets=%" PRIu64 " eps=%" PRIu64 " ms=%.3f"
                " drop_rets=%" PRIu64 " drop_eps=%" PRIu64 " drop_ms=%.3f pend_eps=%"
                PRIu64 " pend_ms=%.3f\n", wp_snap.nret, rets, eps, ns / 1e6,
                wp_snap.ret_drop, wp_snap.ret_drop_eps, wp_snap.ret_drop_ns / 1e6,
                wp_snap.pend_n, wp_snap.pend_ns / 1e6);
    }
    for (int i = 0; i < WP_NRET; i++) {
        WPRet *r = &wp_snap.ret[i];
        if (!r->used) {
            continue;
        }
        fprintf(stderr, "wp-ret nip=%08x asid=%06x lr=%08x lr_other=%u rets=%" PRIu64
                " eps=%" PRIu64 " ms=%.3f cls_ms=%.3f/%.3f/%.3f delay_ms=%.3f dh=%"
                PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64
                " batch=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "\n",
                r->nip, r->asid, r->lr, r->lr_other, r->rets, r->eps, r->ns / 1e6,
                r->cls[0] / 1e6, r->cls[1] / 1e6, r->cls[2] / 1e6, r->delay / 1e6,
                r->dhist[0], r->dhist[1], r->dhist[2], r->dhist[3], r->dhist[4],
                r->dhist[5], r->batch[0], r->batch[1], r->batch[2], r->batch[3]);
    }
    for (int i = 0; i < wp_snap.nas; i++) {
        fprintf(stderr, "wp-as asid=%06x text=%x hdr=%016" PRIx64 " path=%s\n",
                wp_snap.as[i].asid, wp_snap.as[i].text, wp_snap.as[i].hdr,
                wp_snap.as[i].path);
    }
    {
        uint64_t rets = 0, eps = 0, ns = 0, why[4] = { 0 };
        for (int i = 0; i < WP_NSTK; i++) {
            WPStk *s = &wp_snap.stk[i];
            rets += s->rets;
            eps += s->eps;
            ns += s->ns;
            if (s->used) {
                why[s->why] += s->rets;
            }
        }
        fprintf(stderr, "wp-stksum nip=%08x entries=%d rets=%" PRIu64 " eps=%" PRIu64
                " ms=%.3f why=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64
                " drop_rets=%" PRIu64 " drop_ms=%.3f code_drop=%" PRIu64 "\n",
                wp_stk_nip, wp_snap.nstk, rets, eps, ns / 1e6, why[0], why[1], why[2],
                why[3], wp_snap.stk_drop, wp_snap.stk_drop_ns / 1e6, wp_snap.code_drop);
    }
    for (int i = 0; i < WP_NSTK; i++) {
        WPStk *s = &wp_snap.stk[i];
        if (!s->used) {
            continue;
        }
        fprintf(stderr, "wp-stk asid=%06x r1=%08x-%08x depth=%u why=%u f=%08x,%08x,%08x,"
                "%08x,%08x rets=%" PRIu64 " eps=%" PRIu64 " ms=%.3f cls_ms=%.3f/%.3f/%.3f\n",
                s->asid, s->r1min, s->r1max, s->depth, s->why, s->f[0], s->f[1],
                s->f[2], s->f[3], s->f[4], s->rets, s->eps, s->ns / 1e6,
                s->cls[0] / 1e6, s->cls[1] / 1e6, s->cls[2] / 1e6);
    }
    for (int i = 0; i < wp_snap.ncode; i++) {
        WPCode *c = &wp_snap.code[i];
        char hex[sizeof(c->b) * 2 + 1];
        for (unsigned j = 0; j < sizeof(c->b); j++) {
            snprintf(hex + 2 * j, 3, "%02x", c->b[j]);
        }
        fprintf(stderr, "wp-code asid=%06x addr=%08x ok=%d b=%s\n", c->asid, c->addr,
                c->ok, c->ok ? hex : "-");
    }
    for (int i = 0; i < wp_snap.nsites; i++) {
        WPSite *s = &wp_snap.site[i];
        if (!s->n || (s->wait < wp_snap.dt / 2000 && s->hold < wp_snap.dt / 2000)) {
            continue;           /* under 0.05% both ways */
        }
        fprintf(stderr, "wp-site %-44s n=%" PRIu64 " wait_ms=%.2f wait_pct=%.2f "
                "max_us=%.0f h=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%"
                PRIu64 "/%" PRIu64 " hold_pct=%.2f hh=%" PRIu64 "/%" PRIu64 "/%"
                PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 " out_ms=%.2f out_n=%"
                PRIu64 "\n",
                wp_sname(s, a, sizeof(a)), s->n, s->wait / 1e6,
                100.0 * s->wait / wp_snap.dt, s->wait_max / 1e3, s->hist[0],
                s->hist[1], s->hist[2], s->hist[3], s->hist[4], s->hist[5],
                100.0 * s->hold / wp_snap.dt, s->hhist[0], s->hhist[1],
                s->hhist[2], s->hhist[3], s->hhist[4], s->hhist[5], s->out / 1e6,
                s->out_n);
    }
    fprintf(stderr, "wp-check w1_ms=%.2f owned_ms=%.2f free_ms=%.2f ring_trunc=%"
            PRIu64 " pair_drop=%" PRIu64 " site_drop=%" PRIu64 "\n",
            wp_snap.w1 / 1e6, wp_snap.w1_owned / 1e6, wp_snap.w1_free / 1e6,
            wp_snap.ring_trunc, wp_snap.pair_drop, wp_snap.site_drop);
    for (int i = 0; i < wp_snap.npairs; i++) {
        WPPair *p = &wp_snap.pair[i];
        if (p->ns < wp_snap.dt / 2000) {
            continue;
        }
        fprintf(stderr, "wp-pair %-44s <- %-44s ms=%.2f pct=%.2f n=%" PRIu64 "\n",
                wp_sname(&wp_snap.site[p->w], a, sizeof(a)),
                p->o == WP_FREE ? "(free)" : wp_sname(&wp_snap.site[p->o], b, sizeof(b)),
                p->ns / 1e6, 100.0 * p->ns / wp_snap.dt, p->n);
    }
}

/*
 * The BQL is taken from so many places that it is worth profiling the
 * callers directly, instead of funneling them all through a single function.
 */
void bql_lock_impl(const char *file, int line)
{
    QemuMutexLockFunc bql_lock_fn = qatomic_read(&bql_mutex_lock_func);
    int64_t t_req = waitprof_on ? get_clock() : 0;

    g_assert(!bql_locked());
    bql_lock_fn(&bql, file, line);
    set_bql_locked(true);
    if (t_req) {
        if (!wp_last_dump) {
            wp_last_dump = get_clock();
        }
        wp_acquired(file, line, t_req);
    }
}

void bql_unlock(void)
{
    bool dump = false;

    g_assert(bql_locked());
    g_assert(!bql_unlock_blocked);
    if (waitprof_on && wp_release()) {
        wp_snapshot();
        dump = true;
    }
    set_bql_locked(false);
    qemu_mutex_unlock(&bql);
    if (dump) {
        wp_print();
    }
}

void qemu_cond_wait_bql(QemuCond *cond)
{
    if (waitprof_on) {
        wp_release();
    }
    qemu_cond_wait(cond, &bql);
    if (waitprof_on) {
        wp_hold_t0 = get_clock();     /* same site holds again */
    }
}

void qemu_cond_timedwait_bql(QemuCond *cond, int ms)
{
    if (waitprof_on) {
        wp_release();
    }
    qemu_cond_timedwait(cond, &bql, ms);
    if (waitprof_on) {
        wp_hold_t0 = get_clock();
    }
}

/* signal CPU creation */
void cpu_thread_signal_created(CPUState *cpu)
{
    cpu->created = true;
    qemu_cond_signal(&qemu_cpu_cond);
}

/* signal CPU destruction */
void cpu_thread_signal_destroyed(CPUState *cpu)
{
    cpu->created = false;
    qemu_cond_signal(&qemu_cpu_cond);
}

void cpu_pause(CPUState *cpu)
{
    if (qemu_cpu_is_self(cpu)) {
        qemu_cpu_stop(cpu, true);
    } else {
        cpu->stop = true;
        qemu_cpu_kick(cpu);
    }
}

void cpu_resume(CPUState *cpu)
{
    cpu->stop = false;
    cpu->stopped = false;
    qemu_cpu_kick(cpu);
}

static bool all_vcpus_paused(void)
{
    CPUState *cpu;

    CPU_FOREACH(cpu) {
        if (!cpu->stopped) {
            return false;
        }
    }

    return true;
}

void pause_all_vcpus(void)
{
    CPUState *cpu;

    qemu_clock_enable(QEMU_CLOCK_VIRTUAL, false);
    CPU_FOREACH(cpu) {
        cpu_pause(cpu);
    }

    /* We need to drop the replay_lock so any vCPU threads woken up
     * can finish their replay tasks
     */
    replay_mutex_unlock();

    while (!all_vcpus_paused()) {
        qemu_cond_wait(&qemu_pause_cond, &bql);
        CPU_FOREACH(cpu) {
            qemu_cpu_kick(cpu);
        }
    }

    bql_unlock();
    replay_mutex_lock();
    bql_lock();
}

void resume_all_vcpus(void)
{
    CPUState *cpu;

    if (!runstate_is_running()) {
        return;
    }

    qemu_clock_enable(QEMU_CLOCK_VIRTUAL, true);
    CPU_FOREACH(cpu) {
        cpu_resume(cpu);
    }
}

void cpu_remove_sync(CPUState *cpu)
{
    cpu->stop = true;
    cpu->unplug = true;
    qemu_cpu_kick(cpu);
    bql_unlock();
    qemu_thread_join(cpu->thread);
    bql_lock();
}

void cpus_register_accel(const AccelOpsClass *ops)
{
    assert(ops != NULL);
    assert(ops->create_vcpu_thread != NULL); /* mandatory */
    cpus_accel = ops;
}

const AccelOpsClass *cpus_get_accel(void)
{
    /* broken if we call this early */
    assert(cpus_accel);
    return cpus_accel;
}

void qemu_init_vcpu(CPUState *cpu)
{
    MachineState *ms = MACHINE(qdev_get_machine());

    cpu->nr_threads =  ms->smp.threads;
    cpu->stopped = true;
    cpu->random_seed = qemu_guest_random_seed_thread_part1();

    if (!cpu->as) {
        /* If the target cpu hasn't set up any address spaces itself,
         * give it the default one.
         */
        cpu->num_ases = 1;
        cpu_address_space_init(cpu, 0, "cpu-memory", cpu->memory);
    }

    /* accelerators all implement the AccelOpsClass */
    g_assert(cpus_accel != NULL && cpus_accel->create_vcpu_thread != NULL);
    cpus_accel->create_vcpu_thread(cpu);

    while (!cpu->created) {
        qemu_cond_wait(&qemu_cpu_cond, &bql);
    }
}

void cpu_stop_current(void)
{
    if (current_cpu) {
        current_cpu->stop = true;
        cpu_exit(current_cpu);
    }
}

int vm_stop(RunState state)
{
    if (qemu_in_vcpu_thread()) {
        qemu_system_vmstop_request_prepare();
        qemu_system_vmstop_request(state);
        /*
         * FIXME: should not return to device code in case
         * vm_stop() has been requested.
         */
        cpu_stop_current();
        return 0;
    }

    return do_vm_stop(state, true);
}

/**
 * Prepare for (re)starting the VM.
 * Returns 0 if the vCPUs should be restarted, -1 on an error condition,
 * and 1 otherwise.
 */
int vm_prepare_start(bool step_pending)
{
    int ret = vm_was_suspended ? 1 : 0;
    RunState state = vm_was_suspended ? RUN_STATE_SUSPENDED : RUN_STATE_RUNNING;
    RunState requested;

    qemu_vmstop_requested(&requested);
    if (runstate_is_running() && requested == RUN_STATE__MAX) {
        return -1;
    }

    /* Ensure that a STOP/RESUME pair of events is emitted if a
     * vmstop request was pending.  The BLOCK_IO_ERROR event, for
     * example, according to documentation is always followed by
     * the STOP event.
     */
    if (runstate_is_running()) {
        qapi_event_send_stop();
        qapi_event_send_resume();
        return -1;
    }

    /*
     * WHPX accelerator needs to know whether we are going to step
     * any CPUs, before starting the first one.
     */
    if (cpus_accel->synchronize_pre_resume) {
        cpus_accel->synchronize_pre_resume(step_pending);
    }

    /* We are sending this now, but the CPUs will be resumed shortly later */
    qapi_event_send_resume();

    cpu_enable_ticks();
    runstate_set(state);
    vm_state_notify(1, state);
    vm_was_suspended = false;
    return ret;
}

void vm_start(void)
{
    if (!vm_prepare_start(false)) {
        resume_all_vcpus();
    }
}

void vm_resume(RunState state)
{
    if (runstate_is_live(state)) {
        vm_start();
    } else {
        runstate_set(state);
    }
}

/* does a state transition even if the VM is already stopped,
   current state is forgotten forever */
int vm_stop_force_state(RunState state)
{
    if (runstate_is_live(runstate_get())) {
        return vm_stop(state);
    } else {
        int ret;
        runstate_set(state);

        bdrv_drain_all();
        /* Make sure to return an error if the flush in a previous vm_stop()
         * failed. */
        ret = bdrv_flush_all();
        trace_vm_stop_flush_all(ret);
        return ret;
    }
}

void qmp_memsave(uint64_t addr, uint64_t size, const char *filename,
                 bool has_cpu, int64_t cpu_index, Error **errp)
{
    FILE *f;
    uint64_t l;
    CPUState *cpu;
    uint8_t buf[1024];
    uint64_t orig_addr = addr, orig_size = size;

    if (!has_cpu) {
        cpu_index = 0;
    }

    cpu = qemu_get_cpu(cpu_index);
    if (cpu == NULL) {
        error_setg(errp, QERR_INVALID_PARAMETER_VALUE, "cpu-index",
                   "a CPU number");
        return;
    }

    f = fopen(filename, "wb");
    if (!f) {
        error_setg_file_open(errp, errno, filename);
        return;
    }

    while (size != 0) {
        l = sizeof(buf);
        if (l > size)
            l = size;
        if (cpu_memory_rw_debug(cpu, addr, buf, l, 0) != 0) {
            error_setg(errp, "Invalid addr 0x%016" PRIx64 "/size %" PRIu64
                             " specified", orig_addr, orig_size);
            goto exit;
        }
        if (fwrite(buf, 1, l, f) != l) {
            error_setg(errp, "writing memory to '%s' failed",
                       filename);
            goto exit;
        }
        addr += l;
        size -= l;
    }

exit:
    fclose(f);
}

void qmp_pmemsave(uint64_t addr, uint64_t size, const char *filename,
                  Error **errp)
{
    FILE *f;
    uint64_t l;
    uint8_t buf[1024];

    f = fopen(filename, "wb");
    if (!f) {
        error_setg_file_open(errp, errno, filename);
        return;
    }

    while (size != 0) {
        l = sizeof(buf);
        if (l > size)
            l = size;
        cpu_physical_memory_read(addr, buf, l);
        if (fwrite(buf, 1, l, f) != l) {
            error_setg(errp, "writing memory to '%s' failed",
                       filename);
            goto exit;
        }
        addr += l;
        size -= l;
    }

exit:
    fclose(f);
}

void qmp_inject_nmi(Error **errp)
{
    nmi_monitor_handle(monitor_get_cpu_index(monitor_cur()), errp);
}

