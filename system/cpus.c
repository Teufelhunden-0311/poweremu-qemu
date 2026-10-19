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

static void __attribute__((constructor)) waitprof_init(void)
{
    const char *e = getenv("PPC_WAIT_PROF");
    waitprof_on = e && e[0] == '1';
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

void waitprof_idle_enter(void)
{
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

    n = snprintf(key, sizeof(key), "%s |",
                 waitprof_pending ? waitprof_pending(cpu) : "?");
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

