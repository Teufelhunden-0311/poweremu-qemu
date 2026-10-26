/* Test shim: QEMU's qatomic_* as compiler builtins.  AI-generated. */
#ifndef ENCSHIM_ATOMIC_H
#define ENCSHIM_ATOMIC_H
#define qatomic_read(p)            __atomic_load_n(p, __ATOMIC_RELAXED)
#define qatomic_set(p, v)          __atomic_store_n(p, v, __ATOMIC_RELAXED)
#define qatomic_load_acquire(p)    __atomic_load_n(p, __ATOMIC_ACQUIRE)
#define qatomic_store_release(p, v) __atomic_store_n(p, v, __ATOMIC_RELEASE)
#define qatomic_fetch_add(p, v)    __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST)
#define qatomic_inc(p)             ((void)__atomic_fetch_add(p, 1, __ATOMIC_SEQ_CST))
#define smp_mb()                   __atomic_thread_fence(__ATOMIC_SEQ_CST)
#endif
