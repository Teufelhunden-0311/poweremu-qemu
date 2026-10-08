#ifndef VKMOCK_QEMU_ATOMIC_H
#define VKMOCK_QEMU_ATOMIC_H
#define qatomic_read(p)        __atomic_load_n(p, __ATOMIC_SEQ_CST)
#define qatomic_set(p, v)      __atomic_store_n(p, v, __ATOMIC_SEQ_CST)
#define qatomic_inc(p)         ((void)__atomic_fetch_add(p, 1, __ATOMIC_SEQ_CST))
#define qatomic_fetch_inc(p)   __atomic_fetch_add(p, 1, __ATOMIC_SEQ_CST)
#define qatomic_xchg(p, v)     __atomic_exchange_n(p, v, __ATOMIC_SEQ_CST)
/* returns the value found, as QEMU's does */
#define qatomic_cmpxchg(p, old, new)                                    \
    ({                                                                  \
        typeof(*(p)) _old = (old);                                      \
        __atomic_compare_exchange_n(p, &_old, new, false,               \
                                    __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);\
        _old;                                                           \
    })
#endif
