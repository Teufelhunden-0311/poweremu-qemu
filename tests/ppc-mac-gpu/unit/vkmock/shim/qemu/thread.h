#ifndef VKMOCK_QEMU_THREAD_H
#define VKMOCK_QEMU_THREAD_H
#include <pthread.h>

typedef struct QemuMutex { pthread_mutex_t lock; } QemuMutex;
typedef struct QemuCond { pthread_cond_t cond; } QemuCond;
typedef struct QemuThread { pthread_t thread; } QemuThread;

#define QEMU_THREAD_JOINABLE 0
#define QEMU_THREAD_DETACHED 1

static inline void qemu_mutex_init(QemuMutex *m) { pthread_mutex_init(&m->lock, NULL); }
static inline void qemu_mutex_lock(QemuMutex *m) { pthread_mutex_lock(&m->lock); }
static inline void qemu_mutex_unlock(QemuMutex *m) { pthread_mutex_unlock(&m->lock); }
static inline void qemu_cond_init(QemuCond *c) { pthread_cond_init(&c->cond, NULL); }
static inline void qemu_cond_signal(QemuCond *c) { pthread_cond_signal(&c->cond); }
static inline void qemu_cond_broadcast(QemuCond *c) { pthread_cond_broadcast(&c->cond); }
static inline void qemu_cond_wait(QemuCond *c, QemuMutex *m)
{
    pthread_cond_wait(&c->cond, &m->lock);
}

/* The test tags the threads the renderer starts (mockvk.h mock_thread). */
void qemu_thread_create(QemuThread *thread, const char *name,
                        void *(*start_routine)(void *), void *arg, int mode);
#endif
