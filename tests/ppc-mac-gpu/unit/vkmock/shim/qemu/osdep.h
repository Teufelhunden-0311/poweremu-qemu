/*
 * Stand-in for QEMU's osdep.h, for building hw/display/ppc_mac_gpu_vulkan.c
 * outside a QEMU build (the Vulkan failure-path test).  Only what that file
 * uses; the macros are QEMU's own definitions.
 */
#ifndef VKMOCK_QEMU_OSDEP_H
#define VKMOCK_QEMU_OSDEP_H

#include <assert.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <glib.h>

#undef MIN
#define MIN(a, b)                                       \
    ({                                                  \
        typeof(1 ? (a) : (b)) _a = (a), _b = (b);       \
        _a < _b ? _a : _b;                              \
    })
#undef MAX
#define MAX(a, b)                                       \
    ({                                                  \
        typeof(1 ? (a) : (b)) _a = (a), _b = (b);       \
        _a > _b ? _a : _b;                              \
    })
#define ROUND_DOWN(n, d) ((n) & -(0 ? (n) : (d)))
#define ROUND_UP(n, d) ROUND_DOWN((n) + (d) - 1, (d))
#define DIV_ROUND_UP(n, d) (((n) + (d) - 1) / (d))
#ifndef ARRAY_SIZE
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#endif

static inline uintptr_t qemu_real_host_page_size(void)
{
    return getpagesize();
}

#endif
