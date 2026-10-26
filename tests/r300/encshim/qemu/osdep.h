/*
 * Test shim (tests/r300/test_encode.m): just enough of QEMU's osdep.h to
 * build hw/display/ppc_mac_gpu_metal.m outside QEMU.  AI-generated.
 */
#ifndef ENCSHIM_OSDEP_H
#define ENCSHIM_OSDEP_H
#define CONFIG_DARWIN 1
#include <glib.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define QEMU_BUILD_BUG_ON(x) _Static_assert(!(x), "not expecting: " #x)
#ifndef likely
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif
#endif
