/* Test shim: QEMU's log calls go to stderr.  AI-generated. */
#ifndef ENCSHIM_LOG_H
#define ENCSHIM_LOG_H
#include <stdio.h>
#define LOG_GUEST_ERROR (1 << 11)
#define LOG_UNIMP       (1 << 10)
#define qemu_log(...)            fprintf(stderr, __VA_ARGS__)
#define qemu_log_mask(m, ...)    fprintf(stderr, __VA_ARGS__)
#endif
