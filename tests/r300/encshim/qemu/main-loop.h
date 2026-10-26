/* Test shim: no big QEMU lock here.  AI-generated. */
#ifndef ENCSHIM_MAIN_LOOP_H
#define ENCSHIM_MAIN_LOOP_H
#include <stdbool.h>
static inline bool bql_locked(void) { return false; }
#endif
