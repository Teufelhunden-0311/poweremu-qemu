/*
 * Test shim: the big QEMU lock.  The device draws with it held (the
 * PPCGPU_R300_EXTENT planner checks), so the test's draws count as held
 * unless encshim_bql says otherwise.  AI-generated.
 */
#ifndef ENCSHIM_MAIN_LOOP_H
#define ENCSHIM_MAIN_LOOP_H
#include <stdbool.h>
extern bool encshim_bql;
static inline bool bql_locked(void) { return encshim_bql; }
#endif
