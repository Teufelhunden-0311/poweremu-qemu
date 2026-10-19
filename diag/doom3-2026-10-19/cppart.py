#!/usr/bin/env python3
"""cppart.py SAMPLE.txt...: the CP thread's BQL-held execution samples (cpsplit.py's
held-exec) partitioned into disjoint groups: each sample goes to the innermost frame on
its path that matches a group rule, so the groups sum to 100% with no parent/child overlap."""
import re, sys
from collections import Counter
node = re.compile(r'^([\s+!:|]*)(\d+) (.+?)\s*$')
WAIT = re.compile(r'__psynch_cvwait|__psynch_mutexwait|__semwait_signal|__ulock_wait|mach_msg|kevent')
OFF = ('metal_range_busy_r200', 'r200_async_drain', 'metal_flush_r200', 'r200_flush_r200')
# (group, test on (function, parent function)); checked from the leaf upwards
RULES = [
    ('park hand-off: unlock in cp_park/cp_yield', lambda f, p: f == 'qemu_mutex_unlock_impl' and p in ('ppc_mac_gpu_cp_park', 'ppc_mac_gpu_cp_yield')),
    ('fence spin in r200_scratch_write', lambda f, p: f in ('cthread_yield', 'sched_yield') or (f == 'g_get_monotonic_time' and p == 'r200_scratch_write')),
    ('VRAM/RAM pointer lookup (memory_region_get_ram_ptr)', lambda f, p: f == 'memory_region_get_ram_ptr'),
    ('GLSL text copy (strdup) per draw', lambda f, p: f == 'strdup'),
    ('fragment GLSL generation (r300_us_to_glsl)', lambda f, p: f == 'r300_us_to_glsl'),
    ('r300_read_raw (rest: copies, checks)', lambda f, p: f == 'r300_read_raw'),
    ('fetch_vertex (rest: PSC decode, memset)', lambda f, p: f == 'fetch_vertex'),
    ('metal_draw_r300 (renderer enqueue)', lambda f, p: f == 'metal_draw_r300'),
    ('metal_submit_r200 (fence submit)', lambda f, p: f == 'metal_submit_r200'),
    ('r300_us_glsl_cached (memo lookup)', lambda f, p: f == 'r300_us_glsl_cached'),
    ('r300_tex_layout', lambda f, p: f == 'r300_tex_layout'),
    ('gpu_debug_log', lambda f, p: f == 'gpu_debug_log'),
    ('draw_core (rest)', lambda f, p: f == 'draw_core'),
    ('r300_draw_build (rest)', lambda f, p: f == 'r300_draw_build'),
    ('r300_render (rest)', lambda f, p: f == 'r300_render'),
    ('2D blit', lambda f, p: f.startswith('ppc_mac_gpu_2d_blit')),
    ('r200_scratch_write (rest)', lambda f, p: f == 'r200_scratch_write'),
    ('cp_park (rest: clock)', lambda f, p: f == 'ppc_mac_gpu_cp_park'),
    ('mmio_write / process_pm4 / execute_ib (dispatch)', lambda f, p: f.startswith(('ppc_mac_gpu_mmio_write', 'ppc_mac_gpu_process_pm4', 'ppc_mac_gpu_execute_ib'))),
]
grp = Counter(); he = 0
for path in sys.argv[1:]:
    L = open(path, errors='replace').read().split('\n')
    i = next(k for k, l in enumerate(L) if 'ppc_mac_gpu_cp_thread' in l)
    while not re.match(r'^    \d+ Thread_', L[i]): i -= 1
    stack = []; nodes = []
    for l in L[i + 1:]:
        if re.match(r'^    \d+ ', l) or not l.strip(): break
        m = node.match(l)
        if not m: continue
        d = len(m.group(1)); c = int(m.group(2)); fn = m.group(3).split()[0]
        while stack and stack[-1][0] >= d: stack.pop()
        if stack: nodes[stack[-1][1]][2] += c
        nodes.append([fn, c, 0, [n for _, n in stack]])
        stack.append((d, len(nodes) - 1))
    for fn, c, ch, anc in nodes:
        s = c - ch
        if s <= 0: continue
        fns = [nodes[a][0] for a in anc] + [fn]
        if ('ppc_mac_gpu_cp_run' not in fns or any(f.startswith('bql_lock') for f in fns)
                or ('ppc_mac_gpu_cp_park' in fns and 'g_usleep' in fns)
                or (any(f in OFF for f in fns) and 'r200_vram_access' in fns)
                or any(WAIT.search(f) for f in fns[-3:])):
            continue
        he += s
        g = 'other'
        for j in range(len(fns) - 1, 0, -1):
            hit = next((name for name, t in RULES if t(fns[j], fns[j - 1])), None)
            if hit: g = hit; break
        grp[g] += s
print('BQL-held CP execution, %d samples, disjoint groups (innermost matching frame):' % he)
for k, v in grp.most_common(): print('  %5d  %5.1f%%  %s' % (v, 100 * v / he, k))
print('  sum %d' % sum(grp.values()))
