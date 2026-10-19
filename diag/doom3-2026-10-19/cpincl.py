#!/usr/bin/env python3
"""cpincl.py SAMPLE.txt...: inclusive BQL-held execution samples per function on the CP
thread (held-exec as in cpsplit.py; each sample counts once per distinct function)."""
import re, sys
from collections import Counter
node = re.compile(r'^([\s+!:|]*)(\d+) (.+?)\s*$')
WAIT = re.compile(r'__psynch_cvwait|__psynch_mutexwait|__semwait_signal|__ulock_wait|mach_msg|kevent')
OFF = ('metal_range_busy_r200', 'r200_async_drain', 'metal_flush_r200', 'r200_flush_r200')
inc = Counter(); he = 0
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
        for f in set(fns): inc[f] += s
print('held-exec %d samples' % he)
for f, v in inc.most_common(60):
    if v >= 15: print('  %5d  %5.1f%%  %s' % (v, 100 * v / he, f))
