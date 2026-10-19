#!/usr/bin/env python3
"""leafcallers.py SAMPLE.txt...: (inclusive counts) for the CP thread's BQL-held samples (as cpsplit.py), the
QEMU call sites (innermost 3 own-source frames) under each hot library leaf."""
import re, sys
from collections import Counter, defaultdict
node = re.compile(r'^([\s+!:|]*)(\d+) (.+?)\s*$')
LEAVES = ('strdup', 'gpu_debug_log', 'r300_tex_layout', 'MTLPixelFormatGetInfoForDevice', 'memory_region_get_ram_ptr')
callers = defaultdict(Counter)
for path in sys.argv[1:]:
    L = open(path, errors='replace').read().split('\n')
    i = next(k for k, l in enumerate(L) if 'ppc_mac_gpu_cp_thread' in l)
    while not re.match(r'^    \d+ Thread_', L[i]): i -= 1
    stack = []; nodes = []
    for l in L[i + 1:]:
        if re.match(r'^    \d+ ', l) or not l.strip(): break
        m = node.match(l)
        if not m: continue
        d = len(m.group(1)); c = int(m.group(2)); txt = m.group(3)
        fn = txt.split()[0]; ln = re.search(r'(\w+\.[cm]:\d+)\s*$', txt)
        while stack and stack[-1][0] >= d: stack.pop()
        if stack: nodes[stack[-1][1]][2] += c
        nodes.append([fn + (' ' + ln.group(1) if ln else ''), c, 0, [n for _, n in stack]])
        stack.append((d, len(nodes) - 1))
    for name, c, ch, anc in nodes:
        s = c; fn = name.split()[0]
        if fn not in LEAVES: continue
        fns = [nodes[a][0] for a in anc]
        if not any('ppc_mac_gpu_cp_run' in f for f in fns) or any(f.startswith(('bql_lock', 'g_usleep', 'metal_range_busy')) for f in fns): continue
        own = [f for f in fns if re.search(r'\.[cm]:\d+', f)]
        callers[fn][' <- '.join(f.replace('ppc_mac_gpu_', 'g_') for f in own[-3:][::-1])] += s
for fn in LEAVES:
    t = sum(callers[fn].values())
    if not t: continue
    print('%s  %d' % (fn, t))
    for k, v in callers[fn].most_common(5): print('   %5d  %s' % (v, k))
