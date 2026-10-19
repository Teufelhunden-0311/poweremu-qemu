#!/usr/bin/env python3
"""cpsplit.py SAMPLE.txt...: the GPU CP thread's samples (macOS `sample`, 1 ms) split by
lock state, established from the call path against gpu-work fc81963's source:
  idle         qemu_event_wait in ppc_mac_gpu_cp_thread (no work, no BQL)
  acquire      in bql_lock (waiting for the BQL)
  park-sleep   g_usleep in ppc_mac_gpu_cp_park (BQL released)
  off-lock     under r200_vram_access's renderer calls (range_busy / async drain /
               flush), which run between cp_wait_unlock and cp_wait_relock (async renderer)
  held-wait    blocked in a wait primitive with no BQL release on the path (BQL held)
  held-exec    everything else under ppc_mac_gpu_cp_run (BQL held): ranked by self samples
               (leaf function) and by the nearest CP-level phase on the path."""
import re, sys
from collections import Counter
WAIT = re.compile(r'__psynch_cvwait|__psynch_mutexwait|__semwait_signal|__ulock_wait|mach_msg|kevent')
OFF = ('metal_range_busy_r200', 'r200_async_drain', 'metal_flush_r200', 'r200_flush_r200')
PHASES = ['r300_draw_build_indexed', 'r300_draw_build', 'r300_render', 'ppc_mac_gpu_2d_blit_sep',
          'ppc_mac_gpu_2d_blit', 'r200_vram_access', 'ppc_mac_gpu_mmio_write', 'ppc_mac_gpu_execute_ib',
          'ppc_mac_gpu_process_pm4', 'ppc_mac_gpu_process_ring_buffer', 'ppc_mac_gpu_cp_run']
node = re.compile(r'^([\s+!:|]*)(\d+) (.+?)\s*$')
cls = Counter(); self_ = Counter(); phase = Counter(); hw = Counter(); tot = 0
for path in sys.argv[1:]:
    L = open(path, errors='replace').read().split('\n')
    i = next(k for k, l in enumerate(L) if 'ppc_mac_gpu_cp_thread' in l)
    while not re.match(r'^    \d+ Thread_', L[i]): i -= 1
    tot += int(L[i].split()[0]); stack = []; nodes = []
    for l in L[i + 1:]:
        if re.match(r'^    \d+ ', l) or not l.strip(): break
        m = node.match(l)
        if not m: continue
        d = len(m.group(1)); c = int(m.group(2)); txt = m.group(3)
        fn = txt.split()[0]; ln = re.search(r'(\w+\.[cm]:\d+)\s*$', txt)
        while stack and stack[-1][0] >= d: stack.pop()
        if stack: nodes[stack[-1][2]][3] += c
        nodes.append([fn + (' ' + ln.group(1) if ln else ''), c, [n for _, _, n in stack], 0, fn])
        stack.append((d, c, len(nodes) - 1))
    for k, (name, c, anc, ch, fn) in enumerate(nodes):
        s = c - ch
        if s <= 0: continue
        fns = [nodes[a][4] for a in anc] + [fn]
        if 'ppc_mac_gpu_cp_run' not in fns:
            cls['idle' if 'qemu_event_wait' in fns else 'other'] += s; continue
        if any(f.startswith('bql_lock') for f in fns): cls['acquire'] += s
        elif 'ppc_mac_gpu_cp_park' in fns and 'g_usleep' in fns: cls['park-sleep'] += s
        elif any(f in OFF for f in fns) and 'r200_vram_access' in fns: cls['off-lock'] += s
        elif WAIT.search(fn) or any(WAIT.search(f) for f in fns[-3:]):
            cls['held-wait'] += s
            hw[' <- '.join(f for f in fns[-6:-1][::-1] if not f.startswith('_'))[:150]] += s
        else:
            cls['held-exec'] += s; self_[name] += s
            ph = next((p for p in PHASES if p in fns), '?')
            phase[ph] += s
print('CP thread, %d files, %d samples (1 ms each)' % (len(sys.argv) - 1, tot))
for k, v in cls.most_common(): print('  %-11s %6d  %5.1f%% of thread wall time' % (k, v, 100 * v / tot))
he = cls['held-exec']
print('\nheld-exec by nearest phase on the path (each sample counted once, innermost listed phase):')
for k, v in phase.most_common(): print('  %6d  %5.1f%% of held-exec  %s' % (v, 100 * v / he, k))
print('\nheld-exec self samples by leaf (function + line), top 30:')
for k, v in self_.most_common(30): print('  %6d  %5.1f%%  %s' % (v, 100 * v / he, k))
print('\nheld-wait paths (BQL held while blocked):')
for k, v in hw.most_common(10): print('  %6d  %s' % (v, k))
