#!/usr/bin/env python3
"""wpagg.py VM.OUT START END: sum the PPC_WAIT_PROF 10 s windows that lie wholly inside
[START, END] (Unix seconds) and print shares of wall time.
Idle keys are "<pending bits at resume> | <interrupt sources that rose during the episode>":
D = decrementer, E = external, O = other, - = none; sources are OpenPIC numbers in hex
(0d IDE0, 18/09/0a Screamer, 19 PMU/CUDA, 1b-1e PCI/AGP shared), gpu = the GPU's own line."""
import re, sys
from collections import defaultdict
path, start, end = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
wins = []; cur = None
for l in open(path, errors='replace'):
    m = re.search(r'waitprof T=([\d.]+) dt_ms=(\d+) kicks=(\d+) exec_after_idle n=(\d+) avg_us=([\d.]+)', l)
    if m:
        cur = dict(T=float(m.group(1)), dt=float(m.group(2)), kicks=int(m.group(3)),
                   exn=int(m.group(4)), exus=float(m.group(5)), idle=[], site=[], pair=[], chk=[])
        wins.append(cur); continue
    if not cur: continue
    if l.startswith('wp-idle '):
        key = l[8:].split(' n=')[0].strip()
        kv = dict(re.findall(r'(\w+)=([\d./]+)', l[8 + len(key):]))
        cur['idle'].append((key, kv))
    elif l.startswith('wp-site '):
        name = l[8:].split(' n=')[0].strip(); kv = dict(re.findall(r'(\w+)=([\d./]+)', l)); cur['site'].append((name, kv))
    elif l.startswith('wp-check '):
        cur['chk'].append(dict((k, float(v)) for k, v in re.findall(r'(\w+)=([\d.]+)', l)))
    elif l.startswith('wp-pair '):
        m = re.match(r'wp-pair (.+?)\s+<- (.+?)\s+ms=([\d.]+) pct=[\d.]+ n=(\d+)', l)
        if m: cur['pair'].append((m.group(1), m.group(2), float(m.group(3)), int(m.group(4))))
sel = [w for w in wins if w['T'] - w['dt'] / 1000 >= start and w['T'] <= end]
tot = sum(w['dt'] for w in sel)
print('windows used: %d of %d, %.1f s' % (len(sel), len(wins), tot / 1000))
if not sel: sys.exit()
idle = defaultdict(lambda: [0, 0.0, [0] * 6, 0, 0, 0.0, 0, 0.0])
for w in sel:
    for key, kv in w['idle']:
        e = idle[key]; n = int(kv['n'])
        e[0] += n; e[1] += float(kv['ms'])
        for i, h in enumerate(kv['h'].split('/')): e[2][i] += int(h)
        e[3] += int(kv['waits']); e[4] += int(kv['multi'])
        kn = int(kv.get('n', 0))
        m = re.search(r'kick2ret_us=([\d.]+)\(n=(\d+)\)', ' '.join('%s=%s' % x for x in kv.items()))
        e[7] += float(kv['ret2exit_us']) * n
idle_ms = sum(e[1] for e in idle.values())
print('\nIdle episodes: %.1f%% of wall time, %d episodes, %.0f/s' %
      (100 * idle_ms / tot, sum(e[0] for e in idle.values()), sum(e[0] for e in idle.values()) / (tot / 1000)))
print('  share   n/s     avg_ms  dur <1us/<10us/<100us/<1ms/<4ms/>=4ms   multi-wake  key (pending | rose)')
for key, e in sorted(idle.items(), key=lambda x: -x[1][1]):
    if e[1] < tot * 0.0005: continue
    print('  %5.2f%%  %6.1f  %6.3f  %-34s  %4d  %s' % (100 * e[1] / tot, e[0] / (tot / 1000), e[1] / max(e[0], 1),
          '/'.join(map(str, e[2])), e[4], key))
exn = sum(w['exn'] for w in sel); exus = sum(w['exn'] * w['exus'] for w in sel)
print('  resume -> guest execution: %.1f us average over %d resumes; kicks %.0f/s' %
      (exus / max(exn, 1), exn, sum(w['kicks'] for w in sel) / (tot / 1000)))
site = defaultdict(lambda: [0, 0.0, 0.0, [0] * 6, 0.0, [0] * 6, 0.0, 0])
for w in sel:
    for name, kv in w['site']:
        e = site[name]; e[0] += int(kv['n']); e[1] += float(kv['wait_ms'])
        e[2] = max(e[2], float(kv['max_us']))
        for i, h in enumerate(kv['h'].split('/')): e[3][i] += int(h)
        e[4] += float(kv['hold_pct']) * w['dt'] / 100
        if 'hh' in kv:
            for i, h in enumerate(kv['hh'].split('/')): e[5][i] += int(h)
            e[6] += float(kv['out_ms']); e[7] += int(kv['out_n'])
print('\nBQL waits by waiter site (thread): share of wall time')
print('  wait%    n/s       max_us  wait <1us/<10us/<100us/<1ms/<4ms/>=4ms     hold%   site')
for name, e in sorted(site.items(), key=lambda x: -x[1][1]):
    if e[1] < tot * 0.0005 and e[4] < tot * 0.002: continue
    print('  %5.2f%%  %8.0f  %7.0f  %-38s  %5.2f%%  %s' % (100 * e[1] / tot, e[0] / (tot / 1000), e[2],
          '/'.join(map(str, e[3])), 100 * e[4] / tot, name))
print('\nHolds and time outside the BQL by site (sites holding >= 0.2%):')
print('  hold%   holds/s  hold <1us/<10us/<100us/<1ms/<4ms/>=4ms        out%  out_avg_us  site (out = this thread unlocked before this acquire)')
for name, e in sorted(site.items(), key=lambda x: -x[1][4]):
    if e[4] < tot * 0.002 or not sum(e[5]): continue
    print('  %5.2f%%  %7.0f  %-44s  %5.2f%%  %8.1f  %s' % (100 * e[4] / tot, sum(e[5]) / (tot / 1000),
          '/'.join(map(str, e[5])), 100 * e[6] / tot, 1000 * e[6] / max(e[7], 1), name))
chk = defaultdict(float)
for w in sel:
    for c in w['chk']:
        for k, v in c.items(): chk[k] += v
if chk:
    print('\nAccounting check (waits >= 1 us): total %.1f ms = attributed %.1f + free %.1f (diff %.3f ms); '
          'ring truncations %d, pair drops %d, site drops %d' % (chk['w1_ms'], chk['owned_ms'], chk['free_ms'],
          chk['w1_ms'] - chk['owned_ms'] - chk['free_ms'], chk['ring_trunc'], chk['pair_drop'], chk['site_drop']))
pair = defaultdict(lambda: [0.0, 0])
for w in sel:
    for a, b, ms, n in w['pair']: pair[(a, b)][0] += ms; pair[(a, b)][1] += n
print('\nBQL wait split by owner during the wait (waits >= 1 us; "(free)" = lock free, hand-off):')
for (a, b), (ms, n) in sorted(pair.items(), key=lambda x: -x[1][0])[:20]:
    print('  %5.2f%%  %-40s <- %s  (n=%d)' % (100 * ms / tot, a, b, n))
