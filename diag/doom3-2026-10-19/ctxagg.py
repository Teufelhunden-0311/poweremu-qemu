#!/usr/bin/env python3
"""ctxagg.py VM.OUT START END: sum the idlectx wp-ctx lines of the PPC_WAIT_PROF 10 s windows
wholly inside [START, END] (Unix seconds).  Buckets = (NIP, LR, PR) of the halted CPU at idle
entry; cls = wake split: D-only/no source rose, EXT or a source rose, other."""
import re, sys
from collections import defaultdict
path, start, end = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
wins = []; cur = None
for l in open(path, errors='replace'):
    m = re.search(r'waitprof T=([\d.]+) dt_ms=(\d+)', l)
    if m:
        cur = dict(T=float(m.group(1)), dt=float(m.group(2)), ctx=[], sum=None, idle_ms=0.0); wins.append(cur); continue
    if not cur: continue
    if l.startswith('wp-ctxsum '):
        cur['sum'] = dict(re.findall(r'(\w+)=([\d.]+)', l))
    elif l.startswith('wp-ctx '):
        kv = dict(re.findall(r'(\w+)=([\w./-]+)', l)); cur['ctx'].append(kv)
    elif l.startswith('wp-idle '):
        cur['idle_ms'] += float(re.search(r' ms=([\d.]+)', l).group(1))
sel = [w for w in wins if w['T'] - w['dt'] / 1000 >= start and w['T'] <= end]
tot = sum(w['dt'] for w in sel)
print('windows used: %d of %d, %.1f s' % (len(sel), len(wins), tot / 1000))
if not sel: sys.exit()
agg = defaultdict(lambda: dict(n=0, ms=0.0, h=[0] * 6, cls=[0.0] * 3, r1=[0xffffffff, 0], msr=set()))
for w in sel:
    for c in w['ctx']:
        e = agg[(c['nip'], c['lr'], c['pr'])]
        e['n'] += int(c['n']); e['ms'] += float(c['ms'])
        for i, x in enumerate(c['h'].split('/')): e['h'][i] += int(x)
        for i, x in enumerate(c['cls_ms'].split('/')): e['cls'][i] += float(x)
        a, b = c['r1'].split('-'); e['r1'][0] = min(e['r1'][0], int(a, 16)); e['r1'][1] = max(e['r1'][1], int(b, 16))
        e['msr'].add(c['msr'])
ctx_ms = sum(e['ms'] for e in agg.values()); idle_ms = sum(w['idle_ms'] for w in sel)
drop_n = sum(int(w['sum']['drop_n']) for w in sel if w['sum']); drop_ms = sum(float(w['sum']['drop_ms']) for w in sel if w['sum'])
print('idle (wp-idle) %.2f%% of wall; ctx buckets %.2f%%; dropped %d episodes %.1f ms; %d distinct buckets'
      % (100 * idle_ms / tot, 100 * ctx_ms / tot, drop_n, drop_ms, len(agg)))
print('  wall%   idle%     n/s   avg_ms  dur <1us/<10us/<100us/<1ms/<4ms/>=4ms   wake ms% D-only/EXT-or-rose/other  pr  nip       lr        r1 range           msr')
for (nip, lr, pr), e in sorted(agg.items(), key=lambda x: -x[1]['ms'])[:int(sys.argv[4]) if len(sys.argv) > 4 else 25]:
    print('  %5.2f  %6.2f  %6.1f  %6.3f  %-34s  %5.1f/%5.1f/%5.1f   %s   %s  %s  %08x-%08x  %s' % (
        100 * e['ms'] / tot, 100 * e['ms'] / max(ctx_ms, 1e-9), e['n'] / (tot / 1000), e['ms'] / max(e['n'], 1),
        '/'.join(map(str, e['h'])), *(100 * x / max(e['ms'], 1e-9) for x in e['cls']), pr, nip, lr,
        e['r1'][0], e['r1'][1], ','.join(sorted(e['msr']))))
