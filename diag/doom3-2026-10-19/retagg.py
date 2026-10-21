#!/usr/bin/env python3
"""retagg.py VM.OUT START END [N]: sum the idlectx2 wp-ret lines (first return to user mode after
idle, keyed by NIP + SR0 VSID) of the PPC_WAIT_PROF 10 s windows wholly inside [START, END].
Each return carries the batch of idle episodes since the previous return; duration-weighted."""
import re, sys
from collections import defaultdict
path, start, end = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]); N = int(sys.argv[4]) if len(sys.argv) > 4 else 25
wins = []; cur = None; asn = {}
for l in open(path, errors='replace'):
    m = re.search(r'waitprof T=([\d.]+) dt_ms=(\d+)', l)
    if m:
        cur = dict(T=float(m.group(1)), dt=float(m.group(2)), ret=[], sum=None, idle_ms=0.0); wins.append(cur); continue
    if l.startswith('wp-as '):
        m = re.match(r'wp-as asid=(\w+) text=(\w+) path=(.*)', l.strip()); asn[m.group(1)] = (m.group(2), m.group(3)); continue
    if not cur: continue
    if l.startswith('wp-retsum '): cur['sum'] = {k: float(v) for k, v in re.findall(r'(\w+)=([\d.]+)', l)}
    elif l.startswith('wp-ret '): cur['ret'].append(dict(re.findall(r'(\w+)=([\w./]+)', l)))
    elif l.startswith('wp-idle '): cur['idle_ms'] += float(re.search(r' ms=([\d.]+)', l).group(1))
sel = [w for w in wins if w['T'] - w['dt'] / 1000 >= start and w['T'] <= end]
tot = sum(w['dt'] for w in sel); s = lambda k: sum(w['sum'][k] for w in sel if w['sum'])
print('windows used: %d of %d, %.1f s' % (len(sel), len(wins), tot / 1000))
if not sel: sys.exit()
idle = sum(w['idle_ms'] for w in sel)
print('idle %.2f%% of wall (%.0f ms); charged to user returns %.0f ms (%.1f%% of idle); returns %d carrying %d episodes'
      % (100 * idle / tot, idle, s('ms'), 100 * s('ms') / idle, s('rets'), s('eps')))
print('table drops: %d returns, %d episodes, %.1f ms; pending at window ends (approx): %d episodes, %.1f ms total'
      % (s('drop_rets'), s('drop_eps'), s('drop_ms'), s('pend_eps'), s('pend_ms')))
agg = defaultdict(lambda: dict(rets=0, eps=0, ms=0.0, cls=[0.0] * 3, delay=0.0, dh=[0] * 6, batch=[0] * 4, lr=set(), lro=0))
for w in sel:
    for r in w['ret']:
        e = agg[(r['nip'], r['asid'])]
        e['rets'] += int(r['rets']); e['eps'] += int(r['eps']); e['ms'] += float(r['ms']); e['delay'] += float(r['delay_ms'])
        for i, x in enumerate(r['cls_ms'].split('/')): e['cls'][i] += float(x)
        for i, x in enumerate(r['dh'].split('/')): e['dh'][i] += int(x)
        for i, x in enumerate(r['batch'].split('/')): e['batch'][i] += int(x)
        e['lr'].add(r['lr']); e['lro'] += int(r['lr_other'])
byas = defaultdict(lambda: [0.0, 0, 0])
for (nip, a), e in agg.items(): byas[a][0] += e['ms']; byas[a][1] += e['rets']; byas[a][2] += e['eps']
print('\nBy address space (SR0 VSID; path = first "/..." exec string at the stack top, read at first sight):')
print('  idle%   ms        returns  episodes  asid    text     path')
for a, (ms, r, ep) in sorted(byas.items(), key=lambda x: -x[1][0])[:15]:
    t, p = asn.get(a, ('?', 'unresolved'))
    print('  %5.1f  %8.0f  %7d  %8d  %s  %7s  %s' % (100 * ms / idle, ms, r, ep, a, t, p))
print('\nBy return site (NIP, asid), duration-weighted; delay = last idle exit -> this user return:')
print('  idle%   returns  eps/ret  avg_delay_us  delay <1us/<10us/<100us/<1ms/<4ms/>=4ms  batch 1/2/3-4/5+   wake% D/EXT/other  nip       asid    lr(s) [lr_other]')
for (nip, a), e in sorted(agg.items(), key=lambda x: -x[1]['ms'])[:N]:
    print('  %5.1f  %7d  %6.2f  %10.1f   %-30s  %-16s  %3.0f/%3.0f/%3.0f   %s  %s  %s [%d]' % (
        100 * e['ms'] / idle, e['rets'], e['eps'] / max(e['rets'], 1), 1000 * e['delay'] / max(e['rets'], 1),
        '/'.join(map(str, e['dh'])), '/'.join(map(str, e['batch'])), *(100 * x / max(e['ms'], 1e-9) for x in e['cls']),
        nip, a, ','.join(sorted(e['lr']))[:40], e['lro']))
