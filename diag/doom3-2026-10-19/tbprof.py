#!/usr/bin/env python3
"""tbprof.py RUNDIR [N]: host-sampled guest-code profile of the vCPU thread.

Joins macOS `sample` output (RUNDIR/sample-*.txt) with the QEMU_TBMAP log (RUNDIR/tbmap.txt)
and the guest's kextstat / vmmap (RUNDIR/guest.txt).  Denominator: wall-clock samples of
the vCPU thread (sample records every thread every 1 ms whether or not it is on a CPU).

Each vCPU sample is put in one bucket by its leaf frame:
  wait/*         blocked in the host kernel (BQL, halted, other)
  guest/*        leaf in translated code, mapped to the TB's guest code (PR=1 user, PR=0 kernel)
  helper/*       leaf in a QEMU function called from translated code; the calling TB's guest
                 code is in the label (return address - 4)
  translate      inside tb_gen_code
  qemu-other     QEMU outside translated code (dispatch, TB lookup, interrupts, timers)
  jit-unmapped   leaf in the code buffer but in no logged TB (prologue/epilogue or unlogged)
Attribution is whole-TB for modules; guest_pc per sample uses the TB's per-insn host end
offsets (slow paths appended after the last insn are reported as 'tail').
"""
import re, sys, bisect, os
from collections import Counter, defaultdict

D = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 15
WAIT = re.compile(r'__psynch_cvwait|__psynch_mutexwait|mach_msg|semaphore_wait|__select|__sigwait|'
                  r'__workq_kernreturn|kevent|__ulock_wait|__semwait|poll\b|__wait4|__sleep|nanosleep')

# ---- tbmap: list of epochs; each epoch a sorted list of TBs (latest wins on reuse)
epochs = [[0, {}]]          # [start_us, {rx: tb}]
buf = None
for l in open(os.path.join(D, 'tbmap.txt')):
    f = l.split()
    if not f: continue
    if f[0] == 'B': buf = (int(f[1], 16), int(f[2]))
    elif f[0] == 'T':
        rx = int(f[2], 16)
        ends = [int(x) for x in f[8].split(',')] if len(f) > 8 else []
        epochs[-1][1][rx] = dict(t=int(f[1]), rx=rx, size=int(f[3]), pc=int(f[4], 16),
                                 flags=int(f[5], 16), icount=int(f[6]), phys=int(f[7], 16), ends=ends)
    elif f[0] == 'X':
        epochs[-1][1].pop(int(f[2], 16), None)
    elif f[0] == 'F':
        epochs.append([int(f[1]), {}])
print('tbmap: %d epochs (%d flushes), TBs per epoch: %s' %
      (len(epochs), len(epochs) - 1, [len(e[1]) for e in epochs]))

# ---- sample windows -> epoch
win = {}
for l in open(os.path.join(D, 'times.txt')):
    m = re.match(r'sample (\d) (start|end) ([\d.]+)', l)
    if m: win.setdefault(int(m.group(1)), {})[m.group(2)] = float(m.group(3)) * 1e6
def epoch_for(k):
    s, e = win[k]['start'], win[k]['end']
    inside = [ep for ep in epochs[1:] if s <= ep[0] <= e]
    cur = [ep for ep in epochs if ep[0] <= s][-1]
    return cur, inside

# ---- guest modules
user_map, kexts = [], []
g = os.path.join(D, 'guest.txt')
if os.path.exists(g):
    sec = None
    for l in open(g, errors='replace'):
        if l.startswith('==KX'): sec = 'kx'; continue
        if l.startswith('==VM'): sec = 'vm'; continue
        if sec == 'kx':
            m = re.match(r'\s*\d+\s+\d+\s+0x([0-9a-f]+)\s+0x([0-9a-f]+)\s+0x[0-9a-f]+\s+(\S+)', l)
            if m: kexts.append((int(m.group(1), 16), int(m.group(1), 16) + int(m.group(2), 16), m.group(3)))
        elif sec == 'vm':
            m = re.match(r'^(\S.*?)\s+([0-9a-f]{8})-([0-9a-f]{8})\s+\[.*?\]\s+\S+\s+SM=\S+\s*(.*)$', l)
            if m:
                path = m.group(4).strip()
                name = os.path.basename(path) if path else m.group(1).strip()
                user_map.append((int(m.group(2), 16), int(m.group(3), 16), name, m.group(1).strip()))
user_map.sort()
print('guest maps: %d vmmap regions, %d kexts' % (len(user_map), len(kexts)))

def module(pc, pr):
    if pr:
        for lo, hi, name, kind in user_map:
            if lo <= pc < hi: return 'user ' + name + ('' if kind.startswith('__TEXT') else ' (' + kind + ')')
        return 'user ? (no vmmap region)'
    for lo, hi, name in kexts:
        if lo <= pc < hi: return 'kext ' + name
    return 'kernel low vectors' if pc < 0x10000 else 'kernel (mach_kernel or unlisted)'

class Map:
    def __init__(self, tbs):
        self.tbs = sorted(tbs.values(), key=lambda t: t['rx']); self.keys = [t['rx'] for t in self.tbs]
    def find(self, a):
        i = bisect.bisect_right(self.keys, a) - 1
        if i >= 0:
            t = self.tbs[i]
            if a < t['rx'] + t['size']: return t
        return None

def locate(m, a):
    """host addr -> (tb, guest pc or None for tail)"""
    t = m.find(a)
    if not t: return None, None
    off = a - t['rx']
    j = bisect.bisect_right(t['ends'], off)
    return t, (t['pc'] + 4 * j if j < t['icount'] else None)

# ---- sample parsing: per leaf sample, its stack
node = re.compile(r'^[\s+!:|]*(\d+) (.*)$')
def stacks(path):
    lines = open(path, errors='replace').read().split('\n')
    i = next(i for i, l in enumerate(lines) if l.startswith('Call graph:'))
    out = []; th = None; st = []
    def close(depth):
        while st and st[-1][0] >= depth:
            d, c, name, ch, frames = st.pop()
            if c - ch > 0: out.append((th, frames, c - ch))
            if st: st[-1][3] += c
    for l in lines[i + 1:]:
        if l.startswith('Total number') or l.startswith('Sort by'): break
        m = re.match(r'^    (\d+) (Thread_\d+.*)$', l)
        if m: close(0); th = m.group(2); st = []; continue
        m = node.match(l)
        if not m or th is None: continue
        depth = len(l) - len(l.lstrip(' +!:|'))
        close(depth)
        name = m.group(2)
        am = re.search(r'\[(0x[0-9a-f]+)', name)
        fr = (name.split('  (in')[0].split(' (in')[0].strip(), '<unknown binary>' in name,
              int(am.group(1), 16) if am else None)
        frames = (st[-1][4] if st else ()) + (fr,)
        st.append([depth, int(m.group(1)), name, 0, frames])
    close(0)
    return out

tot = Counter(); bucket = Counter(); guestpc = Counter(); tbhits = Counter(); helpers = Counter()
modinc = Counter(); vphys = Counter(); vtotal = 0; per_sample = []
for k in sorted(win):
    p = os.path.join(D, 'sample-%d.txt' % k)
    cur, inside = epoch_for(k)
    m = Map(cur[1])
    nflush = len(inside)
    st = stacks(p)
    vth = [s for s in st if any(f[0] in ('rr_cpu_thread_fn', 'cpu_exec_loop') for f in s[1])]
    names = Counter(s[0] for s in vth)
    vname = names.most_common(1)[0][0]
    vs = [s for s in st if s[0] == vname]
    n = sum(c for _, _, c in vs); vtotal += n
    res = Counter()
    for _, frames, c in vs:
        leaf = frames[-1]
        if WAIT.search(leaf[0]):
            nm = [f[0] for f in frames]
            b = ('wait/BQL' if any('bql_lock' in x for x in nm) else
                 'wait/halted' if any('wait_io_event' in x or 'qemu_cond_wait' in x for x in nm) else 'wait/other')
        elif leaf[1]:
            t, gpc = locate(m, leaf[2])
            if t is None:
                b = 'jit-unmapped' if buf and buf[0] <= leaf[2] < buf[0] + buf[1] else 'unknown-binary'
            else:
                pr = (t['flags'] >> 14) & 1
                mod = module(t['pc'], pr)
                b = 'guest/' + mod
                modinc[mod] += c
                tbhits[(mod, t['pc'])] += c
                guestpc[(mod, gpc)] += c
                vphys[(mod, t['pc'] >> 12, t['phys'] >> 12)] += c
        elif any(f[0] == 'tb_gen_code' for f in frames):
            b = 'translate'
        else:
            jit = [f for f in frames if f[1]]
            if jit:
                t, gpc = locate(m, jit[-1][2] - 4)
                mod = module(t['pc'], (t['flags'] >> 14) & 1) if t else 'unmapped'
                b = 'helper/' + mod
                modinc[mod] += c
                # first QEMU frame below the JIT frame = the helper
                idx = max(i for i, f in enumerate(frames) if f[1])
                h = frames[idx + 1][0] if idx + 1 < len(frames) else leaf[0]
                below = [f[0] for f in frames[idx + 1:]]
                if any(re.search(r'io_readx|io_writex|do_ld_mmio|do_st_mmio|memory_region_dispatch|int_ld_mmio|int_st_mmio', x) for x in below):
                    h += ' [MMIO]'
                elif any(re.search(r'tlb_fill|ppc_cpu_tlb_fill|ppc_xlate|hash32', x) for x in below):
                    h += ' [TLB fill]'
                helpers[(h, mod)] += c
                if t: tbhits[(mod, t['pc'])] += c
            else:
                b = 'qemu-other'
                helpers[(leaf[0], '(outside guest code)')] += c
        bucket[b] += c; res[b] += c
    per_sample.append((k, n, nflush, len(m.tbs), res))

def pct(x, n=None): return '%5.1f%%' % (100.0 * x / (n or vtotal))
print('\nvCPU thread: %d wall-clock samples over %d windows' % (vtotal, len(per_sample)))
for k, n, nf, ntb, res in per_sample:
    print('  window %d: %d samples, %d TBs mapped, %d flushes inside window, unmapped JIT %s' %
          (k, n, ntb, nf, pct(res['jit-unmapped'] + res['unknown-binary'], n)))
print('\nBuckets (share of vCPU wall-clock samples):')
for b, c in sorted(bucket.items(), key=lambda x: -x[1]): print('  %s  %s' % (pct(c), b))
print('\nGuest module, inclusive of helpers it called (waits excluded):')
for mname, c in modinc.most_common(N): print('  %s  %s' % (pct(c), mname))
print('\nTop TBs (guest code + helpers called from it):')
for (mname, pc), c in tbhits.most_common(N): print('  %s  %08x  %s' % (pct(c), pc, mname))
print('\nTop guest instructions (JIT self only; None = TB tail/slow path):')
for (mname, pc), c in guestpc.most_common(N):
    print('  %s  %s  %s' % (pct(c), '%08x' % pc if pc is not None else 'tail    ', mname))
print('\nTop helpers / QEMU functions (non-wait):')
for (h, mname), c in helpers.most_common(N): print('  %s  %s  <- %s' % (pct(c), h, mname))

print('\nHelper classes (all callers):')
cls = Counter()
for (h, mname), c in helpers.items():
    k = ('TLB fill' if '[TLB fill]' in h else 'MMIO' if '[MMIO]' in h else
         'lookup_tb_ptr' if 'lookup_tb_ptr' in h else 'other helpers' if mname != '(outside guest code)' else 'QEMU outside guest code')
    cls[k] += c
for k, c in cls.most_common(): print('  %s  %s' % (pct(c), k))
want = os.environ.get('MODS', '')
for mod in [m for m in want.split(',') if m]:
    print('\nTop TBs in %s:' % mod)
    for (mname, pc), c in [x for x in tbhits.most_common() if mod in x[0][0]][:N]:
        print('  %s  %08x' % (pct(c), pc))

print('\nProcess check (user code: one physical page per virtual page = one process):')
for mod in ('user Doom 3', 'user ATIRadeon9700GLDriver', 'user GLEngine'):
    pages = defaultdict(Counter)
    for (m_, vp, ph), c in vphys.items():
        if m_ == mod: pages[vp][ph] += c
    tot_ = sum(sum(v.values()) for v in pages.values())
    dom = sum(max(v.values()) for v in pages.values())
    print('  %s: %d JIT samples, %.1f%% on the dominant physical page of their virtual page' %
          (mod, tot_, 100.0 * dom / max(tot_, 1)))
