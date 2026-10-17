#!/usr/bin/env python3
"""Build vmx_test.c as a raw mac99 firmware image, run it, report results.

usage: run.py [--fpdiff | --fpdiff-strict | --vfpdiff | --spdiff | --xtdiff |
               --restore | --dsectl] [path/to/qemu-system-ppc]

--fpdiff runs the scalar FP cases with the inline FP ops off and on and
compares results and FPSCR; --fpdiff-strict compares the inline results
with softfloat's (PPC_STRICT_FP=1), whose FPSCR also has the inexact bits.
--vfpdiff does the same for vmaddfp and vnmsubfp with the inline ops off
and on (PPC_NATIVE_VMX; PPC_NATIVE_VNMSUB=0 leaves vnmsubfp on the helper),
comparing results and VSCR bit for bit.
--spdiff runs single-precision chains within one TB on the helper and with
PPC_SP_PROV=0, 1 and 2 (single-precision provenance; 1 also without
forwarding, PPC_SP_FWD=0; 2 also with lfs on the helper, PPC_NATIVE_LFS=0),
comparing results and FPSCR bit for bit.
--restore runs the Dot Product loops (single shadows) while saving the VM,
letting it run on and loading the saved state again; every result must
match the first, which must match the helper's (PPC_NATIVE_FP=0).
--dsectl checks that PPC_SP_SHADOW_DSE=0 keeps the shadow stores that a
TB looping to itself would drop: the read after such a loop then hits
(PPC_SP_SHADOW_STATS), and misses once per call with the removal on; the
results must match the helper's either way.
--xtdiff runs single-precision chains across TBs on the helper and with the
single shadows on (PPC_SP_SHADOW=2 and 1) and off (PPC_SP_BRANCH=0, PPC_SP_INPUTS=0,
PPC_SP_SHADOW_DSE=0 too), comparing results and FPSCR bit for bit.

Needs Homebrew llvm (clang with the PowerPC target) and lld.
"""
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]
QEMU = (ARGS[0] if ARGS else
        os.path.join(HERE, '..', '..', 'build', 'qemu-system-ppc'))
RESULTS = 0x10000
OPS = """vperm vperm(d=a) vperm(d=c) vperm(a=b) vsldoi vsldoi(d=b)
vmrghb vmrghh vmrghw vmrglb vmrglh vmrglw
vpkuhum vpkuwum vpkuhus vpkuwus vpkshus vpkswus vpkshss vpkswss
vupkhsb vupkhsh vupklsb vupklsh
vmuleub vmuloub vmulesb vmulosb vmuleuh vmulouh vmulesh vmulosh
vmsumubm vmsummbm vmsumuhm vmsumshm
vaddfp vsubfp vmaxfp vminfp vmaddfp vnmsubfp vrefp vcfsx vcfux vctsxs vctuxs
""".split()
BENCH = ['vperm x4', 'vmrghb/vmrglh/vsldoi/vpkuhum',
         'vmuleub/vmsumubm/vpkshus/vupkhsb', 'vaddubm x4 (always inline)',
         'vmaddfp x4 (float)', 'lfs/fmadds/stfs x4 (scalar)',
         'fmadd x4 dependent', 'fmadd x4 independent', 'fadd x4 dependent',
         'add x4 dependent (integer)']


def tool(name):
    for d in ('/opt/homebrew/opt/llvm/bin', '/usr/local/opt/llvm/bin'):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    p = shutil.which(name)
    if not p:
        sys.exit(f'{name} not found (brew install llvm lld)')
    return p


def build(out, defs=()):
    cc = [tool('clang'), '--target=powerpc-unknown-none-elf', '-mcpu=7450',
          '-maltivec', '-O2', '-ffreestanding', '-fno-builtin', '-nostdlib',
          '-fno-pic', '-c'] + list(defs)
    objs = []
    for src in ('start.S', 'vmx_test.c'):
        o = os.path.join(out, src + '.o')
        subprocess.check_call(cc + [os.path.join(HERE, src), '-o', o])
        objs.append(o)
    elf = os.path.join(out, 'vmx_test.elf')
    subprocess.check_call([tool('ld.lld'), '-T', os.path.join(HERE, 'link.ld'),
                           '-o', elf] + objs)
    img = os.path.join(out, 'vmx_test.bin')
    subprocess.check_call([tool('llvm-objcopy'), '-O', 'binary', elf, img])
    return img


class Monitor:
    def __init__(self, path):
        self.s = socket.socket(socket.AF_UNIX)
        for _ in range(100):
            try:
                self.s.connect(path)
                break
            except OSError:
                time.sleep(0.1)
        self.read_prompt()

    def read_prompt(self):
        buf = b''
        while not buf.endswith(b'(qemu) '):
            chunk = self.s.recv(65536)
            if not chunk:
                break
            buf += chunk
        return buf.decode(errors='replace')

    def cmd(self, c):
        self.s.sendall(c.encode() + b'\n')
        return self.read_prompt()

    def words(self, addr, n):
        out = self.cmd(f'xp /{n}wx {addr:#x}')
        return [int(x, 16) for x in re.findall(r'0x([0-9a-f]{8})(?!:)', out)][:n]


FPD_BASE = 0x01000000
FPD_REC = 6 * 8
FPD_OPS = ['fadd', 'fadds', 'fsub', 'fsubs', 'fmul', 'fmuls', 'fdiv', 'fdivs',
           'fmadd', 'fmadds', 'fmsub', 'fmsubs', 'fnmadd', 'fnmadds',
           'fnmsub', 'fnmsubs', 'frsp']


def fpdiff(img, out, strict=False):
    """Run the scalar FP cases as the reference and inline; compare them."""
    import struct
    ref = 'softfloat' if strict else 'helper'
    dumps = []
    for native in ('0', '1'):
        sock = os.path.join(out, f'mon{native}.sock')
        env = dict(os.environ, PPC_NATIVE_FP=native)
        if strict and native == '0':
            env['PPC_STRICT_FP'] = '1'
        q = subprocess.Popen([QEMU, '-M', 'mac99', '-cpu', '7450', '-m', '64',
                              '-bios', img, '-display', 'none',
                              '-serial', 'none', '-monitor',
                              f'unix:{sock},server=on,wait=off'],
                             stdout=subprocess.DEVNULL, env=env)
        try:
            mon = Monitor(sock)
            for _ in range(1200):
                if mon.words(RESULTS, 1) == [0x564d5854]:
                    break
                time.sleep(0.25)
            else:
                sys.exit('test did not finish')
            n = mon.words(RESULTS + 0x283 * 4, 1)[0]
            path = os.path.join(out, f'fpd{native}.bin')
            mon.cmd(f'pmemsave {FPD_BASE:#x} {n * FPD_REC} "{path}"')
            dumps.append(open(path, 'rb').read())
        finally:
            q.kill()
    n = len(dumps[0]) // FPD_REC
    bad = 0
    for i in range(n):
        r0 = struct.unpack_from('>6Q', dumps[0], i * FPD_REC)
        r1 = struct.unpack_from('>6Q', dumps[1], i * FPD_REC)
        if r0[:4] != r1[:4] or (not strict and r0 != r1):
            bad += 1
            if bad <= 10:
                st, op, form = (r0[5] >> 16) & 15, (r0[5] >> 8) & 0xff, r0[5] & 0xff
                follow = ('mffs', 'fcmpu', 'mtfsb1', 'mcrfs')[(r0[5] >> 20) & 15]
                print(f'MISMATCH {FPD_OPS[op]} form {form} fpscr_in {st} then {follow} '
                      f'(cr {r0[5] >> 32:08x}/{r1[5] >> 32:08x}): '
                      f'a={r0[0]:016x} c={r0[1]:016x} b={r0[2]:016x}')
                print(f'   {ref}: {r0[3]:016x} fpscr {r0[4] & 0xffffffff:08x}'
                      f'   native: {r1[3]:016x} fpscr {r1[4] & 0xffffffff:08x}')
    print(f'{n} FP cases compared (native vs {ref}), {bad} mismatches')
    return bad


VFD_BASE = 0x02000000
VFD_REC = 6 * 16
VFD_CASES = 2 * 4000 * 2 * 5    # NJ modes x iterations x ops x forms (vfp_diff())


def run_dump(img, out, tag, env, base, rec, count_word):
    """Run the image with env; return count_word's count of rec-byte records."""
    sock = os.path.join(out, f'mon{tag}.sock')
    q = subprocess.Popen([QEMU, '-M', 'mac99', '-cpu', '7450', '-m', '64',
                          '-bios', img, '-display', 'none',
                          '-serial', 'none', '-monitor',
                          f'unix:{sock},server=on,wait=off'],
                         stdout=subprocess.DEVNULL, env=env)
    try:
        mon = Monitor(sock)
        for _ in range(1200):
            if mon.words(RESULTS, 1) == [0x564d5854]:
                break
            time.sleep(0.25)
        else:
            sys.exit('test did not finish')
        n = mon.words(RESULTS + count_word * 4, 1)[0]
        path = os.path.join(out, f'dump{tag}.bin')
        mon.cmd(f'pmemsave {base:#x} {n * rec} "{path}"')
        return open(path, 'rb').read()
    finally:
        q.kill()


def vfpdiff(img, out):
    """Run the vmaddfp/vnmsubfp cases on the helper and inline; compare them."""
    import struct
    # every switch set explicitly, so the caller's environment cannot
    # turn off the path a run is meant to take
    base = {'PPC_NATIVE_VMX': '1', 'PPC_NATIVE_VNMSUB': '1'}
    runs = [('helper', {'PPC_NATIVE_VMX': '0'}), ('native', {}),
            ('native, vnmsubfp on the helper', {'PPC_NATIVE_VNMSUB': '0'})]
    dumps = [run_dump(img, out, f'v{k}', {**os.environ, **base, **env},
                      VFD_BASE, VFD_REC, 0x284) for k, (_, env) in enumerate(runs)]
    n = VFD_CASES
    for (tag, _), d in zip(runs, dumps):
        if len(d) != n * VFD_REC:
            print(f'{tag}: {len(d) // VFD_REC} cases dumped, expected {n}')
            return 1
    total = 0
    forms = ('d distinct', 'd=a', 'd=b', 'd=c', 'all one register')
    ops = ('vmaddfp', 'vnmsubfp')
    for (tag, _), d in zip(runs[1:], dumps[1:]):
        bad = [0, 0]
        for i in range(n):
            r0 = struct.unpack_from('>24I', dumps[0], i * VFD_REC)
            r1 = struct.unpack_from('>24I', d, i * VFD_REC)
            if r0 != r1:
                op = r0[20] >> 16
                bad[op] += 1
                if sum(bad) <= 10:
                    vec = lambda r, k: ' '.join(f'{w:08x}' for w in r[4 * k:4 * k + 4])
                    print(f'MISMATCH ({tag}) {ops[op]} {forms[r0[20] & 0xff]} '
                          f'NJ={(r0[20] >> 8) & 0xff}: '
                          f'a={vec(r0, 0)} b={vec(r0, 1)} c={vec(r0, 2)}')
                    print(f'   helper: {vec(r0, 3)} vscr {r0[19]:08x}'
                          f'   {tag}: {vec(r1, 3)} vscr {r1[19]:08x}')
        print(f'{n} vmaddfp/vnmsubfp cases compared ({tag} vs helper), '
              f'{sum(bad)} mismatches (vmaddfp {bad[0]}, vnmsubfp {bad[1]})')
        total += sum(bad)
    return total


SPD_BASE = 0x03000000
SPD_REC = 10 * 8
SPD_CASES = 250 * 3 * 14 * 6    # iterations x op1 x mid x op3 (sp_diff())


def spdiff(img, out):
    """Run the provenance chains on the helper and with PPC_SP_PROV 0-2."""
    import struct
    # every switch set explicitly, so the caller's environment cannot
    # turn off the path a run is meant to take
    base = {'PPC_NATIVE_FP': '1', 'PPC_STRICT_FP': '0', 'PPC_NATIVE_FRSP': '1',
            'PPC_SP_PROV': '1', 'PPC_SP_FWD': '1', 'PPC_NATIVE_LFS': '1'}
    runs = [('helper', {'PPC_NATIVE_FP': '0', 'PPC_SP_PROV': '0'}),
            ('prov0', {'PPC_SP_PROV': '0'}),
            ('prov1', {}), ('prov2', {'PPC_SP_PROV': '2'}),
            ('prov1-nofwd', {'PPC_SP_FWD': '0'}),
            ('prov2-lfshelper', {'PPC_SP_PROV': '2', 'PPC_NATIVE_LFS': '0'})]
    dumps = {}
    for tag, extra in runs:
        d = run_dump(img, out, tag, {**os.environ, **base, **extra},
                     SPD_BASE, SPD_REC, 0x285)
        if len(d) != SPD_CASES * SPD_REC:
            print(f'{tag}: {len(d) // SPD_REC} cases dumped, expected {SPD_CASES}')
            return 1
        dumps[tag] = d
    o1s = ('fmadds', 'fadds', 'frsp')
    mids = ('-', 'fadd', 'lfd', 'fmr', 'lfs', 'stfd/ori/addi', 'frsp', 'fneg',
            'fadds', 'lfdx', 'lfsx', 'fmul', 'lwarx/stwcx.', 'stwcx.')
    op3s = ('fmadds A', 'fmadds C', 'fmadds B', 'fmadds in place', 'fadds 1,1',
            'fmuls 5,1')
    bad = 0
    for tag, _ in runs[1:]:
        nb = 0
        by_mid = {}
        for i in range(SPD_CASES):
            r0 = struct.unpack_from('>10Q', dumps['helper'], i * SPD_REC)
            r1 = struct.unpack_from('>10Q', dumps[tag], i * SPD_REC)
            if r0 != r1:
                nb += 1
                by_mid[mids[(r0[9] >> 8) & 0xff]] = by_mid.get(mids[(r0[9] >> 8) & 0xff], 0) + 1
                if nb <= 5:
                    k = r0[9]
                    print(f'MISMATCH {tag}: {o1s[k >> 16]} / {mids[(k >> 8) & 0xff]} / '
                          f'{op3s[k & 0xff]}: ' +
                          ' '.join(f'{x:016x}' for x in r0[:6]))
                    print(f'   helper: f1 {r0[6]:016x} f7 {r0[7]:016x} fpscr {r0[8] & 0xffffffff:08x}'
                          f'   {tag}: f1 {r1[6]:016x} f7 {r1[7]:016x} fpscr {r1[8] & 0xffffffff:08x}')
        print(f'{SPD_CASES} chains compared ({tag} vs helper), {nb} mismatches'
              + (f' (by middle instruction: {by_mid})' if nb else ''))
        bad += nb
    return bad


XTD_BASE = 0x03600000
XTD_REC = 12 * 8
XTD_CASES = 120 * 6 * 17 * 7    # iterations x op1 x mid x op3 (xt_diff())


def xtdiff(img, out):
    """Run the cross-TB chains on the helper and with the shadows on and off."""
    import struct
    # every switch set explicitly, so the caller's environment cannot
    # turn off the path a run is meant to take
    base = {'PPC_NATIVE_FP': '1', 'PPC_STRICT_FP': '0', 'PPC_NATIVE_FRSP': '1',
            'PPC_SP_PROV': '2', 'PPC_SP_FWD': '1', 'PPC_NATIVE_LFS': '1',
            'PPC_SP_BRANCH': '1', 'PPC_SP_SHADOW': '2', 'PPC_SP_INPUTS': '1',
            'PPC_SP_SHADOW_DSE': '1', 'PPC_SP_SHADOW_STATS': '0'}
    runs = [('helper', {'PPC_NATIVE_FP': '0', 'PPC_SP_PROV': '0'}),
            ('shadow', {}),
            ('shadow-all', {'PPC_SP_SHADOW': '1'}),
            ('shadow-all-nodse', {'PPC_SP_SHADOW': '1', 'PPC_SP_SHADOW_DSE': '0'}),
            ('shadow-nodse', {'PPC_SP_SHADOW_DSE': '0'}),
            ('shadow-noinputs', {'PPC_SP_INPUTS': '0'}),
            ('shadow-prov1', {'PPC_SP_PROV': '1'}),
            ('shadow-lfshelper', {'PPC_NATIVE_LFS': '0'}),
            ('noshadow', {'PPC_SP_SHADOW': '0'}),
            ('orn', {'PPC_SP_BRANCH': '0', 'PPC_SP_INPUTS': '0'})]
    dumps = {}
    for tag, extra in runs:
        d = run_dump(img, out, 'x' + tag, {**os.environ, **base, **extra},
                     XTD_BASE, XTD_REC, 0x286)
        if len(d) != XTD_CASES * XTD_REC:
            print(f'{tag}: {len(d) // XTD_REC} cases dumped, expected {XTD_CASES}')
            return 1
        dumps[tag] = d
    o1s = ('fmadds', 'fadds', 'frsp', 'fsubs +0', 'fsubs/fneg/fadds -0', 'lfd')
    mids = ('-', 'fadd', 'lfd', 'fmr', 'lfs', 'frsp', 'fneg', 'fadds', 'fmul',
            'same value', 'next double', 'fres', 'fctiwz', 'fsel', 'mtfsfi RZ',
            'fabs', 'fnabs')
    op3s = ('fmadds A', 'fmadds C', 'fmadds B', 'fmadds in place', 'fadds 1,1',
            'fmuls 5,5,1', 'fmadds 1,1,1,1')
    bad = 0
    for tag, _ in runs[1:]:
        nb = 0
        by_mid = {}
        for i in range(XTD_CASES):
            r0 = struct.unpack_from('>12Q', dumps['helper'], i * XTD_REC)
            r1 = struct.unpack_from('>12Q', dumps[tag], i * XTD_REC)
            if r0 != r1:
                nb += 1
                k = r0[11]
                mid = mids[(k >> 8) & 0xff]
                by_mid[mid] = by_mid.get(mid, 0) + 1
                if nb <= 5:
                    print(f'MISMATCH {tag}: {o1s[k >> 16]} / {mid} / '
                          f'{op3s[k & 0xff]}: ' +
                          ' '.join(f'{x:016x}' for x in r0[:6]))
                    print(f'   helper: f1 {r0[6]:016x} f7 {r0[7]:016x} f6 {r0[9]:016x} '
                          f'f5 {r0[10]:016x} fpscr {r0[8] & 0xffffffff:08x}')
                    print(f'   {tag}: f1 {r1[6]:016x} f7 {r1[7]:016x} f6 {r1[9]:016x} '
                          f'f5 {r1[10]:016x} fpscr {r1[8] & 0xffffffff:08x}')
        print(f'{XTD_CASES} cross-TB chains compared ({tag} vs helper), {nb} mismatches'
              + (f' (by middle instruction: {by_mid})' if nb else ''))
        bad += nb
    return bad


def restore(out):
    """Save, run on, load: results across TBs must not change (--restore)."""
    img = build(out, ['-DRESTORE_TEST'])
    disk = os.path.join(out, 'snap.qcow2')
    qimg = os.path.join(os.path.dirname(QEMU), 'qemu-img')
    subprocess.check_call([qimg if os.path.exists(qimg) else 'qemu-img',
                           'create', '-q', '-f', 'qcow2', disk, '16M'])

    def start(tag, env):
        sock = os.path.join(out, f'mon{tag}.sock')
        q = subprocess.Popen([QEMU, '-M', 'mac99', '-cpu', '7450', '-m', '64',
                              '-bios', img, '-display', 'none',
                              '-serial', 'none', '-drive',
                              f'if=none,id=snap,format=qcow2,file={disk}',
                              '-monitor', f'unix:{sock},server=on,wait=off'],
                             stdout=subprocess.DEVNULL, env=env)
        return q, Monitor(sock)

    def wait(mon, cond, what):
        for _ in range(2400):
            w = mon.words(RESULTS + 0x287 * 4, 1)[0]
            if cond(w):
                return w
            time.sleep(0.05)
        sys.exit(f'restore: timed out waiting for {what}')

    # every switch set explicitly, so the caller's environment cannot
    # turn off the path a run is meant to take
    base = {'PPC_NATIVE_FP': '1', 'PPC_STRICT_FP': '0', 'PPC_NATIVE_FRSP': '1',
            'PPC_SP_PROV': '2', 'PPC_SP_FWD': '1', 'PPC_NATIVE_LFS': '1',
            'PPC_SP_BRANCH': '1', 'PPC_SP_INPUTS': '1',
            'PPC_SP_SHADOW_DSE': '1', 'PPC_SP_SHADOW_STATS': '0'}
    q, mon = start('h', {**os.environ, **base, 'PPC_NATIVE_FP': '0',
                         'PPC_SP_PROV': '0', 'PPC_SP_SHADOW': '0'})
    try:
        wait(mon, lambda w: w > 0, 'the helper run')
        href = mon.words(RESULTS + 0x289 * 4, 2)
    finally:
        q.kill()
    failed = 0
    for mode in ('2', '1'):
        q, mon = start('n' + mode, {**os.environ, **base, 'PPC_SP_SHADOW': mode})
        try:
            wait(mon, lambda w: w > 20000, 'the first iterations')
            mon.cmd('stop')
            said = mon.cmd('savevm s1')
            saved = mon.words(RESULTS + 0x287 * 4, 1)[0]
            mon.cmd('cont')
            loads = []
            for k in range(4):
                far = wait(mon, lambda w: w > saved + 30000 * (k + 1), 'running on')
                mon.cmd('stop')
                mon.cmd('loadvm s1')
                loads.append((far, mon.words(RESULTS + 0x287 * 4, 1)[0]))
                mon.cmd('cont')
            done = wait(mon, lambda w: w >= 1500000, 'the end')
            bad = mon.words(RESULTS + 0x288 * 4, 1)[0]
            nref = mon.words(RESULTS + 0x289 * 4, 2)
        finally:
            q.kill()
        print(f'PPC_SP_SHADOW={mode}: saved at iteration {saved}; loaded 4 times, '
              'at iterations ' +
              ', '.join(f'{a} (back to {b})' for a, b in loads) +
              (f'; savevm said {said.strip()!r}' if 'rror' in said else ''))
        print(f'  first results {nref[0]:08x} {nref[1]:08x}, helper {href[0]:08x} '
              f'{href[1]:08x}; {done} iterations, {bad} results differ')
        ok = (bad == 0 and nref == href and all(b == saved for _, b in loads)
              and saved > 0)
        print(f'  restore test (PPC_SP_SHADOW={mode})', 'passed' if ok else 'FAILED')
        failed += not ok
    return failed


def dsectl(img, out):
    """Self-loop shadow-store removal: on, off, and the helper (--dsectl)."""
    nm = subprocess.check_output([tool('llvm-nm'),
                                  os.path.join(out, 'vmx_test.elf')], text=True)
    sym = {l.split()[2]: int(l.split()[0], 16) for l in nm.splitlines()
           if len(l.split()) == 3}
    rng = f"{sym['sp_selfloop']:#x}-{sym['sp_selfloop_end'] - 1:#x}"
    base = {'PPC_NATIVE_FP': '1', 'PPC_STRICT_FP': '0', 'PPC_NATIVE_FRSP': '1',
            'PPC_SP_PROV': '2', 'PPC_SP_FWD': '1', 'PPC_NATIVE_LFS': '1',
            'PPC_SP_BRANCH': '1', 'PPC_SP_INPUTS': '1', 'PPC_SP_SHADOW': '2',
            'PPC_SP_SHADOW_STATS': rng}
    runs = [('helper', {'PPC_NATIVE_FP': '0', 'PPC_SP_PROV': '0',
                        'PPC_SP_SHADOW': '0', 'PPC_SP_SHADOW_STATS': '0'}),
            ('removal on', {'PPC_SP_SHADOW_DSE': '1'}),
            ('PPC_SP_SHADOW_DSE=0', {'PPC_SP_SHADOW_DSE': '0'})]
    got = {}
    for k, (tag, extra) in enumerate(runs):
        sock = os.path.join(out, f'mond{k}.sock')
        errf = os.path.join(out, f'dsectl{k}.err')
        with open(errf, 'w') as err:
            q = subprocess.Popen([QEMU, '-M', 'mac99', '-cpu', '7450', '-m', '64',
                                  '-bios', img, '-display', 'none',
                                  '-serial', 'none', '-monitor',
                                  f'unix:{sock},server=on,wait=off'],
                                 stdout=subprocess.DEVNULL, stderr=err,
                                 env={**os.environ, **base, **extra})
            try:
                mon = Monitor(sock)
                for _ in range(2400):
                    if mon.words(RESULTS, 1) == [0x564d5854]:
                        break
                    time.sleep(0.25)
                else:
                    sys.exit('test did not finish')
                ref, bad = mon.words(RESULTS + 0x28b * 4, 2)
            finally:
                q.kill()
        stat = [l for l in open(errf) if l.startswith('sp-shadow:') and 'kept' in l]
        m = re.search(r'(\d+) kept, (\d+) slow: (\d+) stale', stat[-1]) if stat else None
        got[tag] = (ref, bad, tuple(int(x) for x in m.groups()) if m else None)
        print(f'{tag}: first result {ref:08x}, {bad} results differ'
              + (f'; shadow reads {m.group(1)} kept, {m.group(3)} stale' if m else ''))
    h, on, off = (got[t] for t, _ in runs)
    ok = (on[:2] == h[:2] and off[:2] == h[:2] and h[1] == 0
          and on[2] == (0, 65536, 65536) and off[2] == (65536, 0, 0))
    print('dsectl test', 'passed' if ok else 'FAILED')
    return not ok


def main():
    out = tempfile.mkdtemp(prefix='ppc-vmx-')
    if '--restore' in sys.argv:
        sys.exit(1 if restore(out) else 0)
    img = build(out)
    if '--dsectl' in sys.argv:
        sys.exit(1 if dsectl(img, out) else 0)
    if '--xtdiff' in sys.argv:
        sys.exit(1 if xtdiff(img, out) else 0)
    if '--spdiff' in sys.argv:
        sys.exit(1 if spdiff(img, out) else 0)
    if '--vfpdiff' in sys.argv:
        sys.exit(1 if vfpdiff(img, out) else 0)
    if '--fpdiff' in sys.argv or '--fpdiff-strict' in sys.argv:
        sys.exit(1 if fpdiff(img, out, '--fpdiff-strict' in sys.argv) else 0)
    sock = os.path.join(out, 'mon.sock')
    q = subprocess.Popen([QEMU, '-M', 'mac99', '-cpu', '7450', '-m', '64',
                          '-bios', img, '-display', 'none', '-serial', 'none',
                          '-monitor', f'unix:{sock},server=on,wait=off'],
                         stdout=subprocess.DEVNULL)
    try:
        mon = Monitor(sock)
        for _ in range(600):
            if mon.words(RESULTS, 1) == [0x564d5854]:
                break
            time.sleep(0.25)
        else:
            sys.exit('test did not finish')
        head = mon.words(RESULTS, 3 + len(OPS))
        checks, fails = head[1], head[2]
        for name, n in zip(OPS, head[3:]):
            if n:
                print(f'FAIL {name}: {n} mismatches')
        for k in range(min(fails, 8)):
            p = mon.words(RESULTS + 0x100 + k * 96, 24)
            it, gs, es = p[1] & 0xffff, (p[1] >> 16) & 0xff, p[1] >> 24
            vec = lambda i: ' '.join(f'{w:08x}' for w in p[i:i + 4])
            print(f'  {OPS[p[0]]} #{it}: a={vec(4)} b={vec(8)} c={vec(12)}')
            print(f'    got={vec(16)} sat={gs}  expected={vec(20)} sat={es}')
        tbfreq = 25_000_000  # mac99 timebase (TBFREQ)
        for name, t in zip(BENCH, mon.words(RESULTS + 0x800, len(BENCH))):
            print(f'bench {name:36s} {t / tbfreq * 1e3:8.1f} ms')
        mt, mn = mon.words(RESULTS + 0x20b * 4, 2)
        print(f'bench {"mandelbrot 64x48x256 (fcmpu exits)":36s} {mt / tbfreq * 1e3:8.1f} ms'
              f'  ({mn} iterations)')
        gt, gn = mon.words(RESULTS + 0x20d * 4, 2)
        g1 = mon.words(RESULTS + 0x20f * 4, 1)[0]
        print(f'bench {"GB mandel: inside set (127500 it)":36s} {g1 / tbfreq * 1e3:8.1f} ms')
        print(f'bench {"GB mandel: escaping (50000 pixels)":36s} {(gt - g1) / tbfreq * 1e3:8.1f} ms'
              f'  ({gn} iterations total)')
        for k, name in enumerate(('fmadds x4 dependent', 'fmadds x4 independent',
                                  'lfs x8/transform/stfs (SP chain)')):
            t = mon.words(RESULTS + (0x212 + k) * 4, 1)[0]
            print(f'bench {name:36s} {t / tbfreq * 1e3:8.1f} ms')
        t = mon.words(RESULTS + 0x216 * 4, 1)[0]
        print(f'bench {"vnmsubfp x4 (float)":36s} {t / tbfreq * 1e3:8.1f} ms')
        for k, name in enumerate(('GB dot fs (lfsx/lfsx/addi/fmadds/bdnz)',
                                  'GB dot shape, double (lfdx/fmadd)',
                                  'GB dot fs unrolled x4 (one TB)')):
            t = mon.words(RESULTS + (0x217 + k) * 4, 1)[0]
            print(f'bench {name:36s} {t / tbfreq * 1e3:8.1f} ms')
        t = mon.words(RESULTS + 0x21b * 4, 1)[0]
        print(f'bench {"GB dot fs, cold accumulator (lfd)":36s} {t / tbfreq * 1e3:8.1f} ms')
        t = mon.words(RESULTS + 0x21c * 4, 1)[0]
        print(f'bench {"GB convolution (lfs/fmr, 5x3 fmadds)":36s} {t / tbfreq * 1e3:8.1f} ms')
        ft, ff = mon.words(RESULTS + 0x210 * 4, 2)
        print(f'bench {"GB mandel: whole routine x4":36s} {ft / tbfreq * 1e3:8.1f} ms'
              f'  ({ff // 14} iterations)')
        fchecks, ffails = mon.words(RESULTS + 0xa00, 2)
        for k in range(min(ffails, 4)):
            p = mon.words(RESULTS + 0xa80 + k * 32, 7)
            what = ('lfs', 'stfs')[p[0]]
            print(f'FAIL {what} in={p[1]:08x}{p[2]:08x} '
                  f'got={p[3]:08x}{p[4]:08x} expected={p[5]:08x}{p[6]:08x}')
        print(f'{checks} checks, {fails} failures')
        print(f'{fchecks} lfs/stfs checks, {ffails} failures')
        sys.exit(1 if fails or ffails else 0)
    finally:
        q.kill()


if __name__ == '__main__':
    main()
