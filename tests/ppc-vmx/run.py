#!/usr/bin/env python3
"""Build vmx_test.c as a raw mac99 firmware image, run it, report results.

usage: run.py [--fpdiff | --fpdiff-strict | --vfpdiff] [path/to/qemu-system-ppc]

--fpdiff runs the scalar FP cases with the inline FP ops off and on and
compares results and FPSCR; --fpdiff-strict compares the inline results
with softfloat's (PPC_STRICT_FP=1), whose FPSCR also has the inexact bits.
--vfpdiff does the same for vmaddfp and vnmsubfp with the inline ops off
and on (PPC_NATIVE_VMX; PPC_NATIVE_VNMSUB=0 leaves vnmsubfp on the helper),
comparing results and VSCR bit for bit.

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


def build(out):
    cc = [tool('clang'), '--target=powerpc-unknown-none-elf', '-mcpu=7450',
          '-maltivec', '-O2', '-ffreestanding', '-fno-builtin', '-nostdlib',
          '-fno-pic', '-c']
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


def main():
    out = tempfile.mkdtemp(prefix='ppc-vmx-')
    img = build(out)
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
        t = mon.words(RESULTS + 0x216 * 4, 1)[0]
        print(f'bench {"vnmsubfp x4 (float)":36s} {t / tbfreq * 1e3:8.1f} ms')
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
