#!/usr/bin/env python3
"""gdis.py DUMP.txt [HOT.txt]: disassemble the hex-word dump from gdbgrab.py (PowerPC, big
endian) with llvm-objdump, one block per contiguous range.  If HOT is given (lines
'pct addr'), hot addresses are marked."""
import sys, subprocess, tempfile, os, re
L = '/opt/homebrew/opt/llvm/bin/'
words = []
for l in open(sys.argv[1]):
    m = re.match(r'([0-9a-f]{8}): ([0-9a-f]{8})$', l.strip())
    if m: words.append((int(m.group(1), 16), bytes.fromhex(m.group(2))))
blocks = []
for a, w in words:
    if blocks and blocks[-1][0] + len(blocks[-1][1]) == a: blocks[-1][1] += w
    else: blocks.append([a, bytearray(w)])
td = tempfile.mkdtemp()
for a, b in blocks:
    bi, el = os.path.join(td, 'b.bin'), os.path.join(td, 'b.elf')
    open(bi, 'wb').write(b)
    subprocess.run([L + 'llvm-objcopy', '-I', 'binary', '-O', 'elf32-powerpc', '--rename-section', '.data=.text,code', bi, el], check=True)
    out = subprocess.run([L + 'llvm-objdump', '-d', '--no-show-raw-insn', '--triple=powerpc-apple-darwin', '--adjust-vma=0x%x' % a, el], capture_output=True, text=True).stdout
    print('==== %08x (%d bytes)' % (a, len(b)))
    for l in out.split('\n'):
        m = re.match(r'\s+([0-9a-f]+):\s*(.*)$', l)
        if not m: continue
        ins = re.sub(r'\s*<_binary_[^>]*>', '', m.group(2)).replace('\t', ' ')
        # branch targets are block-relative here: make them absolute
        ins = re.sub(r'(\b(?:b|bl|bt|bf|bdnz|bdz|bc|bne|beq|blt|bgt|ble|bge)\S*\s+(?:\d+,\s*)?)0x([0-9a-f]+)$',
                     lambda x: x.group(1) + '0x%08x' % ((a + int(x.group(2), 16)) & 0xffffffff), ins)
        print('%08x:  %s' % (a + int(m.group(1), 16), ins))
