#!/usr/bin/env python3
"""kgrab.py OUT ADDR:LEN ... : attach to QEMU's gdb stub (port 1234) until the vCPU is stopped
in supervisor mode (MSR PR clear), then dump the given guest virtual ranges (kernel space) as
hex words to OUT.  Prints the stop PC/LR/MSR.  Detach resumes the VM."""
import socket, sys, time, random
out = sys.argv[1]
ranges = [(int(a, 16), int(l, 16)) for a, l in (x.split(':') for x in sys.argv[2:])]
for tries in range(400):
    s = socket.create_connection(('127.0.0.1', 1234)); s.settimeout(10)
    def pkt(d):
        c = sum(d.encode()) & 0xff; s.sendall(f"${d}#{c:02x}".encode()); buf = b''
        while True:
            buf += s.recv(65536); i = buf.find(b'$')
            if i >= 0 and b'#' in buf[i:] and len(buf) >= buf.index(b'#', i) + 3:
                s.sendall(b'+'); return buf[i + 1:buf.index(b'#', i)].decode()
    pkt('?')
    msr = int(pkt('p41'), 16); pc = int(pkt('p40'), 16); lr = int(pkt('p43'), 16)
    if not msr & 0x4000:
        with open(out, 'w') as f:
            f.write(f'# try {tries} pc {pc:08x} lr {lr:08x} msr {msr:08x}\n')
            for a, l in ranges:
                d = pkt(f'm{a:x},{l:x}')
                for k in range(0, len(d), 8): f.write(f'{a + k // 2:08x}: {d[k:k + 8]}\n')
        s.sendall(b'$D#44'); s.close(); print('got it on try %d pc %08x lr %08x msr %08x' % (tries, pc, lr, msr)); break
    s.sendall(b'$D#44'); s.close(); time.sleep(0.05 + random.random() * 0.1)
else:
    print('no supervisor-mode stop')
