#!/usr/bin/env python3
"""gdbgrab.py OUT TEXTSIZE ADDR:LEN ... : attach to QEMU's gdb stub (port 1234) repeatedly
until the vCPU is stopped in user mode (MSR PR) inside a process whose main executable's
__TEXT vmsize (Mach-O header at 0x1000) equals TEXTSIZE, then dump the given guest virtual
ranges as hex words to OUT.  Each attach stops the VM briefly; detach resumes it."""
import socket, sys, time, struct, random
out, want = sys.argv[1], int(sys.argv[2], 16)
ranges = [(int(a, 16), int(l, 16)) for a, l in (x.split(':') for x in sys.argv[3:])]
def session():
    s = socket.create_connection(('127.0.0.1', 1234)); s.settimeout(10)
    def pkt(d):
        c = sum(d.encode()) & 0xff; s.sendall(f"${d}#{c:02x}".encode()); buf = b''
        while True:
            buf += s.recv(65536); i = buf.find(b'$')
            if i >= 0 and b'#' in buf[i:] and len(buf) >= buf.index(b'#', i) + 3:
                s.sendall(b'+'); return buf[i + 1:buf.index(b'#', i)].decode()
    return s, pkt
for tries in range(400):
    s, pkt = session(); pkt('?')
    msr = int(pkt('p41'), 16); pc = int(pkt('p40'), 16)
    ok = False
    if msr & 0x4000:
        h = bytes.fromhex(pkt('m1000,800'))
        if h[:4] == b'\xfe\xed\xfa\xce':
            ncmds = struct.unpack('>I', h[16:20])[0]; o = 28
            for _ in range(ncmds):
                cmd, size = struct.unpack('>II', h[o:o + 8])
                if cmd == 1 and h[o + 8:o + 14] == b'__TEXT':
                    ok = struct.unpack('>I', h[o + 28:o + 32])[0] == want; break
                o += size
                if o + 8 > len(h): break
    if ok:
        with open(out, 'w') as f:
            f.write(f'# try {tries} pc {pc:08x} msr {msr:08x}\n')
            for a, l in ranges:
                d = pkt(f'm{a:x},{l:x}')
                for k in range(0, len(d), 8): f.write(f'{a + k // 2:08x}: {d[k:k + 8]}\n')
        s.sendall(b'$D#44'); s.close(); print('got it on try', tries, 'pc %08x' % pc); break
    s.sendall(b'$D#44'); s.close(); time.sleep(0.05 + random.random() * 0.1)
else:
    print('no user-mode stop in the target process')
