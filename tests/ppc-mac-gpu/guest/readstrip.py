#!/usr/bin/env python3
"""readstrip.py SHOT.png|.ppm: find rbprobe's magenta anchor, check the
0..255 calibration run, and print the text the probe encoded as grey blocks.
Also prints a few texels of the CopyTexImage square (pixel (x,y) of the
pattern at the square's bottom-left origin) as an on-screen cross-check."""
import subprocess, sys, tempfile

SB, SROW = 4, 128
path = sys.argv[1]
if not path.endswith('.ppm'):
    tmp = tempfile.mktemp(suffix='.ppm')
    subprocess.run(['magick', path, '-depth', '8', tmp], check=True)
    path = tmp
d = open(path, 'rb').read()
parts, pos = [], 0
while len(parts) < 4:
    while d[pos:pos+1].isspace(): pos += 1
    if d[pos:pos+1] == b'#':
        while d[pos:pos+1] != b'\n': pos += 1
        continue
    e = pos
    while not d[e:e+1].isspace(): e += 1
    parts.append(d[pos:e]); pos = e
pos += 1
w, h = int(parts[1]), int(parts[2]); px = d[pos:]
def P(x, y): o = 3 * (y * w + x); return px[o], px[o+1], px[o+2]
ax = ay = None
for y in range(h - 8):
    for x in range(w - 8):
        if P(x, y) == (255, 0, 255) and P(x + 7, y) == (255, 0, 255) and P(x, y + 7) == (255, 0, 255) \
                and P(x + 8, y + 8) != (255, 0, 255):
            ax, ay = x, y; break
    if ax is not None: break
if ax is None:
    sys.exit('anchor not found')

def byte(i):
    r, g, b = P(ax + (i % SROW) * SB + SB // 2, ay + 16 + (i // SROW) * SB + SB // 2)
    return r, g, b

bad = [(i, byte(i)) for i in range(256) if byte(i) != (i, i, i)]
if bad:
    print('CALIBRATION: %d of 256 wrong, first %s' % (len(bad), bad[:4]))
    sys.exit(1)
n = 0
for i in range(256, 260):
    n = (n << 8) | byte(i)[0]
text = bytes(byte(i)[0] for i in range(260, 260 + n))
s = 0
for i in range(260 + n, 264 + n):
    s = (s << 8) | byte(i)[0]
ok = s == sum(text) & 0xffffffff
print(text.decode('latin-1'), end='')
print('[strip: %d bytes, checksum %s]' % (n, 'ok' if ok else 'BAD'))

# CopyTexImage square: screen x 528.., its bottom row is pattern row 0
x0, bottom = ax + 528, ay + 16 + 63
def pat(x, y): return ((4 * x + 1) & 255, (4 * y + 2) & 255, (7 * x + 13 * y) & 255)
hits = 0
for (x, y) in [(0, 0), (1, 0), (0, 1), (10, 20), (63, 63), (32, 5)]:
    got = P(x0 + x, bottom - y)
    hits += got == pat(x, y)
    print('  copytex on screen (%d,%d): %s want %s' % (x, y, got, pat(x, y)))
print('[copytex on screen: %d/6 exact]' % hits)
