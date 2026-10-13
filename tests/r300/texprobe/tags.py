#!/usr/bin/env python3
"""tags.py PREFIX [stage]: find texprobe's levels in an R300_CAP_TEX capture.

PREFIX is the R300_CAP prefix (PREFIX.log, PREFIX.t<addr>_d<draw>_f<fmt>_<w>x<h>_p<pitch>.bin).
For every dumped texture that is one of texprobe's, print where each level's
(and cube face's) texels are found in the captured emulator memory, the row
pitch seen there, every further place the same row is found, and whether the
emulator's own layout (the L lines of the log) names one of those places.

That is where the emulator stored the texels while executing the driver's
commands.  A row found in several places is listed in all of them; "ok" means
the emulator's offset is among them, which does not single it out.  The
texel rules are those of texprobe.c.  Exit status 1 when a level differs, 2 when
nothing differs but not every case of the stage was captured and checked (the
capture is best effort; an empty one proves nothing)."""
import sys, re, glob, os, struct

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "texprobe.c")
KINDS = ["rgba8", "1555", "565", "4444", "dxt1", "dxt3", "dxt5", "a8", "l8", "i8", "al88"]
KMAP = {"K_RGBA8": 0, "K_1555": 1, "K_565": 2, "K_4444": 3, "K_DXT1": 4, "K_DXT3": 5,
        "K_DXT5": 6, "K_A8": 7, "K_L8": 8, "K_I8": 9, "K_AL88": 10}
SUBSALT = 77

def cases():
    out = []
    for m in re.finditer(r"\{\s*(\d+),\s*(K_\w+),\s*(\d+),\s*(\d+),\s*(\d),\s*(\d),\s*(\d)(?:,\s*(\d))?\s*\}", open(SRC).read()):
        i, k, w, h, cube, sub, stage, single = m.groups()
        out.append(dict(id=int(i), kind=KMAP[k], w=int(w), h=int(h), cube=int(cube), sub=int(sub),
                        stage=int(stage), single=int(single or 0)))
    return out

def dim(d, lv):
    return max(1, d >> lv)

def is_dxt(k):
    return k in (4, 5, 6)

def hash8(lv, face, x, y, salt):
    M = 0xFFFFFFFF
    h = ((x + 1) * 2654435761 ^ (y + 1) * 2246822519 ^ (lv * 6 + face + 1) * 3266489917 ^ salt * 668265263) & M
    h ^= h >> 15
    h = (h * 2246822519) & M
    h ^= h >> 13
    return h & 0xFF

def texel(kind, salt, face, lv, x, y):
    """The texel's (block's) bytes in the card's little-endian order."""
    if kind == 0:
        r = (lv << 4) | (face << 1) | 1
        g = (x + 7 * salt) & 255
        b = (y + 13 * salt) & 255
        a = 0x40 | (((x >> 8) & 7) << 3) | ((y >> 8) & 7)
        return bytes([b, g, r, a])
    if kind == 1:
        v = (((x >> 5) ^ (y >> 5) ^ 1) & 1) << 15 | ((lv + 1) & 31) << 10 | \
            ((x + 5 * face + 7 * salt) & 31) << 5 | ((y + 11 * face + 13 * salt) & 31)
        return struct.pack("<H", v)
    if kind == 2:
        v = ((lv + 1) & 31) << 11 | (((x + 7 * salt) & 31) << 1) << 5 | ((y + 13 * salt) & 31)
        return struct.pack("<H", v)
    if kind == 3:
        v = (8 | ((x >> 4) & 7)) << 12 | ((lv + 1) & 15) << 8 | ((x + 7 * salt) & 15) << 4 | ((y + 13 * salt) & 15)
        return struct.pack("<H", v)
    if is_dxt(kind):
        c = ((lv + 1) & 31) << 11 | ((x + 5 * face + 7 * salt) & 63) << 5 | ((y + 11 * face + 13 * salt) & 31)
        col = struct.pack("<HH", c, c) + bytes(4)
        if kind == 5:
            a = (8 | (x & 7)) * 0x11
            return bytes([a] * 8) + col
        if kind == 6:
            a = 0x80 | (x & 0x3F)
            return bytes([a, a]) + bytes(6) + col
        return col
    if kind in (7, 8, 9):
        return bytes([hash8(lv, face, x, y, salt)])
    return bytes([hash8(lv, face, x, y, salt), ((lv << 4) | (y & 15)) ^ 0x80])

def subrects(c):
    """texprobe.c subimage(): the rectangles (texels) given the other salt."""
    step = 4 if is_dxt(c["kind"]) else 1
    small = -1
    for lv in range(nlevels(c)):
        lw = dim(c["w"], lv)
        if lw == 8 * step or (small < 0 and 2 * step <= lw <= 8 * step):
            small = lv
    out = []
    for i in range(2):
        lv = small if i else 0
        if lv < 0 or (i and lv == 0):
            continue
        lw, lh = dim(c["w"], lv), dim(c["h"], lv)
        x, y, w, h = (3 * step, step, 5 * step, 3 * step) if i == 0 else (step, 0, 3 * step, 2 * step)
        if x >= lw: x = 0
        if y >= lh: y = 0
        w = min(w, lw - x)
        h = min(h, lh - y)
        out.append((lv, x, y, w, h))
    return out

def row(c, salt, face, lv, y, subs=()):
    n = dim(c["w"], lv)
    step = 1
    if is_dxt(c["kind"]):
        n = (n + 3) // 4
        step = 4
    def s(x):
        for (l, rx, ry, rw, rh) in subs:
            if l == lv and rx <= x * step < rx + rw and ry <= y * step < ry + rh:
                return salt + SUBSALT
        return salt
    return b"".join(texel(c["kind"], s(x), face, lv, x, y) for x in range(n))

def nrows(c, lv):
    n = dim(c["h"], lv)
    return (n + 3) // 4 if is_dxt(c["kind"]) else n

def nlevels(c):
    m, n = max(c["w"], c["h"]), 1
    while m > 1 and not c["single"]:
        m >>= 1
        n += 1
    return n

def swap(d, n):
    if n == 1:
        return d
    d = d[:len(d) // n * n]
    out = bytearray(len(d))
    for k in range(n):
        out[k::n] = d[n - 1 - k::n]
    return bytes(out)

def find_all(hay, needle):
    out, i = [], hay.find(needle)
    while i >= 0:
        out.append(i)
        i = hay.find(needle, i + 1)
    return out

def unswap_off(o, n):
    """An offset in the swapped copy, as an offset in the dump (start of the unit)."""
    return o - o % n if n > 1 else o

def main():
    pre = sys.argv[1]
    stage = int(sys.argv[2]) if len(sys.argv) > 2 else None
    cs = [c for c in cases() if stage is None or c["stage"] == stage]
    lay = {}
    for line in open(pre + ".log", errors="replace"):
        m = re.match(r"L(\d+) t\d+ ([0-9a-f]+) swap (\d+) gart (\d+) size (\d+)((?: \| \d+ \d+ \d+ \d+)*)", line)
        if m:
            lv = [tuple(int(v) for v in g.split()) for g in m.group(6).split("|")[1:]]
            lay[(int(m.group(1)), int(m.group(2), 16))] = dict(swap=int(m.group(3)), size=int(m.group(5)), lv=lv)
    bad = nlev = nolayout = 0
    seen = set()
    for path in sorted(glob.glob(pre + ".t*_d*_f*_*x*_p*.bin"), key=lambda p: int(re.search(r"_d(\d+)_", p).group(1))):
        m = re.search(r"\.t([0-9a-f]+)_d(\d+)_f([0-9a-f]+)_(\d+)x(\d+)_p(\d+)\.bin$", path)
        addr, draw, fmt, w, h = int(m.group(1), 16), int(m.group(2)), m.group(3), int(m.group(4)), int(m.group(5))
        if not any((c["w"], c["h"]) == (w, h) for c in cs):
            continue
        raw = open(path, "rb").read()
        hit = None
        for c in cs:
            if (c["w"], c["h"]) != (w, h):
                continue
            r0 = row(c, c["id"], 0, 0, 0)[:64]
            for sw in (1, 2, 4):
                if swap(raw[:64], sw).startswith(r0):
                    hit = (c, sw)
        if not hit:
            continue
        c, sw = hit
        d = swap(raw, sw)
        L = lay.get((draw, addr))
        first = (c["id"], addr) not in seen
        seen.add((c["id"], addr))
        subs = () if first else subrects(c)
        print("\nT%02d %s%s %dx%d at %06x (draw %d, fmt %s, bytes swapped in %ds%s) emulator size %s" % (
            c["id"], KINDS[c["kind"]], " cube" if c["cube"] else "", w, h, addr, draw, fmt, sw,
            "" if first else ", again after a change", L["size"] if L else "?"))
        end = 0
        for lv in range(nlevels(c)):
            for face in range(6 if c["cube"] else 1):
                r0 = row(c, c["id"], face, lv, 0, subs)
                offs = [unswap_off(o, sw) for o in find_all(d, r0)]
                want = None
                whole = False
                if L and lv < len(L["lv"]):
                    off, lp, rows, facesz = L["lv"][lv]
                    want = (off + face * facesz, lp)
                    # every row of the level, where the sampler will look for it
                    whole = all(d[want[0] + y * lp:want[0] + y * lp + len(r0)] ==
                                row(c, c["id"], face, lv, y, subs) for y in range(nrows(c, lv)))
                if want is None:
                    nolayout += 1
                else:
                    nlev += 1
                    bad += not whole
                if want is not None and whole:
                    end = max(end, want[0] + lp * (nrows(c, lv) - 1) + len(r0))
                print("  lv%-2d f%d %4dx%-4d row %5d B: row 0 found at %s | emulator %s: %s" % (
                    lv, face, dim(w, lv), dim(h, lv), len(r0),
                    ",".join("%#x" % o for o in offs[:8]) + ("..." if len(offs) > 8 else "") if offs else "NOWHERE",
                    ("%#x pitch %d" % want) if want else "?",
                    "whole level there" if whole else "DIFFERS" if want else "not checked"))
        print("  data ends at %#x" % end)
    missing = sorted(c["id"] for c in cs if not any(i == c["id"] for i, _ in seen))
    print("\ntextures examined: %d of %d cases; levels and faces checked: %d; without a layout line: %d" % (
        len(set(i for i, _ in seen)), len(cs), nlev, nolayout))
    if missing:
        print("cases with no capture (not checked): %s" % " ".join("T%02d" % i for i in missing))
    print("levels that differ from the emulator's layout: %d" % bad)
    # 1: a level differs.  2: nothing, or not every case, was examined.
    return 1 if bad else 2 if not nlev or missing or nolayout else 0

sys.exit(main())
