#!/usr/bin/env python3
"""earlyz_mslcheck.py dir [name...]: check the early-rejection shape (design B5) of each
dir/<name>.fs_z.metal (default names: the eight test programs).  In the fragment entry
point: dfdx(z) before the one depth-input read before the conditional discard before
r300_shade at function scope before r300_ztest; exactly one return; r300_zpreview with no
device buffer and no atomic.  Exits nonzero on any failure or missing file.
AI-generated (Claude)."""
import os, re, sys

NAMES = ["qe", "raw16f", "tex", "tex_kil", "tex_mrt", "tex_w", "txb", "txp"]

def check(t):
    i = t.find("\nfragment ")
    if i < 0:
        return "no fragment entry point"
    body = t[i:]
    def pos(p):
        m = re.search(p, body)
        return m.start() if m else -1
    d, z = pos(r"dfdx\(gl_FragCoord\.z\)"), pos(r"zin\.x")
    k, sh, zt = pos(r"discard_fragment\(\)"), pos(r"\n    r300_shade\("), pos(r"r300_ztest\(")
    if not (0 <= d < k and 0 <= z < k and k < sh < zt):
        return "order: dfdx %d read %d discard %d shade %d ztest %d" % (d, z, k, sh, zt)
    if len(re.findall(r"zin\.x", body)) != 1:
        return "depth input read %d times" % len(re.findall(r"zin\.x", body))
    if len(re.findall(r"\breturn\b", body)) != 1:
        return "%d returns" % len(re.findall(r"\breturn\b", body))
    pv = re.search(r"\nuint r300_zpreview\([^)]*\)\n\{.*?\n\}\n", t, re.S)
    if not pv:
        return "no r300_zpreview"
    if "atomic" in pv.group(0) or "device" in pv.group(0).split("\n")[1]:
        return "r300_zpreview has a side effect"
    return None

d = sys.argv[1]
names = sys.argv[2:] or NAMES
bad = 0
for n in names:
    f = os.path.join(d, n + ".fs_z.metal")
    why = check(open(f).read()) if os.path.exists(f) else "missing"
    print("%-10s %s" % (n, "OK" if why is None else "BAD: " + why))
    bad += why is not None
print("early-Z MSL shape: %d of %d OK" % (len(names) - bad, len(names)))
sys.exit(1 if bad or not names else 0)
