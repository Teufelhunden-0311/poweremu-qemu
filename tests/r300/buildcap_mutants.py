#!/usr/bin/env python3
"""buildcap_mutants.py [builddir]: check that test_buildcap notices broken fetch plans and capture reads.

Each mutant breaks one rule of r300_draw_fetch_plan / r300_capture_read in a copy of r300_draw.c;
test_buildcap must fail on every one, and pass on the unmodified copy.  Not part of run.sh (it builds
~10 binaries).  AI-generated (Claude)."""
import os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "../../hw/display/r300")
SRC = open(os.path.join(R, "r300_draw.c")).read()
PLAN = SRC.index("bool r300_draw_fetch_plan(")
END = SRC.index("void r300_draw_parked_begin(void)", PLAN)

# (name, old text, new text): replaced once, inside the plan / capture reader only.
MUTANTS = [
    ("skip ignored", "        cursor += (e >> 4) & 0xF;       /* skip, read or not */\n", ""),
    ("array cursor reset ignored", "            cursor = 0;                 /* each array starts afresh */\n", ""),
    ("element width wrong (FLOAT_8)", "unsigned w = psc_dwords(e & 0xF), a;", "unsigned w = psc_dwords(e & 0xF) & 7, a;"),
    ("extra elements on the wrong array", "a = k < arr->count ? k : arr->count - 1;", "a = k % arr->count;"),
    ("largest index not used", "vmax = v > vmax ? v : vmax;", "vmax = i;"),
    ("VC_SWAP ignored for index buffers", "vc_swap(idx->dw[w], swap)", "idx->dw[w]"),
    ("inline index width ignored", "v = r300_index_at(d + 1, ndw - 1, i32, i);", "v = r300_index_at(d + 1, ndw - 1, false, i);"),
    ("stride ignored at the top", "end = base + ((uint64_t)vmax * stride + hi[a]) * 4;",
     "end = base + ((uint64_t)vmin * stride + hi[a]) * 4;"),
    ("merge shrinks a range", "(e2 > end ? e2 : end)", "e2"),
    ("capture end exclusive by one", "(uint64_t)gpu_addr + len <= (uint64_t)r->gpu + r->len",
     "(uint64_t)gpu_addr + len < (uint64_t)r->gpu + r->len"),
    ("capture offset wrong", "c->buf + at + (gpu_addr - r->gpu)", "c->buf + (gpu_addr - r->gpu)"),
]

def build_and_run(src, bdir):
    d = tempfile.mkdtemp(dir=bdir)
    path = os.path.join(d, "r300_draw.c")
    open(path, "w").write(src)
    exe = os.path.join(d, "test_buildcap")
    cmd = ["cc", "-O1", "-w", "-I", R, "-o", exe, os.path.join(HERE, "test_buildcap.c"), path,
           os.path.join(R, "r300_state.c"), os.path.join(R, "r300_pvs.c"),
           os.path.join(R, "r300_us.c"), "-lm"]
    subprocess.run(cmd, check=True)
    r = subprocess.run([exe, "20000"], cwd=HERE, capture_output=True, text=True)
    return r.returncode, (r.stdout.strip().splitlines() or [""])[-1]

bdir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
os.makedirs(bdir, exist_ok=True)
rc, out = build_and_run(SRC, bdir)
print("unmodified: %s" % out[:140])
if rc:
    sys.exit("unmodified copy fails")
bad = 0
for name, old, new in MUTANTS:
    body = SRC[PLAN:END]
    assert body.count(old) == 1, (name, body.count(old))
    rc, out = build_and_run(SRC[:PLAN] + body.replace(old, new) + SRC[END:], bdir)
    print("%-36s %s  %s" % (name, "caught" if rc else "MISSED", out[:100]))
    bad += rc == 0
print("buildcap mutants: %d of %d caught" % (len(MUTANTS) - bad, len(MUTANTS)))
sys.exit(1 if bad else 0)
