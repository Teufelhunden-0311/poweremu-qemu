#!/usr/bin/env python3
"""vfetch_mutants.py [builddir]: check that test_vfetch notices broken PSC plans.

Each mutant breaks one rule of the r300_vfetch path (fetch_vertex_planned /
psc_plan) in a copy of r300_draw.c; test_vfetch must fail on every one, and
pass on the unmodified copy.  Not part of run.sh (it builds ~10 binaries)."""
import os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "../../hw/display/r300")
SRC = open(os.path.join(R, "r300_draw.c")).read()
PLAN = SRC.index("static void psc_plan(Fetch *f)")
END = SRC.index("/* Fill the input vertex memory for one vertex through the PSC. */", PLAN)

# (name, old text, new text): replaced once, inside psc_plan / fetch_vertex_planned only.
MUTANTS = [
    ("array reset ignored", "p->reset = !f->immd && k < f->arr->count;", "p->reset = false;"),
    ("LAST_VEC ignored", "            k++;\n            break;", "            ;"),
    ("LAST_VEC element dropped", "            k++;\n            break;", "            break;"),
    ("skip ignored", "        cursor += p->skip;", ""),
    ("FLOAT_8 second vector misplaced", "in[(p->dst + 1) % R300_PVS_NUM_INPUTS]", "in[p->dst]"),
    ("sign/normalise swapped", "decode_element(p->type, p->sgn, p->norm, w, c);",
     "decode_element(p->type, p->norm, p->sgn, w, c);"),
    ("swizzle/mask ignored", "            psc_store(p->x, c, in[p->dst]);\n        }\n        cursor",
     "            psc_store(0, c, in[p->dst]);\n        }\n        cursor"),
    ("element size wrong", "p->n = psc_dwords(p->type);", "p->n = psc_dwords(p->type) & 3;"),
    ("zero-size warning lost", "            f->warn |= R300_WARN_VTXFMT;\n        } else if (!fetch_dwords(f, k, vtx, cursor, p->n, w))",
     "            ;\n        } else if (!fetch_dwords(f, k, vtx, cursor, p->n, w))"),
    ("plan kept across draws", "    if (!f->npsc) {\n        psc_plan(f);\n    }",
     "    static Fetch keep;\n    if (!keep.npsc) {\n        psc_plan(f);\n        keep = *f;\n    }\n"
     "    memcpy(f->psc, keep.psc, sizeof(f->psc));\n    f->npsc = keep.npsc;"),
]

def build_and_run(src, tag, bdir):
    d = tempfile.mkdtemp(dir=bdir)
    path = os.path.join(d, "r300_draw.c")
    open(path, "w").write(src)
    exe = os.path.join(d, "test_vfetch")
    cmd = ["cc", "-O1", "-w", "-I", R, "-o", exe, os.path.join(HERE, "test_vfetch.c"), path,
           os.path.join(R, "r300_state.c"), os.path.join(R, "r300_pvs.c"),
           os.path.join(R, "r300_us.c"), "-lm"]
    subprocess.run(cmd, check=True)
    r = subprocess.run([exe, "20000"], cwd=HERE, capture_output=True, text=True)
    return r.returncode, (r.stdout.strip().splitlines() or [""])[-1]

bdir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
os.makedirs(bdir, exist_ok=True)
rc, out = build_and_run(SRC, "base", bdir)
print("unmodified: %s" % out)
if rc:
    sys.exit("unmodified copy fails")
bad = 0
for name, old, new in MUTANTS:
    body = SRC[PLAN:END]
    assert body.count(old) == 1, (name, body.count(old))
    rc, out = build_and_run(SRC[:PLAN] + body.replace(old, new) + SRC[END:], name, bdir)
    print("%-32s %s  %s" % (name, "caught" if rc else "MISSED", out[:110]))
    bad += rc == 0
print("vfetch mutants: %d of %d caught" % (len(MUTANTS) - bad, len(MUTANTS)))
sys.exit(1 if bad else 0)
