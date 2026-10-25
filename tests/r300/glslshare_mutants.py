#!/usr/bin/env python3
"""glslshare_mutants.py [builddir]: check that test_glslshare notices broken text sharing.

Each mutant breaks one ownership rule of r300_text.h in a copy of hw/display/r300; test_glslshare,
built with AddressSanitizer (ThreadSanitizer for the atomicity mutant), must fail on every one and
pass on the unmodified copy.  Not part of run.sh (it builds several binaries).  Leaks (a missing
unref) are not caught: LeakSanitizer is not available on macOS."""
import os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "../../hw/display/r300")

# (name, file, old text, new text, sanitizer)
MUTANTS = [
    ("reference not counted", "r300_text.h",
     "    __atomic_fetch_add(&r300_text_of(s)->ref, 1, __ATOMIC_RELAXED);\n", "", "address"),
    ("copy mode hands out the shared text", "r300_text.h",
     "        return r300_text_new(s);\n", "        return s;\n", "address"),
    ("count not atomic", "r300_text.h",
     "__atomic_sub_fetch(&r300_text_of(s)->ref, 1, __ATOMIC_ACQ_REL) == 0",
     "--r300_text_of(s)->ref == 0", "thread"),
    ("fragment memo frees on eviction", "r300_us.c",
     "    r300_text_unref(last->glsl);\n", "    if (last->glsl) free(r300_text_of(last->glsl));\n", "address"),
    ("vertex memo frees on eviction", "r300_draw.c",
     "    r300_text_unref(last->glsl);\n", "    if (last->glsl) free(r300_text_of(last->glsl));\n", "address"),
    ("fragment hit without a reference", "r300_us.c",
     "            return r300_text_ref(hit.glsl);\n", "            return hit.glsl;\n", "address"),
    ("vertex text without a reference", "r300_draw.c",
     "    pkt->vs_glsl = r300_text_ref(vs->glsl);\n", "    pkt->vs_glsl = vs->glsl;\n", "address"),
]

def build_and_run(tag, mutate, san, bdir):
    root = tempfile.mkdtemp(dir=bdir)
    r = os.path.join(root, "hw/display/r300")
    shutil.copytree(R, r)
    t = os.path.join(root, "tests/r300")
    os.makedirs(t)
    for f in ("test_glslshare.c", "qe_draw1_full.txt"):
        shutil.copy(os.path.join(HERE, f), t)
    if mutate:
        f, old, new = mutate
        p = os.path.join(r, f)
        s = open(p).read()
        assert s.count(old) == 1, (tag, s.count(old))
        open(p, "w").write(s.replace(old, new))
    exe = os.path.join(t, "test_glslshare")
    subprocess.run(["cc", "-O1", "-g", "-fsanitize=" + san, "-w", "-o", exe, "test_glslshare.c"] +
                   [os.path.join(r, x) for x in ("r300_state.c", "r300_pvs.c", "r300_us.c", "r300_draw.c")] +
                   ["-lm"], cwd=t, check=True)
    env = dict(os.environ, TSAN_OPTIONS="halt_on_error=1", ASAN_OPTIONS="halt_on_error=1")
    p = subprocess.run([exe, "3000"], cwd=t, capture_output=True, text=True, env=env)
    out = (p.stdout + p.stderr).strip().splitlines()
    first = next((l for l in out if "ERROR" in l or "WARNING" in l or "FAIL" in l), out[-1] if out else "")
    return p.returncode, first

bdir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
os.makedirs(bdir, exist_ok=True)
for san in ("address", "thread"):
    rc, out = build_and_run("base", None, san, bdir)
    print("unmodified (%s): %s" % (san, out[:110]))
    if rc:
        sys.exit("unmodified copy fails")
bad = 0
for name, f, old, new, san in MUTANTS:
    rc, out = build_and_run(name, (f, old, new), san, bdir)
    print("%-34s %s  %s" % (name, "caught" if rc else "MISSED", out[:100]))
    bad += rc == 0
print("glslshare mutants: %d of %d caught" % (len(MUTANTS) - bad, len(MUTANTS)))
sys.exit(1 if bad else 0)
