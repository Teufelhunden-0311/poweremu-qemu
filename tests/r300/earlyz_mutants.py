#!/usr/bin/env python3
"""earlyz_mutants.py [builddir]: check that test_earlyz notices a broken early rejection.

Each mutant breaks one rule of R300_GLSL_EARLY_Z in a copy of hw/display/r300/r300_us.c
or r300_draw.c; test_earlyz, built on the copy, must then fail.  The unmodified copy must
pass.  Mutants marked "informational" (derivative breakage, which not every GPU shows) are
reported but not required to fail.  Not part of run.sh (it builds a dozen binaries).
AI-generated (Claude)."""
import os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.normpath(os.path.join(HERE, "../../hw/display/r300"))
SRCS = ["r300_state.c", "r300_pvs.c", "r300_us.c", "r300_draw.c"]

# (name, file, old text, new text, required): each old text must occur exactly once.
MUTANTS = [
    # Codex's four (design B8)
    ("no unchanged-word guard", "r300_us.c",
     '"        if (!pre && nz == zb) R300_KILL;\\n"',
     '"        if (!pre) R300_KILL;\\n"', True),
    ("preview counts Z-pass", "r300_us.c",
     'r300_sb_printf(&sb, "%s%s%s", us_zpreview_sig, us_ztest_a, us_ztest_b);',
     'r300_sb_printf(&sb, "%s%s%s%s", us_zpreview_sig, us_ztest_a, us_ztest_count, us_ztest_b);',
     True),
    ("selected shader depth ignored", "r300_us.c",
     '"    if (!(u.depth_src != 0u && R300_WRITES_W)) {\\n"',
     '"    if (true) {\\n"', True),
    ("bit missing from memo identity", "r300_us.c",
     "if (!m->glsl || m->flags != flags ||",
     "if (!m->glsl || (m->flags & ~R300_GLSL_EARLY_Z) != (flags & ~R300_GLSL_EARLY_Z) ||", True),
    # more
    ("preview without polygon offset", "r300_us.c",
     '"        uint nz = r300_zpreview(zb, gz, front, pre);\\n"',
     '"        uint nz = r300_zpreview(zb, gl_FragCoord.z, front, pre);\\n"', True),
    ("full test without polygon offset", "r300_us.c",
     '"    float fz = gz;\\n"',
     '"    float fz = gl_FragCoord.z;\\n"', True),
    ("no per-draw exclusions", "r300_draw.c",
     "    if ((flags & R300_GLSL_EARLY_Z) &&\n        (r300_aa_samples(st) != 1 || (zf != 0 && zf != 2))) {",
     "    if (false) {", True),
    ("GPU-VS call site without exclusions", "r300_draw.c",
     "                                            r300_glsl_flags_for_draw(st, st->glsl_flags),",
     "                                            st->glsl_flags,", True),
    ("CPU-VS call site without exclusions", "r300_draw.c",
     "    pkt->glsl = r300_us_glsl_cached(st, &desc, r300_glsl_flags_for_draw(st, st->glsl_flags),",
     "    pkt->glsl = r300_us_glsl_cached(st, &desc, st->glsl_flags,", True),
    ("interlock not excluded", "r300_us.c",
     "    bool early = (flags & R300_GLSL_EARLY_Z) && !il &&",
     "    bool early = (flags & R300_GLSL_EARLY_Z) &&", True),
    ("rejected fragments return (informational)", "r300_us.c",
     '"        if (!pre && nz == zb) R300_KILL;\\n"',
     '"        if (!pre && nz == zb) { R300_KILL; return; }\\n"', False),
]

def build(files, bdir, tag):
    d = tempfile.mkdtemp(prefix=tag + "-", dir=bdir)
    for f in SRCS:
        open(os.path.join(d, f), "w").write(files[f])
    exe = os.path.join(d, "test_earlyz")
    srcs = " ".join(os.path.join(d, f) for f in SRCS)
    sh = os.path.join(d, "build.sh")
    open(sh, "w").write(r'''#!/bin/sh
set -e
R=../../hw/display/r300
BREW=$(brew --prefix 2>/dev/null || echo /opt/local)
export PKG_CONFIG_PATH="$BREW/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
SPV="$R/r300_spirv.c -DR300_HAVE_SPIRV_CROSS $(pkg-config --cflags --libs shaderc spirv-cross-c)"
SPV="$SPV -lspirv-cross-msl -lspirv-cross-hlsl -lspirv-cross-cpp -lspirv-cross-reflect"
SPV="$SPV -lspirv-cross-glsl -lspirv-cross-util -lspirv-cross-core -lc++"
clang -O1 -w -I$R -fobjc-arc -framework Metal -framework Foundation test_earlyz.m %s $SPV -o "%s"
cc -O1 -w -I$R -o "%s_draw" test_draw.c %s -lm
''' % (srcs, exe, exe, srcs))
    subprocess.run(["sh", sh], cwd=HERE, check=True, stderr=subprocess.DEVNULL)
    return exe

def run(exe):
    r = subprocess.run([exe], cwd=HERE, capture_output=True, text=True)
    d = subprocess.run([exe + "_draw"], cwd=HERE, capture_output=True, text=True)
    fails = [l for l in r.stdout.splitlines() + d.stdout.splitlines() if l.startswith("FAIL")]
    rnd = [l for l in r.stdout.splitlines() if l.startswith("random cases:")]
    rc = r.returncode if r.returncode else d.returncode
    return rc, fails, (rnd[0].split(";")[0][len("random cases: "):] if rnd else "")

bdir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
os.makedirs(bdir, exist_ok=True)
orig = {f: open(os.path.join(R, f)).read() for f in SRCS}
rc, fails, _ = run(build(orig, bdir, "unmodified"))
if rc:
    sys.exit("unmodified copy fails: " + "; ".join(fails[:3]))
print("unmodified: passes")
missed = 0
for name, f, old, new, required in MUTANTS:
    assert orig[f].count(old) == 1, (name, orig[f].count(old))
    files = dict(orig)
    files[f] = orig[f].replace(old, new)
    rc, fails, rnd = run(build(files, bdir, "m"))
    if rc < 0:
        how = "killed by signal %d" % -rc
    elif rc:
        how = "%d check(s) failed: %s" % (len(fails), fails[0][:90] if fails else "")
    else:
        how = ""
    print("%-42s %s  %s" % (name, "caught" if how else ("MISSED" if required else "not seen"), how))
    if rnd:
        print("%-42s random: %s" % ("", rnd))
    missed += required and not how
print("early-Z mutants: %d of %d required caught" %
      (sum(m[4] for m in MUTANTS) - missed, sum(m[4] for m in MUTANTS)))
sys.exit(1 if missed else 0)
