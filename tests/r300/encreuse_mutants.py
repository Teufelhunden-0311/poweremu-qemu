#!/usr/bin/env python3
"""encreuse_mutants.py [builddir]: check that test_encode notices a broken setter cache.

Each mutant breaks one rule of PPCGPU_R300_ENC_REUSE in a copy of
hw/display/ppc_mac_gpu_metal.m; test_encode, built on the copy and run with
the cache on (render queue off and on), must then fail or draw something other
than the unmodified renderer does with the cache off.  The unmodified copy
must pass and match.  Not part of run.sh (it builds a dozen binaries).
AI-generated (Claude)."""
import os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "../../hw/display")
SRC = open(os.path.join(D, "ppc_mac_gpu_metal.m")).read()

# (name, old text, new text): each old text must occur exactly once.
MUTANTS = [
    # Codex's four (encoder-reuse checklist A5)
    ("no reset on a new or ended encoder",
     "    if (!rc) {\n        return;                 /* off: nothing is ever cached */\n    }",
     "    return;"),
    ("line cull not recorded",
     "                if (rc) {\n                    rc->cull = MTLCullModeNone;     /* the next sample or draw restores */\n"
     "                    rc->cull_ok = true;\n                }\n",
     ""),
    ("FS uniforms compared before adjustment",
     "            set = !rc || !rc->fsu_ok || memcmp(&rc->fsu, &uk, sizeof(uk));",
     "            set = !rc || !rc->fsu_ok || memcmp(&rc->fsu, &pkt->uniforms, sizeof(uk));"),
    ("buffer 0 skipped on an equal buffer",
     "        memcpy(dst, data, len);\n        [enc setVertexBuffer:b offset:off atIndex:index];",
     "        memcpy(dst, data, len);\n        static id<MTLBuffer> last0;\n"
     "        if (index != 0 || b != last0) {\n            [enc setVertexBuffer:b offset:off atIndex:index];\n        }\n"
     "        if (index == 0) {\n            last0 = b;\n        }"),
    # more
    ("VS uniforms not compared",
     "            set = !rc || !rc->vsu_ok || memcmp(rc->vsu, pkt->vs_u, sizeof(*pkt->vs_u));",
     "            set = !rc || !rc->vsu_ok;"),
    ("sampler skipped with its texture",
     "            set = !rc || !rc->smp_ok[t] || rc->smp[t] != smp[t];",
     "            set = !rc || !rc->smp_ok[t] || (rc->smp[t] != smp[t] && rc->tex[t] != tex[t]);"),
    ("scissor x ignored (sample columns)",
     "            set = !rc || !rc->rect_ok || memcmp(&rc->rect, &rect, sizeof(rect));",
     "            set = !rc || !rc->rect_ok || rc->rect.y != rect.y ||\n"
     "                  rc->rect.width != rect.width || rc->rect.height != rect.height;"),
    ("winding not compared",
     "        set = !rc || !rc->wind_ok || rc->wind != wind;",
     "        set = !rc || !rc->wind_ok;"),
    ("pipeline not compared",
     "        bool set = !rc || !rc->pipe_ok || rc->pipe != pipe;",
     "        bool set = !rc || !rc->pipe_ok;"),
    ("texture compared on unit 0 only",
     "            set = !rc || !rc->tex_ok[t] || rc->tex[t] != tex[t];",
     "            set = !rc || !rc->tex_ok[t] || (t == 0 && rc->tex[t] != tex[t]);"),
]

def build(src, bdir):
    d = tempfile.mkdtemp(dir=bdir)
    path = os.path.join(d, "ppc_mac_gpu_metal.m")
    open(path, "w").write(src)
    sh = os.path.join(d, "build.sh")
    exe = os.path.join(d, "test_encode")
    open(sh, "w").write(r'''#!/bin/sh
set -e
R=../../hw/display/r300
BREW=$(brew --prefix 2>/dev/null || echo /opt/local)
export PKG_CONFIG_PATH="$BREW/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
SPV="$R/r300_spirv.c -DR300_HAVE_SPIRV_CROSS $(pkg-config --cflags --libs shaderc spirv-cross-c)"
SPV="$SPV -lspirv-cross-msl -lspirv-cross-hlsl -lspirv-cross-cpp -lspirv-cross-reflect"
SPV="$SPV -lspirv-cross-glsl -lspirv-cross-util -lspirv-cross-core -lc++"
clang -O1 -w -Iencshim -I../../hw/display '-DENC_METAL_M="%s"' \
    -framework Metal -framework Foundation -framework QuartzCore test_encode.m \
    $R/r300_state.c $R/r300_pvs.c $R/r300_us.c $R/r300_draw.c $R/r300_msl_cache.c $SPV \
    $(pkg-config --cflags --libs glib-2.0) -lz -o "%s"
''' % (path, exe))
    subprocess.run(["sh", sh], cwd=HERE, check=True)
    return exe, d

def run(exe, d, queue, reuse):
    env = dict(os.environ, ENC_TEST_DIR=d, PPCGPU_RENDER_QUEUE=queue,
               PPCGPU_R300_ENC_REUSE=reuse)
    r = subprocess.run([exe], cwd=HERE, env=env, capture_output=True, text=True)
    steps = [l for l in r.stdout.splitlines() if l.startswith("step")]
    fails = [l for l in r.stdout.splitlines() if l.startswith("FAIL")]
    return r.returncode, steps, fails

bdir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
os.makedirs(bdir, exist_ok=True)
exe, d = build(SRC, bdir)
ref = {}
for q in "01":
    rc, ref[q], _ = run(exe, d, q, "0")
    rc1, steps, _ = run(exe, d, q, "1")
    if rc or rc1 or steps != ref[q]:
        sys.exit("unmodified copy fails or differs (queue %s)" % q)
print("unmodified: passes, cache on = cache off (queue off and on, %d steps)" % len(ref["0"]))
missed = 0
for name, old, new in MUTANTS:
    assert SRC.count(old) == 1, (name, SRC.count(old))
    exe, d = build(SRC.replace(old, new), bdir)
    how = []
    for q in "01":
        rc, steps, fails = run(exe, d, q, "1")
        if steps != ref[q]:
            first = next((i for i, (a, b) in enumerate(zip(steps, ref[q])) if a != b),
                         min(len(steps), len(ref[q])))
            how.append("q%s pixels differ from step %d" % (q, first))
        if rc:
            how.append("q%s exit %d, %d check(s) failed" % (q, rc, len(fails)))
    print("%-40s %s  %s" % (name, "caught" if how else "MISSED", "; ".join(how)[:120]))
    missed += not how
print("encoder-reuse mutants: %d of %d caught" % (len(MUTANTS) - missed, len(MUTANTS)))
sys.exit(1 if missed else 0)
