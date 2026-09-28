#!/bin/sh
# Build and run the offline R300 tests: ./run.sh [builddir]
set -e
cd "$(dirname "$0")"
B=${1:-${TMPDIR:-/tmp}/r300-tests}
mkdir -p "$B"
R=../../hw/display/r300
SRC="$R/r300_state.c $R/r300_pvs.c $R/r300_us.c $R/r300_draw.c"

# The shader toolchain: shaderc (GLSL -> SPIR-V) and SPIRV-Cross (-> MSL).
BREW=$(brew --prefix 2>/dev/null || echo /opt/local)   # Homebrew, else MacPorts
export PKG_CONFIG_PATH="$BREW/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
SPV="$R/r300_spirv.c $(pkg-config --cflags --libs shaderc spirv-cross-c)"
SPV="$SPV -lspirv-cross-msl -lspirv-cross-hlsl -lspirv-cross-cpp -lspirv-cross-reflect"
SPV="$SPV -lspirv-cross-glsl -lspirv-cross-util -lspirv-cross-core -lc++"

for t in test_pvs test_us test_draw test_features; do
    cc -O1 -Wall -Wno-unused-function -o "$B/$t" $t.c $SRC -lm
done
cc -O1 -Wall -o "$B/glsl2msl" glsl2msl.c $SPV
clang -fobjc-arc -framework Metal -framework Foundation mslcheck.m -o "$B/mslcheck"
for t in test_zs test_fmt test_raster test_gpuvs; do
    clang -O1 -fobjc-arc -framework Metal -framework Foundation $t.m $SRC $SPV -o "$B/$t"
done
"$B/test_pvs"
"$B/test_draw"
# Both shader outputs: 0 (Metal: uint texture views), 1 (Vulkan: VRAM SSBO).
for flags in 0 1; do
    "$B/test_us" qe_state_draw1.txt $flags > "$B/qe$flags.glsl" 2>/dev/null
    for s in vert:R300_VS frag:R300_FS frag:R300_FS_Z; do
        glslangValidator -V --target-env vulkan1.1 -S "${s%%:*}" -D"${s#*:}" \
            "$B/qe$flags.glsl" -o "$B/qe$flags.${s#*:}.spv" > "$B/glslang.log" \
            || { cat "$B/glslang.log"; exit 1; }
        spirv-val --target-env vulkan1.1 "$B/qe$flags.${s#*:}.spv"
    done
    echo "SPIR-V OK: flags $flags"
    "$B/glsl2msl" "$B/qe$flags.glsl" "$B/qe$flags"
    for s in vs fs fs_z; do
        "$B/mslcheck" "$B/qe$flags.$s.metal"
    done
done
"$B/test_features"
"$B/test_zs"
"$B/test_fmt"
"$B/test_raster"
"$B/test_gpuvs"
echo "all R300 tests passed"
