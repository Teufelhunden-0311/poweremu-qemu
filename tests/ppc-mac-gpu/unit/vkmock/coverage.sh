#!/bin/sh
# coverage.sh [SRCDIR]: which lines of the renderer does the test reach?
# Builds with clang's source coverage (REV= as for build.sh), runs everything
# for both device variants, prints the renderer's coverage and writes its
# lines that were never executed to uncovered.txt.
# Needs clang, llvm-profdata and llvm-cov (macOS: through xcrun).
set -e
here=$(cd "$(dirname "$0")" && pwd)
src=${1:-${QEMU_SRC:-$here/../../../..}}
tool() { if command -v xcrun >/dev/null 2>&1; then xcrun "$@"; else "$@"; fi; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
if [ -n "$REV" ]; then
    # a copy that stays while llvm-cov reads it
    mkdir -p "$tmp/src/hw/display/r300"
    for f in ppc_mac_gpu_vulkan.c ppc_mac_gpu_renderer.h r300/r300_draw.h r300/r300_us.h \
             r300/r300_state.h r300/r300_spirv.h; do
        git -C "$src" show "$REV:hw/display/$f" > "$tmp/src/hw/display/$f"
    done
    echo "renderer source: $REV"
    src=$tmp/src
    unset REV
fi
COV=1 CC=${CC:-clang} "$here/build.sh" "$src"
export LLVM_PROFILE_FILE="$tmp/vk-%m.profraw"
"$here/test_vk" > "$tmp/plain.txt" 2>&1 || true
"$here/test_vk" -i > "$tmp/interlock.txt" 2>&1 || true
tail -1 "$tmp/plain.txt"; tail -1 "$tmp/interlock.txt"
tool llvm-profdata merge -o "$tmp/vk.profdata" "$tmp"/vk-*.profraw
tool llvm-cov report "$here/test_vk" -instr-profile="$tmp/vk.profdata" \
    "$src/hw/display/ppc_mac_gpu_vulkan.c" 2>/dev/null | sed -n '1p;/ppc_mac_gpu_vulkan/p'
# lines with an execution count of 0
tool llvm-cov show "$here/test_vk" -instr-profile="$tmp/vk.profdata" \
    "$src/hw/display/ppc_mac_gpu_vulkan.c" 2>/dev/null |
    awk -F'|' '$2 ~ /^ *0$/ { print $1 "|" $3 }' > "$here/uncovered.txt"
echo "$(wc -l < "$here/uncovered.txt") renderer lines never executed: $here/uncovered.txt"
