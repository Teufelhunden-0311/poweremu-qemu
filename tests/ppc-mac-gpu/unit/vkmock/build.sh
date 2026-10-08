#!/bin/sh
# build.sh [SRCDIR]: build test_vk against SRCDIR/hw/display/ppc_mac_gpu_vulkan.c
# (default: the tree this directory is in, or $QEMU_SRC).
# With REV=<git revision>, against that revision's renderer source instead
# (taken from the repository SRCDIR is in), so the code under test is the
# committed file and not whatever the work tree holds.
set -e
here=$(cd "$(dirname "$0")" && pwd)
src=${1:-${QEMU_SRC:-$here/../../../..}}
if [ -n "$REV" ]; then
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    mkdir -p "$tmp/hw/display/r300"
    for f in ppc_mac_gpu_vulkan.c ppc_mac_gpu_renderer.h r300/r300_draw.h r300/r300_us.h \
             r300/r300_state.h r300/r300_spirv.h; do
        git -C "$src" show "$REV:hw/display/$f" > "$tmp/hw/display/$f"
    done
    echo "renderer source: $REV, blob $(git -C "$src" rev-parse "$REV:hw/display/ppc_mac_gpu_vulkan.c")"
    src=$tmp
else
    echo "renderer source: $src (work tree), git blob $(git hash-object "$src/hw/display/ppc_mac_gpu_vulkan.c")"
fi
[ -f "$src/hw/display/ppc_mac_gpu_vulkan.c" ] || { echo "no renderer source under $src" >&2; exit 2; }
: "${CC:=cc}"
vkinc=$(pkg-config --cflags vulkan 2>/dev/null || true)
[ -n "$vkinc" ] || { [ -d /opt/homebrew/include/vulkan ] && vkinc=-I/opt/homebrew/include; }
# SAN=1: with the address and undefined-behaviour sanitizers
[ -z "$SAN" ] || san="-fsanitize=address,undefined -fno-omit-frame-pointer"
# COV=1: with clang's source coverage (coverage.sh)
[ -z "$COV" ] || san="$san -fprofile-instr-generate -fcoverage-mapping -DVKMOCK_COVERAGE"
$CC -std=gnu11 -O1 -g $san -Wall -Wno-unused-function -Wno-unused-variable \
    -I"$here/shim" -I"$src/hw/display" $vkinc $(pkg-config --cflags glib-2.0) \
    "$here/test_vk.c" "$here/mockvk.c" -o "$here/test_vk" \
    $(pkg-config --libs glib-2.0) -lpthread
