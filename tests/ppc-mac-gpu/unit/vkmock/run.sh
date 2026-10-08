#!/bin/sh
# run.sh [SRCDIR]: build (build.sh) and run everything, for the plain device
# and for one with the ordered pixel interlock.  Exit status 0 only if all of
# it passed.
#   REV=<revision>   test that revision's renderer source, not the work tree's
#   SAN=1            build with the address and undefined-behaviour sanitizers
# On the test branch the work tree's ppc_mac_gpu_vulkan.c carries test hooks:
# use REV to test the file that is to be merged.
here=$(cd "$(dirname "$0")" && pwd)
"$here/build.sh" "$@" || exit 2
rc=0
echo "== plain device"
"$here/test_vk" || rc=1
echo "== device with the pixel interlock"
"$here/test_vk" -i || rc=1
[ $rc -eq 0 ] && echo "ALL PASSED" || echo "FAILED"
exit $rc
