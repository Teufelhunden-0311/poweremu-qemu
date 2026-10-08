#!/bin/zsh
# rbvkdummy.sh: Vulkan under the Khronos validation layer with synchronization
# validation; the stand-in textures' initialization is cancelled (the batch not
# submitted) or interrupted (one image's creation fails), via the test hooks.
# Each run is checked; exit status is non-zero if any check fails.
# The layer manifest in $VKL must name the layer library by full path.
W=~/ppcosxkvm-work/rbprobe; VKL=${VKL:-$HOME/ppcosxkvm-work/vklayer}
L="VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation VK_LAYER_PATH=$VKL VK_LOADER_DEBUG=error,layer VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT"
ZERO='2D 00000000 3D 00000000 cube face 5 00000000'
bad=0
# run TAG WANT ENV...   WANT: clean | layout | hazard, then log lines that must be present
run() { tag=$1; want=$2; shift 2; need=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do need+=("$1"); shift; done; shift
  O=$W/vm-$tag; rm -rf $O; fail=()
  $W/rbvm.sh $tag ${=L} "$@" -- --gpu vulkan > $O.runner 2>&1; rc=$?
  grep -a RBTEST ~/ppcosxkvm/vm/gpu-trace.log > $O/rbtest.txt
  grep -a -A1 'Validation Error' $O/run.out | grep -a -v '^--' > $O/validation.txt
  ne=$(grep -a -c 'Validation Error' $O/validation.txt); nu=$(grep -a -c 'layout is VK_IMAGE_LAYOUT_UNDEFINED' $O/validation.txt); nh=$(grep -a -c 'SYNC-HAZARD' $O/validation.txt)
  [ $rc -eq 0 ] || fail+=("rbvm status $rc")
  grep -a -q 'checksum ok' $O/read.txt 2>/dev/null || fail+=("strip not ok")
  grep -a -q 'Insert instance layer "VK_LAYER_KHRONOS_validation"' $O/run.out || fail+=("validation layer not loaded")
  case $want in
    clean)  [ $ne -eq 0 ] || fail+=("$ne validation errors, want 0") ;;
    layout) [ $nu -ge 1 ] || fail+=("control: no UNDEFINED-layout error") ;;
    hazard) [ $nh -ge 1 ] || fail+=("control: no SYNC-HAZARD error") ;;
  esac
  for n in $need; do grep -a -q -F -- "$n" $O/rbtest.txt || fail+=("missing: $n"); done
  echo "== $tag ($want): validation errors $ne (undefined layout $nu, sync hazard $nh) -> ${${fail:+FAIL}:-PASS}"
  sed 's/^/   /' $O/rbtest.txt
  for f in $fail; do echo "   FAIL: $f"; bad=1; done
  sed -E 's/0x[0-9a-f]+/H/g' $O/validation.txt | grep -a -v 'Validation Error' | cut -c1-230 | sort | uniq -c | sort -rn | head -4 | sed 's/^/     /'; }
run VD-base   clean  --
run VD-host1  clean  "init 2 recorded in batch 2 (0 images created); contents copied out" "$ZERO" -- R300_VK_FAIL=dummy:-1:1
run VD-dev3   clean  "init 4 recorded in batch 4 (0 images created); contents copied out" "$ZERO" -- R300_VK_FAIL=dummy:-2:3
run VD-nofix  layout -- R300_VK_FAIL=dummy:-1:1 RBTEST_NO_DUMMY_FIX=1
run VD-alloc1 clean  "creating image 1 fails; 1 created, 0 initializations recorded" "2 created, 3 initializations recorded" "$ZERO" -- R300_VK_FAIL=dummyalloc:1
run VD-alloc2 clean  "creating image 2 fails; 2 created, 0 initializations recorded" "1 created, 3 initializations recorded" "$ZERO" -- R300_VK_FAIL=dummyalloc:2
# the retry in the SAME open batch (forced by the hook), fixed order then the old order
run VD-same1  clean  "creating image 1 fails; 1 created, 0 initializations recorded, open batch 1" "batch open before recording 1" "3 initializations recorded in batch 1" "$ZERO" -- R300_VK_FAIL=dummyalloc:1 RBTEST_DUMMY_SAMEBATCH=1
run VD-same2  clean  "creating image 2 fails; 2 created, 0 initializations recorded, open batch 1" "batch open before recording 1" "3 initializations recorded in batch 1" "$ZERO" -- R300_VK_FAIL=dummyalloc:2 RBTEST_DUMMY_SAMEBATCH=1
run VD-old1   hazard "creating image 1 fails; 1 created, 1 initializations recorded, open batch 1" "initializations recorded in batch 1" -- R300_VK_FAIL=dummyalloc:1 RBTEST_DUMMY_SAMEBATCH=1 RBTEST_DUMMY_INTERLEAVED=1
run VD-old2   hazard "creating image 2 fails; 2 created, 2 initializations recorded, open batch 1" "initializations recorded in batch 1" -- R300_VK_FAIL=dummyalloc:2 RBTEST_DUMMY_SAMEBATCH=1 RBTEST_DUMMY_INTERLEAVED=1
[ $bad -eq 0 ] && echo "VD-RESULT all checks passed" || echo "VD-RESULT FAILED"
exit $bad
