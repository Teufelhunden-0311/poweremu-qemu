#!/bin/zsh
# rbvkwait.sh: Vulkan on a real driver, under the Khronos validation layer with
# synchronization validation, with waits for the GPU that give no answer: the
# test hook returns an error (or VK_TIMEOUT) WITHOUT waiting, several times
# running, for every N-th batch waited for -- so the GPU really may still be
# at work when the renderer is told nothing.
# With the fix the probe's readbacks must equal the clean run's and the layer
# must stay silent.  WB-old is the negative control: the unanswered wait is
# taken for a finished batch (as before the fix, but rendering goes on, so
# that the consequence shows) and must produce wrong readbacks or validation
# errors.  Each run is checked; exit status is non-zero if any check fails.
# The layer manifest in $VKL must name the layer library by full path.
W=~/ppcosxkvm-work/rbprobe; VKL=${VKL:-$HOME/ppcosxkvm-work/vklayer}
L="VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation VK_LAYER_PATH=$VKL VK_LOADER_DEBUG=error,layer VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT"
bad=0
results() { grep -a -v 'ms px' $1/read.txt 2>/dev/null; }
# run TAG WANT ENV...   WANT: base | same (as base, layer silent, waits unanswered) | control
run() { tag=$1; want=$2; shift 2
  O=$W/vm-$tag; rm -rf $O; fail=()
  $W/rbvm.sh $tag ${=L} "$@" -- --gpu vulkan > $O.runner 2>&1; rc=$?
  grep -a RBTEST ~/ppcosxkvm/vm/gpu-trace.log > $O/rbtest.txt
  grep -a -A1 'Validation Error' $O/run.out | grep -a -v '^--' > $O/validation.txt
  ne=$(grep -a -c 'Validation Error' $O/validation.txt)
  nw=$(grep -a 'without waiting' $O/rbtest.txt | tail -1 | sed -E 's/.*\(([0-9]+) such so far\).*/\1/'); nw=${nw:-0}
  na=$(grep -a -c 'no answer from the GPU' $O/run.out); ns=$(grep -a -c '3D rendering has stopped' $O/run.out)
  grep -a -q 'Insert instance layer "VK_LAYER_KHRONOS_validation"' $O/run.out || fail+=("validation layer not loaded")
  if [ $want != control ]; then
    [ $rc -eq 0 ] || fail+=("rbvm status $rc")
    grep -a -q 'checksum ok' $O/read.txt 2>/dev/null || fail+=("strip not ok")
    [ $ne -eq 0 ] || fail+=("$ne validation errors, want 0")
    [ $ns -eq 0 ] || fail+=("rendering stopped")
  fi
  same=no; [ $want = base ] || { results $O | diff -q - <(results $W/vm-WB-base) >/dev/null 2>&1 && same=yes; }
  case $want in
    base)    [ $nw -eq 0 ] || fail+=("waits were made to fail in the base run") ;;
    same)    [ $same = yes ] || fail+=("readbacks differ from the base run")
             [ $nw -ge 100 ] || fail+=("only $nw waits were left unanswered")
             [ $na -ge 1 ] || fail+=("the renderer did not say that it had no answer") ;;
    control) [ $nw -ge 100 ] || fail+=("only $nw waits were left unanswered")
             [ $same = no ] || [ $ne -ge 1 ] || fail+=("control: readbacks as the base run's and no validation error") ;;
  esac
  echo "== $tag ($want): waits left unanswered $nw, 'no answer' logged $na, validation errors $ne, readbacks as base: $same, rbvm status $rc -> ${${fail:+FAIL}:-PASS}"
  for f in $fail; do echo "   FAIL: $f"; bad=1; done
  sed -E 's/0x[0-9a-f]+/H/g; s/\[[0-9]+\]/[N]/g' $O/validation.txt | grep -a -v 'Validation Error' | cut -c1-200 | sort | uniq -c | sort -rn | head -4 | sed 's/^/     /'
  [ $want = control ] && results $O | diff - <(results $W/vm-WB-base) | grep -a '^<' | grep -a -m3 'miss' | cut -c1-150 | sed 's/^/     /'; }
run WB-base    base
run WB-dev     same    R300_VK_FAIL=waitbusy:3:-2:4
run WB-host    same    R300_VK_FAIL=waitbusy:2:-1:8
run WB-timeout same    R300_VK_FAIL=waitbusy:2:2:6
run WB-old     control R300_VK_FAIL=waitbusy:3:-2:4 RBTEST_WAIT_NO_RETRY=1
[ $bad -eq 0 ] && echo "WB-RESULT all checks passed" || echo "WB-RESULT FAILED"
exit $bad
