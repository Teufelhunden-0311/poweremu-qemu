#!/bin/zsh
# rbvm.sh TAG [ENV=VAL...] [-- launcher args]: boot Tiger (snapshot) with the
# rbprobe disc, run it, screenshot its results strip, decode it.
W=~/ppcosxkvm-work/rbprobe
cd /Users/Daniel/ppcosxkvm; TAG=$1; shift; O=$W/vm-$TAG; mkdir -p $O
envs=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done; [ "${1:-}" = "--" ] && shift
env $envs ./ppcosx run --monitor --snapshot --attach-dvd $W/rbprobe.iso "$@" > $O/run.out 2>&1 &
until tools/vmctl.py cmd 'info status' 2>/dev/null | grep -q running; do sleep 2; done
tools/vmctl.py cmd 'set_link sungem.0 off' >/dev/null
n=0; prev=; same=0; while [ $n -lt 40 ]; do sleep 8; n=$((n+1)); tools/vmctl.py shot $O/boot.png; h=$(md5 -q $O/boot.png); if [ "$h" = "$prev" ]; then same=$((same+1)); else same=0; fi; prev=$h; [ $n -gt 6 ] && [ $same -ge 3 ] && break; done
tools/vmctl.py click 750 728; sleep 6
L0=$(wc -l < vm/gpu-trace.log)
tools/vmctl.py type "clear; (/Volumes/RBPROBE/rbprobe ${RBARG:-} < /dev/null > /tmp/rbprobe.out 2>&1 &)
" >/dev/null
sleep 25; tools/vmctl.py shot $O/shot.ppm
sed -n "$L0,\$p" vm/gpu-trace.log > $O/trace.txt
[ -n "${KEEP:-}" ] && { echo KEPT; exit 0; }
tools/vmctl.py cmd quit >/dev/null 2>&1; until ! pgrep -x qemu-system-ppc >/dev/null; do sleep 1; done
python3 $W/readstrip.py $O/shot.ppm > $O/read.txt 2>&1 || { echo "RBVM-FAILED $TAG: strip rejected, see $O/read.txt"; exit 1; }
echo RBVM-DONE $TAG
