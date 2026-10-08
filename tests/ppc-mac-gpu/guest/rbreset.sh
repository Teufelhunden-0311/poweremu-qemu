#!/bin/zsh
# rbreset.sh TAG: hold transfer 6 in its first wait (test hook), issue a real
# system_reset from the monitor meanwhile, and record what the device did.
W=~/ppcosxkvm-work/rbprobe
cd /Users/Daniel/ppcosxkvm; TAG=$1; shift; O=$W/vm-$TAG; mkdir -p $O
env R300_SYSRT_PREWAIT=6:12000 ./ppcosx run --monitor --snapshot --attach-dvd $W/rbprobe.iso "$@" > $O/run.out 2>&1 &
until tools/vmctl.py cmd 'info status' 2>/dev/null | grep -q running; do sleep 2; done
tools/vmctl.py cmd 'set_link sungem.0 off' >/dev/null
n=0; prev=; same=0; while [ $n -lt 40 ]; do sleep 8; n=$((n+1)); tools/vmctl.py shot $O/boot.png; h=$(md5 -q $O/boot.png); if [ "$h" = "$prev" ]; then same=$((same+1)); else same=0; fi; prev=$h; [ $n -gt 6 ] && [ $same -ge 3 ] && break; done
tools/vmctl.py click 750 728; sleep 6
L0=$(wc -l < vm/gpu-trace.log)
tools/vmctl.py type "clear; (/Volumes/RBPROBE/rbprobe < /dev/null > /tmp/rbprobe.out 2>&1 &)
" >/dev/null
t=0; until sed -n "$L0,\$p" vm/gpu-trace.log | grep -q "RBTEST prewait transfer" || [ $t -ge 60 ]; do sleep 0.5; t=$((t+1)); done
sleep 2; echo "reset sent at $(date +%H:%M:%S)" > $O/events.txt
tools/vmctl.py cmd 'system_reset' >/dev/null
sleep 20; tools/vmctl.py cmd 'info status' 2>/dev/null | tr -d '\r' | grep -a "VM status" >> $O/events.txt
tools/vmctl.py shot $O/after20.png
sleep 70; tools/vmctl.py shot $O/after90.png
tools/vmctl.py cmd 'info status' 2>/dev/null | tr -d '\r' | grep -a "VM status" >> $O/events.txt
pgrep -x qemu-system-ppc >/dev/null && echo "qemu alive" >> $O/events.txt
sed -n "$L0,\$p" vm/gpu-trace.log | grep -E "RBTEST|system-memory|system memory|reset" > $O/trace.txt
tools/vmctl.py cmd quit >/dev/null 2>&1; until ! pgrep -x qemu-system-ppc >/dev/null; do sleep 1; done
echo RBRESET-DONE $TAG
