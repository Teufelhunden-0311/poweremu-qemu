#!/bin/zsh
# rbvklost.sh TAG: Vulkan; while the probe reads back, make a fence wait report
# DEVICE_LOST (test hook), then ask the guest to shut down and see whether it does.
W=~/ppcosxkvm-work/rbprobe
cd /Users/Daniel/ppcosxkvm; TAG=$1; O=$W/vm-$TAG; mkdir -p $O; rm -f /tmp/rbtest-wait-lost
env R300_VK_FAIL=wait R300_SYSRT_VK=wait ./ppcosx run --monitor --snapshot --attach-dvd $W/rbprobe.iso --gpu vulkan > $O/run.out 2>&1 &
until tools/vmctl.py cmd 'info status' 2>/dev/null | grep -q running; do sleep 2; done
tools/vmctl.py cmd 'set_link sungem.0 off' >/dev/null
n=0; prev=; same=0; while [ $n -lt 40 ]; do sleep 8; n=$((n+1)); tools/vmctl.py shot $O/boot.png; h=$(md5 -q $O/boot.png); if [ "$h" = "$prev" ]; then same=$((same+1)); else same=0; fi; prev=$h; [ $n -gt 6 ] && [ $same -ge 3 ] && break; done
tools/vmctl.py click 750 728; sleep 6
L0=$(wc -l < vm/gpu-trace.log)
tools/vmctl.py type "clear; (/Volumes/RBPROBE/rbprobe < /dev/null > /tmp/rbprobe.out 2>&1 &)
" >/dev/null
sleep 4; echo "probe started $(date +%H:%M:%S)" > $O/events.txt
sleep 25; tools/vmctl.py shot $O/shot.ppm; python3 $W/readstrip.py $O/shot.ppm > $O/read.txt 2>&1
sleep 30; tools/vmctl.py shot $O/later.png
tools/vmctl.py cmd 'info status' 2>/dev/null | tr -d '\r' | grep -a "VM status" >> $O/events.txt
tools/vmctl.py type "sudo shutdown -h now
" >/dev/null; sleep 3; tools/vmctl.py type "test
" >/dev/null
echo "shutdown typed $(date +%H:%M:%S)" >> $O/events.txt
t=0; while pgrep -x qemu-system-ppc >/dev/null && [ $t -lt 150 ]; do sleep 2; t=$((t+2)); done
if pgrep -x qemu-system-ppc >/dev/null; then echo "guest did NOT power off within 150 s" >> $O/events.txt; tools/vmctl.py shot $O/stuck.png; tools/vmctl.py cmd quit >/dev/null 2>&1; else echo "guest powered off after $t s (QEMU exited)" >> $O/events.txt; fi
until ! pgrep -x qemu-system-ppc >/dev/null; do sleep 1; done
sed -n "$L0,\$p" vm/gpu-trace.log | grep -E "RBTEST|system-memory|vulkan" > $O/trace.txt; grep -i "vulkan:" $O/run.out | tail -5 >> $O/trace.txt
rm -f /tmp/rbtest-wait-lost; echo RBVKLOST-DONE $TAG
