#!/bin/zsh
# q41.sh: Codex relay 2026-10-21: one instrumented demo1 run with PPC_WAIT_PROF=1 on idlectx
# (waitprof 31e0c15 + idle-entry NIP/LR/r1/MSR buckets).  Standard switches, slots off,
# counters off, fresh --snapshot VM.  Gate: CPU >= 80% idle.
SW=(PPC_SP_BRANCH=1 PPC_SP_INPUTS=1 PPC_SP_PROV=2 PPC_SP_FWD=1 PPC_SP_SHADOW_DSE=1 PPC_NATIVE_FP=1 PPC_STRICT_FP=0 PPC_NATIVE_LFS=1 PPC_NATIVE_STFS=1 PPC_NATIVE_FRSP=1 PPC_SP_SHADOW=2 PPC_SP_SHADOW_STATS=0)
unset PPC_TLBIE QEMU_PERF_COUNTERS PPC_SR_SLOTS
idle() { top -l 2 -n 0 -s 5 | grep 'CPU usage' | tail -1 | awk '{print int($7)}'; }
shasum -a 256 ~/ppcosxkvm-work/bins/qemu-idlectx; env | grep -E '^(PPC_|QEMU_)' || echo "no PPC_/QEMU_ in parent env"
until (( $(idle) >= 80 )); do python3 -c "import time; time.sleep(15)"; done
echo "start q41ctx $(date +%T) $(top -l 2 -n 0 -s 2 | grep 'CPU usage' | tail -1) | $(sysctl -n vm.loadavg)"
~/ppcosxkvm-work/d3/d3.sh q41ctx PPCOSX_QEMU=$HOME/ppcosxkvm-work/bins/qemu-idlectx PPC_WAIT_PROF=1 $SW | tail -3
echo QUEUE41-DONE
