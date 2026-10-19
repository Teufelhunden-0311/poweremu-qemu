#!/bin/zsh
# q32.sh: Codex 2026-10-19 CP park comparison under waitprof2 (bf247d1): 250 then 50.
SW="PPC_SP_BRANCH=1 PPC_SP_INPUTS=1 PPC_SP_PROV=2 PPC_SP_FWD=1 PPC_SP_SHADOW_DSE=1 PPC_NATIVE_FP=1 PPC_STRICT_FP=0 PPC_NATIVE_LFS=1 PPC_NATIVE_STFS=1 PPC_NATIVE_FRSP=1 PPC_SP_SHADOW=2 PPC_SP_SHADOW_STATS=0"
waitload() { until (( $(sysctl -n vm.loadavg | awk '{print int($2)}') < 4 )); do python3 -c "import time; time.sleep(15)"; done; }
for p in 250 50; do
  waitload; echo "start p$p $(date +%T) load $(sysctl -n vm.loadavg)"
  ~/ppcosxkvm-work/d3/d3.sh q32p$p PPCOSX_QEMU=$HOME/ppcosxkvm-work/bins/qemu-waitprof2 PPC_WAIT_PROF=1 PPCGPU_CP_PARK_US=$p ${=SW} 2>&1 | tail -4
done
echo ALL-DONE
