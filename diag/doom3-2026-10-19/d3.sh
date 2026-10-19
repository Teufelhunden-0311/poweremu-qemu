#!/bin/zsh
# d3.sh TAG [ENV=..]: Doom 3 timedemo demo1 in a snapshot VM; output in /Volumes/Claude/d3-ab/TAG
O=/Volumes/Claude/d3-ab/$1; shift; mkdir -p $O
cd ~/ppcosxkvm
(env "$@" ./ppcosx run --monitor --snapshot --attach-dvd /Volumes/Claude/from-Downloads/Doom_3.cdr > $O/vm.out 2>&1 &)
python3 -c "import time; time.sleep(2)"
python3 ~/ppcosxkvm-work/d3/run.py $O > $O/run.out 2>&1
until ! pgrep -f "^[^ ]*(bins/qemu-[^ ]*|qemu-system-ppc) " >/dev/null; do python3 -c "import time; time.sleep(1)"; done
cp ~/ppcosxkvm/vm/gpu-trace.log $O/qemu.log 2>/dev/null
echo "== $O"; tail -3 $O/run.out; grep -a VSSTAT $O/vm.out | tail -6
