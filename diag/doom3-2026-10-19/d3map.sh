#!/bin/zsh
# d3map.sh TAG [ENV=..]: diagnostic Doom 3 demo1 run with the tbmap binary (QEMU_TBMAP log of
# host code -> guest PC) and three 10 s `sample` captures at ~60/100/140 s, as d3prof.sh.
# The guest's kextstat, vmmap of Doom 3 and td.out arrive on port 5556 (guest.txt).
# Not a timing run.  Output in /Volumes/Claude/d3-ab/TAG.
TAG=$1; shift; O=/Volumes/Claude/d3-ab/$TAG; mkdir -p $O; rm -f $O/*(N)
now() { python3 -c 'import time; print("%.3f" % time.time())'; }
cd ~/ppcosxkvm
nc -l 5556 > $O/guest.txt &
NC=$!
(env QEMU_TBMAP=$O/tbmap.txt "$@" ./ppcosx run --monitor --snapshot --attach-dvd /Volumes/Claude/from-Downloads/Doom_3.cdr > $O/vm.out 2>&1 &)
python3 -c "import time; time.sleep(2)"
python3 ~/ppcosxkvm-work/d3/run-map.py $O > $O/run.out 2>&1 &
R=$!
until grep -q "^3 " $O/run.out 2>/dev/null; do sleep 2; kill -0 $R 2>/dev/null || break; done
QP=$(pgrep -f "^[^ ]*(bins/qemu-[^ ]*|qemu-system-ppc) " | head -1)
for k in 1 2 3; do
    echo "sample $k start $(now)" >> $O/times.txt
    sample $QP 10 -file $O/sample-$k.txt > /dev/null 2>&1
    echo "sample $k end $(now) ($(tail -1 $O/run.out))" >> $O/times.txt
    sleep 30
done
wait $R
until ! pgrep -f "^[^ ]*(bins/qemu-[^ ]*|qemu-system-ppc) " >/dev/null; do python3 -c "import time; time.sleep(1)"; done
kill $NC 2>/dev/null
cp ~/ppcosxkvm/vm/gpu-trace.log $O/qemu.log 2>/dev/null
echo "== $O"; tail -3 $O/run.out; grep -a -i 'frames rendered' $O/guest.txt; grep -c '^T ' $O/tbmap.txt; grep '^F ' $O/tbmap.txt | head
