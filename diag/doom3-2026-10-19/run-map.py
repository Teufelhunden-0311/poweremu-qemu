# run-map.py OUTDIR [MAXSECS]: run.py plus guest kextstat + vmmap of Doom 3 (at ~172 s),
# sent to the host on port 5556 after the demo.  Diagnostic (tbmap) runs only.
# Was: Doom 3 timedemo demo1 in the running VM; screenshots every 20 s.
# The game is started with stdin from /dev/null: from a Terminal it otherwise
# blocks at "Initializing menus".  D3ARGS (env) adds game arguments.
import subprocess, time, sys, os
EXTRA = os.environ.get("D3ARGS", "")  # extra game args, e.g. "+set r_screenFraction 50"
S = sys.argv[1]; SECS = int(sys.argv[2]) if len(sys.argv) > 2 else 900
def v(*a): subprocess.run(["python3", "tools/vmctl.py", *a], check=True, stdout=subprocess.DEVNULL)
def shot():
    v("shot", S + "/b.ppm"); d = open(S + "/b.ppm", "rb").read().split(b"\n", 3)
    return int(d[1].split()[0]), d[3]
def desk():
    try:
        w, p = shot(); o = (100 * w + 250) * 3
        return p[o] > 240 and p[o + 2] > 240
    except Exception: return False
while not desk(): time.sleep(4)
v("cmd", "set_link sungem.0 off"); v("click", "250", "300"); time.sleep(2)
v("type", " kextstat > /tmp/kx.txt 2>&1; which vmmap > /tmp/vm.txt 2>&1; cd '/Applications/Doom 3'; (sleep 172; vmmap $(ps -axco pid,command | awk '/Doom 3/{print $1}') >> /tmp/vm.txt 2>&1) & ./Doom\\ 3.app/Contents/MacOS/Doom\\ 3 +set com_skipIntroVideos 1 " + EXTRA + " +timedemoquit demo1 < /dev/null > /tmp/td.out 2>&1; grep -a -i 'frames rendered' /tmp/td.out; echo TD-DONE")
v("key", "ret")
t0 = time.time(); i = 0; last = None; same = 0
while time.time() - t0 < SECS:
    time.sleep(20); i += 1
    w, p = shot()
    v("shot", "%s/s%02d.png" % (S, i))
    if last is not None and len(last) == len(p):
        diff = sum(1 for k in range(0, len(p), 997) if p[k] != last[k])
        same = same + 1 if diff < 4 else 0
    last = p
    print(i, int(time.time() - t0), same, flush=True)
    # also require the Terminal back (a static game screen must not end the run)
    if same >= 4 and time.time() - t0 > 240 and desk(): break
v("cmd", "set_link sungem.0 on"); time.sleep(5)
v("type", " (cat /tmp/td.out; echo ==KX; cat /tmp/kx.txt; echo ==VM; cat /tmp/vm.txt) | nc 10.0.2.2 5556; echo SENT"); v("key", "ret")
time.sleep(20); v("shot", S + "/sent.png")
v("cmd", "quit")
print("END", flush=True)
