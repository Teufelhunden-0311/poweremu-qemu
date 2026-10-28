#!/usr/bin/env python3
"""tests/ppc-mmu/run.py QEMU [CONFIG...]: build the bare-metal MMU harness and run it under
qemu-system-ppc QEMU (-M mac99 -cpu 7400), in each configuration: base (whole-TLB tlbie),
class (PPC_TLBIE=class), each with PPC_SR_SLOTS off and on.  Exit status 0 if every test
passed in every configuration.  AI-generated (Claude)."""
import os, re, socket, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
LLVM = os.environ.get("LLVM_BIN", "/opt/homebrew/opt/llvm/bin")
LLD = os.environ.get("LLD", "/opt/homebrew/bin/ld.lld")
NAMES = ["class_pair", "other_class(info)", "alias_segments", "alias_sets", "perm_store",
         "code_remap", "code_fallthrough", "perm_fetch", "change_bit", "bat", "pending_sr",
         "local_first", "many_classes", "fill_pending", "victim"]
INFO = 1                                   # t_other_class: 1 = old mapping, 2 = new
CONFIGS = {
    "base":       {},
    "base+slots": {"PPC_SR_SLOTS": "1"},
    "class":      {"PPC_TLBIE": "class"},
    "class+slots": {"PPC_TLBIE": "class", "PPC_SR_SLOTS": "1"},
}


def build(out):
    cflags = ["--target=powerpc-unknown-eabi", "-mcpu=7400", "-mno-altivec", "-msoft-float",
              "-O2", "-ffreestanding", "-fno-builtin", "-fno-common", "-nostdlib",
              "-fno-pic", "-Wall"]
    objs = []
    for src in ["start.S", "mmu.c"]:
        o = os.path.join(out, src + ".o")
        subprocess.run([LLVM + "/clang", *cflags, "-c", os.path.join(HERE, src), "-o", o],
                       check=True)
        objs.append(o)
    elf = os.path.join(out, "mmu.elf")
    subprocess.run([LLD, "-T", os.path.join(HERE, "link.ld"), "-o", elf, *objs], check=True)
    sz = subprocess.run([LLVM + "/llvm-size", "-A", elf], check=True, capture_output=True,
                        text=True).stdout
    m = re.search(r"^\.data\s+(\d+)", sz, re.M)
    if m and int(m.group(1)):
        sys.exit("harness has writable data in ROM:\n" + sz)
    return elf


def hmp(sock, cmd):
    sock.sendall((cmd + "\n").encode())
    buf = b""
    while not buf.rstrip().endswith(b"(qemu)"):
        buf += sock.recv(65536)
    return buf.decode(errors="replace")


def words(text):
    return [int(w, 16) for line in text.splitlines() if re.match(r"\s*[0-9a-f]+:", line)
            for w in line.split(":", 1)[1].split()]


def run(qemu, elf, env, out, tag):
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    log = os.path.join(out, tag + ".log")
    e = dict(os.environ)
    for k in ("PPC_TLBIE", "PPC_SR_SLOTS"):
        e.pop(k, None)
    e.update(env)
    p = subprocess.Popen([qemu, "-M", "mac99", "-cpu", "7400", "-m", "256", "-bios", elf,
                          "-display", "none", "-vga", "none", "-nodefaults",
                          "-audio", "none",
                          "-monitor", "tcp:127.0.0.1:%d,server,nowait" % port,
                          "-d", "in_asm", "-D", log], env=e)
    try:
        s = None
        for _ in range(100):
            try:
                s = socket.create_connection(("127.0.0.1", port))
                break
            except OSError:
                if p.poll() is not None:
                    break
                time.sleep(0.1)
        if s is None:
            sys.exit("%s: QEMU did not start (exit %s)" % (tag, p.poll()))
        buf = b""
        while not buf.rstrip().endswith(b"(qemu)"):    # the banner and first prompt
            buf += s.recv(65536)
        t0 = time.time()
        while True:
            r = words(hmp(s, "xp /24wx 0x4000"))
            if r[0] == 0x4d4d5544 or time.time() - t0 > 20:
                break
            time.sleep(0.2)
        f = words(hmp(s, "xp /8wx 0x3f00"))
        s.sendall(b"quit\n")
    finally:
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            p.kill()
    return r, f, log


def split_ok(log):
    """Code that runs off page 0x30109000 into 0x3010a000: no TB holds both."""
    blocks = open(log, errors="replace").read().split("IN:")
    seen = False
    for b in blocks:
        a = re.findall(r"^0x([0-9a-f]{8}):", b, re.M)
        if "30109ff8" in a:
            seen = True
            if any(x >= "3010a000" for x in a):
                return False
    return seen


def main():
    qemu = sys.argv[1]
    want = sys.argv[2:] or list(CONFIGS)
    out = tempfile.mkdtemp(prefix="ppc-mmu-")
    elf = build(out)
    ok = True
    for tag in want:
        r, f, log = run(qemu, elf, CONFIGS[tag], out, tag)
        done = r[0] == 0x4d4d5544
        n = r[1]
        res = r[4:4 + n]
        line = []
        for i, v in enumerate(res):
            if i == INFO:
                exp = 1 if tag.startswith("class") else 2
                good = v == exp
                line.append("%s=%s" % (NAMES[i], {1: "old", 2: "new"}.get(v, hex(v))))
            else:
                good = v == 0
                if not good:
                    line.append("%s=FAIL(%#x)" % (NAMES[i], v))
            ok &= good
        sp = split_ok(log)
        ok &= done and sp and not r[3]
        print("%-12s %s  passed %d/%d  %s  page-split %s%s" % (
            tag, "done" if done else "NOT DONE (fault record %s)" % [hex(x) for x in f[:7]],
            r[2], n, " ".join(line), "ok" if sp else "NOT SEEN",
            "  harness error %#x" % r[3] if r[3] else ""))
    print("ALL PASS" if ok else "FAILURES", "(%s)" % out)
    sys.exit(0 if ok else 1)


main()
