# Tests for the R300 address rules and system-memory colour buffers

Review material for the `rb-gart` branch.  Not wired into the build, and not
meant to be merged as it is.

## unit/ — host test of the address helpers and the AGP/GART page cache

Compiled from the real source: `extract.sh` copies the helpers
(`r300_span` … `r300_sysmem_rw`) and the page cache (`r200_agp_tc` …
`r200_agp_page`) verbatim out of `hw/display/ppc_mac_gpu.c` into
`helpers.inc` / `cache.inc`, and `test_mem.c` includes them over a mock
device (two apertures, two PCI GART tables, an AGP table, mock guest RAM).

    cd tests/ppc-mac-gpu/unit
    ./extract.sh ../../../hw/display/ppc_mac_gpu.c
    cc -std=gnu11 -O2 -Wall -Wno-unused-function -o test_mem test_mem.c
    ./test_mem          # "54 checks, 0 failed"
    ./test_mem bench    # ns per cached 12-byte AGP read

Covers: range edges, wrap, empty/reversed ranges; MC_FB_LOCATION 0xffff0000
and top < base; VRAM smaller than reported; an AIC end inside a page; VRAM
overlapping an aperture; overlapping apertures (AGP wins; an invalid AGP entry
is not served by the PCI GART); non-contiguous pages with an unaligned start;
VRAM->GART spans; the 4 GB wrap; the page cache across a table-base change,
PCI GART off/on, a moved aperture, an AGP range that comes to cover the
address, a bridge GART generation change; an unaligned PCI GART base.

## guest/ — the probe run inside Mac OS X Tiger

`rbprobe.c` (GLUT; build on Tiger with
`gcc -std=gnu99 -O2 -o rbprobe rbprobe.c -framework GLUT -framework OpenGL`)
draws exact pixel patterns, reads them back in many ways, writes
`/tmp/rbprobe.txt`, and shows the same text as a strip of grey blocks (0..255
calibration run, length, text, checksum) for `readstrip.py` to decode from a
lossless screen dump -- needed while readback itself is what is broken.
Modes: `rbprobe`, `rbprobe mask`, `rbprobe cap`.
`powerbook-rv360-reference.txt` is its output on a PowerBook G4 (Mobility
Radeon 9700, RV360, ATI-1.4.18).  The scripts drive a VM through ppcosxkvm's
`tools/vmctl.py` (paths are the author's): `rbvm.sh TAG [ENV=..] [-- args]`,
`rbreset.sh` (real system_reset during a transfer's first wait),
`rbvklost.sh` (Vulkan device lost during a transfer, then guest shutdown),
`rbvkdummy.sh` (Vulkan under the Khronos validation layer: the batch carrying
the stand-in textures' initialization is not submitted, once and three times
running; `VD-nofix` is the negative control).  `readstrip.py` exits non-zero
on a bad strip checksum and `rbvm.sh` then reports the run as failed.

## Test-only hooks (the commit after this one; never merge)

    R300_SYSRT_TEST=mask        every staged draw: left half of the target, R channel only
    R300_SYSRT_PREWAIT=N:ms     hold transfer N in its first wait, BQL released, for ms
    R300_SYSRT_FAULT=remap      alter the saved translation of every 2nd transfer
                                (exercises the comparison only, not a real remap)
    R300_METAL_FAIL=k           every k-th earlier Metal batch reads as failed where
                                its status is read
    R300_VK_FAIL=submit:N:code  every N-th vkQueueSubmit is not made and returns code
    R300_SYSRT_VK=submit:code   the submission of every 2nd transfer's batch is not made
    R300_VK_FAIL=wait + R300_SYSRT_VK=wait
                                transfer 5's fence wait reports DEVICE_LOST (after
                                really waiting)
    R300_VK_FAIL=dummy:code:reps
                                the batch carrying each of the first reps stand-in
                                initializations is not submitted (returns code); after
                                the next one the three images are copied out and logged
                                (they get TRANSFER_SRC usage for that)
    RBTEST_NO_DUMMY_FIX=1       with the above: a cancelled batch does not clear
                                dummy_ready (the code before the fix), and the images
                                are copied out as they are
