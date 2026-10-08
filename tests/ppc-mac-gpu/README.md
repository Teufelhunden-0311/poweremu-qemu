# Tests for the R300 address rules, system-memory colour buffers and Vulkan failures

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

## unit/vkmock/ — the Vulkan renderer when Vulkan calls fail

`hw/display/ppc_mac_gpu_vulkan.c` built as it is into a test program
(`test_vk.c` includes it), against stand-in QEMU headers (`shim/`, the few
macros and thread/atomic/bitmap calls that file uses) and a stand-in Vulkan
driver (`mockvk.c`).  It needs a C compiler, glib-2.0 and the Vulkan headers:
no QEMU build, no GPU, no Vulkan library.

    REV=<commit> tests/ppc-mac-gpu/unit/vkmock/run.sh    # that commit's renderer
    tests/ppc-mac-gpu/unit/vkmock/run.sh [SRCDIR]        # a work tree's
    SAN=1 ... run.sh                                     # with ASan and UBSan

On this branch the work tree's renderer carries the test hooks listed below;
use `REV` for the file that is to be merged.  `build.sh` prints the git blob
of the renderer it built.  `run.sh` runs everything twice, for the plain
device and for one that offers the ordered pixel interlock, and exits non-zero
unless all of it passed.

**The driver** hands out objects it keeps track of and can make any call that
returns a `VkResult` fail, leaving its outputs as the specification leaves
them: undefined -- a recognisable garbage value -- except where the command
promises a null handle (command buffers, descriptor sets, pipelines).  It
records a *violation* when

- a handle passed in is not a live object of the right type that it returned:
  VK_NULL_HANDLE where one is not allowed, the output of a failed call, an
  object already destroyed;
- an object is destroyed, or a descriptor pool reset, while a submitted command
  buffer that has not been waited for still uses it;
- a command is recorded into a command buffer that is not recording, a command
  buffer is submitted that was not ended successfully, begun or reset while
  pending, or ended with its render pass open; a copy, clear or layout
  transition is recorded inside a render pass;
- a fence is submitted while signaled or in use, reset while its work is
  unfinished, or waited for when nothing can signal it;
- a descriptor is written with a null or dead sampler, view or buffer, or a
  set is used after its pool was reset;
- on submission, when the commands are "executed" in order: an image is not in
  the layout a command names; an image that is read (sampled, loaded as an
  attachment, copied from) has a level, layer or slice never written since it
  last left VK_IMAGE_LAYOUT_UNDEFINED; a layout transition that names no source
  access follows a write to the image with no ordering barrier between;
- the renderer makes more than four million calls or 131072 objects in one run
  (it does not stop).

It does not model shaders (every descriptor of a bound set counts as used),
image contents, formats, queue families or most other valid-usage rules.
Submitted work counts as finished when a `vkWaitForFences` has seen its fence
and succeeded or reported the device lost; after a wait that failed otherwise
it is still in flight.
It is not a substitute for the validation layers on a real driver
(`guest/rbvkdummy.sh`).

**The program** (`test_vk`, no arguments) runs

- `selftest`: each kind of misuse above done to the driver directly, to see
  that it is reported, and a correct batch, to see that it is not;
- `helpers`: `vk_image` and `vk_buffer` with each of their calls failing in
  turn (outputs all null, only what was created destroyed, nothing left) and
  with no memory type fitting; `vk_dummies` with each creation call of each
  stand-in failing, retried in a later batch and in the same open batch, then
  used by a draw; more attachment layouts than render passes are kept;
  programs that do not compile (rejected, compiled once); draws refused for
  what they are (bad buffers, textures out of range ...); every batch in
  flight with a draw waiting for one, and a wait failing for lack of memory
  meanwhile; VRAM mapped off a page boundary at start-up;
- `sweep`: a scenario of some 500 draws (every texture path, lines, GPU vertex
  shading, several colour buffers, multisampling, a buffer rendered to and
  then sampled, batches in flight together, a descriptor pool running out,
  the image, framebuffer and texture caches overflowing, vertex data larger
  than an arena chunk, a texture upload crossing one) run from start-up in a
  fresh process for every case:
    - each call that returns a `VkResult` (31 of them, start-up included; all
      but `vkResetDescriptorPool`, which is specified to succeed) failing
      once, for every time the scenario makes it, and failing from then on;
    - every failable call failing from some point on, as when memory is gone;
    - calls failing at random (0.5 %, 2 %, 10 %; 300 seeds each), with and
      without the two failures after which the renderer stops rendering
      (`vkWaitForFences`, `vkResetFences`).

  After the failing run the scenario is run again with nothing failing.  A
  case passes if the driver recorded no violation; no object exists that the
  renderer's state does not refer to, and it refers to no dead one; the
  renderer did not crash, hang or run away; rendering stopped for good only
  where a lost device or one of those two failures is the cause; a call that
  failed once cost at most one draw; and every draw of the second run was
  accepted.

`test_vk case …` / `test_vk random …` repeat one case, as the sweep's failure
summary names it; `-v` prints the renderer's log and each violation.
`coverage.sh` (clang, llvm-cov) reports how much of the renderer all of this
executes and lists the lines it never does.

The test reads the renderer's private state (`V`, the caches, the batches) to
find what it holds, and calls its static helpers: it has to follow changes to
them.

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
`rbvkdummy.sh` (Vulkan under the Khronos validation layer with synchronization
validation, ten runs, each checked, non-zero exit on any failure: the batch
carrying the stand-in textures' initialization is not submitted, once and three
times; creating the second or third image fails once, with the retry in a later
batch and in the same open batch; `VD-nofix`, `VD-old1`, `VD-old2` are negative
controls that must produce validation errors).  `readstrip.py` exits non-zero
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
    R300_VK_FAIL=dummyalloc:i   creating stand-in image i (1 or 2) fails once (vk_image is
                                not called); after the retry the images are copied out
    RBTEST_DUMMY_SAMEBATCH=1    with dummyalloc: a batch is opened first and the retry
                                follows at once, in that same batch
    RBTEST_DUMMY_INTERLEAVED=1  with dummyalloc: the order before the fix (create,
                                record, create, record ...)
    RBTEST_NO_DUMMY_FIX=1       with the above: a cancelled batch does not clear
                                dummy_ready (the code before the fix), and the images
                                are copied out as they are
