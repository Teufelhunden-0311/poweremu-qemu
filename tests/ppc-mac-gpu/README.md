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
  buffer that is still running uses it;
- a command is recorded into a command buffer that is not recording, a command
  buffer is submitted that was not ended successfully, begun or reset while
  pending, or ended with its render pass open; a copy, clear or layout
  transition is recorded inside a render pass;
- a fence is submitted while signaled or in use, reset while its work is
  running, or waited for when nothing can signal it;
- a descriptor is written with a null or dead sampler, view or buffer, or a
  set is used after its pool was reset;
- on submission, when the commands are "executed" in order: an image is not in
  the layout a command names; an image that is read (sampled, loaded as an
  attachment, copied from) has a level, layer or slice never written since it
  last left VK_IMAGE_LAYOUT_UNDEFINED; a layout transition that names no source
  access follows a write to the image with no ordering barrier between;
- the host touches mapped memory that running work may write, or writes
  memory that it reads (below);
- the renderer makes more than four million calls, 131072 objects, 4096
  mappings or 64 submissions running at once in one run (it does not stop).

*When work has finished.*  What a command buffer does to images is applied
inside `vkQueueSubmit`, at once: those checks are about the order of commands.
Whether the work has *finished* is kept apart, and there the driver is as slow
as the specification allows.  A submission is running until a
`vkWaitForFences` has said that it is not: one that returned VK_SUCCESS for
its fence, or for the fence of a later submission whose command buffer opens
with a barrier over all commands (which orders that one after everything
submitted before); or one that returned VK_ERROR_DEVICE_LOST, which counts as
success for what is pending and in use.  A wait that returned anything else --
an error, VK_TIMEOUT -- leaves it running, and what it uses in use.  A
submission that itself returned VK_ERROR_DEVICE_LOST counts as made.  The test
can also hold the work (`mock_hold`): a wait then does not return at all.

*Memory.*  From the commands recorded the driver knows what a submission
uses: the buffer of a copy to or from an image, a draw's index buffer, the
buffers of its descriptor set (read through a uniform descriptor; read and
possibly written through a storage descriptor, shaders not being modelled).
Mapped memory that running work uses is made inaccessible, or read-only where
the work only reads it, so that the renderer's own accesses -- the Z-pass
counter, a batch's vertex and texture data -- fault when they come too early;
the fault is recorded and the access then let through.  That is exact for
memory used as a whole.  VRAM, which the host and the GPU share piecemeal, is
not guarded by pages: its uses are kept by the byte (a copy's region, and for
the storage descriptor what the test says the draw's shader reads), and every
access to it in the test is declared to the driver first -- the device's
stand-in (`device_access`) and the two stand-in functions through which the
renderer reads texture bytes.  On a lost device memory may be touched whatever
was using it.

It is not a substitute for a real driver under the validation layers
(`guest/rbvkdummy.sh`), and there is much it cannot show:

- It holds no image or buffer contents.  It cannot tell what is on the
  screen, or that VRAM holds the right bytes after a failure -- only that
  nothing was read that was never written, in the layout named, and that
  memory was not touched while in use.
- Shaders, formats beyond the size of a texel, queue families and most
  valid-usage rules are not modelled.
- An access to VRAM that the test does not declare is not seen.  The
  renderer's are (they go through stand-in functions); a guest's own, in a
  real machine, are the guest's to time by the completion it is told of --
  which is why a completion must not be reported early, and that is checked.
- "For good" cannot be run: that the renderer keeps waiting is shown for some
  dozens of attempts and a tenth of a second, and otherwise read in its source.

**Which failures.**  Every call the renderer makes that returns a `VkResult`
can be made to fail (31; not `vkResetDescriptorPool`, for which the registry
lists no error of its own).  Each fails with an error the specification lists
for it (the registry's `errorcodes`, vk.xml 1.4.363): VK_ERROR_OUT_OF_HOST_MEMORY,
or VK_ERROR_OUT_OF_DEVICE_MEMORY for `vkResetFences` and `vkResetCommandBuffer`,
which list only that.  Where the renderer acts on which result it is, the
others listed are run too: device memory and VK_ERROR_DEVICE_LOST for
`vkQueueSubmit`; those two, VK_ERROR_UNKNOWN and VK_TIMEOUT for
`vkWaitForFences`; device memory for `vkAllocateDescriptorSets`.  Injections
*beyond* what a driver may do are run apart and labelled so in the output
("error not listed"): host-memory errors from `vkResetFences` and
`vkResetCommandBuffer`.  Two more liberties, both within the letter of the
specification: a device reported lost by one call may answer later calls with
success, and the same call may fail any number of times running.  The two
generic errors any command may return (VK_ERROR_UNKNOWN,
VK_ERROR_VALIDATION_FAILED) are injected only where named above.

**The program** (`test_vk`, no arguments) runs

- `selftest`: each kind of misuse above done to the driver directly, to see
  that it is reported, and its correct counterpart, to see that it is not;
  what a wait does and does not say about earlier work;
- `helpers`: `vk_image` and `vk_buffer` with each of their calls failing in
  turn (outputs all null, only what was created destroyed, nothing left) and
  with no memory type fitting; `vk_dummies` with each creation call of each
  stand-in failing, retried in a later batch and in the same open batch, then
  used by a draw; more attachment layouts than render passes are kept;
  programs that do not compile (rejected, compiled once); draws refused for
  what they are (bad buffers, textures out of range ...); every batch in
  flight with a draw waiting for one, and a wait failing for lack of memory
  meanwhile; VRAM mapped off a page boundary at start-up; and the waits:
    - *no answer*: a batch in flight, the GPU held, and every wait for it
      returning an error that is not a lost device (host memory, device
      memory, VK_ERROR_UNKNOWN) or VK_TIMEOUT.  While that lasts -- and after
      the waits stop failing, for as long as the GPU has not finished -- the
      pages the batch reads and writes stay busy, no completion is reported,
      a flush does not return and the Z-pass counter is not read; VRAM the
      batch does not use may be written.  Then the GPU finishes: the flush
      returns, the completion is reported, the device writes the pages, and
      rendering goes on;
    - *one of two answered*: the renderer's thread and a flush wait for the
      same batch and only one gets an answer, which settles it for both;
    - *the device lost* with batches in flight, found out by the thread or by
      a flush: completions reported in order, VRAM free to touch, nothing
      submitted and nothing recycled afterwards;
    - five batches in flight and every wait unanswered three times:
      completions once each, in order, none early;

  and which VRAM counts as busy: the whole scenario below with every batch
  it flushes held in flight first, and the device asking about each page
  and then writing or reading it as the answer allows; what a batch writes
  back beyond what its draws touched -- nine scissor rectangles in a buffer
  of which eight are kept apart, a buffer whose pitch is not a multiple of
  4;
- `sweep`: a scenario of some 500 draws (every texture path, lines, GPU vertex
  shading, several colour buffers, multisampling, a buffer rendered to and
  then sampled, batches in flight together, a descriptor pool running out,
  the image, framebuffer and texture caches overflowing, vertex data larger
  than an arena chunk, a texture upload crossing one; the device reading and
  writing VRAM that batches open or in flight use, and VRAM they do not; the
  Z-pass counter read and set) run from start-up in a fresh process for every
  case:
    - each failable call failing once, for every time the scenario makes it,
      and failing from then on, with the results named above;
    - every failable call failing from some point on, as when memory is gone;
    - calls failing at random (0.5 %, 2 %, 10 %; 300 seeds each), with and
      without the one failure after which the renderer stops rendering
      without the device being lost (`vkResetFences`).

  A wait that is made to fail "from then on" fails three times running for a
  fence and is then let through, or the run would not end: the renderer waits
  it out.

  After the failing run the scenario is run again with nothing failing.  A
  case passes if the driver recorded no violation; no batch was reported
  complete (the callback, or `flush_r200` returning) before the driver had
  finished every submission made up to it; completions came once each and in
  order; no object exists that the renderer's state does not refer to, and it
  refers to no dead one; the renderer did not crash, hang or run away;
  rendering stopped for good only where the device was lost or a fence could
  not be reset, and did stop where a wait or a submission reported the device
  lost; and every draw of the second run was accepted.

  What a failure costs is two different things, and both are checked for a
  call that failed once (rendering going on): at most one draw was *refused*
  (`draw_r300` returned an error), and at most one batch was *cancelled* --
  only when it was its recording or its submission that failed.  A cancelled
  batch takes with it every draw already accepted into it, of which the
  caller learns only through `gpu_failures`; the sweep prints how many that
  was.

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
controls that must produce validation errors), `rbvkwait.sh` (the same layer:
waits for the GPU that give no answer -- the hook returns an error or
VK_TIMEOUT without waiting, several times running, so the GPU may really still
be at work; five runs, each checked: a clean one, three with device-memory,
host-memory and VK_TIMEOUT results, whose readbacks must equal the clean
run's with the layer silent, and `WB-old`, the negative control, in which an
unanswered wait is taken for a finished batch and wrong readbacks or
validation errors must follow).  `readstrip.py` exits non-zero on a bad strip
checksum and `rbvm.sh` then reports the run as failed.

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
    R300_VK_FAIL=waitbusy:N:code:R
                                for every N-th batch waited for (the renderer's thread
                                and flushes counted apart) the first R attempts return
                                code (-1, -2; 2 is VK_TIMEOUT) without waiting
    RBTEST_WAIT_NO_RETRY=1      with waitbusy: such a result is taken for the batch
                                having finished, as before the fix -- but rendering
                                goes on, so that what follows from it shows
