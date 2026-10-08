/*
 * A stand-in Vulkan driver for testing how hw/display/ppc_mac_gpu_vulkan.c
 * behaves when Vulkan calls fail.  It implements the calls that file makes,
 * tracks every object it hands out, and reports as a "violation" anything a
 * real driver would be entitled to crash on or the validation layers would
 * reject, as far as this file models it (see mockvk.c).
 *
 * Any call that can fail can be made to: it then returns the chosen error
 * and leaves its outputs as the specification leaves them -- undefined (a
 * recognizable garbage handle) unless the command says otherwise.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */
#ifndef MOCKVK_H
#define MOCKVK_H

#include <stdbool.h>
#include <stddef.h>
#include <vulkan/vulkan.h>

/* Calls that can be made to fail: every one the renderer makes that returns
 * a VkResult, but for vkResetDescriptorPool, for which the specification
 * lists no error of its own (only the two any command may return,
 * VK_ERROR_UNKNOWN and VK_ERROR_VALIDATION_FAILED). */
enum {
    MF_CreateInstance, MF_CreateDevice, MF_CreateCommandPool,
    MF_CreateDescriptorSetLayout, MF_CreatePipelineLayout,
    MF_AllocateCommandBuffers, MF_CreateFence,
    MF_CreateBuffer, MF_AllocateMemory, MF_BindBufferMemory, MF_MapMemory,
    MF_CreateImage, MF_BindImageMemory, MF_CreateImageView,
    MF_CreateRenderPass, MF_CreateFramebuffer, MF_CreateShaderModule,
    MF_CreateGraphicsPipelines, MF_CreateSampler, MF_CreateDescriptorPool,
    MF_AllocateDescriptorSets, MF_BeginCommandBuffer, MF_EndCommandBuffer,
    MF_QueueSubmit, MF_WaitForFences, MF_ResetFences, MF_ResetCommandBuffer,
    MF_EnumerateInstanceExtensionProperties, MF_EnumeratePhysicalDevices,
    MF_EnumerateDeviceExtensionProperties, MF_EnumerateInstanceVersion,
    MF_COUNT
};
extern const char *const mock_fn_name[MF_COUNT];

/* Object types. */
enum {
    MT_NONE, MT_INSTANCE, MT_PHYSDEV, MT_DEVICE, MT_QUEUE, MT_CMDPOOL, MT_DSL,
    MT_PLAYOUT, MT_CMDBUF, MT_FENCE, MT_BUFFER, MT_MEMORY, MT_IMAGE, MT_VIEW,
    MT_RENDERPASS, MT_FRAMEBUFFER, MT_SHADER, MT_PIPELINE, MT_SAMPLER,
    MT_DPOOL, MT_DSET, MT_COUNT
};
extern const char *const mock_type_name[MT_COUNT];

/* 0 on the thread that drives the renderer, 1 on threads it starts. */
extern __thread int mock_thread;

/* The device offers the ordered-pixel-interlock path (storage images). */
extern bool mock_interlock;

/*
 * The error a call is made to fail with when the plan names none: one the
 * specification lists for it (the registry's errorcodes, vk.xml 1.4.363) --
 * VK_ERROR_OUT_OF_HOST_MEMORY, or VK_ERROR_OUT_OF_DEVICE_MEMORY for the two
 * calls that list only that (vkResetFences, vkResetCommandBuffer).
 */
VkResult mock_listed_error(int fn);

/*
 * Make call number nth (1-based) of fn on `thread` (0, 1, or -1 for either)
 * return res (VK_SUCCESS: mock_listed_error(fn)); with persistent, every
 * call from nth on.  For vkWaitForFences res may be VK_TIMEOUT: not an
 * error, but no answer either.
 * With fn MF_COUNT: every failable call, on any thread, from the nth such
 * call on thread 0 on -- as when memory has run out.
 */
void mock_plan(int fn, unsigned nth, bool persistent, VkResult res, int thread);
/*
 * Or: every failable call fails with mock_listed_error() with probability
 * permille / 1000, from a generator per thread seeded with seed -- except
 * the calls whose bit (1 << MF_...) is set in except.
 */
void mock_plan_random(unsigned seed, unsigned permille, unsigned long except);
void mock_plan_clear(void);
/*
 * A wait that a persistent plan makes fail (other than with
 * VK_ERROR_DEVICE_LOST) fails n times running for a fence, then is let
 * through, so that a run in which every wait fails still ends.  0: never.
 */
void mock_wait_patience(unsigned n);
unsigned mock_fired(void);                  /* times the plan made a call fail */
unsigned mock_failed(int fn);               /* ... this call */
unsigned mock_calls(int fn, int thread);    /* thread 0, 1, or 2 for both */
unsigned mock_failable_calls(void);         /* failable calls on thread 0 so far */

unsigned mock_violations(void);
const char *mock_violation(unsigned i);
void mock_violation_add(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

/* Stop a runaway: after n calls in total, report a violation and call fn. */
void mock_set_budget(unsigned long n, void (*fn)(void));
unsigned long mock_total_calls(void);

/* While set, memory requirements name no memory type at all. */
void mock_no_memory_type(bool none);

/* While set, vkMapMemory returns addresses that are not page-aligned. */
void mock_misalign_maps(bool misalign);

/* While closed, vkWaitForFences on thread 1 does not return. */
void mock_gate(bool closed);

/*
 * Submitted work, and whether it has finished.
 *
 * A submission is "running" until the driver has said that it is not: until
 * a vkWaitForFences that returned VK_SUCCESS for its fence -- or for the
 * fence of a later submission whose command buffer opens with a barrier
 * over all commands, which orders that one after everything submitted
 * before -- or one that returned VK_ERROR_DEVICE_LOST, which counts as
 * success for what is pending and in use.  A wait that returned anything
 * else (an error, VK_TIMEOUT) leaves it running.  So work finishes as late
 * as the specification allows: nothing may rely on it having finished
 * sooner.  A submission that itself returned VK_ERROR_DEVICE_LOST counts as
 * made, and runs until such a wait.
 *
 * Work found finished through later work has freed what it used, but its
 * own fence has not been seen signaled: resetting that fence before a wait
 * for it has been answered is a violation.  (The specification's wording
 * asks only that the work have finished; a driver may signal late.)
 */
unsigned mock_running(void);                /* submissions still running */
unsigned mock_submitted(void);              /* submissions made so far; they are numbered 1 ... */
unsigned mock_finished_upto(void);          /* every submission up to this number has
                                               finished; UINT_MAX once the device is lost */
bool mock_device_lost(void);                /* a call has returned VK_ERROR_DEVICE_LOST */
/* While held, running work does not finish: a wait for it without a time
 * limit does not return, one with a limit returns VK_TIMEOUT. */
void mock_hold(bool held);

/*
 * Memory that running work uses must not be touched by the host: not at all
 * where the work may write it, not written where the work reads it.  (On a
 * lost device it may be: its contents are undefined, the access is valid.)
 * Uses are taken from the commands recorded: the buffer of a copy to or
 * from an image, a draw's index buffer and the buffers of its descriptor
 * set -- read through a uniform descriptor, read and possibly written
 * through a storage descriptor, since shaders are not modelled.
 *
 * Mapped memory in such use is made inaccessible (or read-only), and an
 * access is a violation: the fault handler asks mock_trap(), which says
 * whether the fault was that -- the access is then allowed to go on -- or a
 * crash.  That is exact for memory used as a whole.  Memory that the host
 * and the GPU share piecemeal (VRAM) cannot be guarded by pages: declare
 * its buffer with mock_shared_buffer().  Its uses are then kept by the byte
 * (a copy's region; for a storage descriptor only what mock_shader_reads()
 * declared for the draws recorded while the declaration stands), and
 * whoever touches it says so with mock_host_access(), which reports a
 * violation and returns false if running work uses those bytes
 * conflictingly.  For memory that is not shared, or not the driver's at
 * all, mock_host_access() checks the same and does no harm.
 */
void mock_shared_buffer(VkBuffer buf);
void mock_shader_reads(VkBuffer buf, VkDeviceSize off, VkDeviceSize len);
void mock_shader_reads_clear(void);
bool mock_host_access(const void *p, size_t n, bool write, const char *who);
bool mock_trap(void *addr);

/*
 * Which objects does the renderer still hold?  mock_mark_begin(), then
 * mock_mark() each handle it holds: a handle that is not a live object of
 * that type is a violation (VK_NULL_HANDLE is ignored).  mock_unmarked()
 * then counts the live objects nothing holds -- leaks.
 */
void mock_mark_begin(void);
void mock_mark(void *handle, int type, const char *what);
unsigned mock_unmarked(int type);
unsigned mock_alive(int type);
unsigned mock_destroyed(int type);          /* objects of that type destroyed so far */
bool mock_is_garbage(const void *handle);   /* the output of a call that failed */

#endif
