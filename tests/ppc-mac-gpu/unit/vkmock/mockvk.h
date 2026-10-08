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
 * a VkResult, but for vkResetDescriptorPool, which is specified to succeed. */
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
 * Make call number nth (1-based) of fn on `thread` (0, 1, or -1 for either)
 * fail with res; with persistent, every call from nth on.
 * With fn MF_COUNT: every failable call, on any thread, from the nth such
 * call on thread 0 on -- as when memory has run out.
 */
void mock_plan(int fn, unsigned nth, bool persistent, VkResult res, int thread);
/*
 * Or: every failable call fails with VK_ERROR_OUT_OF_HOST_MEMORY with
 * probability permille / 1000, from a generator per thread seeded with
 * seed -- except the calls whose bit (1 << MF_...) is set in except.
 */
void mock_plan_random(unsigned seed, unsigned permille, unsigned long except);
void mock_plan_clear(void);
unsigned mock_fired(void);                  /* times the plan made a call fail */
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
