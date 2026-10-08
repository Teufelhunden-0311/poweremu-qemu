/*
 * A stand-in Vulkan driver for failure-path tests; see mockvk.h.
 *
 * What it checks (each breach is recorded as a violation):
 *  - every handle passed in is a live object of the right type that this
 *    driver returned: not VK_NULL_HANDLE where one is not allowed, not the
 *    undefined output of a failed call, not a destroyed object;
 *  - nothing is destroyed twice, or while a submitted command buffer that
 *    has not been waited for still uses it;
 *  - command buffer states: recording commands only while recording,
 *    submitting only what was ended successfully, no begin or reset while
 *    pending, transfer commands and layout transitions outside render passes;
 *  - fences: not submitted while signaled or in use, not reset while their
 *    work is unfinished, not waited for when they can never signal;
 *  - when a command buffer is submitted its commands are "executed": every
 *    image is in the layout the command names, images that are read
 *    (sampled, attachments loaded, copied from) have had every subresource
 *    written since they last left VK_IMAGE_LAYOUT_UNDEFINED, and a layout
 *    transition that names no source access is not preceded by an unordered
 *    write to the image;
 *  - descriptor sets are written with valid samplers, views and buffers and
 *    are not used after their pool was reset.
 *
 * It does not model: shaders (every descriptor of a set counts as used),
 * image contents, queue families, formats, most valid-usage rules.
 * Submitted work "runs" inside vkQueueSubmit and counts as finished once
 * vkWaitForFences has seen its fence -- a wait that succeeded, or one that
 * reported the device lost; after a wait that failed otherwise the work is
 * still in flight.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "mockvk.h"

#define NOBJ    (1u << 17)
#define MAXVIOL 256

const char *const mock_fn_name[MF_COUNT] = {
    "vkCreateInstance", "vkCreateDevice", "vkCreateCommandPool",
    "vkCreateDescriptorSetLayout", "vkCreatePipelineLayout",
    "vkAllocateCommandBuffers", "vkCreateFence",
    "vkCreateBuffer", "vkAllocateMemory", "vkBindBufferMemory", "vkMapMemory",
    "vkCreateImage", "vkBindImageMemory", "vkCreateImageView",
    "vkCreateRenderPass", "vkCreateFramebuffer", "vkCreateShaderModule",
    "vkCreateGraphicsPipelines", "vkCreateSampler", "vkCreateDescriptorPool",
    "vkAllocateDescriptorSets", "vkBeginCommandBuffer", "vkEndCommandBuffer",
    "vkQueueSubmit", "vkWaitForFences", "vkResetFences", "vkResetCommandBuffer",
    "vkEnumerateInstanceExtensionProperties", "vkEnumeratePhysicalDevices",
    "vkEnumerateDeviceExtensionProperties", "vkEnumerateInstanceVersion",
};

const char *const mock_type_name[MT_COUNT] = {
    "none", "instance", "physical device", "device", "queue", "command pool",
    "descriptor set layout", "pipeline layout", "command buffer", "fence",
    "buffer", "memory", "image", "image view", "render pass", "framebuffer",
    "shader module", "pipeline", "sampler", "descriptor pool", "descriptor set",
};

typedef struct Obj Obj;

typedef struct Ent {
    uint32_t binding, elem;
    VkDescriptorType type;
    Obj *sampler, *view, *buffer;
    VkImageLayout layout;
} Ent;

enum { C_SYNC, C_TRANSITION, C_CLEAR, C_COPY_B2I, C_COPY_I2B, C_BEGIN_PASS,
       C_END_PASS, C_DRAW };
typedef struct Cmd {
    int op;
    Obj *a, *b;
    uint32_t p[6];
} Cmd;

enum { CB_INITIAL, CB_RECORDING, CB_EXECUTABLE, CB_PENDING, CB_INVALID };
static const char *const cb_state_name[] = {
    "initial", "recording", "executable", "pending", "invalid" };

struct Obj {
    int type;
    bool alive, marked;
    unsigned serial;
    unsigned pending;           /* submitted, unfinished command buffers using it */
    Obj *ref_cb;                /* the command buffer that last referenced it ... */
    uint64_t ref_epoch;         /* ... in this recording (no double counting) */

    Obj *mem;                   /* image, buffer: the memory bound */
    /* image */
    bool is3d;
    uint32_t levels, layers, depth, slots;
    VkImageLayout layout;
    uint8_t *written;           /* per level, per layer or slice */
    bool wseen;
    uint64_t wepoch;            /* g_epoch of the last transfer write */
    /* image view */
    Obj *image;
    /* memory */
    void *map;
    VkDeviceSize msize;
    /* buffer */
    VkDeviceSize bsize;
    /* render pass */
    uint32_t rp_natt;
    bool dep_in, dep_out;       /* external dependencies that order everything */
    /* framebuffer */
    uint32_t natt;
    Obj *att[8];
    Obj *rp;
    /* descriptor pool */
    uint32_t max_sets, used_sets, gen;
    /* descriptor set */
    Obj *pool;
    uint32_t set_gen;
    Ent *ent;
    unsigned nent, capent;
    /* command buffer */
    int state;
    Cmd *cmd;
    unsigned ncmd, capcmd;
    bool in_pass;
    Obj *cur_rp, *bound_set, *bound_pipe;
    Obj **ref;
    unsigned nref, capref;
    uint64_t epoch;
    /* fence */
    bool signaled;
    Obj *fence_cb[4];
    unsigned nfence_cb;
};

static Obj g_obj[NOBJ];
static unsigned g_nobj;
static unsigned g_serial[MT_COUNT], g_destroyed[MT_COUNT];
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_gate_cv = PTHREAD_COND_INITIALIZER;
static bool g_gate;
static char *g_viol[MAXVIOL];
static unsigned g_nviol;
static struct {
    int fn;
    unsigned nth;
    bool persistent;
    VkResult res;
    int thread;
} g_plan = { -1, 0, false, VK_SUCCESS, -1 };
static unsigned g_calls[MF_COUNT][3], g_fired, g_failable;
static bool g_all_fail;
static struct {
    unsigned permille;
    unsigned long except;
    uint64_t state[2];          /* per thread */
} g_rand;
static unsigned long g_total, g_budget;
static void (*g_budget_fn)(void);
static uint64_t g_epoch = 1, g_cb_epoch, g_garbage;

__thread int mock_thread;
bool mock_interlock;
static bool g_no_memtype, g_misalign;

void mock_no_memory_type(bool none)
{
    g_no_memtype = none;
}

void mock_misalign_maps(bool misalign)
{
    g_misalign = misalign;
}

/* ---- bookkeeping ------------------------------------------------------ */

static void violv(const char *fmt, va_list ap)
{
    char buf[400];

    vsnprintf(buf, sizeof(buf), fmt, ap);
    if (g_nviol < MAXVIOL) {
        g_viol[g_nviol] = strdup(buf);
    }
    g_nviol++;
    if (getenv("MOCK_TRACE")) {
        fprintf(stderr, "VIOLATION: %s\n", buf);
    }
}

static void viol(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void viol(const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    violv(fmt, ap);
    va_end(ap);
}

static void enter(void)
{
    pthread_mutex_lock(&g_mu);
    if (g_budget && ++g_total > g_budget) {
        void (*fn)(void) = g_budget_fn;
        viol("more than %lu Vulkan calls: the renderer does not stop", g_budget);
        g_budget = 0;
        pthread_mutex_unlock(&g_mu);
        if (fn) {
            fn();
        }
        _exit(97);
    }
}

#define ENTER() enter()
#define LEAVE() pthread_mutex_unlock(&g_mu)

static void *garbage(void);

/* A failed count-then-fill call: its count and what it wrote are undefined. */
static void garbage_list(uint32_t *n, void *array, size_t size)
{
    if (array) {
        memset(array, 0x5a, (size_t)*n * size);    /* no more than the caller offered */
        *n += 2;
    } else {
        *n = 3;
    }
}

static void *garbage(void)
{
    return (void *)(uintptr_t)(0xDEAD0000BAD00000ull + 16 * ++g_garbage);
}

bool mock_is_garbage(const void *h)
{
    return ((uintptr_t)h >> 32) == 0xDEAD0000u;
}

static Obj *new_obj(int type)
{
    Obj *o;

    if (g_nobj == NOBJ) {
        void (*fn)(void) = g_budget_fn;
        viol("more than %u objects created: the renderer does not stop", NOBJ);
        pthread_mutex_unlock(&g_mu);
        if (fn) {
            fn();
        }
        _exit(97);
    }
    o = &g_obj[g_nobj++];
    memset(o, 0, sizeof(*o));
    o->type = type;
    o->alive = true;
    o->serial = ++g_serial[type];
    return o;
}

static Obj *get(const void *h, int type, const char *fn, const char *what)
{
    Obj *o = (Obj *)h;

    if (!h) {
        viol("%s: %s is VK_NULL_HANDLE", fn, what);
        return NULL;
    }
    if (mock_is_garbage(h)) {
        viol("%s: %s is the undefined output of a call that failed (%p)", fn, what, h);
        return NULL;
    }
    if (o < g_obj || o >= g_obj + g_nobj ||
        ((uintptr_t)o - (uintptr_t)g_obj) % sizeof(Obj)) {
        viol("%s: %s (%p) is not a handle this driver returned", fn, what, h);
        return NULL;
    }
    if (o->type != type) {
        viol("%s: %s is a %s, not a %s", fn, what, mock_type_name[o->type],
             mock_type_name[type]);
        return NULL;
    }
    if (!o->alive) {
        viol("%s: %s (%s #%u) is used after it was destroyed", fn, what,
             mock_type_name[type], o->serial);
        return NULL;
    }
    return o;
}

/* Count the call; should it fail? */
static bool fail(int fn, VkResult *res)
{
    int t = mock_thread ? 1 : 0;
    unsigned n_t = ++g_calls[fn][t], n_any = ++g_calls[fn][2];
    unsigned n = g_plan.thread < 0 ? n_any : n_t;

    if (!t) {
        g_failable++;
    }
    if (g_rand.permille) {
        uint64_t x = g_rand.state[t];
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        g_rand.state[t] = x;
        if (!((g_rand.except >> fn) & 1) && (x >> 11) % 1000 < g_rand.permille) {
            g_fired++;
            *res = VK_ERROR_OUT_OF_HOST_MEMORY;
            return true;
        }
        return false;
    }
    if (g_plan.fn == MF_COUNT) {
        g_all_fail |= !t && g_failable >= g_plan.nth;
        if (g_all_fail) {
            g_fired++;
            *res = g_plan.res;
        }
        return g_all_fail;
    }
    if (g_plan.fn != fn || (g_plan.thread >= 0 && g_plan.thread != t)) {
        return false;
    }
    if (n == g_plan.nth || (g_plan.persistent && n >= g_plan.nth)) {
        g_fired++;
        *res = g_plan.res;
        return true;
    }
    return false;
}

void mock_plan(int fn, unsigned nth, bool persistent, VkResult res, int thread)
{
    pthread_mutex_lock(&g_mu);
    g_plan.fn = fn;
    g_plan.nth = nth;
    g_plan.persistent = persistent;
    g_plan.res = res;
    g_plan.thread = thread;
    pthread_mutex_unlock(&g_mu);
}

void mock_plan_random(unsigned seed, unsigned permille, unsigned long except)
{
    pthread_mutex_lock(&g_mu);
    g_rand.permille = permille;
    g_rand.except = except;
    g_rand.state[0] = 0x9E3779B97F4A7C15ull * (seed + 1);
    g_rand.state[1] = 0xD1B54A32D192ED03ull * (seed + 1);
    pthread_mutex_unlock(&g_mu);
}

void mock_plan_clear(void)
{
    pthread_mutex_lock(&g_mu);
    g_plan.fn = -1;
    g_all_fail = false;
    g_rand.permille = 0;
    pthread_mutex_unlock(&g_mu);
}

unsigned mock_fired(void)
{
    return g_fired;
}

unsigned mock_calls(int fn, int thread)
{
    return g_calls[fn][thread];
}

unsigned mock_failable_calls(void)
{
    return g_failable;
}

unsigned mock_violations(void)
{
    return g_nviol;
}

const char *mock_violation(unsigned i)
{
    return i < g_nviol && i < MAXVIOL ? g_viol[i] : "";
}

void mock_violation_add(const char *fmt, ...)
{
    va_list ap;

    pthread_mutex_lock(&g_mu);
    va_start(ap, fmt);
    violv(fmt, ap);
    va_end(ap);
    pthread_mutex_unlock(&g_mu);
}

void mock_set_budget(unsigned long n, void (*fn)(void))
{
    pthread_mutex_lock(&g_mu);
    g_budget = n ? g_total + n : 0;
    g_budget_fn = fn;
    pthread_mutex_unlock(&g_mu);
}

unsigned long mock_total_calls(void)
{
    return g_total;
}

void mock_gate(bool closed)
{
    pthread_mutex_lock(&g_mu);
    g_gate = closed;
    pthread_cond_broadcast(&g_gate_cv);
    pthread_mutex_unlock(&g_mu);
}

void mock_mark_begin(void)
{
    pthread_mutex_lock(&g_mu);
    for (unsigned i = 0; i < g_nobj; i++) {
        g_obj[i].marked = false;
    }
    pthread_mutex_unlock(&g_mu);
}

void mock_mark(void *h, int type, const char *what)
{
    Obj *o;

    if (!h) {
        return;
    }
    pthread_mutex_lock(&g_mu);
    o = get(h, type, "the renderer still holds", what);
    if (o) {
        o->marked = true;
    }
    pthread_mutex_unlock(&g_mu);
}

unsigned mock_unmarked(int type)
{
    unsigned n = 0;

    pthread_mutex_lock(&g_mu);
    for (unsigned i = 0; i < g_nobj; i++) {
        n += g_obj[i].type == type && g_obj[i].alive && !g_obj[i].marked;
    }
    pthread_mutex_unlock(&g_mu);
    return n;
}

unsigned mock_alive(int type)
{
    unsigned n = 0;

    pthread_mutex_lock(&g_mu);
    for (unsigned i = 0; i < g_nobj; i++) {
        n += g_obj[i].type == type && g_obj[i].alive;
    }
    pthread_mutex_unlock(&g_mu);
    return n;
}

unsigned mock_destroyed(int type)
{
    return g_destroyed[type];
}

static void destroy(const void *h, int type, const char *fn)
{
    Obj *o;

    if (!h) {
        return;                         /* destroying VK_NULL_HANDLE is allowed */
    }
    o = get(h, type, fn, "the object to destroy");
    if (!o) {
        return;
    }
    if (o->pending) {
        viol("%s: %s #%u is destroyed while a submitted command buffer that has "
             "not finished still uses it", fn, mock_type_name[type], o->serial);
    }
    o->alive = false;
    g_destroyed[type]++;
    free(o->written);
    o->written = NULL;
    free(o->map);
    o->map = NULL;
    free(o->ent);
    o->ent = NULL;
}

/* ---- instance and device --------------------------------------------- */

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(
    const char *layer, uint32_t *n, VkExtensionProperties *props)
{
    VkResult r;

    ENTER();
    if (fail(MF_EnumerateInstanceExtensionProperties, &r)) {
        garbage_list(n, props, sizeof(*props));
        LEAVE();
        return r;
    }
    *n = 0;
    LEAVE();
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceVersion(uint32_t *v)
{
    VkResult r;

    ENTER();
    if (fail(MF_EnumerateInstanceVersion, &r)) {
        *v = 0xBAD00BADu;
        LEAVE();
        return r;
    }
    *v = VK_API_VERSION_1_3;
    LEAVE();
    return VK_SUCCESS;
}

static Obj *g_physdev;

VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(
    const VkInstanceCreateInfo *ci, const VkAllocationCallbacks *a, VkInstance *out)
{
    VkResult r;

    ENTER();
    if (fail(MF_CreateInstance, &r)) {
        *out = garbage();
        LEAVE();
        return r;
    }
    *out = (VkInstance)new_obj(MT_INSTANCE);
    g_physdev = new_obj(MT_PHYSDEV);
    LEAVE();
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumeratePhysicalDevices(
    VkInstance inst, uint32_t *n, VkPhysicalDevice *pd)
{
    VkResult r;

    ENTER();
    get(inst, MT_INSTANCE, __func__, "instance");
    if (fail(MF_EnumeratePhysicalDevices, &r)) {
        uint32_t cap = pd ? *n : 0;
        garbage_list(n, pd, sizeof(*pd));
        for (uint32_t i = 0; i < cap; i++) {
            pd[i] = garbage();
        }
        LEAVE();
        return r;
    }
    if (pd && *n >= 1) {
        pd[0] = (VkPhysicalDevice)g_physdev;
    }
    *n = 1;
    LEAVE();
    return VK_SUCCESS;
}

static void fill_props(VkPhysicalDeviceProperties *p)
{
    memset(p, 0, sizeof(*p));
    p->apiVersion = VK_API_VERSION_1_3;
    p->deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    snprintf(p->deviceName, sizeof(p->deviceName), "mock Vulkan driver");
    p->limits.maxImageDimension2D = 16384;
    p->limits.minUniformBufferOffsetAlignment = 16;
    p->limits.minStorageBufferOffsetAlignment = 16;
    p->limits.maxSamplerAnisotropy = 16;
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties(
    VkPhysicalDevice pd, VkPhysicalDeviceProperties *p)
{
    ENTER();
    get(pd, MT_PHYSDEV, __func__, "physicalDevice");
    fill_props(p);
    LEAVE();
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties2(
    VkPhysicalDevice pd, VkPhysicalDeviceProperties2 *p)
{
    ENTER();
    get(pd, MT_PHYSDEV, __func__, "physicalDevice");
    fill_props(&p->properties);
    for (VkBaseOutStructure *s = p->pNext; s; s = s->pNext) {
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES) {
            ((VkPhysicalDeviceDriverProperties *)s)->driverID = VK_DRIVER_ID_MESA_LLVMPIPE;
        }
    }
    LEAVE();
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties(
    VkPhysicalDevice pd, VkPhysicalDeviceMemoryProperties *p)
{
    ENTER();
    get(pd, MT_PHYSDEV, __func__, "physicalDevice");
    memset(p, 0, sizeof(*p));
    p->memoryTypeCount = 1;
    p->memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    p->memoryHeapCount = 1;
    p->memoryHeaps[0].size = 1ull << 32;
    LEAVE();
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(
    VkPhysicalDevice pd, const char *layer, uint32_t *n, VkExtensionProperties *props)
{
    static const char *const ext[2] = {
        VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME,
        VK_EXT_SHADER_DEMOTE_TO_HELPER_INVOCATION_EXTENSION_NAME,
    };
    uint32_t have = mock_interlock ? 2 : 0;
    VkResult r;

    ENTER();
    get(pd, MT_PHYSDEV, __func__, "physicalDevice");
    if (fail(MF_EnumerateDeviceExtensionProperties, &r)) {
        garbage_list(n, props, sizeof(*props));
        LEAVE();
        return r;
    }
    if (props) {
        for (uint32_t i = 0; i < have && i < *n; i++) {
            memset(&props[i], 0, sizeof(props[i]));
            snprintf(props[i].extensionName, sizeof(props[i].extensionName), "%s", ext[i]);
        }
    }
    *n = have;
    LEAVE();
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2(
    VkPhysicalDevice pd, VkPhysicalDeviceFeatures2 *f)
{
    ENTER();
    get(pd, MT_PHYSDEV, __func__, "physicalDevice");
    memset(&f->features, 0, sizeof(f->features));
    f->features.fragmentStoresAndAtomics = VK_TRUE;
    f->features.depthClamp = VK_TRUE;
    f->features.shaderClipDistance = VK_TRUE;
    f->features.samplerAnisotropy = VK_TRUE;
    f->features.textureCompressionBC = VK_TRUE;
    f->features.shaderStorageImageExtendedFormats = VK_TRUE;
    for (VkBaseOutStructure *s = f->pNext; s; s = s->pNext) {
        switch ((int)s->sType) {
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT:
            ((VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT *)s)
                ->fragmentShaderPixelInterlock = mock_interlock;
            break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES_EXT:
            ((VkPhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT *)s)
                ->shaderDemoteToHelperInvocation = VK_TRUE;
            break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES:
            ((VkPhysicalDeviceVulkan12Features *)s)->samplerMirrorClampToEdge = VK_TRUE;
            break;
        }
    }
    LEAVE();
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFormatProperties(
    VkPhysicalDevice pd, VkFormat fmt, VkFormatProperties *p)
{
    ENTER();
    get(pd, MT_PHYSDEV, __func__, "physicalDevice");
    memset(p, 0, sizeof(*p));
    p->optimalTilingFeatures = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
    LEAVE();
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties(
    VkPhysicalDevice pd, uint32_t *n, VkQueueFamilyProperties *p)
{
    ENTER();
    get(pd, MT_PHYSDEV, __func__, "physicalDevice");
    if (p && *n >= 1) {
        memset(&p[0], 0, sizeof(p[0]));
        p[0].queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT;
        p[0].queueCount = 1;
    }
    *n = 1;
    LEAVE();
}

static Obj *g_queue;

VKAPI_ATTR VkResult VKAPI_CALL vkCreateDevice(
    VkPhysicalDevice pd, const VkDeviceCreateInfo *ci, const VkAllocationCallbacks *a,
    VkDevice *out)
{
    VkResult r;

    ENTER();
    get(pd, MT_PHYSDEV, __func__, "physicalDevice");
    if (fail(MF_CreateDevice, &r)) {
        *out = garbage();
        LEAVE();
        return r;
    }
    *out = (VkDevice)new_obj(MT_DEVICE);
    g_queue = new_obj(MT_QUEUE);
    LEAVE();
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkGetDeviceQueue(VkDevice dev, uint32_t fam, uint32_t idx,
                                            VkQueue *out)
{
    ENTER();
    get(dev, MT_DEVICE, __func__, "device");
    *out = (VkQueue)g_queue;
    LEAVE();
}

/* Create an object that has nothing to it but its existence. */
#define SIMPLE_CREATE(fnid, mtype, out)                                 \
    do {                                                                \
        VkResult r_;                                                    \
        ENTER();                                                        \
        get(dev, MT_DEVICE, __func__, "device");                        \
        if (fail(fnid, &r_)) {                                          \
            *(out) = garbage();                                         \
            LEAVE();                                                    \
            return r_;                                                  \
        }                                                               \
        *(out) = (void *)new_obj(mtype);                                \
        LEAVE();                                                        \
        return VK_SUCCESS;                                              \
    } while (0)

VKAPI_ATTR VkResult VKAPI_CALL vkCreateCommandPool(
    VkDevice dev, const VkCommandPoolCreateInfo *ci, const VkAllocationCallbacks *a,
    VkCommandPool *out)
{
    SIMPLE_CREATE(MF_CreateCommandPool, MT_CMDPOOL, out);
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorSetLayout(
    VkDevice dev, const VkDescriptorSetLayoutCreateInfo *ci,
    const VkAllocationCallbacks *a, VkDescriptorSetLayout *out)
{
    SIMPLE_CREATE(MF_CreateDescriptorSetLayout, MT_DSL, out);
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreatePipelineLayout(
    VkDevice dev, const VkPipelineLayoutCreateInfo *ci, const VkAllocationCallbacks *a,
    VkPipelineLayout *out)
{
    ENTER();
    for (uint32_t i = 0; i < ci->setLayoutCount; i++) {
        get(ci->pSetLayouts[i], MT_DSL, __func__, "pSetLayouts[]");
    }
    LEAVE();
    SIMPLE_CREATE(MF_CreatePipelineLayout, MT_PLAYOUT, out);
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateShaderModule(
    VkDevice dev, const VkShaderModuleCreateInfo *ci, const VkAllocationCallbacks *a,
    VkShaderModule *out)
{
    SIMPLE_CREATE(MF_CreateShaderModule, MT_SHADER, out);
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateSampler(
    VkDevice dev, const VkSamplerCreateInfo *ci, const VkAllocationCallbacks *a,
    VkSampler *out)
{
    SIMPLE_CREATE(MF_CreateSampler, MT_SAMPLER, out);
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateFence(
    VkDevice dev, const VkFenceCreateInfo *ci, const VkAllocationCallbacks *a,
    VkFence *out)
{
    VkResult r;
    Obj *o;

    ENTER();
    get(dev, MT_DEVICE, __func__, "device");
    if (fail(MF_CreateFence, &r)) {
        *out = garbage();
        LEAVE();
        return r;
    }
    o = new_obj(MT_FENCE);
    o->signaled = ci->flags & VK_FENCE_CREATE_SIGNALED_BIT;
    *out = (VkFence)o;
    LEAVE();
    return VK_SUCCESS;
}

/* ---- memory, buffers, images ----------------------------------------- */

VKAPI_ATTR VkResult VKAPI_CALL vkCreateBuffer(
    VkDevice dev, const VkBufferCreateInfo *ci, const VkAllocationCallbacks *a,
    VkBuffer *out)
{
    VkResult r;
    Obj *o;

    ENTER();
    get(dev, MT_DEVICE, __func__, "device");
    if (fail(MF_CreateBuffer, &r)) {
        *out = garbage();
        LEAVE();
        return r;
    }
    o = new_obj(MT_BUFFER);
    o->bsize = ci->size;
    *out = (VkBuffer)o;
    LEAVE();
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkGetBufferMemoryRequirements(
    VkDevice dev, VkBuffer buf, VkMemoryRequirements *req)
{
    Obj *o;

    ENTER();
    get(dev, MT_DEVICE, __func__, "device");
    o = get(buf, MT_BUFFER, __func__, "buffer");
    req->size = o ? (o->bsize + 4095) & ~(VkDeviceSize)4095 : 4096;
    req->alignment = 4096;
    req->memoryTypeBits = g_no_memtype ? 0 : 1;
    LEAVE();
}

VKAPI_ATTR void VKAPI_CALL vkGetImageMemoryRequirements(
    VkDevice dev, VkImage img, VkMemoryRequirements *req)
{
    ENTER();
    get(dev, MT_DEVICE, __func__, "device");
    get(img, MT_IMAGE, __func__, "image");
    req->size = 4096;
    req->alignment = 4096;
    req->memoryTypeBits = g_no_memtype ? 0 : 1;
    LEAVE();
}

VKAPI_ATTR VkResult VKAPI_CALL vkAllocateMemory(
    VkDevice dev, const VkMemoryAllocateInfo *ai, const VkAllocationCallbacks *a,
    VkDeviceMemory *out)
{
    VkResult r;
    Obj *o;

    ENTER();
    get(dev, MT_DEVICE, __func__, "device");
    if (ai->memoryTypeIndex != 0) {
        viol("%s: memory type %u does not exist", __func__, ai->memoryTypeIndex);
    }
    if (fail(MF_AllocateMemory, &r)) {
        *out = garbage();
        LEAVE();
        return r;
    }
    o = new_obj(MT_MEMORY);
    o->msize = ai->allocationSize;
    *out = (VkDeviceMemory)o;
    LEAVE();
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkFreeMemory(VkDevice dev, VkDeviceMemory mem,
                                        const VkAllocationCallbacks *a)
{
    ENTER();
    get(dev, MT_DEVICE, __func__, "device");
    destroy(mem, MT_MEMORY, __func__);
    LEAVE();
}

VKAPI_ATTR VkResult VKAPI_CALL vkBindBufferMemory(VkDevice dev, VkBuffer buf,
                                                  VkDeviceMemory mem, VkDeviceSize off)
{
    VkResult r;
    Obj *b, *m;

    ENTER();
    get(dev, MT_DEVICE, __func__, "device");
    b = get(buf, MT_BUFFER, __func__, "buffer");
    m = get(mem, MT_MEMORY, __func__, "memory");
    if (b && b->mem) {
        viol("%s: buffer #%u already has memory", __func__, b->serial);
    }
    if (fail(MF_BindBufferMemory, &r)) {
        LEAVE();
        return r;
    }
    if (b && m) {
        b->mem = m;
    }
    LEAVE();
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkMapMemory(VkDevice dev, VkDeviceMemory mem,
                                           VkDeviceSize off, VkDeviceSize size,
                                           VkMemoryMapFlags flags, void **out)
{
    VkResult r;
    Obj *m;

    ENTER();
    get(dev, MT_DEVICE, __func__, "device");
    m = get(mem, MT_MEMORY, __func__, "memory");
    if (fail(MF_MapMemory, &r)) {
        *out = garbage();
        LEAVE();
        return r;
    }
    if (m && !m->map &&
        posix_memalign(&m->map, getpagesize(), (m->msize ? m->msize : 16) + 64)) {
        fprintf(stderr, "mockvk: no memory for a mapping\n");
        abort();
    }
    *out = !m ? NULL : g_misalign ? (char *)m->map + 8 : m->map;
    LEAVE();
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyBuffer(VkDevice dev, VkBuffer buf,
                                           const VkAllocationCallbacks *a)
{
    ENTER();
    get(dev, MT_DEVICE, __func__, "device");
    destroy(buf, MT_BUFFER, __func__);
    LEAVE();
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateImage(
    VkDevice dev, const VkImageCreateInfo *ci, const VkAllocationCallbacks *a,
    VkImage *out)
{
    VkResult r;
    Obj *o;

    ENTER();
    get(dev, MT_DEVICE, __func__, "device");
    if (!ci->extent.width || !ci->extent.height || !ci->extent.depth || !ci->mipLevels ||
        !ci->arrayLayers) {
        viol("%s: an extent, the level count or the layer count is 0", __func__);
    }
    if (fail(MF_CreateImage, &r)) {
        *out = garbage();
        LEAVE();
        return r;
    }
    o = new_obj(MT_IMAGE);
    o->is3d = ci->imageType == VK_IMAGE_TYPE_3D;
    o->levels = ci->mipLevels ? ci->mipLevels : 1;
    o->layers = ci->arrayLayers ? ci->arrayLayers : 1;
    o->depth = ci->extent.depth ? ci->extent.depth : 1;
    o->slots = o->is3d ? o->depth : o->layers;
    o->written = calloc((size_t)o->levels * o->slots, 1);
    o->layout = ci->initialLayout;
    *out = (VkImage)o;
    LEAVE();
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkBindImageMemory(VkDevice dev, VkImage img,
                                                 VkDeviceMemory mem, VkDeviceSize off)
{
    VkResult r;
    Obj *i, *m;

    ENTER();
    get(dev, MT_DEVICE, __func__, "device");
    i = get(img, MT_IMAGE, __func__, "image");
    m = get(mem, MT_MEMORY, __func__, "memory");
    if (i && i->mem) {
        viol("%s: image #%u already has memory", __func__, i->serial);
    }
    if (fail(MF_BindImageMemory, &r)) {
        LEAVE();
        return r;
    }
    if (i && m) {
        i->mem = m;
    }
    LEAVE();
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateImageView(
    VkDevice dev, const VkImageViewCreateInfo *ci, const VkAllocationCallbacks *a,
    VkImageView *out)
{
    VkResult r;
    Obj *o, *i;

    ENTER();
    get(dev, MT_DEVICE, __func__, "device");
    i = get(ci->image, MT_IMAGE, __func__, "image");
    if (i && (!i->mem || !i->mem->alive)) {
        viol("%s: image #%u has no live memory bound", __func__, i->serial);
    }
    if (fail(MF_CreateImageView, &r)) {
        *out = garbage();
        LEAVE();
        return r;
    }
    o = new_obj(MT_VIEW);
    o->image = i;
    *out = (VkImageView)o;
    LEAVE();
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyImage(VkDevice dev, VkImage img,
                                          const VkAllocationCallbacks *a)
{
    ENTER();
    get(dev, MT_DEVICE, __func__, "device");
    destroy(img, MT_IMAGE, __func__);
    LEAVE();
}

VKAPI_ATTR void VKAPI_CALL vkDestroyImageView(VkDevice dev, VkImageView view,
                                              const VkAllocationCallbacks *a)
{
    ENTER();
    get(dev, MT_DEVICE, __func__, "device");
    destroy(view, MT_VIEW, __func__);
    LEAVE();
}

/* ---- render passes, framebuffers, pipelines --------------------------- */

VKAPI_ATTR VkResult VKAPI_CALL vkCreateRenderPass(
    VkDevice dev, const VkRenderPassCreateInfo *ci, const VkAllocationCallbacks *a,
    VkRenderPass *out)
{
    VkResult r;
    Obj *o;

    ENTER();
    get(dev, MT_DEVICE, __func__, "device");
    if (fail(MF_CreateRenderPass, &r)) {
        *out = garbage();
        LEAVE();
        return r;
    }
    o = new_obj(MT_RENDERPASS);
    o->rp_natt = ci->attachmentCount;
    for (uint32_t i = 0; i < ci->dependencyCount; i++) {
        const VkSubpassDependency *d = &ci->pDependencies[i];
        bool all = (d->srcStageMask & VK_PIPELINE_STAGE_ALL_COMMANDS_BIT) &&
                   (d->dstStageMask & VK_PIPELINE_STAGE_ALL_COMMANDS_BIT) &&
                   (d->srcAccessMask & VK_ACCESS_MEMORY_WRITE_BIT);
        o->dep_in |= all && d->srcSubpass == VK_SUBPASS_EXTERNAL;
        o->dep_out |= all && d->dstSubpass == VK_SUBPASS_EXTERNAL;
    }
    *out = (VkRenderPass)o;
    LEAVE();
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateFramebuffer(
    VkDevice dev, const VkFramebufferCreateInfo *ci, const VkAllocationCallbacks *a,
    VkFramebuffer *out)
{
    VkResult r;
    Obj *o, *rp, *att[8] = { 0 };
    uint32_t n = ci->attachmentCount < 8 ? ci->attachmentCount : 8;

    ENTER();
    get(dev, MT_DEVICE, __func__, "device");
    rp = get(ci->renderPass, MT_RENDERPASS, __func__, "renderPass");
    if (rp && rp->rp_natt != ci->attachmentCount) {
        viol("%s: %u attachments for a render pass of %u", __func__, ci->attachmentCount,
             rp->rp_natt);
    }
    for (uint32_t i = 0; i < n; i++) {
        att[i] = get(ci->pAttachments[i], MT_VIEW, __func__, "pAttachments[]");
    }
    if (!ci->width || !ci->height) {
        viol("%s: zero size", __func__);
    }
    if (fail(MF_CreateFramebuffer, &r)) {
        *out = garbage();
        LEAVE();
        return r;
    }
    o = new_obj(MT_FRAMEBUFFER);
    o->rp = rp;
    o->natt = n;
    memcpy(o->att, att, sizeof(att));
    *out = (VkFramebuffer)o;
    LEAVE();
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyFramebuffer(VkDevice dev, VkFramebuffer fb,
                                                const VkAllocationCallbacks *a)
{
    ENTER();
    get(dev, MT_DEVICE, __func__, "device");
    destroy(fb, MT_FRAMEBUFFER, __func__);
    LEAVE();
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateGraphicsPipelines(
    VkDevice dev, VkPipelineCache cache, uint32_t n, const VkGraphicsPipelineCreateInfo *ci,
    const VkAllocationCallbacks *a, VkPipeline *out)
{
    VkResult r;

    ENTER();
    get(dev, MT_DEVICE, __func__, "device");
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t k = 0; k < ci[i].stageCount; k++) {
            get(ci[i].pStages[k].module, MT_SHADER, __func__, "pStages[].module");
        }
        get(ci[i].layout, MT_PLAYOUT, __func__, "layout");
        get(ci[i].renderPass, MT_RENDERPASS, __func__, "renderPass");
    }
    if (fail(MF_CreateGraphicsPipelines, &r)) {
        for (uint32_t i = 0; i < n; i++) {
            out[i] = VK_NULL_HANDLE;    /* the specification's promise */
        }
        LEAVE();
        return r;
    }
    for (uint32_t i = 0; i < n; i++) {
        out[i] = (VkPipeline)new_obj(MT_PIPELINE);
    }
    LEAVE();
    return VK_SUCCESS;
}

/* ---- descriptors ------------------------------------------------------ */

VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorPool(
    VkDevice dev, const VkDescriptorPoolCreateInfo *ci, const VkAllocationCallbacks *a,
    VkDescriptorPool *out)
{
    VkResult r;
    Obj *o;

    ENTER();
    get(dev, MT_DEVICE, __func__, "device");
    if (fail(MF_CreateDescriptorPool, &r)) {
        *out = garbage();
        LEAVE();
        return r;
    }
    o = new_obj(MT_DPOOL);
    o->max_sets = ci->maxSets;
    *out = (VkDescriptorPool)o;
    LEAVE();
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkAllocateDescriptorSets(
    VkDevice dev, const VkDescriptorSetAllocateInfo *ai, VkDescriptorSet *out)
{
    VkResult r;
    Obj *pool;

    ENTER();
    get(dev, MT_DEVICE, __func__, "device");
    pool = get(ai->descriptorPool, MT_DPOOL, __func__, "descriptorPool");
    for (uint32_t i = 0; i < ai->descriptorSetCount; i++) {
        get(ai->pSetLayouts[i], MT_DSL, __func__, "pSetLayouts[]");
        out[i] = VK_NULL_HANDLE;        /* the specification's promise on failure */
    }
    if (fail(MF_AllocateDescriptorSets, &r)) {
        LEAVE();
        return r;
    }
    if (!pool || pool->used_sets + ai->descriptorSetCount > pool->max_sets) {
        LEAVE();
        return VK_ERROR_OUT_OF_POOL_MEMORY;
    }
    for (uint32_t i = 0; i < ai->descriptorSetCount; i++) {
        Obj *o = new_obj(MT_DSET);
        o->pool = pool;
        o->set_gen = pool->gen;
        out[i] = (VkDescriptorSet)o;
    }
    pool->used_sets += ai->descriptorSetCount;
    LEAVE();
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkResetDescriptorPool(VkDevice dev, VkDescriptorPool p,
                                                     VkDescriptorPoolResetFlags flags)
{
    Obj *pool;

    ENTER();
    get(dev, MT_DEVICE, __func__, "device");
    pool = get(p, MT_DPOOL, __func__, "descriptorPool");
    if (pool) {
        if (pool->pending) {
            viol("%s: descriptor pool #%u is reset while a submitted command buffer "
                 "that has not finished uses its sets", __func__, pool->serial);
        }
        pool->gen++;
        pool->used_sets = 0;
    }
    LEAVE();
    return VK_SUCCESS;
}

static Obj *get_set(const void *h, const char *fn, const char *what)
{
    Obj *s = get(h, MT_DSET, fn, what);

    if (s && s->set_gen != s->pool->gen) {
        viol("%s: %s (descriptor set #%u) is used after its pool was reset", fn, what,
             s->serial);
        return NULL;
    }
    return s;
}

VKAPI_ATTR void VKAPI_CALL vkUpdateDescriptorSets(
    VkDevice dev, uint32_t nw, const VkWriteDescriptorSet *w, uint32_t nc,
    const VkCopyDescriptorSet *c)
{
    ENTER();
    get(dev, MT_DEVICE, __func__, "device");
    for (uint32_t i = 0; i < nw; i++) {
        Obj *set = get_set(w[i].dstSet, __func__, "dstSet");
        for (uint32_t k = 0; k < w[i].descriptorCount; k++) {
            Ent e = { w[i].dstBinding, w[i].dstArrayElement + k, w[i].descriptorType };
            Ent *slot = NULL;

            switch (w[i].descriptorType) {
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                e.sampler = get(w[i].pImageInfo[k].sampler, MT_SAMPLER, __func__,
                                "a combined image sampler's sampler");
                /* fall through */
            case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                e.view = get(w[i].pImageInfo[k].imageView, MT_VIEW, __func__,
                             "a descriptor's imageView");
                e.layout = w[i].pImageInfo[k].imageLayout;
                break;
            default:
                e.buffer = get(w[i].pBufferInfo[k].buffer, MT_BUFFER, __func__,
                               "a descriptor's buffer");
                break;
            }
            if (!set) {
                continue;
            }
            for (unsigned j = 0; j < set->nent; j++) {
                if (set->ent[j].binding == e.binding && set->ent[j].elem == e.elem) {
                    slot = &set->ent[j];
                }
            }
            if (!slot) {
                if (set->nent == set->capent) {
                    set->capent = set->capent ? set->capent * 2 : 32;
                    set->ent = realloc(set->ent, set->capent * sizeof(Ent));
                }
                slot = &set->ent[set->nent++];
            }
            *slot = e;
        }
    }
    LEAVE();
}

/* ---- command buffers -------------------------------------------------- */

VKAPI_ATTR VkResult VKAPI_CALL vkAllocateCommandBuffers(
    VkDevice dev, const VkCommandBufferAllocateInfo *ai, VkCommandBuffer *out)
{
    VkResult r;

    ENTER();
    get(dev, MT_DEVICE, __func__, "device");
    get(ai->commandPool, MT_CMDPOOL, __func__, "commandPool");
    if (fail(MF_AllocateCommandBuffers, &r)) {
        for (uint32_t i = 0; i < ai->commandBufferCount; i++) {
            out[i] = VK_NULL_HANDLE;    /* the specification's promise */
        }
        LEAVE();
        return r;
    }
    for (uint32_t i = 0; i < ai->commandBufferCount; i++) {
        out[i] = (VkCommandBuffer)new_obj(MT_CMDBUF);
    }
    LEAVE();
    return VK_SUCCESS;
}

static void cb_clear(Obj *cb)
{
    cb->ncmd = 0;
    cb->nref = 0;
    cb->in_pass = false;
    cb->cur_rp = cb->bound_set = cb->bound_pipe = NULL;
    cb->epoch = ++g_cb_epoch;
}

VKAPI_ATTR VkResult VKAPI_CALL vkBeginCommandBuffer(VkCommandBuffer h,
                                                    const VkCommandBufferBeginInfo *bi)
{
    VkResult r;
    Obj *cb;

    ENTER();
    cb = get(h, MT_CMDBUF, __func__, "commandBuffer");
    if (cb && (cb->state == CB_RECORDING || cb->state == CB_PENDING)) {
        viol("%s: command buffer #%u is %s", __func__, cb->serial, cb_state_name[cb->state]);
    }
    if (fail(MF_BeginCommandBuffer, &r)) {
        LEAVE();
        return r;                       /* the buffer is not recording */
    }
    if (cb && cb->state != CB_PENDING) {
        cb_clear(cb);
        cb->state = CB_RECORDING;
    }
    LEAVE();
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkEndCommandBuffer(VkCommandBuffer h)
{
    VkResult r;
    Obj *cb;

    ENTER();
    cb = get(h, MT_CMDBUF, __func__, "commandBuffer");
    if (cb && cb->state != CB_RECORDING) {
        viol("%s: command buffer #%u is not recording (it is %s)", __func__, cb->serial,
             cb_state_name[cb->state]);
        cb = NULL;
    }
    if (cb && cb->in_pass) {
        viol("%s: command buffer #%u still has a render pass open", __func__, cb->serial);
    }
    if (fail(MF_EndCommandBuffer, &r)) {
        if (cb) {
            cb->state = CB_INVALID;
        }
        LEAVE();
        return r;
    }
    if (cb) {
        cb->state = CB_EXECUTABLE;
    }
    LEAVE();
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkResetCommandBuffer(VkCommandBuffer h,
                                                    VkCommandBufferResetFlags flags)
{
    VkResult r;
    Obj *cb;

    ENTER();
    cb = get(h, MT_CMDBUF, __func__, "commandBuffer");
    if (cb && cb->state == CB_PENDING) {
        viol("%s: command buffer #%u is pending", __func__, cb->serial);
        cb = NULL;
    }
    if (fail(MF_ResetCommandBuffer, &r)) {
        LEAVE();
        return r;
    }
    if (cb) {
        cb_clear(cb);
        cb->state = CB_INITIAL;
    }
    LEAVE();
    return VK_SUCCESS;
}

/* The command buffer a command is recorded into, if it may be. */
static Obj *rec(VkCommandBuffer h, const char *fn)
{
    Obj *cb = get(h, MT_CMDBUF, fn, "commandBuffer");

    if (cb && cb->state != CB_RECORDING) {
        viol("%s: command buffer #%u is not recording (it is %s)", fn, cb->serial,
             cb_state_name[cb->state]);
        return NULL;
    }
    return cb;
}

static void add_ref(Obj *cb, Obj *o)
{
    if (!o || (o->ref_cb == cb && o->ref_epoch == cb->epoch)) {
        return;
    }
    o->ref_cb = cb;
    o->ref_epoch = cb->epoch;
    if (cb->nref == cb->capref) {
        cb->capref = cb->capref ? cb->capref * 2 : 64;
        cb->ref = realloc(cb->ref, cb->capref * sizeof(Obj *));
    }
    cb->ref[cb->nref++] = o;
}

static void add_ref_image(Obj *cb, Obj *im)
{
    if (im) {
        add_ref(cb, im);
        add_ref(cb, im->mem);
    }
}

static void add_ref_view(Obj *cb, Obj *v)
{
    if (v) {
        add_ref(cb, v);
        add_ref_image(cb, v->image);
    }
}

static void add_ref_buffer(Obj *cb, Obj *b)
{
    if (b) {
        add_ref(cb, b);
        add_ref(cb, b->mem);
    }
}

static Cmd *add_cmd(Obj *cb, int op, Obj *a, Obj *b)
{
    Cmd *c;

    if (cb->ncmd == cb->capcmd) {
        cb->capcmd = cb->capcmd ? cb->capcmd * 2 : 256;
        cb->cmd = realloc(cb->cmd, cb->capcmd * sizeof(Cmd));
    }
    c = &cb->cmd[cb->ncmd++];
    memset(c, 0, sizeof(*c));
    c->op = op;
    c->a = a;
    c->b = b;
    return c;
}

VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier(
    VkCommandBuffer h, VkPipelineStageFlags src, VkPipelineStageFlags dst,
    VkDependencyFlags dep, uint32_t nm, const VkMemoryBarrier *m, uint32_t nb,
    const VkBufferMemoryBarrier *b, uint32_t ni, const VkImageMemoryBarrier *im)
{
    Obj *cb;

    ENTER();
    cb = rec(h, __func__);
    for (uint32_t i = 0; cb && i < nm; i++) {
        if ((src & VK_PIPELINE_STAGE_ALL_COMMANDS_BIT) &&
            (dst & VK_PIPELINE_STAGE_ALL_COMMANDS_BIT) &&
            (m[i].srcAccessMask & VK_ACCESS_MEMORY_WRITE_BIT) &&
            (m[i].dstAccessMask & (VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT))) {
            if (cb->in_pass) {
                viol("%s: a barrier over all commands inside a render pass", __func__);
            }
            add_cmd(cb, C_SYNC, NULL, NULL);
        }
    }
    for (uint32_t i = 0; i < ni; i++) {
        Obj *img = get(im[i].image, MT_IMAGE, __func__, "an image barrier's image");
        if (!cb || !img) {
            continue;
        }
        if (cb->in_pass && im[i].oldLayout != im[i].newLayout) {
            viol("%s: a layout transition of image #%u inside a render pass", __func__,
                 img->serial);
        }
        Cmd *c = add_cmd(cb, C_TRANSITION, img, NULL);
        c->p[0] = im[i].oldLayout;
        c->p[1] = im[i].newLayout;
        c->p[2] = im[i].subresourceRange.baseMipLevel;
        c->p[3] = im[i].subresourceRange.levelCount;
        c->p[4] = im[i].srcAccessMask;
        add_ref_image(cb, img);
    }
    LEAVE();
}

VKAPI_ATTR void VKAPI_CALL vkCmdClearColorImage(
    VkCommandBuffer h, VkImage image, VkImageLayout layout, const VkClearColorValue *color,
    uint32_t n, const VkImageSubresourceRange *range)
{
    Obj *cb, *img;

    ENTER();
    cb = rec(h, __func__);
    img = get(image, MT_IMAGE, __func__, "image");
    if (cb && cb->in_pass) {
        viol("%s: inside a render pass", __func__);
    }
    for (uint32_t i = 0; cb && img && i < n; i++) {
        Cmd *c = add_cmd(cb, C_CLEAR, img, NULL);
        c->p[0] = layout;
        c->p[1] = range[i].baseMipLevel;
        c->p[2] = range[i].levelCount;
        c->p[3] = range[i].baseArrayLayer;
        c->p[4] = range[i].layerCount;
        add_ref_image(cb, img);
    }
    LEAVE();
}

static void rec_copy(Obj *cb, int op, Obj *img, Obj *buf, VkImageLayout layout,
                     uint32_t n, const VkBufferImageCopy *r, const char *fn)
{
    if (cb && cb->in_pass) {
        viol("%s: inside a render pass", fn);
    }
    for (uint32_t i = 0; cb && img && buf && i < n; i++) {
        Cmd *c;
        if (!r[i].imageExtent.width || !r[i].imageExtent.height || !r[i].imageExtent.depth) {
            viol("%s: region %u has a zero extent", fn, i);
        }
        c = add_cmd(cb, op, img, buf);
        c->p[0] = layout;
        c->p[1] = r[i].imageSubresource.mipLevel;
        c->p[2] = r[i].imageSubresource.baseArrayLayer;
        c->p[3] = r[i].imageSubresource.layerCount;
        c->p[4] = r[i].imageOffset.z;
        c->p[5] = r[i].imageExtent.depth;
        add_ref_image(cb, img);
        add_ref_buffer(cb, buf);
    }
}

VKAPI_ATTR void VKAPI_CALL vkCmdCopyBufferToImage(
    VkCommandBuffer h, VkBuffer src, VkImage dst, VkImageLayout layout, uint32_t n,
    const VkBufferImageCopy *r)
{
    Obj *cb, *img, *buf;

    ENTER();
    cb = rec(h, __func__);
    buf = get(src, MT_BUFFER, __func__, "srcBuffer");
    img = get(dst, MT_IMAGE, __func__, "dstImage");
    rec_copy(cb, C_COPY_B2I, img, buf, layout, n, r, __func__);
    LEAVE();
}

VKAPI_ATTR void VKAPI_CALL vkCmdCopyImageToBuffer(
    VkCommandBuffer h, VkImage src, VkImageLayout layout, VkBuffer dst, uint32_t n,
    const VkBufferImageCopy *r)
{
    Obj *cb, *img, *buf;

    ENTER();
    cb = rec(h, __func__);
    img = get(src, MT_IMAGE, __func__, "srcImage");
    buf = get(dst, MT_BUFFER, __func__, "dstBuffer");
    rec_copy(cb, C_COPY_I2B, img, buf, layout, n, r, __func__);
    LEAVE();
}

VKAPI_ATTR void VKAPI_CALL vkCmdBeginRenderPass(
    VkCommandBuffer h, const VkRenderPassBeginInfo *bi, VkSubpassContents contents)
{
    Obj *cb, *rp, *fb;

    ENTER();
    cb = rec(h, __func__);
    rp = get(bi->renderPass, MT_RENDERPASS, __func__, "renderPass");
    fb = get(bi->framebuffer, MT_FRAMEBUFFER, __func__, "framebuffer");
    if (cb && cb->in_pass) {
        viol("%s: a render pass is already open", __func__);
    }
    if (cb && rp && fb) {
        add_cmd(cb, C_BEGIN_PASS, rp, fb);
        add_ref(cb, rp);
        add_ref(cb, fb);
        for (uint32_t i = 0; i < fb->natt; i++) {
            add_ref_view(cb, fb->att[i]);
        }
        cb->in_pass = true;
        cb->cur_rp = rp;
    }
    LEAVE();
}

VKAPI_ATTR void VKAPI_CALL vkCmdEndRenderPass(VkCommandBuffer h)
{
    Obj *cb;

    ENTER();
    cb = rec(h, __func__);
    if (cb && !cb->in_pass) {
        viol("%s: no render pass is open", __func__);
    } else if (cb) {
        add_cmd(cb, C_END_PASS, cb->cur_rp, NULL);
        cb->in_pass = false;
    }
    LEAVE();
}

VKAPI_ATTR void VKAPI_CALL vkCmdBindPipeline(VkCommandBuffer h, VkPipelineBindPoint bp,
                                             VkPipeline pipe)
{
    Obj *cb, *p;

    ENTER();
    cb = rec(h, __func__);
    p = get(pipe, MT_PIPELINE, __func__, "pipeline");
    if (cb) {
        cb->bound_pipe = p;
        add_ref(cb, p);
    }
    LEAVE();
}

VKAPI_ATTR void VKAPI_CALL vkCmdBindDescriptorSets(
    VkCommandBuffer h, VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t first,
    uint32_t n, const VkDescriptorSet *sets, uint32_t ndyn, const uint32_t *dyn)
{
    Obj *cb, *s;

    ENTER();
    cb = rec(h, __func__);
    get(layout, MT_PLAYOUT, __func__, "layout");
    s = n ? get_set(sets[0], __func__, "pDescriptorSets[0]") : NULL;
    if (cb) {
        cb->bound_set = s;
    }
    LEAVE();
}

VKAPI_ATTR void VKAPI_CALL vkCmdBindIndexBuffer(VkCommandBuffer h, VkBuffer buffer,
                                                VkDeviceSize off, VkIndexType type)
{
    Obj *cb, *b;

    ENTER();
    cb = rec(h, __func__);
    b = get(buffer, MT_BUFFER, __func__, "buffer");
    if (cb) {
        add_ref_buffer(cb, b);
    }
    LEAVE();
}

VKAPI_ATTR void VKAPI_CALL vkCmdSetViewport(VkCommandBuffer h, uint32_t first, uint32_t n,
                                            const VkViewport *vp)
{
    ENTER();
    rec(h, __func__);
    LEAVE();
}

VKAPI_ATTR void VKAPI_CALL vkCmdSetScissor(VkCommandBuffer h, uint32_t first, uint32_t n,
                                           const VkRect2D *sc)
{
    ENTER();
    rec(h, __func__);
    LEAVE();
}

static void rec_draw(VkCommandBuffer h, const char *fn)
{
    Obj *cb = rec(h, fn);

    if (!cb) {
        return;
    }
    if (!cb->in_pass) {
        viol("%s: no render pass is open", fn);
    }
    if (!cb->bound_pipe || !cb->bound_set) {
        viol("%s: no %s bound", fn, cb->bound_pipe ? "descriptor set" : "pipeline");
        return;
    }
    add_cmd(cb, C_DRAW, cb->bound_set, cb->bound_pipe);
    add_ref(cb, cb->bound_set);
    add_ref(cb, cb->bound_set->pool);
    for (unsigned i = 0; i < cb->bound_set->nent; i++) {
        Ent *e = &cb->bound_set->ent[i];
        add_ref(cb, e->sampler);
        add_ref_view(cb, e->view);
        add_ref_buffer(cb, e->buffer);
    }
}

VKAPI_ATTR void VKAPI_CALL vkCmdDraw(VkCommandBuffer h, uint32_t nv, uint32_t ni,
                                     uint32_t fv, uint32_t fi)
{
    ENTER();
    rec_draw(h, __func__);
    LEAVE();
}

VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexed(VkCommandBuffer h, uint32_t nidx, uint32_t ni,
                                            uint32_t fidx, int32_t voff, uint32_t fi)
{
    ENTER();
    rec_draw(h, __func__);
    LEAVE();
}

/* ---- execution -------------------------------------------------------- */

static const char *layout_name(uint32_t l)
{
    static __thread char buf[2][24];
    static __thread int k;

    switch (l) {
    case VK_IMAGE_LAYOUT_UNDEFINED: return "UNDEFINED";
    case VK_IMAGE_LAYOUT_GENERAL:   return "GENERAL";
    }
    k ^= 1;
    snprintf(buf[k], sizeof(buf[k]), "%u", l);
    return buf[k];
}

static uint32_t img_slots_at(const Obj *im, uint32_t level)
{
    if (!im->is3d) {
        return im->layers;
    }
    return im->depth >> level ? im->depth >> level : 1;
}

static void img_set_written(Obj *im, uint32_t l0, uint32_t ln, uint32_t s0, uint32_t sn,
                            bool v)
{
    if (!im->written) {
        return;
    }
    for (uint32_t l = l0; l < im->levels && (ln == VK_REMAINING_MIP_LEVELS || l - l0 < ln);
         l++) {
        for (uint32_t s = s0; s < im->slots && (sn == VK_REMAINING_ARRAY_LAYERS || s - s0 < sn);
             s++) {
            im->written[(size_t)l * im->slots + s] = v;
        }
    }
}

static bool img_all_written(const Obj *im)
{
    if (!im->written) {
        return false;
    }
    for (uint32_t l = 0; l < im->levels; l++) {
        for (uint32_t s = 0; s < img_slots_at(im, l); s++) {
            if (!im->written[(size_t)l * im->slots + s]) {
                return false;
            }
        }
    }
    return true;
}

/* An image as a command executes: there, in the layout named, and (when it
 * is read) written everywhere since it was last without a layout. */
static void use_image(const Obj *im, uint32_t layout, bool read, const char *how)
{
    if (!im) {
        return;
    }
    if (!im->alive) {
        viol("executing: %s image #%u, which was destroyed", how, im->serial);
        return;
    }
    if (!im->mem || !im->mem->alive) {
        viol("executing: %s image #%u, which has no live memory", how, im->serial);
    }
    if (im->layout != layout) {
        viol("executing: %s image #%u as in layout %s, but it is in layout %s", how,
             im->serial, layout_name(layout), layout_name(im->layout));
    } else if (read && !img_all_written(im)) {
        viol("executing: %s image #%u, parts of which were never written", how, im->serial);
    }
}

static void use_view(const Obj *v, uint32_t layout, bool read, const char *how)
{
    if (!v) {
        return;
    }
    if (!v->alive) {
        viol("executing: %s image view #%u, which was destroyed", how, v->serial);
        return;
    }
    use_image(v->image, layout, read, how);
}

static void use_buffer(const Obj *b, const char *how)
{
    if (b && !b->alive) {
        viol("executing: %s buffer #%u, which was destroyed", how, b->serial);
    } else if (b && (!b->mem || !b->mem->alive)) {
        viol("executing: %s buffer #%u, which has no live memory", how, b->serial);
    }
}

static void execute(Obj *cb)
{
    for (unsigned i = 0; i < cb->ncmd; i++) {
        Cmd *c = &cb->cmd[i];
        Obj *im;

        switch (c->op) {
        case C_SYNC:
            g_epoch++;
            break;
        case C_TRANSITION:
            im = c->a;
            if (!im->alive) {
                viol("executing: a layout transition of image #%u, which was destroyed",
                     im->serial);
                break;
            }
            if (c->p[0] != VK_IMAGE_LAYOUT_UNDEFINED && c->p[0] != im->layout) {
                viol("executing: a transition of image #%u from layout %s, but it is in "
                     "layout %s", im->serial, layout_name(c->p[0]), layout_name(im->layout));
            }
            if (!(c->p[4] & (VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT)) &&
                im->wseen && im->wepoch == g_epoch) {
                viol("executing: a layout transition of image #%u that nothing orders "
                     "after an earlier write to it", im->serial);
            }
            im->layout = c->p[1];
            if (c->p[0] == VK_IMAGE_LAYOUT_UNDEFINED) {
                /* its contents may be discarded */
                img_set_written(im, c->p[2], c->p[3], 0, VK_REMAINING_ARRAY_LAYERS, false);
            }
            break;
        case C_CLEAR:
            im = c->a;
            use_image(im, c->p[0], false, "a clear of");
            img_set_written(im, c->p[1], c->p[2],
                            im->is3d ? 0 : c->p[3],
                            im->is3d ? VK_REMAINING_ARRAY_LAYERS : c->p[4], true);
            im->wseen = true;
            im->wepoch = g_epoch;
            break;
        case C_COPY_B2I:
            im = c->a;
            use_buffer(c->b, "a copy from");
            use_image(im, c->p[0], false, "a copy to");
            img_set_written(im, c->p[1], 1, im->is3d ? c->p[4] : c->p[2],
                            im->is3d ? c->p[5] : c->p[3], true);
            im->wseen = true;
            im->wepoch = g_epoch;
            break;
        case C_COPY_I2B:
            use_image(c->a, c->p[0], true, "a copy from");
            use_buffer(c->b, "a copy to");
            break;
        case C_BEGIN_PASS:
            if (!c->a->alive || !c->b->alive) {
                viol("executing: a render pass whose %s was destroyed",
                     c->a->alive ? "framebuffer" : "render pass object");
                break;
            }
            if (c->a->dep_in) {
                g_epoch++;
            }
            for (uint32_t k = 0; k < c->b->natt; k++) {
                use_view(c->b->att[k], VK_IMAGE_LAYOUT_GENERAL, true, "a render pass loading");
            }
            break;
        case C_END_PASS:
            if (c->a && c->a->dep_out) {
                g_epoch++;
            }
            break;
        case C_DRAW:
            if (!c->b->alive) {
                viol("executing: a draw with pipeline #%u, which was destroyed", c->b->serial);
            }
            if (!c->a->alive || c->a->set_gen != c->a->pool->gen) {
                viol("executing: a draw with descriptor set #%u, whose pool was reset",
                     c->a->serial);
                break;
            }
            for (unsigned k = 0; k < c->a->nent; k++) {
                Ent *e = &c->a->ent[k];
                if (e->sampler && !e->sampler->alive) {
                    viol("executing: a draw with sampler #%u, which was destroyed",
                         e->sampler->serial);
                }
                use_view(e->view, e->layout, true, "a draw reading");
                use_buffer(e->buffer, "a draw using");
            }
            break;
        }
    }
}

static void complete(Obj *cb)
{
    for (unsigned i = 0; i < cb->nref; i++) {
        if (cb->ref[i]->pending) {
            cb->ref[i]->pending--;
        }
    }
    cb->state = CB_INVALID;             /* one-time submit */
}

VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit(VkQueue queue, uint32_t n,
                                             const VkSubmitInfo *si, VkFence fence)
{
    VkResult r;
    Obj *f = NULL;

    ENTER();
    get(queue, MT_QUEUE, __func__, "queue");
    if (fence) {
        f = get(fence, MT_FENCE, __func__, "fence");
    }
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t k = 0; k < si[i].commandBufferCount; k++) {
            get(si[i].pCommandBuffers[k], MT_CMDBUF, __func__, "pCommandBuffers[]");
        }
    }
    if (fail(MF_QueueSubmit, &r)) {
        LEAVE();
        return r;                       /* nothing was submitted */
    }
    if (f && f->signaled) {
        viol("%s: fence #%u is signaled", __func__, f->serial);
    }
    if (f && f->nfence_cb) {
        viol("%s: fence #%u belongs to unfinished work", __func__, f->serial);
    }
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t k = 0; k < si[i].commandBufferCount; k++) {
            Obj *cb = (Obj *)si[i].pCommandBuffers[k];
            if (!cb || mock_is_garbage(cb) || cb < g_obj || cb >= g_obj + g_nobj ||
                cb->type != MT_CMDBUF || !cb->alive) {
                continue;               /* reported above */
            }
            if (cb->state != CB_EXECUTABLE) {
                viol("%s: command buffer #%u is not executable (it is %s)", __func__,
                     cb->serial, cb_state_name[cb->state]);
                continue;
            }
            execute(cb);
            cb->state = CB_PENDING;
            for (unsigned j = 0; j < cb->nref; j++) {
                cb->ref[j]->pending++;
            }
            if (f && f->nfence_cb < 4) {
                f->fence_cb[f->nfence_cb++] = cb;
            } else {
                complete(cb);
            }
        }
    }
    LEAVE();
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkWaitForFences(VkDevice dev, uint32_t n, const VkFence *fences,
                                               VkBool32 all, uint64_t timeout)
{
    VkResult r = VK_SUCCESS;
    bool failed;

    ENTER();
    while (mock_thread && g_gate) {
        pthread_cond_wait(&g_gate_cv, &g_mu);
    }
    get(dev, MT_DEVICE, __func__, "device");
    failed = fail(MF_WaitForFences, &r);
    for (uint32_t i = 0; i < n; i++) {
        Obj *f = get(fences[i], MT_FENCE, __func__, "pFences[]");
        if (!f) {
            continue;
        }
        if (f->nfence_cb) {
            if (failed && r != VK_ERROR_DEVICE_LOST) {
                /* The wait did not take place: the work goes on, and what it
                 * uses stays in use.  (A lost device has dropped it.) */
                continue;
            }
            for (unsigned k = 0; k < f->nfence_cb; k++) {
                complete(f->fence_cb[k]);
            }
            f->nfence_cb = 0;
            f->signaled = true;
        } else if (!f->signaled && !failed) {
            viol("%s: fence #%u is not signaled and nothing was submitted with it: "
                 "this wait would never return", __func__, f->serial);
        }
    }
    LEAVE();
    return failed ? r : VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkResetFences(VkDevice dev, uint32_t n, const VkFence *fences)
{
    VkResult r;
    bool failed;

    ENTER();
    get(dev, MT_DEVICE, __func__, "device");
    failed = fail(MF_ResetFences, &r);
    for (uint32_t i = 0; i < n; i++) {
        Obj *f = get(fences[i], MT_FENCE, __func__, "pFences[]");
        if (!f) {
            continue;
        }
        if (f->nfence_cb) {
            viol("%s: fence #%u belongs to unfinished work", __func__, f->serial);
        }
        if (!failed) {
            f->signaled = false;
        }
    }
    LEAVE();
    return failed ? r : VK_SUCCESS;
}
