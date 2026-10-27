/*
 * GPU time of the generated r300_fs_z without and with early depth/stencil
 * rejection (R300_GLSL_EARLY_Z), in scenes that are all kept, all rejected,
 * half rejected per pixel (every 2x2 quad mixed) and half in 16x16 blocks.
 * Times are command-buffer GPU intervals (GPUEndTime - GPUStartTime) only;
 * pipelines, textures and geometry are made beforehand and both variants
 * warmed; off/on alternate ABBA.  No depth writes, so a scene stays what it
 * is across repetitions.  Not part of run.sh.  AI-generated (Claude).
 */
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../hw/display/r300/r300_draw.h"
#include "r300_mtl.h"

#define W 1024
#define H 1024
#define LAYERS 8        /* full-screen rectangles per draw */
#define REPS 20         /* command buffers per measurement */
#define ROUNDS 4        /* ABBA rounds */

static id<MTLDevice> dev;
static id<MTLCommandQueue> q;
static id<MTLBuffer> g_zp;

/* One node: a chain of ntex dependent fetches t0 -> t1 -> t2 -> ... (the
 * last into t1), out A = t1. */
static void program_chain(R300State *st, unsigned ntex)
{
    r300_state_reset(st);
    r300_state_write(st, 0x4600, 1u << 3);
    r300_state_write(st, 0x4608, 0);
    r300_state_write(st, 0x461C, (ntex - 1) << 17);
    for (unsigned k = 0; k < ntex; k++) {
        unsigned src = k == 0 ? 0 : k + 1, dst = k + 1 == ntex ? 1 : k + 2;
        r300_state_write(st, 0x4620 + 4 * k, src | (dst << 6) | (0u << 11) | (1u << 15));
    }
    r300_state_write(st, 0x46C0, 1 | (7u << 26));
    r300_state_write(st, 0x48C0, 0 | (21u << 7) | (20u << 14));
    r300_state_write(st, 0x47C0, 1 | (1u << 24));
    r300_state_write(st, 0x49C0, 9 | (17u << 7) | (16u << 14));
    r300_state_write(st, 0x4E38, (6u << 21) | W);
    r300_state_write(st, 0x46A4, 0 | (1u << 8) | (2u << 10) | (3u << 12));
}

static id<MTLRenderPipelineState> make_pipe(R300State *st, uint32_t flags)
{
    R300FSDesc d;
    const char *err;
    memset(d.route, -1, sizeof(d.route));
    d.route[0] = 0;
    char *m = r300_us_to_glsl(st, &d, flags, &err);
    if (!m) { printf("glsl: %s\n", err); exit(1); }
    if ((strstr(m, "r300_zpreview(zb") != NULL) != ((flags & R300_GLSL_EARLY_Z) != 0)) {
        printf("variant text wrong\n"); exit(1);
    }
    NSArray<id<MTLFunction>> *lib = r300_test_lib(dev, m);
    free(m);
    if (!lib) exit(1);
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = lib[R300_STAGE_VS];
    pd.fragmentFunction = lib[R300_STAGE_FS_Z];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    pd.colorAttachments[1].pixelFormat = MTLPixelFormatR32Uint;
    NSError *e = nil;
    id<MTLRenderPipelineState> p = [dev newRenderPipelineStateWithDescriptor:pd error:&e];
    if (!p) { printf("pipeline: %s\n", e.localizedDescription.UTF8String); exit(1); }
    return p;
}

static uint32_t zword(float z)
{
    return (uint32_t)(z * 16777215.0f) << 8;
}

typedef struct Target { id<MTLTexture> col, z; id<MTLBuffer> zb; } Target;

static Target target(int scene)
{
    Target t;
    MTLTextureDescriptor *d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:W height:H mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget;
    d.storageMode = MTLStorageModeShared;
    t.col = [dev newTextureWithDescriptor:d];
    {
        static uint8_t grey[W * H * 4];         /* the same start for every target */
        memset(grey, 0x40, sizeof(grey));
        [t.col replaceRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0 withBytes:grey
                 bytesPerRow:W * 4];
    }
    t.zb = [dev newBufferWithLength:W * H * 4 options:MTLResourceStorageModeShared];
    uint32_t *z = t.zb.contents;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            bool reject;
            switch (scene) {
            case 0: reject = false; break;
            case 1: reject = true; break;
            case 2: reject = (x ^ y) & 1; break;
            default: reject = ((x >> 4) ^ (y >> 4)) & 1; break;
            }
            z[y * W + x] = zword(reject ? 0.25f : 0.75f);
        }
    }
    MTLTextureDescriptor *zd = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint width:W height:H mipmapped:NO];
    zd.usage = MTLTextureUsageRenderTarget;
    zd.storageMode = MTLStorageModeShared;
    t.z = [t.zb newTextureWithDescriptor:zd offset:0 bytesPerRow:W * 4];
    return t;
}

static id<MTLBuffer> g_vb;
static id<MTLTexture> g_tex;
static id<MTLSamplerState> g_smp;
static R300FSUniforms g_u;

/* REPS command buffers; returns their summed GPU time in ms. */
static double measure(Target *t, id<MTLRenderPipelineState> p)
{
    double sum = 0;
    for (int r = 0; r < REPS; r++) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = t->col;
        rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        rp.colorAttachments[1].texture = t->z;
        rp.colorAttachments[1].loadAction = MTLLoadActionLoad;
        rp.colorAttachments[1].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
        [e setRenderPipelineState:p];
        [e setVertexBuffer:g_vb offset:0 atIndex:0];
        [e setVertexBytes:(float[4]){ 0 } length:16 atIndex:1];
        [e setFragmentBytes:&g_u length:sizeof(g_u) atIndex:0];
        [e setFragmentBuffer:g_zp offset:0 atIndex:1];
        [e setFragmentTexture:g_tex atIndex:0];
        [e setFragmentSamplerState:g_smp atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6 * LAYERS];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            printf("command buffer failed\n");
            exit(1);
        }
        if (!(cb.GPUStartTime > 0 && cb.GPUEndTime >= cb.GPUStartTime)) {
            printf("invalid GPU interval %f..%f\n", cb.GPUStartTime, cb.GPUEndTime);
            exit(1);
        }
        sum += (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
    }
    return sum;
}

int main(void)
{
    @autoreleasepool {
        dev = MTLCreateSystemDefaultDevice();
        q = [dev newCommandQueue];
        g_zp = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];

        /* LAYERS full-screen rectangles at depth 0.5, uv over 0..4. */
        static R300Vertex v[6 * LAYERS];
        float p[6][2] = { { -1, 1 }, { -1, -1 }, { 1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };
        for (int l = 0; l < LAYERS; l++) {
            for (int i = 0; i < 6; i++) {
                R300Vertex *x = &v[l * 6 + i];
                memset(x, 0, sizeof(*x));
                x->pos[0] = p[i][0]; x->pos[1] = p[i][1]; x->pos[2] = 0.5f; x->pos[3] = 1;
                x->v[0][0] = (p[i][0] + 1) * 2 + l * 0.37f;
                x->v[0][1] = (1 - p[i][1]) * 2;
                x->v[0][3] = 1;
                for (int k = 0; k < 8; k++) x->ucp[k] = 1;
            }
        }
        g_vb = [dev newBufferWithBytes:v length:sizeof(v) options:MTLResourceStorageModeShared];
        MTLTextureDescriptor *td = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:256 height:256
                                     mipmapped:YES];
        g_tex = [dev newTextureWithDescriptor:td];
        static uint32_t texels[256 * 256];
        uint32_t rng = 1;
        for (unsigned l = 0; l < g_tex.mipmapLevelCount; l++) {
            unsigned w = 256 >> l;
            for (unsigned i = 0; i < w * w; i++) {
                rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                texels[i] = rng;
            }
            [g_tex replaceRegion:MTLRegionMake2D(0, 0, w, w) mipmapLevel:l withBytes:texels
                     bytesPerRow:w * 4];
        }
        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
        sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear;
        sd.mipFilter = MTLSamplerMipFilterLinear;
        sd.sAddressMode = sd.tAddressMode = MTLSamplerAddressModeRepeat;
        g_smp = [dev newSamplerStateWithDescriptor:sd];

        memset(&g_u, 0, sizeof(g_u));
        g_u.out_sel[0] = 1; g_u.out_sel[1] = 2; g_u.out_sel[2] = 3; g_u.out_sel[3] = 0;
        g_u.chanmask = 0xF;
        g_u.clip_rule = 0xFFFF;
        for (int i = 0; i < 4; i++) g_u.cliprect[i][2] = g_u.cliprect[i][3] = 4095;
        for (int k = 0; k < R300_NUM_TEX_UNITS; k++) {
            g_u.tex_swz[k][0] = 0; g_u.tex_swz[k][1] = 1; g_u.tex_swz[k][2] = 2; g_u.tex_swz[k][3] = 3;
        }
        g_u.tex_info[0][0] = 1; g_u.tex_info[0][2] = 256; g_u.tex_info[0][3] = 256;
        g_u.tex_lod[0][2] = 8; g_u.tex_lod[0][3] = 2;
        g_u.zinfo[0] = 2;               /* Z test, no Z write, no stencil */
        g_u.zinfo[1] = 1;               /* LESS */

        static const char *scenes[4] = { "all kept", "all rejected", "half, per pixel",
                                         "half, 16x16 blocks" };
        static const unsigned chains[2] = { 1, 4 };
        int differ = 0;
        printf("GPU ms per %d command buffers of %d full-screen %dx%d layers; "
               "mean of %d ABBA rounds (min-max)\n", REPS, LAYERS, W, H, ROUNDS);
        for (int ci = 0; ci < 2; ci++) {
            static R300State st;
            program_chain(&st, chains[ci]);
            id<MTLRenderPipelineState> pv[2] = { make_pipe(&st, 0), make_pipe(&st, R300_GLSL_EARLY_Z) };
            for (int sc = 0; sc < 4; sc++) {
                Target t = target(sc);
                measure(&t, pv[0]);                     /* warm both */
                measure(&t, pv[1]);
                double s[2] = { 0, 0 }, lo[2] = { 1e9, 1e9 }, hi[2] = { 0, 0 };
                for (int r = 0; r < ROUNDS; r++) {
                    int order[4] = { 0, 1, 1, 0 };
                    for (int k = 0; k < 4; k++) {
                        int v = r & 1 ? 1 - order[k] : order[k];
                        double ms = measure(&t, pv[v]) / 1;
                        s[v] += ms / 2;
                        if (ms < lo[v]) lo[v] = ms;
                        if (ms > hi[v]) hi[v] = ms;
                    }
                }
                double off = s[0] / ROUNDS, on = s[1] / ROUNDS;
                /* Outputs: both variants from fresh targets, compared. */
                Target a = target(sc), b = target(sc);
                measure(&a, pv[0]);
                measure(&b, pv[1]);
                static uint8_t ca[W * H * 4], cbuf[W * H * 4];
                [a.col getBytes:ca bytesPerRow:W * 4 fromRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0];
                [b.col getBytes:cbuf bytesPerRow:W * 4 fromRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0];
                bool same = !memcmp(ca, cbuf, sizeof(ca)) && !memcmp(a.zb.contents, b.zb.contents, W * H * 4);
                printf("%u fetch%s  %-20s off %8.2f (%.2f-%.2f)  on %8.2f (%.2f-%.2f)  on/off %.3f  %s\n",
                       chains[ci], chains[ci] > 1 ? "es" : "  ", scenes[sc], off, lo[0], hi[0],
                       on, lo[1], hi[1], on / off, same ? "outputs equal" : "OUTPUTS DIFFER");
                differ += !same;
            }
        }
        if (differ) {
            printf("FAIL: %d scene(s) with different outputs\n", differ);
            return 1;
        }
    }
    return 0;
}
