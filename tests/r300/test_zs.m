/*
 * Render through the generated r300_fs_z with Metal, the way
 * metal_draw_r300 binds it, and check depth, stencil (incl. two-sided),
 * DEPTHENDIAN, Z16, culling and the Z-pass counter.
 */
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>
#include "../../hw/display/r300/r300_draw.h"
#include "r300_mtl.h"

#define W 64
#define H 64

static id<MTLDevice> dev;
static id<MTLCommandQueue> q;
static int fails;

#define CHECK(c, ...) do { if (!(c)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); \
    printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

/* US program: out = t0 (the colour varying), rgb and alpha. */
static char *make_msl(void)
{
    static R300State st;
    R300FSDesc d;
    const char *err;

    r300_state_reset(&st);
    r300_state_write(&st, 0x4600, 0);                   /* US_CONFIG: 1 node */
    r300_state_write(&st, 0x4608, 0);                   /* US_CODE_OFFSET */
    r300_state_write(&st, 0x461C, 0);                   /* CODE_ADDR_3: alu 0, 1 inst */
    r300_state_write(&st, 0x46C0, (7u << 26));          /* rgb: src0 t0, out rgb */
    r300_state_write(&st, 0x48C0, 0 | (21u << 7) | (20u << 14)); /* MAD t0.rgb * 1 + 0 */
    r300_state_write(&st, 0x47C0, (1u << 24));          /* alpha: src0 t0, out a */
    r300_state_write(&st, 0x49C0, 9 | (17u << 7) | (16u << 14));
    memset(d.route, -1, sizeof(d.route));
    d.route[0] = 0;
    char *m = r300_us_to_glsl(&st, &d, 0, &err);
    if (!m) {
        printf("msl: %s\n", err);
    }
    return m;
}

typedef struct Target {
    id<MTLTexture> col, z;
    id<MTLBuffer> zbuf;     /* backing of z (linear, like VRAM) */
    bool z16;
} Target;

static Target make_target(bool z16, uint32_t zinit)
{
    Target t = { 0 };
    MTLTextureDescriptor *d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:W height:H mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    d.storageMode = MTLStorageModeShared;
    t.col = [dev newTextureWithDescriptor:d];
    uint32_t zero[W * H];
    memset(zero, 0, sizeof(zero));
    [t.col replaceRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0 withBytes:zero bytesPerRow:W * 4];
    MTLPixelFormat zpf = z16 ? MTLPixelFormatR16Uint : MTLPixelFormatR32Uint;
    size_t bpp = z16 ? 2 : 4;
    t.zbuf = [dev newBufferWithLength:W * H * bpp options:MTLResourceStorageModeShared];
    for (int i = 0; i < W * H; i++) {
        if (z16) ((uint16_t *)t.zbuf.contents)[i] = zinit;
        else ((uint32_t *)t.zbuf.contents)[i] = zinit;
    }
    MTLTextureDescriptor *zd = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:zpf width:W height:H mipmapped:NO];
    zd.usage = MTLTextureUsageRenderTarget;
    zd.storageMode = MTLStorageModeShared;
    t.z = [t.zbuf newTextureWithDescriptor:zd offset:0 bytesPerRow:W * bpp];
    t.z16 = z16;
    return t;
}

static id<MTLRenderPipelineState> pipe_for(const char *msl, bool z16)
{
    NSError *e = nil;
    NSArray<id<MTLFunction>> *lib = r300_test_lib(dev, msl);
    if (!lib) return nil;
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = lib[R300_STAGE_VS];
    pd.fragmentFunction = lib[R300_STAGE_FS_Z];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    pd.colorAttachments[1].pixelFormat = z16 ? MTLPixelFormatR16Uint : MTLPixelFormatR32Uint;
    id<MTLRenderPipelineState> p = [dev newRenderPipelineStateWithDescriptor:pd error:&e];
    if (!p) printf("pipeline: %s\n", e.localizedDescription.UTF8String);
    return p;
}

static id<MTLBuffer> g_zp;

static R300FSUniforms base_uniforms(void)
{
    R300FSUniforms u;
    memset(&u, 0, sizeof(u));
    u.out_sel[0] = 1; u.out_sel[1] = 2; u.out_sel[2] = 3; u.out_sel[3] = 0;
    u.chanmask = 0xF;
    u.clip_rule = 0xFFFF;
    for (int i = 0; i < 4; i++) {
        u.cliprect[i][0] = u.cliprect[i][1] = 0;
        u.cliprect[i][2] = u.cliprect[i][3] = 4095;
    }
    return u;
}

/* A rectangle in pixels as two triangles; ccw = counter-clockwise on screen. */
static void rect(R300Vertex *v, float x0, float y0, float x1, float y1, float z,
                 const float c[4], bool ccw)
{
    float X0 = 2 * x0 / W - 1, X1 = 2 * x1 / W - 1;
    float Y0 = 1 - 2 * y0 / H, Y1 = 1 - 2 * y1 / H;     /* y up */
    /* on screen (y down): TL, BL, BR is counter-clockwise */
    float p[6][2] = { { X0, Y0 }, { X0, Y1 }, { X1, Y1 }, { X0, Y0 }, { X1, Y1 }, { X1, Y0 } };
    for (int i = 0; i < 6; i++) {
        int k = ccw ? i : (i / 3) * 3 + (2 - i % 3);
        memset(&v[i], 0, sizeof(v[i]));
        v[i].pos[0] = p[k][0]; v[i].pos[1] = p[k][1]; v[i].pos[2] = z; v[i].pos[3] = 1;
        memcpy(v[i].v[0], c, 16);
    }
}

static void draw(Target *t, id<MTLRenderPipelineState> p, const R300FSUniforms *u,
                 const R300Vertex *v, int n, uint32_t cull)
{
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
    [e setDepthClipMode:MTLDepthClipModeClamp];
    [e setFrontFacingWinding:r300_front_ccw(cull) ? MTLWindingCounterClockwise
                                                  : MTLWindingClockwise];
    [e setCullMode:(cull & R300_CULL_FRONT) ? MTLCullModeFront :
                   (cull & R300_CULL_BACK) ? MTLCullModeBack : MTLCullModeNone];
    [e setVertexBytes:v length:sizeof(R300Vertex) * n atIndex:0];
    [e setVertexBytes:(float[4]){ 0 } length:16 atIndex:1];
    [e setFragmentBytes:u length:sizeof(*u) atIndex:0];
    [e setFragmentBuffer:g_zp offset:0 atIndex:1];
    [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:n];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

static uint32_t px(Target *t, int x, int y)
{
    uint32_t v;
    [t->col getBytes:&v bytesPerRow:W * 4 fromRegion:MTLRegionMake2D(x, y, 1, 1) mipmapLevel:0];
    return v;   /* bytes R,G,B,A -> 0xAABBGGRR */
}

static uint32_t zw(Target *t, int x, int y)
{
    return t->z16 ? ((uint16_t *)t->zbuf.contents)[y * W + x]
                  : ((uint32_t *)t->zbuf.contents)[y * W + x];
}

static const float RED[4] = { 1, 0, 0, 1 }, GREEN[4] = { 0, 1, 0, 1 }, BLUE[4] = { 0, 0, 1, 1 };
#define PRED 0xFF0000FFu
#define PGREEN 0xFF00FF00u
#define PBLUE 0xFFFF0000u

int main(void)
{
    @autoreleasepool {
        dev = MTLCreateSystemDefaultDevice();
        q = [dev newCommandQueue];
        g_zp = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
        char *msl = make_msl();
        if (!msl) return 1;
        id<MTLRenderPipelineState> p32 = pipe_for(msl, false), p16 = pipe_for(msl, true);
        if (!p32 || !p16) return 1;
        R300Vertex v[12];

        /* 1. Z LESS + write, Z24S8, no swap: near red first, far green after. */
        {
            Target t = make_target(false, 0xFFFFFF00u);
            R300FSUniforms u = base_uniforms();
            u.zinfo[0] = 2 | 4;             /* Z test + write */
            u.zinfo[1] = 1;                 /* LESS */
            u.zinfo[2] = 0x00FFFF00;
            rect(v, 8, 8, 40, 40, 0.25f, RED, true);
            draw(&t, p32, &u, v, 6, 0);
            rect(v, 24, 24, 56, 56, 0.75f, GREEN, true);
            draw(&t, p32, &u, v, 6, 0);
            CHECK(px(&t, 30, 30) == PRED, "overlap %08x want red", px(&t, 30, 30));
            CHECK(px(&t, 50, 50) == PGREEN, "far only %08x", px(&t, 50, 50));
            CHECK(zw(&t, 30, 30) >> 8 == 0x400000, "z word %08x", zw(&t, 30, 30));
            CHECK((zw(&t, 30, 30) & 0xFF) == 0, "stencil kept");
            CHECK(zw(&t, 2, 2) == 0xFFFFFF00u, "untouched z %08x", zw(&t, 2, 2));
            /* no Z write: a nearer blue passes but leaves Z */
            u.zinfo[0] = 2;
            rect(v, 44, 44, 60, 60, 0.5f, BLUE, true);
            draw(&t, p32, &u, v, 6, 0);
            CHECK(px(&t, 50, 50) == PBLUE, "blue %08x", px(&t, 50, 50));
            CHECK(zw(&t, 50, 50) >> 8 == 0xBFFFFF || zw(&t, 50, 50) >> 8 == 0xC00000,
                  "z not written %08x", zw(&t, 50, 50));
        }
        /* 2. DEPTHENDIAN = 2 (dword swap): memory holds big-endian words. */
        {
            Target t = make_target(false, __builtin_bswap32(0xFFFFFF00u));
            R300FSUniforms u = base_uniforms();
            u.zinfo[0] = 2 | 4; u.zinfo[1] = 2; u.zinfo[2] = 0x00FFFF00;
            u.zinfo[3] = 2;
            rect(v, 0, 0, 32, 32, 0.5f, RED, true);
            draw(&t, p32, &u, v, 6, 0);
            rect(v, 0, 0, 32, 32, 0.6f, GREEN, true);
            draw(&t, p32, &u, v, 6, 0);
            uint32_t w = __builtin_bswap32(zw(&t, 5, 5));
            CHECK(px(&t, 5, 5) == PRED, "swapped depth test %08x", px(&t, 5, 5));
            CHECK((w >> 8) == 0x800000 || (w >> 8) == 0x7FFFFF, "swapped word %08x", w);
        }
        /* 3. Z16 GEQUAL. */
        {
            Target t = make_target(true, 0);
            R300FSUniforms u = base_uniforms();
            u.zinfo[0] = 2 | 4; u.zinfo[1] = 4; u.zinfo[3] = R300_ZFMT_Z16;
            rect(v, 0, 0, 32, 32, 0.5f, RED, true);
            draw(&t, p16, &u, v, 6, 0);
            rect(v, 0, 0, 32, 32, 0.25f, GREEN, true);
            draw(&t, p16, &u, v, 6, 0);
            CHECK(px(&t, 5, 5) == PRED, "z16 %08x", px(&t, 5, 5));
            CHECK(zw(&t, 5, 5) == 0x8000 || zw(&t, 5, 5) == 0x7FFF, "z16 word %04x", zw(&t, 5, 5));
        }
        /* 4. Stencil: write 1 with ALWAYS/REPLACE, then draw where EQUAL 1. */
        {
            Target t = make_target(false, 0xFFFFFF00u);
            R300FSUniforms u = base_uniforms();
            u.zinfo[0] = 1;                             /* stencil only */
            u.zinfo[1] = (7u << 3) | (2u << 9);         /* ALWAYS, zpass REPLACE */
            u.zinfo[2] = 1 | (0xFFu << 8) | (0xFFu << 16);
            u.chanmask = 0;                             /* no colour writes */
            rect(v, 16, 16, 48, 48, 0.5f, RED, true);
            draw(&t, p32, &u, v, 6, 0);
            CHECK((zw(&t, 20, 20) & 0xFF) == 1, "stencil written %08x", zw(&t, 20, 20));
            CHECK(zw(&t, 20, 20) >> 8 == 0xFFFFFF, "z kept %08x", zw(&t, 20, 20));
            CHECK(px(&t, 20, 20) == 0, "masked colour %08x", px(&t, 20, 20));
            u.chanmask = 0xF;
            u.zinfo[1] = (3u << 3);                     /* EQUAL, keep */
            rect(v, 0, 0, 64, 64, 0.5f, BLUE, true);
            draw(&t, p32, &u, v, 6, 0);
            CHECK(px(&t, 20, 20) == PBLUE, "inside %08x", px(&t, 20, 20));
            CHECK(px(&t, 4, 4) == 0, "outside %08x", px(&t, 4, 4));
            /* write mask: INCR through a 0x02 write mask leaves bit 0 */
            u.zinfo[1] = (7u << 3) | (3u << 9);
            u.zinfo[2] = 0 | (0xFFu << 8) | (0x02u << 16);
            rect(v, 16, 16, 48, 48, 0.5f, RED, true);
            draw(&t, p32, &u, v, 6, 0);
            CHECK((zw(&t, 20, 20) & 0xFF) == 3, "masked incr %02x", zw(&t, 20, 20) & 0xFF);
        }
        /* 5. Two-sided stencil and front_facing vs SU_CULL_MODE.FACE. */
        {
            Target t = make_target(false, 0xFFFFFF00u);
            R300FSUniforms u = base_uniforms();
            u.zinfo[0] = 1 | 16;                        /* stencil, separate back */
            u.zinfo[1] = (0u << 3) | (7u << 15);        /* front NEVER, back ALWAYS */
            u.zinfo[2] = 0x00FFFF00;
            rect(v, 0, 0, 32, 32, 0.5f, RED, true);     /* CCW: front */
            rect(v + 6, 32, 0, 64, 32, 0.5f, GREEN, false); /* CW: back */
            draw(&t, p32, &u, v, 12, 0);                /* FACE_CCW */
            CHECK(px(&t, 5, 5) == 0, "front face ran back test %08x", px(&t, 5, 5));
            CHECK(px(&t, 40, 5) == PGREEN, "back face %08x", px(&t, 40, 5));
            draw(&t, p32, &u, v, 12, R300_FACE_CW);     /* now CW is front */
            CHECK(px(&t, 5, 5) == PRED, "FACE_CW back %08x", px(&t, 5, 5));
            /* without FRONT_BACK both faces use the front state */
            Target t2 = make_target(false, 0xFFFFFF00u);
            u.zinfo[0] = 1;
            draw(&t2, p32, &u, v, 12, 0);
            CHECK(px(&t2, 5, 5) == 0 && px(&t2, 40, 5) == 0, "single-sided %08x %08x",
                  px(&t2, 5, 5), px(&t2, 40, 5));
        }
        /* 6. Culling. */
        {
            R300FSUniforms u = base_uniforms();
            rect(v, 0, 0, 32, 32, 0.5f, RED, true);
            rect(v + 6, 32, 0, 64, 32, 0.5f, GREEN, false);
            Target t = make_target(false, 0xFFFFFF00u);
            draw(&t, p32, &u, v, 12, R300_CULL_BACK);
            CHECK(px(&t, 5, 5) == PRED && px(&t, 40, 5) == 0, "cull back %08x %08x",
                  px(&t, 5, 5), px(&t, 40, 5));
            Target t2 = make_target(false, 0xFFFFFF00u);
            draw(&t2, p32, &u, v, 12, R300_CULL_FRONT);
            CHECK(px(&t2, 5, 5) == 0 && px(&t2, 40, 5) == PGREEN, "cull front %08x %08x",
                  px(&t2, 5, 5), px(&t2, 40, 5));
            Target t3 = make_target(false, 0xFFFFFF00u);
            draw(&t3, p32, &u, v, 12, R300_CULL_BACK | R300_FACE_CW);
            CHECK(px(&t3, 5, 5) == 0 && px(&t3, 40, 5) == PGREEN, "cull back, CW front");
        }
        /* 7. Z-pass counter: a 10x10 quad, half hidden. */
        {
            Target t = make_target(false, 0xFFFFFF00u);
            R300FSUniforms u = base_uniforms();
            u.zinfo[0] = 2 | 4; u.zinfo[1] = 1; u.zinfo[2] = 0x00FFFF00;
            rect(v, 0, 0, 5, 10, 0.1f, RED, true);
            draw(&t, p32, &u, v, 6, 0);
            *(uint32_t *)g_zp.contents = 0;
            u.zpass_count = 1;
            rect(v, 0, 0, 10, 10, 0.5f, GREEN, true);
            draw(&t, p32, &u, v, 6, 0);
            uint32_t n = *(uint32_t *)g_zp.contents;
            CHECK(n == 50, "zpass count %u want 50", n);
        }
        printf(fails ? "test_zs: %d FAILED\n" : "test_zs: PASS\n", fails);
    }
    return fails != 0;
}
