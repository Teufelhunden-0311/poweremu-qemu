/*
 * Rendering-completeness features of the generated MSL, run with Metal:
 * mip level selection, 3D and cube lookups for shader-decoded formats,
 * border colours, fog, logic ops, multiple render targets, depth from the
 * program (OMASK_W), polygon offset and RB3D discard-src-pixels.
 */
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../../hw/display/r300/r300_draw.h"

#define W 64
#define H 64

static id<MTLDevice> dev;
static id<MTLCommandQueue> q;
static id<MTLBuffer> g_zp;
static int fails;

#define CHECK(c, ...) do { if (!(c)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); \
    printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)
#define NEAR(a, b, e) (fabsf((a) - (b)) < (e))

/* US program: tex ? t1 = sample(unit 0, t0) : t1 = t0; out A = t1.  With
 * mrt, a second instruction writes white to target B; with w, the alpha
 * of constant 0 goes to the depth output. */
static void program(R300State *st, bool tex, bool mrt, bool w)
{
    unsigned nalu = 1 + mrt + w;
    r300_state_reset(st);
    r300_state_write(st, 0x4600, tex ? 1u << 3 : 0);
    r300_state_write(st, 0x4608, 0);
    r300_state_write(st, 0x461C, (nalu - 1) << 6);
    r300_state_write(st, 0x4620, 0 | (1u << 6) | (0u << 11) | (1u << 15));
    unsigned src = tex ? 1 : 0;
    r300_state_write(st, 0x46C0, src | (7u << 26));
    r300_state_write(st, 0x48C0, 0 | (21u << 7) | (20u << 14));
    r300_state_write(st, 0x47C0, src | (1u << 24));
    r300_state_write(st, 0x49C0, 9 | (17u << 7) | (16u << 14));
    unsigned i = 1;
    if (mrt) {          /* target B: 0 * 1 + 1 */
        r300_state_write(st, 0x46C0 + 4 * i, (7u << 26) | (1u << 29));
        r300_state_write(st, 0x48C0 + 4 * i, 20 | (21u << 7) | (21u << 14));
        r300_state_write(st, 0x47C0 + 4 * i, (1u << 24) | (1u << 25));
        r300_state_write(st, 0x49C0 + 4 * i, 16 | (17u << 7) | (17u << 14));
        i++;
    }
    if (w) {            /* depth = c0.a */
        r300_state_write(st, 0x46C0 + 4 * i, 0);
        r300_state_write(st, 0x48C0 + 4 * i, 20 | (20u << 7) | (20u << 14));
        r300_state_write(st, 0x47C0 + 4 * i, 32 | (1u << 27));
        r300_state_write(st, 0x49C0 + 4 * i, 9 | (17u << 7) | (16u << 14));
    }
    r300_state_write(st, 0x4E38, (6u << 21) | W);
    r300_state_write(st, 0x46A4, 0 | (1u << 8) | (2u << 10) | (3u << 12));
    r300_state_write(st, 0x46A8, 15);
    r300_state_write(st, 0x46AC, 15);
    r300_state_write(st, 0x46B0, 15);
    if (mrt) {
        r300_state_write(st, 0x4E3C, (6u << 21) | W);
        r300_state_write(st, 0x46A8, 0 | (1u << 8) | (2u << 10) | (3u << 12));
    }
}

static id<MTLLibrary> lib_for(R300State *st)
{
    R300FSDesc d;
    const char *err;
    NSError *e = nil;
    memset(d.route, -1, sizeof(d.route));
    d.route[0] = 0;
    char *m = r300_us_to_msl(st, &d, &err);
    if (!m) { printf("msl: %s\n", err); fails++; return nil; }
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(m) options:nil error:&e];
    if (!lib) { printf("compile: %s\n%s\n", e.localizedDescription.UTF8String, m); fails++; }
    free(m);
    return lib;
}

static id<MTLRenderPipelineState> pipe_for(id<MTLLibrary> lib, unsigned ncb, bool z)
{
    NSError *e = nil;
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [lib newFunctionWithName:@"r300_vs"];
    pd.fragmentFunction = [lib newFunctionWithName:z ? @"r300_fs_z" : @"r300_fs"];
    for (unsigned k = 0; k < ncb; k++) {
        pd.colorAttachments[k].pixelFormat = MTLPixelFormatRGBA8Unorm;
    }
    if (z) {
        pd.colorAttachments[ncb].pixelFormat = MTLPixelFormatR32Uint;
    }
    id<MTLRenderPipelineState> p = [dev newRenderPipelineStateWithDescriptor:pd error:&e];
    if (!p) { printf("pipeline: %s\n", e.localizedDescription.UTF8String); fails++; }
    return p;
}

static R300FSUniforms base_uniforms(void)
{
    R300FSUniforms u;
    memset(&u, 0, sizeof(u));
    u.out_sel[0] = 1; u.out_sel[1] = 2; u.out_sel[2] = 3; u.out_sel[3] = 0;
    u.chanmask = 0xF;
    u.clip_rule = 0xFFFF;
    for (int i = 0; i < 4; i++) {
        u.cliprect[i][2] = u.cliprect[i][3] = 4095;
    }
    for (int k = 0; k < R300_NUM_TEX_UNITS; k++) {
        u.tex_swz[k][0] = 0; u.tex_swz[k][1] = 1; u.tex_swz[k][2] = 2; u.tex_swz[k][3] = 3;
    }
    return u;
}

/* A counter-clockwise quad over pixels [0, px) with v0 = c, or v0 = uv
 * spanning 0..1 when c is NULL; depth z. */
static void quad(R300Vertex *v, float px, const float c[4], float z)
{
    float e = -1 + 2 * px / W, f = 1 - 2 * px / H;
    float p[6][2] = { { -1, 1 }, { -1, f }, { e, f }, { -1, 1 }, { e, f }, { e, 1 } };
    for (int i = 0; i < 6; i++) {
        memset(&v[i], 0, sizeof(v[i]));
        v[i].pos[0] = p[i][0]; v[i].pos[1] = p[i][1]; v[i].pos[2] = z; v[i].pos[3] = 1;
        if (c) {
            memcpy(v[i].v[0], c, 16);
        } else {
            v[i].v[0][0] = (p[i][0] + 1) / (e + 1);
            v[i].v[0][1] = (1 - p[i][1]) / (1 - f);
            v[i].v[0][3] = 1;
        }
        for (int k = 0; k < 8; k++) v[i].ucp[k] = 1;
    }
}

static id<MTLTexture> rgba8(uint8_t fill)
{
    MTLTextureDescriptor *d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:W height:H mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    d.storageMode = MTLStorageModeShared;
    id<MTLTexture> t = [dev newTextureWithDescriptor:d];
    static uint8_t buf[W * H * 4];
    memset(buf, fill, sizeof(buf));
    [t replaceRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0 withBytes:buf bytesPerRow:W * 4];
    return t;
}

static void px(id<MTLTexture> t, int x, int y, uint8_t o[4])
{
    [t getBytes:o bytesPerRow:4 fromRegion:MTLRegionMake2D(x, y, 1, 1) mipmapLevel:0];
}

typedef struct Pass {
    id<MTLTexture> cb[2];
    unsigned ncb;
    id<MTLTexture> z;
    id<MTLTexture> tex;
    id<MTLSamplerState> smp;
} Pass;

static void run(Pass *ps, id<MTLRenderPipelineState> p, const R300FSUniforms *u,
                const R300Vertex *v, unsigned nv)
{
    id<MTLCommandBuffer> cb = [q commandBuffer];
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    for (unsigned k = 0; k < ps->ncb; k++) {
        rp.colorAttachments[k].texture = ps->cb[k];
        rp.colorAttachments[k].loadAction = MTLLoadActionLoad;
        rp.colorAttachments[k].storeAction = MTLStoreActionStore;
    }
    if (ps->z) {
        rp.colorAttachments[ps->ncb].texture = ps->z;
        rp.colorAttachments[ps->ncb].loadAction = MTLLoadActionLoad;
        rp.colorAttachments[ps->ncb].storeAction = MTLStoreActionStore;
    }
    id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
    [e setRenderPipelineState:p];
    [e setFrontFacingWinding:MTLWindingCounterClockwise];
    [e setVertexBytes:v length:sizeof(R300Vertex) * nv atIndex:0];
    [e setFragmentBytes:u length:sizeof(*u) atIndex:0];
    [e setFragmentBuffer:g_zp offset:0 atIndex:1];
    if (ps->tex) {
        [e setFragmentTexture:ps->tex atIndex:0];
        [e setFragmentSamplerState:ps->smp ? ps->smp :
            [dev newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]] atIndex:0];
    }
    [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:nv];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

/* Shader-decoded texture: bytes as they lie in VRAM, viewed as uint
 * dwords rowel to a row (r300_texture_raw's layout). */
static id<MTLTexture> raw_tex(const uint8_t *bytes, uint32_t size, uint32_t rowel)
{
    uint32_t rows = (size + rowel * 4 - 1) / (rowel * 4);
    MTLTextureDescriptor *d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint width:rowel height:rows mipmapped:NO];
    id<MTLTexture> t = [dev newTextureWithDescriptor:d];
    static uint8_t pad[65536];
    memset(pad, 0, sizeof(pad));
    memcpy(pad, bytes, size);
    [t replaceRegion:MTLRegionMake2D(0, 0, rowel, rows) mipmapLevel:0 withBytes:pad
         bytesPerRow:rowel * 4];
    return t;
}

static void put_be(uint8_t *b, uint32_t v)
{
    b[0] = v >> 24; b[1] = v >> 16; b[2] = v >> 8; b[3] = v;
}

static uint32_t fb32(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

/* One float result of a raw-texture program: R of pixel (0, 0). */
static float raw_sample(uint32_t fmt, uint32_t dim, const uint8_t *bytes, uint32_t size,
                        R300FSUniforms *u, float px_, const float *coord)
{
    static R300State st;
    program(&st, true, false, false);
    r300_state_write(&st, 0x4104, 1);
    r300_state_write(&st, 0x44C0, fmt | (dim << 25));
    r300_state_write(&st, 0x4E38, (7u << 21) | W);          /* ARGB32323232 */
    r300_state_write(&st, 0x46A4, 21);                      /* C4_32_FP */
    id<MTLLibrary> lib = lib_for(&st);
    NSError *e = nil;
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [lib newFunctionWithName:@"r300_vs"];
    pd.fragmentFunction = [lib newFunctionWithName:@"r300_fs"];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Uint;
    id<MTLRenderPipelineState> p = [dev newRenderPipelineStateWithDescriptor:pd error:&e];
    if (!p) { printf("pipeline: %s\n", e.localizedDescription.UTF8String); fails++; return NAN; }
    MTLTextureDescriptor *d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Uint width:W height:H mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    Pass ps = { { [dev newTextureWithDescriptor:d] }, 1, nil,
                raw_tex(bytes, size, u->tex_info[0][1]), nil };
    R300Vertex v[6];
    quad(v, px_, coord, 0);
    run(&ps, p, u, v, 6);
    uint32_t o[4];
    [ps.cb[0] getBytes:o bytesPerRow:16 fromRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0];
    float f;
    memcpy(&f, &o[0], 4);
    return f;
}

int main(void)
{
    @autoreleasepool {
        dev = MTLCreateSystemDefaultDevice();
        q = [dev newCommandQueue];
        g_zp = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];

        /* ---- 16F 8x8, 4 levels, level L filled with L: the level follows
         * the texel footprint (mip nearest) ---- */
        {
            R300TexDesc td;
            memset(&td, 0, sizeof(td));
            td.width = td.height = 8; td.levels = 4; td.depth = 1;
            r300_tex_layout(&td, 2, false, false);
            static uint8_t mem[1024];
            memset(mem, 0, sizeof(mem));
            for (uint32_t l = 0; l < 4; l++) {
                __fp16 h = (float)l;
                uint16_t hb;
                memcpy(&hb, &h, 2);
                for (uint32_t b = 0; b < td.lvl_pitch[l] * td.lvl_rows[l]; b += 4) {
                    put_be(mem + td.lvl_off[l] + b, hb | (uint32_t)hb << 16);
                }
            }
            R300FSUniforms u = base_uniforms();
            u.tex_info[0][0] = 1; u.tex_info[0][1] = td.lvl_pitch[0] / 4;
            u.tex_info[0][2] = 8; u.tex_info[0][3] = 8;
            u.tex_lod[0][0] = 0; u.tex_lod[0][1] = 0; u.tex_lod[0][2] = 3; u.tex_lod[0][3] = 1;
            u.tex_dim[0][0] = 0; u.tex_dim[0][1] = 1; u.tex_dim[0][2] = td.lvl_pitch[0];
            u.tex_dim[0][3] = R300_TEXF_POT_ROWS;
            float l0 = raw_sample(0x18, 0, mem, td.size_bytes, &u, 64, NULL);
            float l1 = raw_sample(0x18, 0, mem, td.size_bytes, &u, 4, NULL);
            float l3 = raw_sample(0x18, 0, mem, td.size_bytes, &u, 1, NULL);
            CHECK(l0 == 0 && l1 == 1 && l3 == 3, "mip levels %g %g %g", l0, l1, l3);
            u.tex_lod[0][0] = 1.0f;                         /* LOD bias +1 */
            float b1 = raw_sample(0x18, 0, mem, td.size_bytes, &u, 4, NULL);
            CHECK(b1 == 2, "lod bias level %g", b1);
            u.tex_lod[0][0] = 0; u.tex_lod[0][3] = 0; u.tex_lod[0][1] = 2;   /* no mips: MAX_MIP_LEVEL */
            float nm = raw_sample(0x18, 0, mem, td.size_bytes, &u, 64, NULL);
            CHECK(nm == 2, "mip filter none uses MAX_MIP_LEVEL %g", nm);
        }

        /* ---- 32F 2x2x2 volume: slice from r ---- */
        {
            R300TexDesc td;
            memset(&td, 0, sizeof(td));
            td.width = td.height = 2; td.depth = 2; td.levels = 1; td.dim = R300_TEXDIM_3D;
            r300_tex_layout(&td, 4, false, false);
            static uint8_t mem[256];
            for (uint32_t z = 0; z < 2; z++)
                for (uint32_t y = 0; y < 2; y++)
                    for (uint32_t x = 0; x < 2; x++)
                        put_be(mem + z * td.lvl_pitch[0] * td.lvl_rows[0] + y * td.lvl_pitch[0] + x * 4,
                               fb32(10 * z + 2 * y + x));
            R300FSUniforms u = base_uniforms();
            u.tex_info[0][0] = 1; u.tex_info[0][1] = td.lvl_pitch[0] / 4;
            u.tex_info[0][2] = 2; u.tex_info[0][3] = 2;
            u.tex_dim[0][0] = 1; u.tex_dim[0][1] = 2; u.tex_dim[0][2] = td.lvl_pitch[0];
            u.tex_dim[0][3] = R300_TEXF_POT_ROWS;
            const float c[4] = { 0.75f, 0.25f, 0.75f, 1 };
            float r = raw_sample(0x1B, 1, mem, td.size_bytes, &u, 64, c);
            CHECK(r == 11, "3D texel %g (want slice 1, x 1, y 0)", r);
        }

        /* ---- 32F 2x2 cube map: face from the major axis ---- */
        {
            R300TexDesc td;
            memset(&td, 0, sizeof(td));
            td.width = td.height = 2; td.depth = 1; td.levels = 1; td.dim = R300_TEXDIM_CUBE;
            r300_tex_layout(&td, 4, false, false);
            static uint8_t mem[1024];
            for (uint32_t f = 0; f < 6; f++)
                for (uint32_t b = 0; b < td.lvl_pitch[0] * td.lvl_rows[0]; b += 4)
                    put_be(mem + f * td.lvl_pitch[0] * td.lvl_rows[0] + b, fb32(f));
            R300FSUniforms u = base_uniforms();
            u.tex_info[0][0] = 1; u.tex_info[0][1] = td.lvl_pitch[0] / 4;
            u.tex_info[0][2] = 2; u.tex_info[0][3] = 2;
            u.tex_dim[0][0] = 2; u.tex_dim[0][1] = 1; u.tex_dim[0][2] = td.lvl_pitch[0];
            u.tex_dim[0][3] = R300_TEXF_POT_ROWS;
            const float px_[4] = { 1, 0.1f, 0.2f, 1 }, ny[4] = { 0.1f, -1, 0.2f, 1 },
                        nz[4] = { 0.2f, 0.1f, -1, 1 };
            float a = raw_sample(0x1B, 2, mem, td.size_bytes, &u, 64, px_);
            float b = raw_sample(0x1B, 2, mem, td.size_bytes, &u, 64, ny);
            float cz = raw_sample(0x1B, 2, mem, td.size_bytes, &u, 64, nz);
            CHECK(a == 0 && b == 3 && cz == 5, "cube faces %g %g %g", a, b, cz);
        }

        /* ---- border colour through a float texture ---- */
        {
            static R300State st;
            program(&st, true, false, false);
            r300_state_write(&st, 0x4104, 1);
            r300_state_write(&st, 0x44C0, 0x0C);
            r300_state_write(&st, 0x4400, 6 | (6u << 3) | (1u << 9) | (1u << 11));
            id<MTLLibrary> lib = lib_for(&st);
            id<MTLRenderPipelineState> p = pipe_for(lib, 1, false);
            MTLTextureDescriptor *d = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:4 height:4 mipmapped:NO];
            id<MTLTexture> tx = [dev newTextureWithDescriptor:d];
            uint8_t green[64];
            for (int i = 0; i < 16; i++) { green[4 * i] = 0; green[4 * i + 1] = 255; green[4 * i + 2] = 0; green[4 * i + 3] = 255; }
            [tx replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:green bytesPerRow:16];
            MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
            sd.sAddressMode = sd.tAddressMode = MTLSamplerAddressModeClampToBorderColor;
            sd.borderColor = MTLSamplerBorderColorTransparentBlack;
            Pass ps = { { rgba8(0) }, 1, nil, tx, [dev newSamplerStateWithDescriptor:sd] };
            R300FSUniforms u = base_uniforms();
            u.tex_info[0][0] = 1; u.tex_info[0][2] = 4; u.tex_info[0][3] = 4;
            u.tex_border[0][0] = 1; u.tex_border[0][3] = 1;         /* red */
            R300Vertex v[6];
            const float out_[4] = { 1.5f, 0.5f, 0, 1 }, in_[4] = { 0.5f, 0.5f, 0, 1 };
            quad(v, 64, out_, 0);
            run(&ps, p, &u, v, 6);
            uint8_t o[4];
            px(ps.cb[0], 3, 3, o);
            CHECK(o[0] == 255 && o[1] == 0 && o[3] == 255, "border %u %u %u %u", o[0], o[1], o[2], o[3]);
            quad(v, 64, in_, 0);
            run(&ps, p, &u, v, 6);
            px(ps.cb[0], 3, 3, o);
            CHECK(o[0] == 0 && o[1] == 255, "inside %u %u %u %u", o[0], o[1], o[2], o[3]);
        }

        /* ---- fog (linear), ROP XOR, discard src pixels ---- */
        {
            static R300State st;
            program(&st, false, false, false);
            id<MTLLibrary> lib = lib_for(&st);
            id<MTLRenderPipelineState> p = pipe_for(lib, 1, false);
            const float red[4] = { 1, 0, 0, 1 };
            R300Vertex v[6];
            uint8_t o[4];

            quad(v, 64, red, 0);
            for (int i = 0; i < 6; i++) v[i].aux[0] = 0.25f;
            R300FSUniforms u = base_uniforms();
            u.fog_blend = 1;
            u.fog_color[2] = 1;
            Pass ps = { { rgba8(0) }, 1 };
            run(&ps, p, &u, v, 6);
            px(ps.cb[0], 5, 5, o);
            CHECK(NEAR(o[0], 64, 1.5f) && o[1] == 0 && NEAR(o[2], 191, 1.5f), "fog %u %u %u",
                  o[0], o[1], o[2]);

            u = base_uniforms();
            u.rop = 4 | (6u << 8);                                  /* XOR */
            Pass pr = { { rgba8(0xFF) }, 1 };
            run(&pr, p, &u, v, 6);
            px(pr.cb[0], 5, 5, o);
            CHECK(o[0] == 0 && o[1] == 255 && o[2] == 255 && o[3] == 0, "rop xor %u %u %u %u",
                  o[0], o[1], o[2], o[3]);

            const float clear_[4] = { 1, 1, 1, 0 };
            quad(v, 64, clear_, 0);
            u = base_uniforms();
            u.cblend = 1 | (1u << 3) | (33u << 16) | (32u << 24);    /* ONE, ZERO, discard a == 0 */
            u.ablend = u.cblend;
            Pass pd = { { rgba8(0x40) }, 1 };
            run(&pd, p, &u, v, 6);
            px(pd.cb[0], 5, 5, o);
            CHECK(o[0] == 0x40 && o[3] == 0x40, "discard src %u %u", o[0], o[3]);
        }

        /* ---- two render targets ---- */
        {
            static R300State st;
            program(&st, false, true, false);
            CHECK(r300_us_num_targets(&st) == 2, "targets %u", r300_us_num_targets(&st));
            id<MTLLibrary> lib = lib_for(&st);
            id<MTLRenderPipelineState> p = pipe_for(lib, 2, false);
            const float c[4] = { 0, 0, 1, 1 };
            R300Vertex v[6];
            quad(v, 64, c, 0);
            R300FSUniforms u = base_uniforms();
            Pass ps = { { rgba8(0), rgba8(0) }, 2 };
            run(&ps, p, &u, v, 6);
            uint8_t a[4], b[4];
            px(ps.cb[0], 7, 7, a);
            px(ps.cb[1], 7, 7, b);
            CHECK(a[2] == 255 && a[0] == 0 && b[0] == 255 && b[1] == 255 && b[2] == 255,
                  "mrt A %u %u %u B %u %u %u", a[0], a[1], a[2], b[0], b[1], b[2]);
        }

        /* ---- depth from the program; polygon offset ---- */
        {
            MTLTextureDescriptor *zd = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint width:W height:H mipmapped:NO];
            zd.usage = MTLTextureUsageRenderTarget;
            zd.storageMode = MTLStorageModeShared;
            static uint32_t zinit[W * H];
            for (int i = 0; i < W * H; i++) zinit[i] = 0xFFFFFF00u;

            static R300State st;
            program(&st, false, false, true);
            id<MTLLibrary> lib = lib_for(&st);
            id<MTLRenderPipelineState> p = pipe_for(lib, 1, true);
            const float c[4] = { 1, 1, 1, 1 };
            R300Vertex v[6];
            quad(v, 64, c, 0.9f);
            R300FSUniforms u = base_uniforms();
            u.consts[0][3] = 0.25f;
            u.depth_src = 1;
            u.zinfo[0] = 6;                 /* Z test + write */
            u.zinfo[1] = 7;                 /* ALWAYS */
            Pass ps = { { rgba8(0) }, 1, [dev newTextureWithDescriptor:zd] };
            [ps.z replaceRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0 withBytes:zinit bytesPerRow:W * 4];
            run(&ps, p, &u, v, 6);
            uint32_t z;
            [ps.z getBytes:&z bytesPerRow:4 fromRegion:MTLRegionMake2D(9, 9, 1, 1) mipmapLevel:0];
            CHECK((z >> 8) == (uint32_t)lrintf(0.25f * 16777215), "shader depth %06x", z >> 8);

            u.depth_src = 0;                /* from the rasterizer, offset by 256 units */
            u.poly_en = 1;
            u.poly_offset[1] = 256.0f / 16777216.0f;
            quad(v, 64, c, 0.5f);
            [ps.z replaceRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0 withBytes:zinit bytesPerRow:W * 4];
            run(&ps, p, &u, v, 6);
            [ps.z getBytes:&z bytesPerRow:4 fromRegion:MTLRegionMake2D(9, 9, 1, 1) mipmapLevel:0];
            uint32_t want = (uint32_t)lrintf(0.5f * 16777215) + 256;
            CHECK(abs((int)(z >> 8) - (int)want) <= 2, "poly offset %06x want %06x", z >> 8, want);
        }
    }
    printf(fails ? "test_raster: %d failure(s)\n" : "test_raster: ok\n", fails);
    return fails != 0;
}
