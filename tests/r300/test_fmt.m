/*
 * Render through the generated r300_fs with Metal, the way metal_draw_r300
 * binds it, and check the colour-buffer formats other than ARGB8888
 * (float, 16-bit, 565/1555/4444, I8, UV88; COLOR_ENDIAN) and the texture
 * formats the shader decodes itself (float, 16-bit, YUV 4:2:2).
 */
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../../hw/display/r300/r300_draw.h"

#define W 16
#define H 16

static id<MTLDevice> dev;
static id<MTLCommandQueue> q;
static id<MTLBuffer> g_zp;
static int fails;

#define CHECK(c, ...) do { if (!(c)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); \
    printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

/* Program: tex ? t1 = sample(unit 0, t0.xy), out = t1 : out = t0. */
static void program(R300State *st, bool tex)
{
    r300_state_reset(st);
    r300_state_write(st, 0x4600, tex ? 1u << 3 : 0);    /* US_CONFIG */
    r300_state_write(st, 0x4608, 0);
    r300_state_write(st, 0x461C, 0);                    /* 1 tex, 1 alu */
    r300_state_write(st, 0x4620, 0 | (1u << 6) | (0u << 11) | (1u << 15));
    unsigned src = tex ? 1 : 0;
    r300_state_write(st, 0x46C0, src | (7u << 26));
    r300_state_write(st, 0x48C0, 0 | (21u << 7) | (20u << 14));
    r300_state_write(st, 0x47C0, src | (1u << 24));
    r300_state_write(st, 0x49C0, 9 | (17u << 7) | (16u << 14));
}

static char *msl_for(R300State *st)
{
    R300FSDesc d;
    const char *err;
    memset(d.route, -1, sizeof(d.route));
    d.route[0] = 0;
    char *m = r300_us_to_msl(st, &d, &err);
    if (!m) printf("msl: %s\n", err);
    return m;
}

static id<MTLRenderPipelineState> pipe_for(const char *msl, MTLPixelFormat pf)
{
    NSError *e = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(msl) options:nil error:&e];
    if (!lib) { printf("compile: %s\n%s\n", e.localizedDescription.UTF8String, msl); return nil; }
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [lib newFunctionWithName:@"r300_vs"];
    pd.fragmentFunction = [lib newFunctionWithName:@"r300_fs"];
    pd.colorAttachments[0].pixelFormat = pf;
    id<MTLRenderPipelineState> p = [dev newRenderPipelineStateWithDescriptor:pd error:&e];
    if (!p) printf("pipeline: %s\n", e.localizedDescription.UTF8String);
    return p;
}

static R300FSUniforms base_uniforms(void)
{
    R300FSUniforms u;
    memset(&u, 0, sizeof(u));
    /* Mesa's RGBA outputs: C0 R, C1 G, C2 B, C3 A */
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

/* Full-target quad; v0 = colour, or texcoords spanning the texture. */
static void quad(R300Vertex *v, const float c[4], bool uv)
{
    float p[6][2] = { { -1, 1 }, { -1, -1 }, { 1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };
    for (int i = 0; i < 6; i++) {
        memset(&v[i], 0, sizeof(v[i]));
        v[i].pos[0] = p[i][0]; v[i].pos[1] = p[i][1]; v[i].pos[3] = 1;
        if (uv) {
            v[i].v[0][0] = (p[i][0] + 1) / 2; v[i].v[0][1] = (1 - p[i][1]) / 2;
            v[i].v[0][3] = 1;
        } else {
            memcpy(v[i].v[0], c, 16);
        }
    }
}

typedef struct Target {
    id<MTLBuffer> buf;
    id<MTLTexture> tex;
    uint32_t bpp;
} Target;

static Target target(MTLPixelFormat pf, uint32_t bpp, uint8_t fill)
{
    Target t = { 0 };
    t.bpp = bpp;
    t.buf = [dev newBufferWithLength:W * H * bpp options:MTLResourceStorageModeShared];
    memset(t.buf.contents, fill, W * H * bpp);
    MTLTextureDescriptor *d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:pf width:W height:H mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    d.storageMode = MTLStorageModeShared;
    t.tex = [t.buf newTextureWithDescriptor:d offset:0 bytesPerRow:W * bpp];
    return t;
}

static void draw(Target *t, id<MTLRenderPipelineState> p, const R300FSUniforms *u,
                 const R300Vertex *v, id<MTLTexture> tex)
{
    id<MTLCommandBuffer> cb = [q commandBuffer];
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = t->tex;
    rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
    [e setRenderPipelineState:p];
    [e setVertexBytes:v length:sizeof(R300Vertex) * 6 atIndex:0];
    [e setVertexBytes:(float[4]){ 0 } length:16 atIndex:1];
    [e setFragmentBytes:u length:sizeof(*u) atIndex:0];
    [e setFragmentBuffer:g_zp offset:0 atIndex:1];
    if (tex) {
        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
        [e setFragmentTexture:tex atIndex:0];
        [e setFragmentSamplerState:[dev newSamplerStateWithDescriptor:sd] atIndex:0];
    }
    [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

static const uint8_t *pix(Target *t, int x, int y)
{
    return (const uint8_t *)t->buf.contents + (y * W + x) * t->bpp;
}

static uint32_t be32(const uint8_t *b) { return (uint32_t)b[0] << 24 | b[1] << 16 | b[2] << 8 | b[3]; }
static uint32_t le32(const uint8_t *b) { return (uint32_t)b[3] << 24 | b[2] << 16 | b[1] << 8 | b[0]; }
static float f32(uint32_t v) { float f; memcpy(&f, &v, 4); return f; }
static uint32_t u32(float f) { uint32_t v; memcpy(&v, &f, 4); return v; }

/* A colour-buffer case: render colour c into format cf/of/endian. */
static Target render_cb(uint32_t cf, uint32_t of, uint32_t endian, const float c[4],
                        R300FSUniforms *uu, uint8_t fill)
{
    static R300State st;
    uint32_t bpp, view;
    program(&st, false);
    r300_state_write(&st, 0x4E38, (cf << 21) | (endian << 19) | W);
    r300_state_write(&st, 0x46A4, of);
    view = r300_cb_view(cf, of, &bpp);
    MTLPixelFormat pf = view == R300_RTV_R8U ? MTLPixelFormatR8Uint :
                        view == R300_RTV_R16U ? MTLPixelFormatR16Uint :
                        view == R300_RTV_R32U ? MTLPixelFormatR32Uint :
                        view == R300_RTV_RG32U ? MTLPixelFormatRG32Uint :
                        view == R300_RTV_RGBA32U ? MTLPixelFormatRGBA32Uint :
                                                   MTLPixelFormatRGBA8Unorm;
    Target t = target(pf, bpp, fill);
    char *m = msl_for(&st);
    id<MTLRenderPipelineState> p = m ? pipe_for(m, pf) : nil;
    free(m);
    if (!p) { fails++; return t; }
    R300FSUniforms u = uu ? *uu : base_uniforms();
    u.rt_endian = endian;
    R300Vertex v[6];
    quad(v, c, false);
    draw(&t, p, &u, v, nil);
    return t;
}

/* A texture case: sample a W x H texture of format fmt (texel bytes as
 * given, VRAM rule) into an ARGB32323232 / C4_32_FP target. */
static Target render_tex(uint32_t fmt, uint32_t swap, bool yuv, const void *texels,
                         uint32_t pitch_bytes, uint32_t filter0)
{
    static R300State st;
    program(&st, true);
    r300_state_write(&st, 0x4E38, (7u << 21) | W);
    r300_state_write(&st, 0x46A4, 21);
    r300_state_write(&st, 0x4104, 1);
    r300_state_write(&st, 0x44C0, fmt | (yuv ? 1u << 22 : 0));
    r300_state_write(&st, 0x4400, filter0);
    r300_state_write(&st, 0x4540, swap);
    Target t = target(MTLPixelFormatRGBA32Uint, 16, 0);
    char *m = msl_for(&st);
    id<MTLRenderPipelineState> p = m ? pipe_for(m, MTLPixelFormatRGBA32Uint) : nil;
    free(m);
    if (!p) { fails++; return t; }

    uint32_t vb = r300_tex_raw_bpp(fmt) < 4 ? 4 : r300_tex_raw_bpp(fmt);
    MTLPixelFormat rpf = vb == 16 ? MTLPixelFormatRGBA32Uint :
                         vb == 8 ? MTLPixelFormatRG32Uint : MTLPixelFormatR32Uint;
    MTLTextureDescriptor *d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:rpf width:pitch_bytes / vb height:H mipmapped:NO];
    id<MTLTexture> tx = [dev newTextureWithDescriptor:d];
    [tx replaceRegion:MTLRegionMake2D(0, 0, pitch_bytes / vb, H) mipmapLevel:0
            withBytes:texels bytesPerRow:pitch_bytes];
    R300FSUniforms u = base_uniforms();
    u.tex_info[0][0] = 1;
    u.tex_info[0][1] = pitch_bytes / vb;    /* view elements per row */
    u.tex_info[0][2] = W;
    u.tex_info[0][3] = H;
    u.tex_dim[0][2] = pitch_bytes;          /* level 0 pitch */
    R300Vertex v[6];
    quad(v, NULL, true);
    draw(&t, p, &u, v, tx);
    return t;
}

static void out4(Target *t, int x, int y, float o[4])
{
    const uint8_t *b = pix(t, x, y);
    for (int i = 0; i < 4; i++) o[i] = f32(le32(b + 4 * i));
}

static uint16_t half_bits(float f)
{
    __fp16 h = f;
    uint16_t v;
    memcpy(&v, &h, 2);
    return v;
}

#define NEAR(a, b) (fabsf((a) - (b)) < 1.0f / 512)

int main(void)
{
    @autoreleasepool {
        dev = MTLCreateSystemDefaultDevice();
        q = [dev newCommandQueue];
        g_zp = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
        const float c1[4] = { 0.5f, 0.25f, 2.0f, 1.0f };
        const float red[4] = { 1, 0, 0, 1 };

        /* ---- colour buffers ---- */
        /* ARGB16161616 / C4_16_FP, no swap: halves C0..C3 little-endian, unclamped */
        {
            Target t = render_cb(10, 18, 0, c1, NULL, 0);
            const uint8_t *b = pix(&t, 3, 3);
            uint16_t h[4];
            memcpy(h, b, 8);
            CHECK(h[0] == half_bits(0.5f) && h[1] == half_bits(0.25f) &&
                  h[2] == half_bits(2.0f) && h[3] == half_bits(1.0f),
                  "16F x4: %04x %04x %04x %04x", h[0], h[1], h[2], h[3]);
        }
        /* ARGB32323232 / C4_32_FP, 32-bit swap: big-endian floats */
        {
            Target t = render_cb(7, 21, 2, c1, NULL, 0);
            const uint8_t *b = pix(&t, 5, 9);
            CHECK(f32(be32(b)) == 0.5f && f32(be32(b + 4)) == 0.25f &&
                  f32(be32(b + 8)) == 2.0f && f32(be32(b + 12)) == 1.0f,
                  "32F x4 BE: %g %g %g %g", f32(be32(b)), f32(be32(b + 4)),
                  f32(be32(b + 8)), f32(be32(b + 12)));
        }
        /* ARGB16161616 / C4_16 (unorm): clamped */
        {
            Target t = render_cb(10, 5, 0, c1, NULL, 0);
            uint16_t h[4];
            memcpy(h, pix(&t, 0, 0), 8);
            CHECK(h[0] == 32768 && h[1] == 16384 && h[2] == 65535 && h[3] == 65535,
                  "C4_16: %u %u %u %u", h[0], h[1], h[2], h[3]);
        }
        /* ARGB8888 / C_32_FP: one float */
        {
            Target t = render_cb(6, 19, 0, c1, NULL, 0);
            CHECK(f32(le32(pix(&t, 1, 1))) == 0.5f, "C_32_FP %g", f32(le32(pix(&t, 1, 1))));
        }
        /* ARGB8888 / C2_16_FP */
        {
            Target t = render_cb(6, 17, 0, c1, NULL, 0);
            uint32_t w = le32(pix(&t, 1, 1));
            CHECK((w & 0xFFFF) == half_bits(0.5f) && (w >> 16) == half_bits(0.25f),
                  "C2_16_FP %08x", w);
        }
        /* RGB565, Mesa's BGRA outputs (C0 B, C1 G, C2 R), 16-bit swap */
        {
            R300FSUniforms u = base_uniforms();
            u.out_sel[0] = 3; u.out_sel[1] = 2; u.out_sel[2] = 1; u.out_sel[3] = 0;
            Target t = render_cb(4, 0, 1, red, &u, 0);
            const uint8_t *b = pix(&t, 2, 2);
            CHECK(b[0] == 0xF8 && b[1] == 0x00, "565 BE red %02x %02x", b[0], b[1]);
            /* ARGB1555 no swap, and blending: 50% over the red just drawn is
             * checked below for 4444. */
            Target t2 = render_cb(3, 0, 0, red, &u, 0);
            b = pix(&t2, 2, 2);
            CHECK(b[0] == 0x00 && b[1] == 0xFC, "1555 LE red %02x %02x", b[0], b[1]);
        }
        /* ARGB4444 with blending: src alpha 0.5 over 0x0000 */
        {
            R300FSUniforms u = base_uniforms();
            u.out_sel[0] = 3; u.out_sel[1] = 2; u.out_sel[2] = 1; u.out_sel[3] = 0;
            /* RB3D_CBLEND: enable, ADD, src SRC_ALPHA (38), dst ONE_MINUS_SRC_ALPHA (39) */
            u.cblend = 1 | (38u << 16) | (39u << 24);
            const float half_red[4] = { 1, 0, 0, 0.5f };
            Target t = render_cb(15, 0, 0, half_red, &u, 0);
            uint16_t w;
            memcpy(&w, pix(&t, 4, 4), 2);
            /* R = 0.5 -> 8 of 15 at 11:8; A = 0.5*0.5 + 0 = 0.25 -> 4 */
            CHECK(w == 0x4800, "4444 blend %04x", w);
        }
        /* I8 stores C2 */
        {
            R300FSUniforms u = base_uniforms();
            u.out_sel[2] = 1;                       /* C2_SEL_R */
            Target t = render_cb(9, 0, 0, c1, &u, 0);
            CHECK(*pix(&t, 7, 7) == 128, "I8 %u", *pix(&t, 7, 7));
        }
        /* UV88 / C4_8: C2 in the low byte, C0 in the high byte */
        {
            R300FSUniforms u = base_uniforms();
            u.out_sel[0] = 2; u.out_sel[2] = 1;     /* R8G8: C0 G, C2 R */
            Target t = render_cb(13, 0, 0, c1, &u, 0);
            const uint8_t *b = pix(&t, 7, 7);
            CHECK(b[0] == 128 && b[1] == 64, "UV88 %02x %02x", b[0], b[1]);
        }
        /* Channel mask keeps the old value */
        {
            R300FSUniforms u = base_uniforms();
            u.chanmask = 4;                          /* red only */
            Target t = render_cb(10, 18, 0, c1, &u, 0);
            uint16_t h[4];
            memcpy(h, pix(&t, 3, 3), 8);
            CHECK(h[0] == half_bits(0.5f) && h[1] == 0 && h[3] == 0,
                  "mask: %04x %04x %04x %04x", h[0], h[1], h[2], h[3]);
        }

        /* ---- textures (VRAM rule: dword = TXO swap of the big-endian dword) ---- */
        uint32_t nearest = 2 | (2u << 3) | (1u << 9) | (1u << 11);
        /* 32F x4, swap 0: big-endian floats as the CPU wrote them */
        {
            static uint8_t tx[W * H * 16];
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    for (int c = 0; c < 4; c++) {
                        uint32_t v = u32(x + 0.25f * c + 100.0f * y);
                        uint8_t *b = tx + (y * W + x) * 16 + c * 4;
                        b[0] = v >> 24; b[1] = v >> 16; b[2] = v >> 8; b[3] = v;
                    }
            Target t = render_tex(0x1D, 0, false, tx, W * 16, nearest);
            float o[4];
            out4(&t, 5, 3, o);
            CHECK(o[0] == 305.0f && o[1] == 305.25f && o[2] == 305.5f && o[3] == 305.75f,
                  "32F x4 texel %g %g %g %g", o[0], o[1], o[2], o[3]);
        }
        /* 16F x4, swap 3 (half-dword): big-endian halves in order */
        {
            static uint8_t tx[W * H * 8];
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    for (int c = 0; c < 4; c++) {
                        uint16_t v = half_bits(x * 0.5f + c);
                        uint8_t *b = tx + (y * W + x) * 8 + c * 2;
                        b[0] = v >> 8; b[1] = v;
                    }
            Target t = render_tex(0x1A, 3, false, tx, W * 8, nearest);
            float o[4];
            out4(&t, 6, 1, o);
            CHECK(o[0] == 3.0f && o[1] == 4.0f && o[2] == 5.0f && o[3] == 6.0f,
                  "16F x4 texel %g %g %g %g", o[0], o[1], o[2], o[3]);
        }
        /* 32F, swap 0: X only, W defaults to 1 */
        {
            static uint8_t tx[W * H * 4];
            for (int i = 0; i < W * H; i++) {
                uint32_t v = u32(-1.5f * i);
                tx[4 * i] = v >> 24; tx[4 * i + 1] = v >> 16; tx[4 * i + 2] = v >> 8; tx[4 * i + 3] = v;
            }
            Target t = render_tex(0x1B, 0, false, tx, W * 4, nearest);
            float o[4];
            out4(&t, 2, 1, o);
            CHECK(o[0] == -1.5f * 18 && o[1] == 0 && o[3] == 1.0f, "32F texel %g %g %g %g",
                  o[0], o[1], o[2], o[3]);
        }
        /* 16F, swap 3: two texels per dword */
        {
            static uint8_t tx[W * H * 2];
            for (int i = 0; i < W * H; i++) {
                uint16_t v = half_bits(i * 0.25f);
                tx[2 * i] = v >> 8; tx[2 * i + 1] = v;
            }
            Target t = render_tex(0x18, 3, false, tx, W * 2, nearest);
            float o[4];
            out4(&t, 3, 2, o);
            CHECK(o[0] == 35 * 0.25f, "16F odd texel %g", o[0]);
            out4(&t, 4, 2, o);
            CHECK(o[0] == 36 * 0.25f, "16F even texel %g", o[0]);
        }
        /* Y16X16 unorm with bilinear filtering between two columns */
        {
            static uint8_t tx[W * H * 4];
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++) {
                    uint32_t v = (x & 1 ? 65535u : 0u) | (32768u << 16);
                    uint8_t *b = tx + (y * W + x) * 4;
                    b[0] = v >> 24; b[1] = v >> 16; b[2] = v >> 8; b[3] = v;
                }
            Target t = render_tex(0x04, 0, false, tx, W * 4, nearest);
            float o[4];
            out4(&t, 3, 3, o);
            CHECK(o[0] == 1.0f && NEAR(o[1], 0.5f), "Y16X16 %g %g", o[0], o[1]);
            /* 1:1 sampling at texel centres: linear gives the same */
            uint32_t linear = 2 | (2u << 3) | (2u << 9) | (2u << 11);
            t = render_tex(0x04, 0, false, tx, W * 4, linear);
            out4(&t, 3, 3, o);
            CHECK(NEAR(o[0], 1.0f), "Y16X16 linear at centre %g", o[0]);
        }
        /* YVYU422 (UYVY bytes U Y0 V Y1, swap 2) with YUV_TO_RGB: grey and red */
        {
            static uint8_t tx[W * H * 2];
            for (int i = 0; i < W * H / 2; i++) {
                uint8_t *b = tx + 4 * i;
                if ((i % (W / 2)) < 4) { b[0] = 128; b[1] = 126; b[2] = 128; b[3] = 126; }
                else { b[0] = 90; b[1] = 81; b[2] = 240; b[3] = 81; }       /* BT.601 red */
            }
            Target t = render_tex(0x15, 2, true, tx, W * 2, nearest);
            float o[4];
            out4(&t, 1, 1, o);
            CHECK(NEAR(o[0], o[1]) && NEAR(o[1], o[2]) && fabsf(o[0] - 0.5f) < 0.02f,
                  "YUV grey %g %g %g", o[0], o[1], o[2]);
            out4(&t, 13, 1, o);
            /* X = B, Y = G, Z = R */
            CHECK(o[2] > 0.95f && o[1] < 0.05f && o[0] < 0.05f, "YUV red %g %g %g",
                  o[0], o[1], o[2]);
        }
        /* W16Z16Y16X16 */
        {
            static uint8_t tx[W * H * 8];
            for (int i = 0; i < W * H; i++) {
                uint32_t lo = 0x0000FFFFu, hi = 0xFFFF8000u;   /* X 1, Y 0, Z .5, W 1 */
                uint8_t *b = tx + 8 * i;
                b[0] = lo >> 24; b[1] = lo >> 16; b[2] = lo >> 8; b[3] = lo;
                b[4] = hi >> 24; b[5] = hi >> 16; b[6] = hi >> 8; b[7] = hi;
            }
            Target t = render_tex(0x0E, 0, false, tx, W * 8, nearest);
            float o[4];
            out4(&t, 1, 1, o);
            CHECK(o[0] == 1 && o[1] == 0 && NEAR(o[2], 0.5f) && o[3] == 1,
                  "16x4 unorm %g %g %g %g", o[0], o[1], o[2], o[3]);
        }
    }
    if (fails) {
        printf("test_fmt: %d failure(s)\n", fails);
        return 1;
    }
    printf("test_fmt: ok\n");
    return 0;
}
