/*
 * Early depth/stencil rejection (R300_GLSL_EARLY_Z): the generated r300_fs_z
 * with the bit draws exactly what it draws without it.  Every case runs twice
 * from the same starting contents, once per variant, and compares every
 * colour target byte, every raw depth/stencil word and the Z-pass count.
 *   - random cases: depth/stencil state, both formats and all DEPTHENDIAN
 *     swaps, polygon offset, shader depth, alpha test, clip rule, blending,
 *     ROP, channel mask, two render targets, overlapping primitives in one
 *     draw and across draws, some across a command-buffer boundary;
 *   - witness cases with known results (a failing fragment whose stencil
 *     changes, a preview pass then an alpha kill, shader depth against
 *     geometric depth, polygon offset across the comparison);
 *   - derivatives in 2x2 quads that mix rejected and kept fragments, for a
 *     sampled and a shader-decoded texture (whose LOD uses dFdx/dFdy);
 *   - the text: memo identity off/on/off, the per-draw exclusions, the
 *     threshold and the interlock.
 * AI-generated (Claude).
 */
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../hw/display/r300/r300_draw.h"
#include "../../hw/display/r300/r300_text.h"
#include "r300_mtl.h"

#define W 32
#define H 32

static id<MTLDevice> dev;
static id<MTLCommandQueue> q;
static id<MTLBuffer> g_zp;
static int fails;

#define CHECK(c, ...) do { if (!(c)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); \
    printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

static uint32_t rng = 0x2545F491u;
static uint32_t rnd(void)
{
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return rng;
}
static float frnd(void) { return (rnd() >> 8) / 16777216.0f; }

/* US program (test_raster.m's): op ? t1 = sample(unit 0, t0) with TEX
 * opcode op (1 LD, 3 TXP, 4 TXB) : t1 = t0; out A = t1.  kil: then KIL t0
 * (killed where a component of t0 is negative).  mrt: a second instruction
 * writes white to target B; w: the alpha of constant 0 goes to the depth
 * output. */
static void program_op(R300State *st, unsigned op, bool kil, bool mrt, bool w)
{
    bool tex = op != 0;
    unsigned nalu = 1 + mrt + w;
    r300_state_reset(st);
    r300_state_write(st, 0x4600, tex ? 1u << 3 : 0);
    r300_state_write(st, 0x4608, 0);
    r300_state_write(st, 0x461C, ((nalu - 1) << 6) | (kil ? 1u << 17 : 0));
    r300_state_write(st, 0x4620, 0 | (1u << 6) | (0u << 11) | (op << 15));
    if (kil) {
        r300_state_write(st, 0x4624, 0 | (2u << 15));      /* KIL t0 */
    }
    unsigned src = tex ? 1 : 0;
    r300_state_write(st, 0x46C0, src | (7u << 26));
    r300_state_write(st, 0x48C0, 0 | (21u << 7) | (20u << 14));
    r300_state_write(st, 0x47C0, src | (1u << 24));
    r300_state_write(st, 0x49C0, 9 | (17u << 7) | (16u << 14));
    unsigned i = 1;
    if (mrt) {
        r300_state_write(st, 0x46C0 + 4 * i, (7u << 26) | (1u << 29));
        r300_state_write(st, 0x48C0 + 4 * i, 20 | (21u << 7) | (21u << 14));
        r300_state_write(st, 0x47C0 + 4 * i, (1u << 24) | (1u << 25));
        r300_state_write(st, 0x49C0 + 4 * i, 16 | (17u << 7) | (17u << 14));
        i++;
    }
    if (w) {
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

/* No texture: n ALU instructions, the last writing out A = t0 (work n). */
static void alu_program(R300State *st, unsigned n)
{
    program_op(st, 0, false, false, false);
    r300_state_write(st, 0x461C, (n - 1) << 6);
    for (unsigned i = 0; i < n; i++) {
        bool last = i + 1 == n;
        r300_state_write(st, 0x46C0 + 4 * i, last ? 7u << 26 : 0);
        r300_state_write(st, 0x48C0 + 4 * i, 0 | (21u << 7) | (20u << 14));
        r300_state_write(st, 0x47C0 + 4 * i, last ? 1u << 24 : 0);
        r300_state_write(st, 0x49C0 + 4 * i, 9 | (17u << 7) | (16u << 14));
    }
}

static void program(R300State *st, bool tex, bool mrt, bool w)
{
    program_op(st, tex ? 1 : 0, false, mrt, w);
}

static char *glsl(R300State *st, uint32_t flags)
{
    R300FSDesc d;
    const char *err;
    memset(d.route, -1, sizeof(d.route));
    d.route[0] = 0;
    char *m = r300_us_to_glsl(st, &d, flags, &err);
    if (!m) { printf("glsl: %s\n", err); fails++; }
    return m;
}

/* With $EARLYZ_MSL_DIR set: write st's early r300_fs_z MSL as
 * <dir>/<name>.fs_z.metal for earlyz_mslcheck.py (run.sh). */
static void dump_msl(R300State *st, const char *name)
{
    const char *dir = getenv("EARLYZ_MSL_DIR");
    if (!dir) return;
    char *m = glsl(st, R300_GLSL_EARLY_Z), *err = NULL, path[1024];
    char *msl = m ? r300_glsl_to_msl(m, R300_STAGE_FS_Z, &err) : NULL;
    snprintf(path, sizeof(path), "%s/%s.fs_z.metal", dir, name);
    FILE *f = msl ? fopen(path, "w") : NULL;
    CHECK(f != NULL, "dump %s: %s", path, err ? err : "cannot write");
    if (f) { fputs(msl, f); fclose(f); }
    free(m); free(msl); free(err);
}

typedef struct Prog {
    const char *name;
    unsigned ncb;
    MTLPixelFormat cfmt;
    uint32_t xflags;                        /* more r300_us_to_glsl flags (byte-order model) */
    id<MTLRenderPipelineState> p[2][2];     /* [early][z16] */
} Prog;

static bool make_prog(Prog *pg, R300State *st)
{
    for (int early = 0; early < 2; early++) {
        char *m = glsl(st, pg->xflags | (early ? R300_GLSL_EARLY_Z : 0));
        if (!m) return false;
        CHECK((strstr(m, "r300_zpreview(zb") != NULL) == early, "%s: preview in text %d",
              pg->name, early);
        NSArray<id<MTLFunction>> *lib = r300_test_lib(dev, m);
        free(m);
        if (!lib) { fails++; return false; }
        for (int z16 = 0; z16 < 2; z16++) {
            NSError *e = nil;
            MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
            pd.vertexFunction = lib[R300_STAGE_VS];
            pd.fragmentFunction = lib[R300_STAGE_FS_Z];
            for (unsigned k = 0; k < pg->ncb; k++) {
                pd.colorAttachments[k].pixelFormat = pg->cfmt;
            }
            pd.colorAttachments[pg->ncb].pixelFormat = z16 ? MTLPixelFormatR16Uint
                                                            : MTLPixelFormatR32Uint;
            pg->p[early][z16] = [dev newRenderPipelineStateWithDescriptor:pd error:&e];
            if (!pg->p[early][z16]) {
                printf("pipeline: %s\n", e.localizedDescription.UTF8String);
                fails++;
                return false;
            }
        }
    }
    return true;
}

/* One variant's targets: colour buffers and the depth/stencil words (a
 * linear buffer, as in VRAM). */
typedef struct Arm {
    id<MTLTexture> cb[2];
    id<MTLBuffer> zbuf;
    id<MTLTexture> z;
    unsigned ncb, cbpp;
    bool z16;
} Arm;

static Arm make_arm(const Prog *pg, bool z16)
{
    Arm a = { .ncb = pg->ncb, .z16 = z16 };
    a.cbpp = pg->cfmt == MTLPixelFormatRGBA32Uint ? 16 : 4;
    for (unsigned k = 0; k < pg->ncb; k++) {
        MTLTextureDescriptor *d = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:pg->cfmt width:W height:H mipmapped:NO];
        d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        d.storageMode = MTLStorageModeShared;
        a.cb[k] = [dev newTextureWithDescriptor:d];
    }
    size_t bpp = z16 ? 2 : 4;
    a.zbuf = [dev newBufferWithLength:W * H * bpp options:MTLResourceStorageModeShared];
    MTLTextureDescriptor *zd = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:z16 ? MTLPixelFormatR16Uint : MTLPixelFormatR32Uint
                                     width:W height:H mipmapped:NO];
    zd.usage = MTLTextureUsageRenderTarget;
    zd.storageMode = MTLStorageModeShared;
    a.z = [a.zbuf newTextureWithDescriptor:zd offset:0 bytesPerRow:W * bpp];
    return a;
}

typedef struct Contents {
    uint8_t col[2][W * H * 16];
    uint32_t z[W * H];              /* words as stored; Z16 uses the low half */
} Contents;

static void load(Arm *a, const Contents *c)
{
    for (unsigned k = 0; k < a->ncb; k++) {
        [a->cb[k] replaceRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0
                      withBytes:c->col[k] bytesPerRow:W * a->cbpp];
    }
    for (int i = 0; i < W * H; i++) {
        if (a->z16) ((uint16_t *)a->zbuf.contents)[i] = c->z[i];
        else ((uint32_t *)a->zbuf.contents)[i] = c->z[i];
    }
}

static void save(Arm *a, Contents *c)
{
    memset(c, 0, sizeof(*c));
    for (unsigned k = 0; k < a->ncb; k++) {
        [a->cb[k] getBytes:c->col[k] bytesPerRow:W * a->cbpp
                fromRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0];
    }
    for (int i = 0; i < W * H; i++) {
        c->z[i] = a->z16 ? ((uint16_t *)a->zbuf.contents)[i] : ((uint32_t *)a->zbuf.contents)[i];
    }
}

/* Draws: one or two, each a list of triangles, in one encoder unless split. */
#define MAXV 48
typedef struct Case {
    R300FSUniforms u;
    R300Vertex v[2][MAXV];
    unsigned nv[2], ndraw;
    bool split;
} Case;

static uint32_t render(Arm *a, id<MTLRenderPipelineState> p, const Case *c,
                       id<MTLTexture> tex, id<MTLSamplerState> smp)
{
    *(uint32_t *)g_zp.contents = 0;
    id<MTLCommandBuffer> cb = nil;
    id<MTLRenderCommandEncoder> e = nil;
    for (unsigned d = 0; d < c->ndraw; d++) {
        if (!e) {
            cb = [q commandBuffer];
            MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
            for (unsigned k = 0; k <= a->ncb; k++) {
                rp.colorAttachments[k].texture = k < a->ncb ? a->cb[k] : a->z;
                rp.colorAttachments[k].loadAction = MTLLoadActionLoad;
                rp.colorAttachments[k].storeAction = MTLStoreActionStore;
            }
            e = [cb renderCommandEncoderWithDescriptor:rp];
            [e setRenderPipelineState:p];
            [e setDepthClipMode:MTLDepthClipModeClamp];
            [e setFrontFacingWinding:MTLWindingCounterClockwise];
            [e setCullMode:MTLCullModeNone];
            [e setVertexBytes:(float[4]){ 0 } length:16 atIndex:1];
            [e setFragmentBytes:&c->u length:sizeof(c->u) atIndex:0];
            [e setFragmentBuffer:g_zp offset:0 atIndex:1];
            if (tex) {
                [e setFragmentTexture:tex atIndex:0];
                [e setFragmentSamplerState:smp atIndex:0];
            }
        }
        id<MTLBuffer> vb = [dev newBufferWithBytes:c->v[d] length:sizeof(R300Vertex) * c->nv[d]
                                           options:MTLResourceStorageModeShared];
        [e setVertexBuffer:vb offset:0 atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:c->nv[d]];
        if (c->split || d + 1 == c->ndraw) {
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            CHECK(cb.status == MTLCommandBufferStatusCompleted, "command buffer: %s",
                  cb.error.localizedDescription.UTF8String);
            e = nil;
        }
    }
    return *(uint32_t *)g_zp.contents;
}

typedef struct Result {
    Contents c;
    uint32_t zpass;
} Result;

/* Both variants from the same contents; returns whether they agree. */
static bool both(const Prog *pg, bool z16, const Case *c, const Contents *init,
                 id<MTLTexture> tex, id<MTLSamplerState> smp, Result r[2])
{
    for (int early = 0; early < 2; early++) {
        Arm a = make_arm(pg, z16);
        load(&a, init);
        r[early].zpass = render(&a, pg->p[early][z16], c, tex, smp);
        save(&a, &r[early].c);
    }
    return memcmp(&r[0].c, &r[1].c, sizeof(r[0].c)) == 0 && r[0].zpass == r[1].zpass;
}

/* r300_zswap and the depth quantization, on the CPU. */
static uint32_t zswap(uint32_t v, uint32_t e, bool z16)
{
    if (z16) return e == 1 || e == 2 ? ((v >> 8) & 0xff) | ((v & 0xff) << 8) : v;
    switch (e) {
    case 1: return ((v & 0x00ff00ffu) << 8) | ((v >> 8) & 0x00ff00ffu);
    case 2: return (v >> 24) | ((v >> 8) & 0xff00u) | ((v << 8) & 0xff0000u) | (v << 24);
    case 3: return (v << 16) | (v >> 16);
    default: return v;
    }
}
static uint32_t quant(float z, bool z16)
{
    uint32_t zmax = z16 ? 0xffff : 0xffffff;
    float f = rintf(fminf(fmaxf(z, 0), 1) * (float)zmax);
    return f > zmax ? zmax : (uint32_t)f;
}
/* A stored word from depth and stencil. */
static uint32_t word(uint32_t zq, uint32_t s, uint32_t endian, bool z16)
{
    return zswap(z16 ? zq : ((zq << 8) | (s & 0xff)), endian, z16);
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
    u.tex_info[0][0] = 1; u.tex_info[0][2] = W; u.tex_info[0][3] = H;
    u.tex_lod[0][0] = 0; u.tex_lod[0][1] = 0; u.tex_lod[0][2] = 5; u.tex_lod[0][3] = 1;
    return u;
}

/* A rectangle [x0, x1) x [y0, y1) in pixels as two triangles, with depth
 * z00 at the top left changing by dzx, dzy per pixel; v0 = uv over it
 * scaled by s.  ccw: counter-clockwise on screen (front facing). */
static unsigned rect(R300Vertex *v, float x0, float y0, float x1, float y1, float z00,
                     float dzx, float dzy, float s, bool ccw)
{
    float px[4][2] = { { x0, y0 }, { x0, y1 }, { x1, y1 }, { x1, y0 } };
    int idx[6] = { 0, 1, 2, 0, 2, 3 };
    for (int i = 0; i < 6; i++) {
        int k = idx[ccw ? i : (i / 3) * 3 + (2 - i % 3)];
        float x = px[k][0], y = px[k][1];
        memset(&v[i], 0, sizeof(v[i]));
        v[i].pos[0] = 2 * x / W - 1;
        v[i].pos[1] = 1 - 2 * y / H;
        v[i].pos[2] = z00 + dzx * (x - x0) + dzy * (y - y0);
        v[i].pos[3] = 1;
        v[i].v[0][0] = s * (x - x0) / W;
        v[i].v[0][1] = s * (y - y0) / H;
        v[i].v[0][2] = (x - (x0 + x1) / 2) / W;    /* KIL t0: the left part */
        v[i].v[0][3] = 1;
        for (int j = 0; j < 8; j++) v[i].ucp[j] = 1;
    }
    return 6;
}

static uint32_t pick8(void)
{
    switch (rnd() % 4) {
    case 0: return 0;
    case 1: return 0xff;
    default: return rnd() & 0xff;
    }
}

/* A random case for pg (w: it writes depth). */
static void random_case(Case *c, Contents *init, const Prog *pg, bool z16, bool w)
{
    memset(c, 0, sizeof(*c));
    R300FSUniforms *u = &c->u;
    *u = base_uniforms();
    uint32_t endian = rnd() & 3, ref = rnd() & 0xff;
    u->zinfo[0] = rnd() & (1 | 2 | 4 | 16);
    u->zinfo[1] = rnd() & 0x7FFFFFF;
    u->zinfo[2] = ref | (pick8() << 8) | (pick8() << 16);
    u->zinfo[3] = (z16 ? R300_ZFMT_Z16 : 0) | endian;
    u->zpass_count = rnd() & 1;
    u->poly_en = rnd() & 3;
    float unit = 1.0f / (z16 ? 65535 : 16777215);
    u->poly_offset[0] = ((int)(rnd() % 9) - 4) * 0.5f;
    u->poly_offset[1] = ((int)(rnd() % 9) - 4) * unit;
    u->poly_offset[2] = ((int)(rnd() % 9) - 4) * 0.5f;
    u->poly_offset[3] = ((int)(rnd() % 9) - 4) * unit;
    u->depth_src = w && (rnd() & 1);
    u->consts[0][3] = frnd();
    if (rnd() & 1) {
        u->alpha_func = (1u << 11) | ((rnd() & 7) << 8) | (rnd() & 0xff);
    }
    if (rnd() % 4 == 0) {                   /* keep only inside clip rectangle 0 */
        u->cliprect[0][0] = rnd() % W; u->cliprect[0][1] = rnd() % H;
        u->cliprect[0][2] = rnd() % W; u->cliprect[0][3] = rnd() % H;
        u->clip_rule = 0xAAAA;
    }
    if (rnd() % 3 == 0) {                   /* blending, ROP and channel mask */
        u->cblend = rnd(); u->ablend = rnd();
        u->chanmask = rnd() & 0xF;
        u->rop = rnd() % 2 ? rnd() : 0;
        for (int i = 0; i < 4; i++) u->blend_color[i] = frnd();
    }

    /* Primitives: flat or sloped, front or back facing, overlapping. */
    float zs[8];
    unsigned nz = 0;
    c->ndraw = 1 + (rnd() & 1);
    c->split = c->ndraw == 2 && (rnd() & 1);
    for (unsigned d = 0; d < c->ndraw; d++) {
        unsigned n = 1 + rnd() % 4;
        for (unsigned i = 0; i < n; i++) {
            float x0 = rnd() % (W - 4), y0 = rnd() % (H - 4);
            float x1 = x0 + 4 + rnd() % (W - (unsigned)x0 - 3);
            float y1 = y0 + 4 + rnd() % (H - (unsigned)y0 - 3);
            float z = frnd();
            bool flat = rnd() & 1;
            float dzx = flat ? 0 : (frnd() - 0.5f) * 0.02f, dzy = flat ? 0 : (frnd() - 0.5f) * 0.02f;
            if (flat && nz < 8) zs[nz++] = z;
            c->nv[d] += rect(&c->v[d][c->nv[d]], x0, y0, x1, y1, z, dzx, dzy,
                             1 + rnd() % 4, rnd() & 1);
        }
    }

    /* Starting contents: random colours; depth words random or at a flat
     * primitive's quantized depth (equal or one off), stencil near ref. */
    for (unsigned k = 0; k < 2; k++) {
        for (size_t i = 0; i < sizeof(init->col[k]); i++) init->col[k][i] = rnd();
    }
    for (int i = 0; i < W * H; i++) {
        uint32_t zq;
        if (nz && rnd() % 3) {
            zq = quant(zs[rnd() % nz], z16) + (rnd() % 3) - 1;
        } else {
            zq = rnd();
        }
        zq &= z16 ? 0xffff : 0xffffff;
        static const int ds[6] = { 0, 0, 1, -1, 0, 0 };
        uint32_t s;
        switch (rnd() % 4) {
        case 0: s = 0; break;
        case 1: s = 255; break;
        case 2: s = ref + ds[rnd() % 6]; break;
        default: s = rnd(); break;
        }
        init->z[i] = word(zq, s, endian, z16);
    }
    (void)pg;
}

/* A mipmapped RGBA8 texture whose level L is colour lvl[L] (with alpha a). */
static const uint32_t LVL[6] = { 0xFF0000FFu, 0xFF00FF00u, 0xFFFF0000u, 0xFF00FFFFu,
                                 0xFFFF00FFu, 0xFFFFFF00u };
static id<MTLTexture> mip_tex(bool random_texels)
{
    MTLTextureDescriptor *d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:W height:H mipmapped:YES];
    id<MTLTexture> t = [dev newTextureWithDescriptor:d];
    static uint32_t buf[W * H];
    for (unsigned l = 0; l < t.mipmapLevelCount; l++) {
        unsigned w = W >> l, h = H >> l;
        for (unsigned i = 0; i < w * h; i++) buf[i] = random_texels ? rnd() : LVL[l % 6];
        [t replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:l withBytes:buf
             bytesPerRow:w * 4];
    }
    return t;
}

static id<MTLSamplerState> sampler(void)
{
    MTLSamplerDescriptor *s = [MTLSamplerDescriptor new];
    s.minFilter = s.magFilter = MTLSamplerMinMagFilterNearest;
    s.mipFilter = MTLSamplerMipFilterNearest;
    s.sAddressMode = s.tAddressMode = MTLSamplerAddressModeRepeat;
    return [dev newSamplerStateWithDescriptor:s];
}

static uint32_t col32(const Contents *c, unsigned k, int x, int y)
{
    uint32_t v;
    memcpy(&v, &c->col[k][(y * W + x) * 4], 4);
    return v;
}

/* A one-draw case: depth/stencil state, one rectangle over the target. */
static void one(Case *c, uint32_t cntl, uint32_t zs, uint32_t rm, bool z16, float z,
                bool ccw)
{
    memset(c, 0, sizeof(*c));
    c->u = base_uniforms();
    c->u.zinfo[0] = cntl; c->u.zinfo[1] = zs; c->u.zinfo[2] = rm;
    c->u.zinfo[3] = z16 ? R300_ZFMT_Z16 : 0;
    c->u.zpass_count = 1;
    c->ndraw = 1;
    c->nv[0] = rect(c->v[0], 0, 0, W, H, z, 0, 0, 1, ccw);
}

static void fill(Contents *init, uint32_t zword)
{
    memset(init->col, 0, sizeof(init->col));
    for (int i = 0; i < W * H; i++) init->z[i] = zword;
}

int main(void)
{
    @autoreleasepool {
        dev = MTLCreateSystemDefaultDevice();
        q = [dev newCommandQueue];
        g_zp = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
        static R300State st;
        static Case c;
        static Contents init;
        static Result r[2];

        /* The compiler's default language version has discard as MSL 2.3
         * defines it (helpers go on for derivatives). */
        {
            MTLCompileOptions *o = [MTLCompileOptions new];
            CHECK(o.languageVersion >= MTLLanguageVersion2_3, "default MSL version %lu",
                  (unsigned long)o.languageVersion);
        }

        /* ---- text and identity ---- */
        {
            R300FSDesc d;
            const char *err;
            uint32_t id0, id1, id2;
            memset(d.route, -1, sizeof(d.route));
            d.route[0] = 0;
            program(&st, true, false, false);
            char *t0 = r300_us_glsl_cached(&st, &d, 0, &id0, &err);
            char *t1 = r300_us_glsl_cached(&st, &d, R300_GLSL_EARLY_Z, &id1, &err);
            char *t2 = r300_us_glsl_cached(&st, &d, 0, &id2, &err);
            CHECK(id0 && id1 && id0 != id1 && id2 == id0, "memo ids %u %u %u", id0, id1, id2);
            CHECK(!strstr(t0, "r300_zpreview") && strstr(t1, "r300_zpreview(zb") &&
                  strcmp(t0, t2) == 0, "memo texts");
            /* Interlock (Vulkan): the bit is dropped and the text is the
             * interlock text. */
            char *a = glsl(&st, R300_GLSL_FB_INTERLOCK | R300_GLSL_VRAM_SSBO);
            char *b = glsl(&st, R300_GLSL_FB_INTERLOCK | R300_GLSL_VRAM_SSBO | R300_GLSL_EARLY_Z);
            CHECK(a && b && strcmp(a, b) == 0, "interlock text with the bit");
            free(a); free(b);
            /* Below the threshold (one ALU instruction, no texture): no
             * preview, the off text. */
            program(&st, false, false, false);
            a = glsl(&st, 0);
            b = glsl(&st, R300_GLSL_EARLY_Z);
            CHECK(a && b && strcmp(a, b) == 0, "short program text with the bit");
            free(a); free(b);
            /* The work threshold without textures: 7 ALU instructions stay
             * on the off text, 8 get the preview (and compile). */
            for (unsigned n = 7; n <= 8; n++) {
                alu_program(&st, n);
                a = glsl(&st, 0);
                b = glsl(&st, R300_GLSL_EARLY_Z);
                bool pre = b && strstr(b, "r300_zpreview(zb") != NULL;
                CHECK(a && b && strstr(a, "alu 0+") && pre == (n >= 8) &&
                      (n >= 8 || strcmp(a, b) == 0), "ALU-only threshold at %u: preview %d", n, pre);
                CHECK(strstr(b, n == 7 ? "alu 0+7" : "alu 0+8") != NULL, "ALU count %u in text", n);
                if (n == 8) {
                    NSArray<id<MTLFunction>> *lib = r300_test_lib(dev, b);
                    CHECK(lib != nil, "8-instruction early text compiles");
                }
                free(a); free(b);
            }
            /* Per draw: single-sampled Z16 / Z24S8 only. */
            program(&st, true, false, false);
            static const struct { uint32_t aa, zf; bool keep; } ex[] = {
                { 0, 0, true }, { 0, 2, true }, { 0, 1, false }, { 0, 3, false },
                { 0, 0xF, false }, { 1, 0, false }, { 3, 2, false }, { 5, 2, false },
                { 7, 0, false }, { 6, 2, true },    /* AA config bits without enable */
            };
            for (unsigned i = 0; i < sizeof(ex) / sizeof(ex[0]); i++) {
                r300_state_write(&st, 0x4020, ex[i].aa);        /* GB_AA_CONFIG */
                r300_state_write(&st, 0x4F10, ex[i].zf);        /* ZB_FORMAT */
                uint32_t f = r300_glsl_flags_for_draw(&st, R300_GLSL_EARLY_Z | R300_GLSL_GPU_VS);
                CHECK(f == (R300_GLSL_GPU_VS | (ex[i].keep ? R300_GLSL_EARLY_Z : 0)),
                      "exclusion aa %x zf %x: flags %x", ex[i].aa, ex[i].zf, f);
                CHECK(r300_glsl_flags_for_draw(&st, R300_GLSL_GPU_VS) == R300_GLSL_GPU_VS,
                      "exclusion without the bit");
            }
            r300_text_unref(t0); r300_text_unref(t1); r300_text_unref(t2);
        }

        /* ---- programs ---- */
        Prog ptex = { "tex", 1, MTLPixelFormatRGBA8Unorm };
        Prog pw = { "tex+W", 1, MTLPixelFormatRGBA8Unorm };
        Prog pmrt = { "tex+MRT", 2, MTLPixelFormatRGBA8Unorm };
        Prog pmw = { "tex+multiwrite", 2, MTLPixelFormatRGBA8Unorm };
        Prog pkil = { "tex+KIL", 1, MTLPixelFormatRGBA8Unorm };
        Prog ptxp = { "TXP", 1, MTLPixelFormatRGBA8Unorm };
        Prog ptxb = { "TXB", 1, MTLPixelFormatRGBA8Unorm };
        /* The default byte-order model: both colour buffers in VRAM, COLORENDIAN
         * 0 (big-endian stores; rt_swap32 for target A, as r300_draw sets it). */
        const uint32_t MODEL = R300_GLSL_ENDIAN_MODEL | R300_GLSL_CB_VRAM(0) | R300_GLSL_CB_VRAM(1);
        Prog ptex_m = { "tex model", 1, MTLPixelFormatRGBA8Unorm, MODEL };
        Prog pmrt_m = { "MRT model", 2, MTLPixelFormatRGBA8Unorm, MODEL };
        Prog pmw_m = { "multiwrite model", 2, MTLPixelFormatRGBA8Unorm, MODEL };
        program(&st, true, false, false);
        {
            char *m = glsl(&st, 0);
            CHECK(m && strstr(m, "uniform sampler2D tex0"), "unit 0 is a sampled texture");
            free(m);
        }
        dump_msl(&st, "tex");
        if (!make_prog(&ptex, &st)) return 1;
        program(&st, true, false, true);
        dump_msl(&st, "tex_w");
        if (!make_prog(&pw, &st)) return 1;
        program(&st, true, true, false);
        dump_msl(&st, "tex_mrt");
        if (!make_prog(&pmrt, &st)) return 1;
        r300_state_write(&st, 0x4E00, 1u << 5);            /* RB3D_CCTL: output A to both */
        if (!make_prog(&pmw, &st)) return 1;
        program_op(&st, 1, true, false, false);
        dump_msl(&st, "tex_kil");
        if (!make_prog(&pkil, &st)) return 1;
        program(&st, true, false, false);
        if (!make_prog(&ptex_m, &st)) return 1;
        program(&st, true, true, false);
        {
            char *a = glsl(&st, 0), *b = glsl(&st, MODEL);
            CHECK(a && b && strcmp(a, b) != 0, "the model changes the MRT text");
            free(a); free(b);
        }
        if (!make_prog(&pmrt_m, &st)) return 1;
        r300_state_write(&st, 0x4E00, 1u << 5);
        if (!make_prog(&pmw_m, &st)) return 1;
        program_op(&st, 3, false, false, false);
        dump_msl(&st, "txp");
        if (!make_prog(&ptxp, &st)) return 1;
        program_op(&st, 4, false, false, false);
        dump_msl(&st, "txb");
        if (!make_prog(&ptxb, &st)) return 1;
        id<MTLTexture> rtex = mip_tex(true), ltex = mip_tex(false);
        id<MTLSamplerState> smp = sampler();
        const float unit24 = 1.0f / 16777215;

        /* ---- witness cases (known results) ---- */
        /* 1. Fails Z LESS, stencil Z-fail INCR: the word changes, so the
         *    fragment takes the full test. */
        one(&c, 1 | 2 | 4, 1 | (7u << 3) | (3u << 12), 0xFFFF00, false, 0.75f, true);
        fill(&init, word(quant(0.5f, false), 5, 0, false));
        CHECK(both(&ptex, false, &c, &init, rtex, smp, r), "witness 1 variants differ");
        CHECK((r[1].c.z[100] & 0xff) == 6 && r[1].c.z[100] >> 8 == quant(0.5f, false) &&
              col32(&r[1].c, 0, 4, 3) == 0 && r[1].zpass == 0,
              "witness 1: word %08x colour %08x count %u", r[1].c.z[100], col32(&r[1].c, 0, 4, 3),
              r[1].zpass);
        /* 2. Passes the preview, killed by the alpha test (NEVER): nothing
         *    written, nothing counted.  ALWAYS: every pixel counted once. */
        one(&c, 2 | 4, 1, 0, false, 0.25f, true);
        c.u.alpha_func = (1u << 11) | (0u << 8);
        fill(&init, word(quant(0.75f, false), 0, 0, false));
        CHECK(both(&ptex, false, &c, &init, rtex, smp, r), "witness 2 variants differ");
        CHECK(r[1].c.z[77] == init.z[77] && col32(&r[1].c, 0, 13, 2) == 0 && r[1].zpass == 0,
              "witness 2 (alpha NEVER): word %08x count %u", r[1].c.z[77], r[1].zpass);
        c.u.alpha_func = (1u << 11) | (7u << 8);
        CHECK(both(&ptex, false, &c, &init, rtex, smp, r), "witness 2b variants differ");
        CHECK(r[1].c.z[77] >> 8 == quant(0.25f, false) && r[1].zpass == W * H,
              "witness 2b (alpha ALWAYS): word %08x count %u", r[1].c.z[77], r[1].zpass);
        /* 3. Shader depth selected: geometric 0.9 fails LESS against 0.5,
         *    the program's 0.1 passes, and the reverse. */
        one(&c, 2 | 4, 1, 0, false, 0.9f, true);
        c.u.depth_src = 1;
        c.u.consts[0][3] = 0.1f;
        fill(&init, word(quant(0.5f, false), 0, 0, false));
        CHECK(both(&pw, false, &c, &init, rtex, smp, r), "witness 3 variants differ");
        CHECK(r[1].c.z[200] >> 8 == quant(0.1f, false) && r[1].zpass == W * H,
              "witness 3 (shader depth passes): word %08x count %u", r[1].c.z[200], r[1].zpass);
        c.u.consts[0][3] = 0.9f;
        c.nv[0] = rect(c.v[0], 0, 0, W, H, 0.1f, 0, 0, 1, true);
        CHECK(both(&pw, false, &c, &init, rtex, smp, r), "witness 3b variants differ");
        CHECK(r[1].c.z[200] == init.z[200] && r[1].zpass == 0,
              "witness 3b (shader depth fails): word %08x count %u", r[1].c.z[200], r[1].zpass);
        /* 4. Polygon offset across the comparison: depth 1.5 units behind
         *    the buffer fails LESS, the offset (-3 units) brings it in front;
         *    front faces (enable bit 0, .x/.y) and back faces (bit 1, .z/.w).
         *    At 0.25, where a float step is a quarter unit (at 0.5 it is one
         *    unit, and the offset rounded back onto the buffer's word). */
        for (int back = 0; back < 2; back++) {
            float z = (quant(0.25f, false) + 1.5f) * unit24;
            one(&c, 2 | 4, 1, 0, false, z, !back);
            c.u.poly_en = back ? 2 : 1;
            c.u.poly_offset[back ? 3 : 1] = -3 * unit24;
            c.u.poly_offset[back ? 1 : 3] = 100 * unit24;   /* the other face's: unused */
            fill(&init, word(quant(0.25f, false), 0, 0, false));
            CHECK(both(&ptex, false, &c, &init, rtex, smp, r), "witness 4 variants differ");
            CHECK(r[1].zpass == W * H &&
                  abs((int)(r[1].c.z[300] >> 8) - (int)quant(z - 3 * unit24, false)) <= 1,
                  "witness 4 (%s offset): word %08x count %u", back ? "back" : "front",
                  r[1].c.z[300], r[1].zpass);
        }
        /* 5. Depth and stencil off with the depth buffer attached: every
         *    fragment lands, the word stays. */
        one(&c, 0, 1, 0xFFFF00, false, 0.75f, true);
        fill(&init, word(quant(0.5f, false), 9, 0, false));
        CHECK(both(&ptex, false, &c, &init, rtex, smp, r), "witness 5 variants differ");
        CHECK(r[1].c.z[400] == init.z[400] && r[1].zpass == W * H,
              "witness 5 (depthless): word %08x count %u", r[1].c.z[400], r[1].zpass);
        /* 6. Passes the preview, then KIL on the left half (t0.z < 0): the
         *    left half keeps its word and is not counted. */
        one(&c, 2 | 4, 1, 0, false, 0.25f, true);
        fill(&init, word(quant(0.75f, false), 0, 0, false));
        CHECK(both(&pkil, false, &c, &init, rtex, smp, r), "witness 6 variants differ");
        CHECK(r[1].c.z[3] == init.z[3] && r[1].c.z[W - 3] >> 8 == quant(0.25f, false) &&
              r[1].zpass == W * H / 2, "witness 6 (KIL): words %08x %08x count %u",
              r[1].c.z[3], r[1].c.z[W - 3], r[1].zpass);
        /* 4b. Sloped: the slope term (factor -4) brings a surface 1 unit
         *     behind the buffer in front of it. */
        {
            float dz = 16 * unit24;
            one(&c, 2 | 4, 1, 0, false, 0.5f, true);
            c.nv[0] = rect(c.v[0], 0, 0, W, H, 0.5f, dz, 0, 1, true);
            c.u.poly_en = 1;
            c.u.poly_offset[0] = -4;
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    init.z[y * W + x] = word(quant(0.5f + dz * (x + 0.5f), false) - 1, 0, 0, false);
                }
            }
            memset(init.col, 0, sizeof(init.col));
            CHECK(both(&ptex, false, &c, &init, rtex, smp, r), "witness 4b variants differ");
            CHECK(r[1].zpass == W * H, "witness 4b (slope offset): count %u", r[1].zpass);
        }

        /* ---- derivatives: every 2x2 quad mixes kept and rejected fragments ---- */
        for (int s = 1; s <= 3; s++) {
            one(&c, 2, 1, 0, false, 0.5f, true);            /* LESS, no Z write */
            c.u.zpass_count = 1;
            c.nv[0] = rect(c.v[0], 0, 0, W, H, 0.5f, 0, 0, 1 << s, true);
            memset(init.col, 0, sizeof(init.col));
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    init.z[y * W + x] = word(quant((x ^ y) & 1 ? 0.25f : 0.75f, false), 0, 0, false);
                }
            }
            CHECK(both(&ptex, false, &c, &init, ltex, smp, r), "mip %d variants differ", s);
            int bad = 0;
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    uint32_t want = (x ^ y) & 1 ? 0 : LVL[s];
                    bad += col32(&r[1].c, 0, x, y) != want;
                }
            }
            CHECK(bad == 0 && r[1].zpass == W * H / 2, "mip level %d: %d pixels wrong, count %u",
                  s, bad, r[1].zpass);
            c.u.tex_lod[0][0] = 1.0f;                       /* LOD bias +1 */
            CHECK(both(&ptex, false, &c, &init, ltex, smp, r), "mip %d bias variants differ", s);
            CHECK(col32(&r[1].c, 0, 0, 0) == LVL[s + 1], "mip level %d bias +1: %08x", s,
                  col32(&r[1].c, 0, 0, 0));
            c.u.tex_lod[0][0] = 0;
            const struct { Prog *pg; int dl; } ops[] = { { &ptxp, 0 }, { &ptxb, 1 } };
            for (unsigned o = 0; o < 2; o++) {
                CHECK(both(ops[o].pg, false, &c, &init, ltex, smp, r), "mip %d %s variants differ",
                      s, ops[o].pg->name);
                bad = 0;
                for (int y = 0; y < H; y++) {
                    for (int x = 0; x < W; x++) {
                        uint32_t want = (x ^ y) & 1 ? 0 : LVL[s + ops[o].dl];
                        bad += col32(&r[1].c, 0, x, y) != want;
                    }
                }
                CHECK(bad == 0, "mip level %d %s: %d pixels wrong", s, ops[o].pg->name, bad);
            }
        }
        /* Shader-decoded texture (16F, 8x8, 4 levels, level L holds L): the
         * level comes from dFdx/dFdy of the coordinate. */
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
                    uint32_t v = hb | (uint32_t)hb << 16;
                    mem[td.lvl_off[l] + b] = v >> 24; mem[td.lvl_off[l] + b + 1] = v >> 16;
                    mem[td.lvl_off[l] + b + 2] = v >> 8; mem[td.lvl_off[l] + b + 3] = v;
                }
            }
            program(&st, true, false, false);
            r300_state_write(&st, 0x4104, 1);
            r300_state_write(&st, 0x44C0, 0x18);
            r300_state_write(&st, 0x4E38, (7u << 21) | W);  /* ARGB32323232 */
            r300_state_write(&st, 0x46A4, 21);              /* C4_32_FP */
            Prog praw = { "raw 16F", 1, MTLPixelFormatRGBA32Uint };
            dump_msl(&st, "raw16f");
            if (!make_prog(&praw, &st)) return 1;
            uint32_t rowel = td.lvl_pitch[0] / 4, rows = (td.size_bytes + rowel * 4 - 1) / (rowel * 4);
            MTLTextureDescriptor *d = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint width:rowel height:rows
                                         mipmapped:NO];
            id<MTLTexture> raw = [dev newTextureWithDescriptor:d];
            static uint8_t pad[65536];
            memcpy(pad, mem, td.size_bytes);
            [raw replaceRegion:MTLRegionMake2D(0, 0, rowel, rows) mipmapLevel:0 withBytes:pad
                   bytesPerRow:rowel * 4];
            /* A quad of side px pixels over the 8x8 level 0: 8 / px texels
             * per pixel, level log2(8 / px). */
            static const struct { int px, level; } rq[] = { { 4, 1 }, { 2, 2 } };
            for (unsigned i = 0; i < 2; i++) {
                one(&c, 2, 1, 0, false, 0.5f, true);
                c.u.tex_info[0][0] = 1; c.u.tex_info[0][1] = rowel;
                c.u.tex_info[0][2] = 8; c.u.tex_info[0][3] = 8;
                c.u.tex_lod[0][0] = 0; c.u.tex_lod[0][1] = 0; c.u.tex_lod[0][2] = 3;
                c.u.tex_lod[0][3] = 1;
                c.u.tex_dim[0][0] = 0; c.u.tex_dim[0][1] = 1; c.u.tex_dim[0][2] = td.lvl_pitch[0];
                c.u.tex_dim[0][3] = R300_TEXF_POT_ROWS;
                c.nv[0] = rect(c.v[0], 0, 0, rq[i].px, rq[i].px, 0.5f, 0, 0,
                               (float)W / rq[i].px, true);
                memset(init.col, 0, sizeof(init.col));
                for (int y = 0; y < H; y++) {
                    for (int x = 0; x < W; x++) {
                        init.z[y * W + x] = word(quant((x ^ y) & 1 ? 0.25f : 0.75f, false), 0, 0,
                                                 false);
                    }
                }
                CHECK(both(&praw, false, &c, &init, raw, smp, r), "raw px %d variants differ",
                      rq[i].px);
                float f;
                memcpy(&f, &r[1].c.col[0][0], 4);           /* pixel (0, 0): kept */
                CHECK(f == rq[i].level && r[1].zpass == (unsigned)(rq[i].px * rq[i].px / 2),
                      "raw level %g want %d, count %u", f, rq[i].level, r[1].zpass);
            }
        }

        /* ---- the byte-order model (default configuration) ---- */
        {
            /* Colour bytes that no swap leaves alone. */
            static const uint8_t pat[8] = { 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF };
            /* M1: Z LESS fails everywhere, KEEP: every fragment is rejected
             *     early; nothing changes, nothing counted. */
            one(&c, 1 | 2 | 4, 1 | (7u << 3), 0xFFFF00, false, 0.75f, true);
            c.u.rt_swap32 = 1;
            fill(&init, word(quant(0.5f, false), 5, 0, false));
            for (size_t i = 0; i < sizeof(init.col); i++) ((uint8_t *)init.col)[i] = pat[i % 8];
            CHECK(both(&ptex_m, false, &c, &init, rtex, smp, r), "model M1 variants differ");
            CHECK(memcmp(r[1].c.col[0], init.col[0], W * H * 4) == 0 &&
                  memcmp(r[1].c.z, init.z, sizeof(init.z)) == 0 && r[1].zpass == 0,
                  "model M1: contents changed, count %u", r[1].zpass);
            /* M2: passes: every fragment survives, colour and depth written. */
            c.u.zinfo[1] = 4 | (7u << 3);                   /* GEQUAL */
            CHECK(both(&ptex_m, false, &c, &init, rtex, smp, r), "model M2 variants differ");
            CHECK(memcmp(r[1].c.col[0], init.col[0], W * H * 4) != 0 && r[1].zpass == W * H &&
                  r[1].c.z[5] >> 8 == quant(0.75f, false), "model M2: count %u word %08x",
                  r[1].zpass, r[1].c.z[5]);
            /* M3: MRT and multiwrite, channel mask and blending, half the
             *     pixels rejected (checkerboard). */
            const Prog *mp[2] = { &pmrt_m, &pmw_m };
            for (int k = 0; k < 2; k++) {
                one(&c, 2, 1, 0, false, 0.5f, true);
                c.u.rt_swap32 = 1;
                c.u.chanmask = 0x5;
                c.u.cblend = 1 | (4u << 16) | (5u << 24);       /* enable, src alpha / 1 - src alpha */
                c.u.ablend = c.u.cblend;
                for (int y = 0; y < H; y++) {
                    for (int x = 0; x < W; x++) {
                        init.z[y * W + x] = word(quant((x ^ y) & 1 ? 0.25f : 0.75f, false), 0, 0,
                                                 false);
                    }
                }
                CHECK(both(mp[k], false, &c, &init, rtex, smp, r), "model M3 %s variants differ",
                      mp[k]->name);
                CHECK(r[1].zpass == W * H / 2 &&
                      memcmp(r[1].c.col[1], init.col[1], W * H * 4) != 0 &&
                      memcmp(&r[1].c.col[0][4], &init.col[0][4], 4) == 0,   /* (1, 0): rejected */
                      "model M3 %s: count %u", mp[k]->name, r[1].zpass);
            }
            /* Random cases under the model. */
            const Prog *rp[3] = { &ptex_m, &pmrt_m, &pmw_m };
            unsigned n = getenv("EARLYZ_CASES") ? atoi(getenv("EARLYZ_CASES")) / 5 : 1000;
            unsigned differ = 0;
            for (unsigned i = 0; i < n; i++) {
                bool z16 = (i / 3) & 1;
                random_case(&c, &init, rp[i % 3], z16, false);
                c.u.rt_swap32 = 1;
                differ += !both(rp[i % 3], z16, &c, &init, rtex, smp, r);
            }
            CHECK(differ == 0, "%u of %u model random cases differ", differ, n);
            printf("byte-order model: %u random cases, %u differ\n", n, differ);
        }

        /* ---- random cases ---- */
        {
            const struct { Prog *pg; bool w; } progs[] = { { &ptex, false }, { &pw, true },
                                                           { &pmrt, false }, { &pmw, false },
                                                           { &pkil, false } };
            unsigned n = getenv("EARLYZ_CASES") ? atoi(getenv("EARLYZ_CASES")) : 5000;
            unsigned differ = 0, written = 0, counted = 0;
            for (unsigned i = 0; i < n; i++) {
                unsigned k = i % 5;
                bool z16 = (i / 5) & 1;
                random_case(&c, &init, progs[k].pg, z16, progs[k].w);
                if (!both(progs[k].pg, z16, &c, &init, rtex, smp, r)) {
                    if (differ++ < 5) {
                        int zd = -1, cd = -1;
                        for (int j = 0; j < W * H && zd < 0; j++) {
                            if (r[0].c.z[j] != r[1].c.z[j]) zd = j;
                        }
                        for (int j = 0; j < (int)sizeof(r[0].c.col) && cd < 0; j++) {
                            if (((uint8_t *)r[0].c.col)[j] != ((uint8_t *)r[1].c.col)[j]) cd = j;
                        }
                        printf("FAIL random case %u (%s, %s): first z word differs at %d, "
                               "colour byte at %d, count %u / %u\n", i, progs[k].pg->name,
                               z16 ? "Z16" : "Z24S8", zd, cd, r[0].zpass, r[1].zpass);
                    }
                }
                written += memcmp(r[0].c.z, init.z, sizeof(init.z)) != 0;
                counted += r[0].zpass != 0;
            }
            CHECK(differ == 0, "%u of %u random cases differ", differ, n);
            printf("random cases: %u run, %u differ; %u changed depth/stencil, %u counted Z-pass\n",
                   n, differ, written, counted);
        }
        printf(fails ? "test_earlyz: %d FAILED\n" : "test_earlyz: PASS\n", fails);
    }
    return fails != 0;
}
