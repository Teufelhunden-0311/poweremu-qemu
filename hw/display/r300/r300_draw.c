/*
 * R300-family draw assembly.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#include "r300_draw.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "r300_pvs.h"

/* Registers used here */
#define SE_VPORT_XSCALE             0x1D98  /* then XOFFSET, YSCALE, ... ZOFFSET */
#define VAP_CNTL_STATUS             0x2140
#define VAP_OUTPUT_VTX_FMT_0        0x2090
#define VAP_OUTPUT_VTX_FMT_1        0x2094
#define VAP_VTE_CNTL                0x20B0
#define VAP_VTX_SIZE                0x20B4
#define VAP_PROG_STREAM_CNTL_0      0x2150
#define VAP_PROG_STREAM_CNTL_EXT_0  0x21E0
#define VAP_PVS_CODE_CNTL_0         0x22D0
#define VAP_PVS_CONST_CNTL          0x22D4
#define TX_ENABLE                   0x4104
#define RS_COUNT                    0x4300
#define RS_INST_COUNT               0x4304
#define RS_IP_0                     0x4310
#define RS_INST_0                   0x4330
#define SC_CLIPRECT_TL_0            0x43B0  /* then BR_0, TL_1, BR_1, ... */
#define SC_CLIP_RULE                0x43D0
#define SC_SCISSORS_TL              0x43E0
#define SC_SCISSORS_BR              0x43E4
#define TX_FILTER0_0                0x4400
#define TX_FORMAT0_0                0x4480
#define TX_FORMAT1_0                0x44C0
#define TX_FORMAT2_0                0x4500
#define TX_OFFSET_0                 0x4540
#define US_OUT_FMT_0                0x46A4
#define FG_ALPHA_FUNC               0x4BD4
#define PFS_PARAM_0_X               0x4C00
#define RB3D_CBLEND                 0x4E04
#define RB3D_ABLEND                 0x4E08
#define RB3D_COLOR_CHANNEL_MASK     0x4E0C
#define RB3D_BLEND_COLOR            0x4E10
#define RB3D_COLOROFFSET0           0x4E28
#define RB3D_COLORPITCH0            0x4E38
#define SU_CULL_MODE                0x42B8
#define ZB_CNTL                     0x4F00
#define ZB_ZSTENCILCNTL             0x4F04
#define ZB_STENCILREFMASK           0x4F08
#define ZB_FORMAT                   0x4F10
#define ZB_DEPTHOFFSET              0x4F20
#define ZB_DEPTHPITCH               0x4F24
#define GA_POINT_S0                 0x4200  /* then T0, S1, T1 */
#define GA_POINT_T0                 0x4204
#define GA_POINT_S1                 0x4208
#define GA_POINT_T1                 0x420C
#define GA_POINT_SIZE               0x421C

#define SC_COORD_BIAS               1440    /* scissor/cliprect origin */

static float bits_to_float(uint32_t v)
{
    float f;
    memcpy(&f, &v, 4);
    return f;
}

static uint32_t vc_swap(uint32_t v, unsigned mode)
{
    switch (mode & 3) {
    case 1:  return ((v & 0x00FF00FFu) << 8) | ((v >> 8) & 0x00FF00FFu);
    case 2:  return __builtin_bswap32(v);
    case 3:  return (v << 16) | (v >> 16);
    default: return v;
    }
}

static float half_to_float(uint16_t h)
{
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    float f;

    if (e == 0) {
        f = ldexpf((float)m, -24);
    } else if (e == 31) {
        f = m ? NAN : INFINITY;
    } else {
        f = ldexpf(1.0f + m / 1024.0f, (int)e - 15);
    }
    return s ? -f : f;
}

void r300_load_vbpntr(R300Arrays *arr, const uint32_t *d, uint32_t ndw)
{
    unsigned n, i;

    memset(arr, 0, sizeof(*arr));
    if (ndw < 1) {
        return;
    }
    n = d[0] & 0x1F;
    for (i = 0; i < n && i < R300_MAX_ARRAYS; i++) {
        unsigned pair = i / 2, half = i % 2;
        unsigned base = 1 + pair * 3;
        uint32_t fmt;

        if (base + 1 + half >= ndw) {
            break;
        }
        fmt = d[base] >> (16 * half);
        arr->a[i].size_dw = fmt & 0x7F;
        arr->a[i].stride_dw = (fmt >> 8) & 0x7F;
        arr->a[i].addr = d[base + 1 + half];
    }
    arr->count = i;
}

/* ---- vertex fetch --------------------------------------------------- */

typedef struct Fetch {
    const R300State *st;
    const R300Arrays *arr;
    R300ReadFn read;
    void *opaque;
    const uint32_t *immd;       /* immediate vertex data, or NULL */
    uint32_t immd_dw;
    uint32_t vtx_size;          /* VAP_VTX_SIZE, dwords */
    unsigned swap;              /* VAP_CNTL_STATUS.VC_SWAP */
    uint32_t warn;
    const char *why;            /* first fetch failure */
} Fetch;

static unsigned psc_dwords(unsigned type)
{
    static const unsigned n[16] = { 1, 2, 3, 4, 1, 1, 1, 2, 1, 1, 8, 1, 2, 0, 0, 0 };
    return n[type & 15];
}

/* Fetch dwords for PSC element k of vertex vtx starting at *cursor. */
static bool fetch_dwords(Fetch *f, unsigned k, uint32_t vtx, unsigned cursor,
                         unsigned count, uint32_t *out)
{
    if (f->immd) {
        uint64_t at = (uint64_t)vtx * f->vtx_size + cursor;
        if (at + count > f->immd_dw) {
            f->why = "immediate vertex data shorter than VAP_VTX_SIZE says";
            return false;
        }
        memcpy(out, &f->immd[at], count * 4);   /* CP data: already in order */
        return true;
    }
    if (!f->arr->count) {
        f->why = "vertex element with no 3D_LOAD_VBPNTR array";
        return false;
    }
    /* More elements than arrays: the rest are interleaved in the last one
     * (fetch_vertex keeps the cursor running for them). */
    if (k >= f->arr->count) {
        k = f->arr->count - 1;
    }
    uint32_t addr = f->arr->a[k].addr +
                    (vtx * f->arr->a[k].stride_dw + cursor) * 4;
    if (!f->read(f->opaque, addr, out, count * 4)) {
        f->why = "vertex array outside VRAM and the GART";
        return false;
    }
    for (unsigned i = 0; i < count; i++) {
        out[i] = vc_swap(out[i], f->swap);
    }
    return true;
}

static void decode_element(unsigned type, bool sgn, bool norm,
                           const uint32_t *w, float c[4])
{
    c[0] = c[1] = c[2] = 0.0f;
    c[3] = 1.0f;
    switch (type) {
    case 0: case 1: case 2: case 3:         /* FLOAT_1..4 */
        for (unsigned i = 0; i <= type; i++) {
            c[i] = bits_to_float(w[i]);
        }
        break;
    case 4:                                 /* BYTE x4 */
    case 5:                                 /* D3DCOLOR */
        for (unsigned i = 0; i < 4; i++) {
            unsigned b = (w[0] >> (8 * i)) & 0xFF;
            float x = sgn ? (float)(int8_t)b : (float)b;
            if (norm || type == 5) {
                x = sgn ? fmaxf(x / 127.0f, -1.0f) : x / 255.0f;
            }
            c[i] = x;
        }
        if (type == 5) {                    /* stored B,G,R,A */
            float t = c[0];
            c[0] = c[2];
            c[2] = t;
        }
        break;
    case 6:                                 /* SHORT_2 */
    case 7:                                 /* SHORT_4 */
        for (unsigned i = 0; i < (type == 6 ? 2u : 4u); i++) {
            unsigned h = (w[i / 2] >> (16 * (i % 2))) & 0xFFFF;
            float x = sgn ? (float)(int16_t)h : (float)h;
            if (norm) {
                x = sgn ? fmaxf(x / 32767.0f, -1.0f) : x / 65535.0f;
            }
            c[i] = x;
        }
        break;
    case 11:                                /* FLT16_2 */
    case 12:                                /* FLT16_4 */
        for (unsigned i = 0; i < (type == 11 ? 2u : 4u); i++) {
            c[i] = half_to_float((w[i / 2] >> (16 * (i % 2))) & 0xFFFF);
        }
        break;
    default:
        break;
    }
}

/* Fill the input vertex memory for one vertex through the PSC. */
static bool fetch_vertex(Fetch *f, uint32_t vtx, float in[R300_PVS_NUM_INPUTS][4])
{
    unsigned cursor = 0;

    memset(in, 0, sizeof(float) * 4 * R300_PVS_NUM_INPUTS);
    for (unsigned k = 0; k < 16; k++) {
        uint32_t reg = r300_reg(f->st, VAP_PROG_STREAM_CNTL_0 + 4 * (k / 2));
        uint32_t ext = r300_reg(f->st, VAP_PROG_STREAM_CNTL_EXT_0 + 4 * (k / 2));
        unsigned e = (reg >> (16 * (k % 2))) & 0xFFFF;
        unsigned x = (ext >> (16 * (k % 2))) & 0xFFFF;
        unsigned type = e & 0xF, skip = (e >> 4) & 0xF, dst = (e >> 8) & 0x1F;
        unsigned n = psc_dwords(type);
        uint32_t w[8];
        float c[4];

        if (!f->immd && k < f->arr->count) {
            cursor = 0;                 /* each array starts afresh */
        }
        cursor += skip;
        if (!n || type == 8 || type == 9 || type == 10) {
            f->warn |= R300_WARN_VTXFMT;
        } else if (!fetch_dwords(f, k, vtx, cursor, n, w)) {
            return false;
        } else {
            decode_element(type, (e >> 14) & 1, (e >> 15) & 1, w, c);
            for (unsigned i = 0; i < 4; i++) {
                if ((x >> (12 + i)) & 1) {
                    unsigned s = (x >> (3 * i)) & 7;
                    in[dst][i] = s < 4 ? c[s] : s == 5 ? 1.0f : 0.0f;
                }
            }
        }
        cursor += n;
        if (e & (1u << 13)) {           /* LAST_VEC */
            break;
        }
    }
    return true;
}

/* ---- output routing ------------------------------------------------- */

typedef struct Layout {
    int pos, color[4];
    int tex_slot[8];
    unsigned tex_n[8];
    /* texture interpolants as one scalar list: (slot, component) */
    unsigned nscal;
    uint8_t scal_slot[32], scal_comp[32];
} Layout;

static void output_layout(const R300State *st, Layout *l)
{
    uint32_t f0 = r300_reg(st, VAP_OUTPUT_VTX_FMT_0);
    uint32_t f1 = r300_reg(st, VAP_OUTPUT_VTX_FMT_1);
    int slot = 0;

    memset(l, -1, sizeof(*l));
    l->nscal = 0;
    l->pos = slot++;                    /* always present */
    if (f0 & (1u << 16)) {
        slot++;                         /* point size */
    }
    for (int c = 0; c < 4; c++) {
        if (f0 & (1u << (1 + c))) {
            l->color[c] = slot++;
        }
    }
    for (int t = 0; t < 8; t++) {
        unsigned n = (f1 >> (3 * t)) & 7;
        l->tex_n[t] = n;
        if (n) {
            l->tex_slot[t] = slot++;
            for (unsigned i = 0; i < n && l->nscal < 32; i++) {
                l->scal_slot[l->nscal] = l->tex_slot[t];
                l->scal_comp[l->nscal++] = i;
            }
        }
    }
}

typedef struct Route {
    unsigned nvary;
    struct {
        bool is_tex;
        unsigned ip;
    } vary[R300_NUM_VARYINGS];
} Route;

/* RS_INST -> varyings and the temp routing for the fragment program. */
static void rs_route(const R300State *st, Route *r, R300FSDesc *desc,
                     uint32_t *warn)
{
    unsigned count = (r300_reg(st, RS_INST_COUNT) & 0xF) + 1;

    memset(r, 0, sizeof(*r));
    memset(desc->route, -1, sizeof(desc->route));
    for (unsigned i = 0; i < count; i++) {
        uint32_t inst = r300_reg(st, RS_INST_0 + 4 * i);

        for (int pass = 0; pass < 2; pass++) {
            bool tex = pass == 0;
            bool write = tex ? (inst >> 3) & 1 : (inst >> 14) & 1;
            unsigned ip = tex ? inst & 7 : (inst >> 11) & 7;
            unsigned addr = tex ? (inst >> 6) & 0x1F : (inst >> 17) & 0x1F;

            if (!write) {
                continue;
            }
            if (r->nvary == R300_NUM_VARYINGS) {
                *warn |= R300_WARN_VARYINGS;
                continue;
            }
            r->vary[r->nvary].is_tex = tex;
            r->vary[r->nvary].ip = ip;
            desc->route[addr] = r->nvary++;
        }
    }
}

static float rs_col_fmt(unsigned fmt, const float *c, unsigned i)
{
    static const int8_t tbl[16][4] = {
        /* per component: 0-3 = channel, -1 = 0.0, -2 = 1.0 */
        [0] = { 0, 1, 2, 3 },  [1] = { 0, 1, 2, -1 }, [2] = { 0, 1, 2, -2 },
        [4] = { -1, -1, -1, 3 }, [5] = { -1, -1, -1, -1 }, [6] = { -1, -1, -1, -2 },
        [8] = { -2, -2, -2, 3 }, [9] = { -2, -2, -2, -1 }, [10] = { -2, -2, -2, -2 },
    };
    int s = tbl[fmt & 15][i];
    return s >= 0 ? c[s] : s == -2 ? 1.0f : 0.0f;
}

static void rs_values(const R300State *st, const Layout *l, const Route *r,
                      float out[][4], const float *sprite,
                      float v[R300_NUM_VARYINGS][4])
{
    static const float zero[4];

    for (unsigned k = 0; k < r->nvary; k++) {
        uint32_t ip = r300_reg(st, RS_IP_0 + 4 * r->vary[k].ip);

        if (r->vary[k].is_tex) {
            unsigned ptr = ip & 0x3F;
            for (unsigned i = 0; i < 4; i++) {
                unsigned sel = (ip >> (13 + 3 * i)) & 7;
                float x = 0.0f;
                if (sel < 4) {
                    unsigned s = ptr + sel;
                    if (s < l->nscal) {
                        x = out[l->scal_slot[s]][l->scal_comp[s]];
                    } else if (sprite) {
                        /* Point sprite: the rasterizer generates (s, t). */
                        unsigned c = s - l->nscal;
                        x = c < 2 ? sprite[c] : c == 3 ? 1.0f : 0.0f;
                    }
                } else if (sel == 5) {
                    x = 1.0f;
                }
                v[k][i] = x;
            }
        } else {
            unsigned cp = (ip >> 6) & 7;
            int slot = cp < 4 ? l->color[cp] : -1;
            const float *c = slot >= 0 ? out[slot] : zero;
            for (unsigned i = 0; i < 4; i++) {
                v[k][i] = rs_col_fmt((ip >> 9) & 0xF, c, i);
            }
        }
    }
}

/* ---- draw ----------------------------------------------------------- */

static void set_uniforms(const R300State *st, R300DrawPacket *pkt)
{
    R300FSUniforms *u = &pkt->uniforms;
    uint32_t outfmt = r300_reg(st, US_OUT_FMT_0);
    uint32_t bc = r300_reg(st, RB3D_BLEND_COLOR);

    memset(u, 0, sizeof(*u));
    for (int i = 0; i < R300_US_NUM_CONSTS; i++) {
        for (int c = 0; c < 4; c++) {
            u->consts[i][c] = r300_float24(r300_reg(st, PFS_PARAM_0_X + 16 * i + 4 * c));
        }
    }
    u->blend_color[0] = ((bc >> 16) & 0xFF) / 255.0f;
    u->blend_color[1] = ((bc >> 8) & 0xFF) / 255.0f;
    u->blend_color[2] = (bc & 0xFF) / 255.0f;
    u->blend_color[3] = (bc >> 24) / 255.0f;
    u->cblend = r300_reg(st, RB3D_CBLEND);
    u->ablend = r300_reg(st, RB3D_ABLEND);
    u->chanmask = r300_reg(st, RB3D_COLOR_CHANNEL_MASK) & 0xF;
    u->alpha_func = r300_reg(st, FG_ALPHA_FUNC);
    for (int n = 0; n < 4; n++) {
        u->out_sel[n] = (outfmt >> (8 + 2 * n)) & 3;
    }
    u->rt_swap32 = ((r300_reg(st, RB3D_COLORPITCH0) >> 19) & 3) == 2;
    u->rt_endian = (r300_reg(st, RB3D_COLORPITCH0) >> 19) & 3;
    u->clip_rule = r300_reg(st, SC_CLIP_RULE) & 0xFFFF;
    for (int i = 0; i < 4; i++) {
        uint32_t tl = r300_reg(st, SC_CLIPRECT_TL_0 + 8 * i);
        uint32_t br = r300_reg(st, SC_CLIPRECT_TL_0 + 8 * i + 4);
        u->cliprect[i][0] = (int32_t)(tl & 0x1FFF) - SC_COORD_BIAS;
        u->cliprect[i][1] = (int32_t)((tl >> 13) & 0x1FFF) - SC_COORD_BIAS;
        u->cliprect[i][2] = (int32_t)(br & 0x1FFF) - SC_COORD_BIAS;
        u->cliprect[i][3] = (int32_t)((br >> 13) & 0x1FFF) - SC_COORD_BIAS;
    }
}

static void set_textures(const R300State *st, R300DrawPacket *pkt)
{
    uint32_t enable = r300_reg(st, TX_ENABLE);

    for (unsigned k = 0; k < R300_NUM_TEX_UNITS; k++) {
        R300TexDesc *t = &pkt->tex[k];
        uint32_t f0 = r300_reg(st, TX_FORMAT0_0 + 4 * k);
        uint32_t f1 = r300_reg(st, TX_FORMAT1_0 + 4 * k);
        uint32_t f2 = r300_reg(st, TX_FORMAT2_0 + 4 * k);
        uint32_t off = r300_reg(st, TX_OFFSET_0 + 4 * k);
        uint32_t fmt = f1 & 0x1F;

        memset(t, 0, sizeof(*t));
        /* swizzle: R 14:12, G 17:15, B 20:18, A 11:9 */
        pkt->uniforms.tex_swz[k][0] = (f1 >> 12) & 7;
        pkt->uniforms.tex_swz[k][1] = (f1 >> 15) & 7;
        pkt->uniforms.tex_swz[k][2] = (f1 >> 18) & 7;
        pkt->uniforms.tex_swz[k][3] = (f1 >> 9) & 7;
        if (!(enable & (1u << k))) {
            continue;
        }
        unsigned bpp, decode = 0;
        switch (fmt) {
        case 0x0:                       /* X8 */
            t->kind = R300_TEXK_R8; bpp = 1;
            break;
        case 0x3:                       /* Y8X8: big-endian halfwords */
            t->kind = R300_TEXK_RG8; bpp = 2; decode = 2;
            break;
        case 0x1: case 0x6: case 0x7: case 0xA: case 0xB:
            t->kind = R300_TEXK_CONVERT16; bpp = 2;
            break;
        case 0xC:                       /* W8Z8Y8X8 */
            /* VRAM holds the guest CPU's big-endian words (see r300_tex). */
            t->kind = R300_TEXK_RGBA8; bpp = 4; decode = (off & 3) == 0;
            break;
        case 0xF:  t->kind = R300_TEXK_DXT1; bpp = 8;  break;   /* per block */
        case 0x10: t->kind = R300_TEXK_DXT3; bpp = 16; break;
        case 0x11: t->kind = R300_TEXK_DXT5; bpp = 16; break;
        default:
            bpp = r300_tex_raw_bpp(fmt);
            if (!bpp) {
                pkt->warn |= R300_WARN_TEXFMT;
                continue;
            }
            t->kind = R300_TEXK_RAW;
            t->view_bpp = bpp < 4 ? 4 : bpp;
            break;
        }
        t->bound = true;
        t->gpu_addr = off & ~0x1Fu;
        t->width = (f0 & 0x7FF) + 1;
        t->height = ((f0 >> 11) & 0x7FF) + 1;
        {
            uint32_t texels = (f0 >> 31) ? (f2 & 0x3FFF) + 1 : t->width;
            t->pitch_bytes = t->kind >= R300_TEXK_DXT1 ? ((texels + 3) / 4) * bpp
                                                       : texels * bpp;
        }
        t->format = fmt;
        t->filter0 = r300_reg(st, TX_FILTER0_0 + 4 * k);
        pkt->uniforms.tex_info[k][0] = 1;
        pkt->uniforms.tex_info[k][1] = decode;
        pkt->uniforms.tex_info[k][2] = t->width;
        pkt->uniforms.tex_info[k][3] = t->height;
    }
}

static void set_depth(const R300State *st, R300DrawPacket *pkt)
{
    uint32_t cntl = r300_reg(st, ZB_CNTL);
    uint32_t fmt = r300_reg(st, ZB_FORMAT) & 0xF;
    uint32_t pitch = r300_reg(st, ZB_DEPTHPITCH);
    R300DepthDesc *z = &pkt->depth;
    R300FSUniforms *u = &pkt->uniforms;

    memset(z, 0, sizeof(*z));
    /* STENCIL_ENABLE or Z_ENABLE; Z_WRITE_ENABLE alone does nothing. */
    if (!(cntl & 3)) {
        return;
    }
    z->attach = true;
    z->gpu_addr = r300_reg(st, ZB_DEPTHOFFSET) & ~0x1Fu;
    z->pitch = pitch & 0x3FFC;
    z->format = fmt;
    z->bpp = fmt == 2 ? 4 : 2;
    if (fmt > 2) {
        pkt->warn |= R300_WARN_DEPTH;       /* reserved format: as Z24S8 */
        z->bpp = 4;
    } else if (fmt == 1) {
        pkt->warn |= R300_WARN_DEPTH;       /* 13E3 float: kept as 16-bit unorm */
    }
    u->zinfo[0] = cntl;
    u->zinfo[1] = r300_reg(st, ZB_ZSTENCILCNTL);
    u->zinfo[2] = r300_reg(st, ZB_STENCILREFMASK);
    u->zinfo[3] = ((pitch >> 19) & R300_ZFMT_ENDIAN_MASK) |
                  (z->bpp == 2 ? R300_ZFMT_Z16 : 0);
}

static void viewport_xform(const R300State *st, const R300DrawPacket *pkt,
                           const float c[4], float p[4])
{
    uint32_t vte = r300_reg(st, VAP_VTE_CNTL);
    float vp[6];
    float W = pkt->rt_width, H = pkt->rt_height;

    for (int i = 0; i < 6; i++) {
        bool en = (vte >> i) & 1;
        vp[i] = en ? bits_to_float(r300_reg(st, SE_VPORT_XSCALE + 4 * i))
                   : (i % 2 ? 0.0f : 1.0f);
    }
    /* vp: xscale, xoffset, yscale, yoffset, zscale, zoffset */
    if (!((vte >> 8) & 1)) {
        /* Perspective divide by the VTE: stay linear in clip space so Metal
         * clips and interpolates exactly as the card would. */
        float w = c[3];
        p[0] = (2.0f * vp[0] / W) * c[0] + (2.0f * vp[1] / W - 1.0f) * w;
        p[1] = -(2.0f * vp[2] / H) * c[1] + (1.0f - 2.0f * vp[3] / H) * w;
        p[2] = vp[4] * c[2] + vp[5] * w;
        p[3] = w;
    } else {
        float wx = vp[0] * c[0] + vp[1], wy = vp[2] * c[1] + vp[3];
        p[0] = 2.0f * wx / W - 1.0f;
        p[1] = 1.0f - 2.0f * wy / H;
        p[2] = vp[4] * c[2] + vp[5];
        p[3] = 1.0f;
    }
}

/* Emit a primitive list as expanded triangles/lines/points. */
uint32_t r300_assemble(unsigned prim, uint32_t n, uint32_t *list,
                       uint32_t *cls)
{
    uint32_t m = 0;

#define PUT(a) (list[m++] = (a))
    switch (prim) {
    case 1:                                     /* points */
        *cls = 2;
        for (uint32_t i = 0; i < n; i++) PUT(i);
        break;
    case 2:                                     /* lines */
        *cls = 1;
        for (uint32_t i = 0; i + 1 < n; i += 2) { PUT(i); PUT(i + 1); }
        break;
    case 3:                                     /* line strip */
    case 12:                                    /* line loop */
        *cls = 1;
        for (uint32_t i = 0; i + 1 < n; i++) { PUT(i); PUT(i + 1); }
        if (prim == 12 && n > 2) { PUT(n - 1); PUT(0); }
        break;
    case 4:                                     /* triangles */
        *cls = 0;
        for (uint32_t i = 0; i + 2 < n; i += 3) { PUT(i); PUT(i + 1); PUT(i + 2); }
        break;
    case 5:                                     /* fan */
    case 15:                                    /* polygon */
        *cls = 0;
        for (uint32_t i = 1; i + 1 < n; i++) { PUT(0); PUT(i); PUT(i + 1); }
        break;
    case 6:                                     /* strip */
        *cls = 0;
        for (uint32_t i = 0; i + 2 < n; i++) {
            if (i & 1) { PUT(i + 1); PUT(i); PUT(i + 2); }
            else       { PUT(i); PUT(i + 1); PUT(i + 2); }
        }
        break;
    case 13:                                    /* quads */
        *cls = 0;
        for (uint32_t i = 0; i + 3 < n; i += 4) {
            PUT(i); PUT(i + 1); PUT(i + 2);
            PUT(i); PUT(i + 2); PUT(i + 3);
        }
        break;
    case 14:                                    /* quad strip */
        *cls = 0;
        for (uint32_t i = 0; i + 3 < n; i += 2) {
            PUT(i); PUT(i + 1); PUT(i + 3);
            PUT(i); PUT(i + 3); PUT(i + 2);
        }
        break;
    default:
        return 0;
    }
#undef PUT
    return m;
}

/*
 * Everything after vertex ordering.  order[] (n entries, malloc'd) is
 * taken over; immd is the DRAW_IMMD_2 vertex data or NULL.
 */
static bool draw_core(const R300State *st, const R300Arrays *arr,
                      uint32_t vf, uint32_t *order, uint32_t n,
                      const uint32_t *immd, uint32_t immd_dw,
                      R300ReadFn read, void *opaque,
                      R300DrawPacket *pkt, const char **err)
{
    uint32_t prim, nidx = 0;
    uint32_t *list = NULL;
    Fetch f = { 0 };
    Layout lay;
    Route route;
    R300FSDesc desc;
    uint32_t pitch_reg = r300_reg(st, RB3D_COLORPITCH0);
    uint32_t cntl0 = r300_reg(st, VAP_PVS_CODE_CNTL_0);
    uint32_t cc = r300_reg(st, VAP_PVS_CONST_CNTL);
    bool bypass = (r300_reg(st, VAP_CNTL_STATUS) >> 8) & 1;
    uint32_t tl = r300_reg(st, SC_SCISSORS_TL), br = r300_reg(st, SC_SCISSORS_BR);
    R300PVSProgram prog;
    float consts[256][4];

    prim = vf & 0xF;

    /* Colour buffer */
    pkt->rt_gpu_addr = r300_reg(st, RB3D_COLOROFFSET0) & ~0x1Fu;
    pkt->rt_pitch = pitch_reg & 0x3FFE;
    pkt->rt_format = (pitch_reg >> 21) & 0xF;
    pkt->rt_view = r300_cb_view(pkt->rt_format, r300_reg(st, US_OUT_FMT_0),
                                &pkt->rt_bpp);
    if ((pkt->rt_bpp == 1 && (pitch_reg >> 19) & 3) ||
        (pkt->rt_bpp == 2 && ((pitch_reg >> 19) & 3) >= 2)) {
        pkt->warn |= R300_WARN_RTFMT;   /* swap across pixels: not done */
    }
    if (pkt->rt_view == R300_RTV_NONE) {
        pkt->warn |= R300_WARN_RTFMT;
        *err = "colour buffer format not supported";
        free(order);
        return false;
    }
    pkt->scissor[0] = ((tl & 0x1FFF) > SC_COORD_BIAS) ? (tl & 0x1FFF) - SC_COORD_BIAS : 0;
    pkt->scissor[1] = (((tl >> 13) & 0x1FFF) > SC_COORD_BIAS) ?
                      ((tl >> 13) & 0x1FFF) - SC_COORD_BIAS : 0;
    pkt->scissor[2] = ((br & 0x1FFF) >= SC_COORD_BIAS) ?
                      (br & 0x1FFF) - SC_COORD_BIAS + 1 : 0;
    pkt->scissor[3] = (((br >> 13) & 0x1FFF) >= SC_COORD_BIAS) ?
                      ((br >> 13) & 0x1FFF) - SC_COORD_BIAS + 1 : 0;
    pkt->rt_width = pkt->rt_pitch;
    pkt->rt_height = pkt->scissor[3];
    if (!pkt->rt_width || !pkt->rt_height) {
        *err = "empty colour buffer";
        free(order);
        return false;
    }

    set_uniforms(st, pkt);
    set_textures(st, pkt);
    set_depth(st, pkt);
    /* The setup unit culls polygons only (triangles, fans, strips, quads,
     * quad strips, polygons). */
    pkt->cull = ((1u << prim) & 0xE0F0u) ? r300_reg(st, SU_CULL_MODE) & 7 : 0;

    /* Vertex order */
    f.st = st;
    f.arr = arr;
    f.read = read;
    f.opaque = opaque;
    f.swap = r300_reg(st, VAP_CNTL_STATUS) & 3;
    f.vtx_size = r300_reg(st, VAP_VTX_SIZE) & 0x7F;
    f.immd = immd;
    f.immd_dw = immd_dw;

    /* Vertex program */
    for (int i = 0; i < 256; i++) {
        memcpy(consts[i], &st->pvs_mem[(R300_PVS_CONST_START + (cc & 0xFF) + i) * 4],
               sizeof(consts[i]));
        if (R300_PVS_CONST_START + (cc & 0xFF) + i + 1 >= R300_PVS_MEM_VECS) {
            break;
        }
    }
    prog.code = &st->pvs_mem[R300_PVS_CODE_START * 4];
    prog.first_inst = cntl0 & 0x3FF;
    prog.last_inst = (cntl0 >> 20) & 0x3FF;
    prog.consts = (const float (*)[4])consts;
    prog.max_const = (cc >> 16) & 0xFF;

    output_layout(st, &lay);
    rs_route(st, &route, &desc, &pkt->warn);

    /* Transform every vertex in draw order (cheap next to the GPU work). */
    R300Vertex *xv = calloc(n, sizeof(R300Vertex));
    float (*outs)[R300_PVS_NUM_OUTPUTS][4] = calloc(n, sizeof(*outs));
    for (uint32_t i = 0; i < n; i++) {
        float in[R300_PVS_NUM_INPUTS][4], out[R300_PVS_NUM_OUTPUTS][4];

        memset(out, 0, sizeof(out));
        if (!fetch_vertex(&f, order[i], in)) {
            *err = f.why ? f.why : "vertex data out of range";
            free(outs);
            free(xv);
            free(order);
            return false;
        }
        if (bypass) {
            memcpy(out, in, sizeof(out));
        } else if (r300_pvs_run(&prog, (const float (*)[4])in, out)) {
            pkt->warn |= R300_WARN_PVS;
        }
        viewport_xform(st, pkt, out[lay.pos], xv[i].pos);
        rs_values(st, &lay, &route, out, NULL, xv[i].v);
        memcpy(outs[i], out, sizeof(out));
    }
    pkt->warn |= f.warn;

    if (prim == 1) {
        /*
         * Points are screen-aligned squares GA_POINT_SIZE wide (width in
         * the high half, height in the low half, in sixths of a pixel)
         * with texture coordinates S0..S1 left to right and T0..T1 bottom
         * to top.  Apple's compositor presents the screen as one such
         * point covering the whole frame buffer.
         */
        uint32_t ps = r300_reg(st, GA_POINT_SIZE);
        float half_w = ((ps >> 16) & 0xFFFF) / 12.0f;
        float half_h = (ps & 0xFFFF) / 12.0f;
        float s0 = bits_to_float(r300_reg(st, GA_POINT_S0));
        float t0 = bits_to_float(r300_reg(st, GA_POINT_T0));
        float s1 = bits_to_float(r300_reg(st, GA_POINT_S1));
        float t1 = bits_to_float(r300_reg(st, GA_POINT_T1));
        float W = pkt->rt_width, H = pkt->rt_height;
        static const int corner[6] = { 0, 1, 2, 0, 2, 3 };

        pkt->verts = malloc(sizeof(R300Vertex) * n * 6);
        pkt->num_verts = 0;
        pkt->prim_class = 0;
        for (uint32_t i = 0; i < n; i++) {
            const float *p = xv[i].pos;
            float w = p[3] != 0.0f ? p[3] : 1.0f;
            float cx = (p[0] / w + 1.0f) * W * 0.5f;
            float cy = (1.0f - p[1] / w) * H * 0.5f;
            float z = p[2] / w;
            /* corners: top-left, top-right, bottom-right, bottom-left */
            float x[4] = { cx - half_w, cx + half_w, cx + half_w, cx - half_w };
            float y[4] = { cy - half_h, cy - half_h, cy + half_h, cy + half_h };
            float sc[4] = { s0, s1, s1, s0 };
            float tc[4] = { t1, t1, t0, t0 };

            for (int k = 0; k < 6; k++) {
                int c = corner[k];
                R300Vertex *o = &pkt->verts[pkt->num_verts++];
                float st2[2] = { sc[c], tc[c] };

                o->pos[0] = 2.0f * x[c] / W - 1.0f;
                o->pos[1] = 1.0f - 2.0f * y[c] / H;
                o->pos[2] = z;
                o->pos[3] = 1.0f;
                rs_values(st, &lay, &route, outs[i], st2, o->v);
            }
        }
        free(outs);
        free(xv);
        free(order);
        pkt->msl = r300_us_to_msl(st, &desc, err);
        if (!pkt->msl) {
            r300_draw_free(pkt);
            return false;
        }
        return true;
    }

    list = malloc(sizeof(uint32_t) * (n * 3 + 6));
    nidx = r300_assemble(prim, n, list, &pkt->prim_class);
    if (!nidx) {
        if (prim != 0) {
            pkt->warn |= R300_WARN_PRIM;
        }
        free(list);
        free(outs);
        free(xv);
        free(order);
        *err = "primitive type not supported";
        return false;
    }
    pkt->verts = malloc(sizeof(R300Vertex) * nidx);
    for (uint32_t i = 0; i < nidx; i++) {
        pkt->verts[i] = xv[list[i]];
    }
    pkt->num_verts = nidx;
    free(list);
    free(outs);
    free(xv);
    free(order);

    pkt->msl = r300_us_to_msl(st, &desc, err);
    if (!pkt->msl) {
        r300_draw_free(pkt);
        return false;
    }
    return true;
}

bool r300_draw_build(const R300State *st, const R300Arrays *arr,
                     uint32_t opcode, const uint32_t *d, uint32_t ndw,
                     R300ReadFn read, void *opaque,
                     R300DrawPacket *pkt, const char **err)
{
    const uint32_t *payload = d + 1;
    uint32_t payload_dw = ndw ? ndw - 1 : 0;
    uint32_t vf, n, *order;

    memset(pkt, 0, sizeof(*pkt));
    *err = NULL;
    if (ndw < 1) {
        *err = "empty draw packet";
        return false;
    }
    vf = d[0];
    n = vf >> 16;
    if (!n) {
        *err = "no vertices";
        return false;
    }
    order = malloc(sizeof(uint32_t) * n);
    if (opcode == 0x36) {                       /* DRAW_INDX_2, inline */
        bool i32 = (vf >> 11) & 1;
        for (uint32_t i = 0; i < n; i++) {
            order[i] = r300_index_at(payload, payload_dw, i32, i);
        }
    } else {                                    /* DRAW_VBUF_2, DRAW_IMMD_2 */
        for (uint32_t i = 0; i < n; i++) {
            order[i] = i;
        }
    }
    return draw_core(st, arr, vf, order, n, opcode == 0x35 ? payload : NULL,
                     opcode == 0x35 ? payload_dw : 0, read, opaque, pkt, err);
}

uint32_t r300_msaa_offset(uint32_t x, uint32_t y, uint32_t ns,
                          uint32_t pitch_px, uint32_t bpp, uint32_t sample)
{
    uint32_t sh = bpp == 2 ? 1 : 2;             /* log2(bytes per sample) */
    uint32_t pitch = pitch_px * ns;             /* in samples, as the driver keeps it */
    uint32_t base, idx;

    if ((ns != 2 && ns != 4) || sample >= ns) {
        return ~0u;
    }
    base = 32 * ((pitch / ns / 4) * ns * sh * 2 * (y >> 3) +
                 ((((x >> 2) << 1) | ((y >> 2) & 1)) * ns * sh));
    if (ns == 2) {
        idx = (((y >> 1) & 1) << 4) | (((x >> 1) & 1) << 3) | (sample << 2);
    } else {
        idx = (((((x >> 2) & 1) ^ ((y >> 1) & 1))) << 5) |
              (((x >> 1) & 1) << 4) | (sample << 2);
    }
    idx |= ((y & 1) << 1) | (x & 1);
    return base + (idx << sh);
}

uint32_t r300_index_at(const uint32_t *dw, uint32_t ndw, bool i32, uint32_t i)
{
    if (i32) {
        return i < ndw ? dw[i] : 0;
    }
    return i / 2 < ndw ? (dw[i / 2] >> (16 * (i & 1))) & 0xFFFF : 0;
}

bool r300_draw_build_indexed(const R300State *st, const R300Arrays *arr,
                             uint32_t vf, const R300Indices *idx,
                             R300ReadFn read, void *opaque,
                             R300DrawPacket *pkt, const char **err)
{
    uint32_t n = vf >> 16, *order, *sw;
    unsigned swap = r300_reg(st, VAP_CNTL_STATUS) & 3;
    bool i32 = (vf >> 11) & 1;

    memset(pkt, 0, sizeof(*pkt));
    *err = NULL;
    if (!n) {
        *err = "no vertices";
        return false;
    }
    if ((uint64_t)(i32 ? n : (n + 1) / 2) > idx->ndw) {
        *err = "index buffer shorter than the draw";
        return false;
    }
    sw = malloc(sizeof(uint32_t) * (idx->ndw ? idx->ndw : 1));
    for (uint32_t k = 0; k < idx->ndw; k++) {
        sw[k] = vc_swap(idx->dw[k], swap);
    }
    order = malloc(sizeof(uint32_t) * n);
    for (uint32_t i = 0; i < n; i++) {
        order[i] = r300_index_at(sw, idx->ndw, i32, i);
    }
    free(sw);
    return draw_core(st, arr, vf, order, n, NULL, 0, read, opaque, pkt, err);
}

void r300_draw_free(R300DrawPacket *pkt)
{
    free(pkt->msl);
    free(pkt->verts);
    pkt->msl = NULL;
    pkt->verts = NULL;
    for (int t = 0; t < R300_NUM_TEX_UNITS; t++) {
        free(pkt->tex[t].host_data);
        pkt->tex[t].host_data = NULL;
    }
}
