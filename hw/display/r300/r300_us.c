/*
 * R300-family fragment shader unit (US) -> Metal Shading Language.
 *
 * Program layout (US_CONFIG, US_CODE_OFFSET, US_CODE_ADDR_0..3): up to four
 * nodes, stored so that the last node is always in CODE_ADDR_3.  Each node
 * runs a block of texture instructions and then a block of ALU
 * instructions; starts are relative to the offsets in US_CODE_OFFSET.
 *
 * ALU instructions are split into an RGB half and an alpha half, each with
 * three source addresses (temporaries 0-31, constants 32-63), an operand
 * select per argument, a pre-subtract, an op, an output modifier and a
 * clamp.  See AMD "R3xx 3D Registers", US_ALU_{RGB,ALPHA}_{ADDR,INST}.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#include "r300_us.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "r300_sb.h"

#define US_CONFIG           0x4600
#define US_CODE_OFFSET      0x4608
#define US_CODE_ADDR_0      0x4610
#define US_TEX_INST_0       0x4620
#define US_ALU_RGB_ADDR_0   0x46C0
#define US_ALU_ALPHA_ADDR_0 0x47C0
#define US_ALU_RGB_INST_0   0x48C0
#define US_ALU_ALPHA_INST_0 0x49C0

float r300_float24(uint32_t v)
{
    uint32_t mant = v & 0xFFFF;
    uint32_t exp = (v >> 16) & 0x7F;
    uint32_t sign = (v >> 23) & 1;
    float f;

    if (!exp && !mant) {
        return sign ? -0.0f : 0.0f;
    }
    /* Bias 63; treat the top exponent as a large finite value, not inf. */
    f = ldexpf(1.0f + mant / 65536.0f, (int)exp - 63);
    return sign ? -f : f;
}

typedef struct USNode {
    unsigned alu_start, alu_count;
    unsigned tex_start, tex_count;
} USNode;

/* Decode the node list; returns the number of nodes. */
static unsigned us_nodes(const R300State *st, USNode nodes[4])
{
    uint32_t config = r300_reg(st, US_CONFIG);
    uint32_t off = r300_reg(st, US_CODE_OFFSET);
    unsigned n = (config & 3) + 1;
    unsigned alu_off = off & 0x3F, tex_off = (off >> 13) & 0x1F;

    for (unsigned i = 0; i < n; i++) {
        uint32_t a = r300_reg(st, US_CODE_ADDR_0 + 4 * (4 - n + i));
        USNode *nd = &nodes[i];

        nd->alu_start = alu_off + (a & 0x3F);
        nd->alu_count = ((a >> 6) & 0x3F) + 1;
        nd->tex_start = tex_off + ((a >> 12) & 0x1F);
        nd->tex_count = ((a >> 17) & 0x1F) + 1;
        if (i == 0 && !(config & (1u << 3))) {
            nd->tex_count = 0;          /* FIRST_NODE_HAS_TEX clear */
        }
    }
    return n;
}

/* ---- source expressions --------------------------------------------- */

static void us_addr_expr(char *buf, size_t len, unsigned addr)
{
    if (addr & 32) {
        snprintf(buf, len, "u.consts[%u]", addr & 31);
    } else {
        snprintf(buf, len, "t[%u]", addr & 31);
    }
}

/* RGB operand select (US_ALU_RGB_INST.SEL_*) as a float3 expression. */
static void us_rgb_sel(char *buf, size_t len, unsigned sel)
{
    static const char *comp[] = { "rgb", "rrr", "ggg", "bbb" };

    if (sel < 12) {
        snprintf(buf, len, "cs%u.%s", sel / 4 == 0 ? 0 : sel / 4 == 1 ? 1 : 2,
                 comp[sel % 4]);
    } else if (sel < 15) {
        snprintf(buf, len, "float3(as%u.a)", sel - 12);
    } else if (sel < 19) {
        snprintf(buf, len, "ps.%s", comp[sel - 15]);
    } else if (sel == 19) {
        snprintf(buf, len, "float3(psa)");
    } else if (sel == 20) {
        snprintf(buf, len, "float3(0.0)");
    } else if (sel == 21) {
        snprintf(buf, len, "float3(1.0)");
    } else if (sel == 22) {
        snprintf(buf, len, "float3(0.5)");
    } else if (sel < 26) {
        snprintf(buf, len, "cs%u.gbr", sel - 23);
    } else if (sel < 29) {
        snprintf(buf, len, "cs%u.brg", sel - 26);
    } else {
        snprintf(buf, len, "float3(as%u.a, cs%u.b, cs%u.g)",
                 sel - 29, sel - 29, sel - 29);
    }
}

/* Alpha operand select (US_ALU_ALPHA_INST.SEL_*) as a float expression. */
static void us_alpha_sel(char *buf, size_t len, unsigned sel)
{
    static const char comp[] = "rgb";

    if (sel < 9) {
        snprintf(buf, len, "cs%u.%c", sel / 3, comp[sel % 3]);
    } else if (sel < 12) {
        snprintf(buf, len, "as%u.a", sel - 9);
    } else if (sel < 15) {
        snprintf(buf, len, "ps.%c", comp[sel - 12]);
    } else if (sel == 15) {
        snprintf(buf, len, "psa");
    } else if (sel == 16) {
        snprintf(buf, len, "0.0");
    } else if (sel == 17) {
        snprintf(buf, len, "1.0");
    } else {
        snprintf(buf, len, "0.5");
    }
}

static void us_mod(char *out, size_t len, const char *expr, unsigned mod)
{
    switch (mod) {
    case 1:  snprintf(out, len, "(-%s)", expr); break;
    case 2:  snprintf(out, len, "abs(%s)", expr); break;
    case 3:  snprintf(out, len, "(-abs(%s))", expr); break;
    default: snprintf(out, len, "(%s)", expr); break;
    }
}

static const char *us_omod(unsigned omod)
{
    static const char *m[] = { "", " * 2.0", " * 4.0", " * 8.0",
                               " * 0.5", " * 0.25", " * 0.125", "" };
    return m[omod & 7];
}

static const char *us_presub_rgb(unsigned op)
{
    static const char *p[] = { "1.0 - 2.0 * cs0.rgb", "cs1.rgb - cs0.rgb",
                               "cs1.rgb + cs0.rgb", "1.0 - cs0.rgb" };
    return p[op & 3];
}

static const char *us_presub_alpha(unsigned op)
{
    static const char *p[] = { "1.0 - 2.0 * as0.a", "as1.a - as0.a",
                               "as1.a + as0.a", "1.0 - as0.a" };
    return p[op & 3];
}

static void us_mask(char *buf, unsigned m)
{
    int n = 0;

    if (m & 1) buf[n++] = 'r';
    if (m & 2) buf[n++] = 'g';
    if (m & 4) buf[n++] = 'b';
    buf[n] = 0;
}

/* ---- instructions --------------------------------------------------- */

static bool us_emit_tex(R300Sb *sb, uint32_t inst, uint32_t *units_used,
                        const char **err)
{
    unsigned src = inst & 0x1F, dst = (inst >> 6) & 0x1F;
    unsigned unit = (inst >> 11) & 0xF, op = (inst >> 15) & 7;

    switch (op) {
    case 0:         /* NOP */
        return true;
    case 2:         /* KIL */
        r300_sb_printf(sb, "    if (any(t[%u] < 0.0)) discard_fragment();\n", src);
        return true;
    case 1:         /* LD */
        r300_sb_printf(sb, "    t[%u] = r300_tex(tex%u, smp%u, u, %u, t[%u].xy, 0.0);\n",
                       dst, unit, unit, unit, src);
        break;
    case 3:         /* TXP */
        r300_sb_printf(sb, "    t[%u] = r300_tex(tex%u, smp%u, u, %u, t[%u].xy / t[%u].w, 0.0);\n",
                       dst, unit, unit, unit, src, src);
        break;
    case 4:         /* TXB */
        r300_sb_printf(sb, "    t[%u] = r300_tex(tex%u, smp%u, u, %u, t[%u].xy, t[%u].w);\n",
                       dst, unit, unit, unit, src, src);
        break;
    default:
        *err = "unknown texture instruction";
        return false;
    }
    *units_used |= 1u << unit;
    return true;
}

static void us_emit_alu(R300Sb *sb, const R300State *st, unsigned i)
{
    uint32_t ra = r300_reg(st, US_ALU_RGB_ADDR_0 + 4 * i);
    uint32_t aa = r300_reg(st, US_ALU_ALPHA_ADDR_0 + 4 * i);
    uint32_t ri = r300_reg(st, US_ALU_RGB_INST_0 + 4 * i);
    uint32_t ai = r300_reg(st, US_ALU_ALPHA_INST_0 + 4 * i);
    char s[3][64], e[3][96], sel[96], m[4];
    unsigned rop = (ri >> 23) & 0xF, aop = (ai >> 23) & 0xF;

    r300_sb_printf(sb, "    { // alu %u: rgb %08x/%08x alpha %08x/%08x\n",
                   i, ra, ri, aa, ai);
    for (int k = 0; k < 3; k++) {
        us_addr_expr(s[k], sizeof(s[k]), (ra >> (6 * k)) & 0x3F);
        r300_sb_printf(sb, "        float4 cs%d = %s;", k, s[k]);
        us_addr_expr(s[k], sizeof(s[k]), (aa >> (6 * k)) & 0x3F);
        r300_sb_printf(sb, " float4 as%d = %s;\n", k, s[k]);
    }
    r300_sb_printf(sb, "        float3 ps = %s; float psa = %s;\n",
                   us_presub_rgb((ri >> 21) & 3), us_presub_alpha((ai >> 21) & 3));

    /* RGB arguments A, B, C */
    for (int k = 0; k < 3; k++) {
        us_rgb_sel(sel, sizeof(sel), (ri >> (7 * k)) & 0x1F);
        us_mod(e[k], sizeof(e[k]), sel, (ri >> (7 * k + 5)) & 3);
        r300_sb_printf(sb, "        float3 r%c = %s;\n", 'A' + k, e[k]);
    }
    for (int k = 0; k < 3; k++) {
        us_alpha_sel(sel, sizeof(sel), (ai >> (7 * k)) & 0x1F);
        us_mod(e[k], sizeof(e[k]), sel, (ai >> (7 * k + 5)) & 3);
        r300_sb_printf(sb, "        float a%c = %s;\n", 'A' + k, e[k]);
    }

    /* The dot product feeds both halves (alpha OP_DP takes it). */
    r300_sb_printf(sb, "        float dp = %s;\n",
                   rop == 2 ? "dot(rA, rB) + aA * aB" :
                   rop == 3 ? "rA.r * rB.r + rA.g * rB.g + rC.b" : "dot(rA, rB)");

    /* Alpha result first: OP_SOP hands it to the RGB half. */
    switch (aop) {
    case 0:  r300_sb_printf(sb, "        float ar = aA * aB + aC;\n"); break;
    case 1:  r300_sb_printf(sb, "        float ar = dp;\n"); break;
    case 2:  r300_sb_printf(sb, "        float ar = min(aA, aB);\n"); break;
    case 3:  r300_sb_printf(sb, "        float ar = max(aA, aB);\n"); break;
    case 5:  r300_sb_printf(sb, "        float ar = aC > 0.5 ? aA : aB;\n"); break;
    case 6:  r300_sb_printf(sb, "        float ar = aC >= 0.0 ? aA : aB;\n"); break;
    case 7:  r300_sb_printf(sb, "        float ar = fract(aA);\n"); break;
    case 8:  r300_sb_printf(sb, "        float ar = exp2(aA);\n"); break;
    case 9:  r300_sb_printf(sb, "        float ar = log2(aA);\n"); break;
    case 10: r300_sb_printf(sb, "        float ar = 1.0 / aA;\n"); break;
    case 11: r300_sb_printf(sb, "        float ar = rsqrt(aA);\n"); break;
    default: r300_sb_printf(sb, "        float ar = 0.0;\n"); break;
    }
    switch (rop) {
    case 0:  r300_sb_printf(sb, "        float3 rr = rA * rB + rC;\n"); break;
    case 1:
    case 2:
    case 3:  r300_sb_printf(sb, "        float3 rr = float3(dp);\n"); break;
    case 4:  r300_sb_printf(sb, "        float3 rr = min(rA, rB);\n"); break;
    case 5:  r300_sb_printf(sb, "        float3 rr = max(rA, rB);\n"); break;
    case 7:  r300_sb_printf(sb, "        float3 rr = select(rB, rA, rC > 0.5);\n"); break;
    case 8:  r300_sb_printf(sb, "        float3 rr = select(rB, rA, rC >= 0.0);\n"); break;
    case 9:  r300_sb_printf(sb, "        float3 rr = fract(rA);\n"); break;
    case 10: r300_sb_printf(sb, "        float3 rr = float3(ar);\n"); break;
    default: r300_sb_printf(sb, "        float3 rr = float3(0.0);\n"); break;
    }
    r300_sb_printf(sb, "        rr = rr%s; ar = ar%s;\n",
                   us_omod((ri >> 27) & 7), us_omod((ai >> 27) & 7));
    if ((ri >> 30) & 1) {
        r300_sb_printf(sb, "        rr = saturate(rr);\n");
    }
    if ((ai >> 30) & 1) {
        r300_sb_printf(sb, "        ar = saturate(ar);\n");
    }

    /* Writes: temporaries, then the output FIFO (render target A only). */
    us_mask(m, (ra >> 23) & 7);
    if (m[0]) {
        r300_sb_printf(sb, "        t[%u].%s = rr.%s;\n", (ra >> 18) & 0x1F, m, m);
    }
    us_mask(m, (ra >> 26) & 7);
    if (m[0] && !((ra >> 29) & 3)) {
        r300_sb_printf(sb, "        oc.%s = rr.%s;\n", m, m);
    }
    if ((aa >> 23) & 1) {
        r300_sb_printf(sb, "        t[%u].a = ar;\n", (aa >> 18) & 0x1F);
    }
    if ((aa >> 24) & 1 && !((aa >> 25) & 3)) {
        r300_sb_printf(sb, "        oc.a = ar;\n");
    }
    r300_sb_printf(sb, "    }\n");
}

/* ---- library -------------------------------------------------------- */

static const char us_prelude[] =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct R300Vertex { float4 pos; float4 v[10]; };\n"
"struct R300VOut {\n"
"    float4 pos [[position]];\n"
"    float4 v0 [[user(v0)]]; float4 v1 [[user(v1)]]; float4 v2 [[user(v2)]];\n"
"    float4 v3 [[user(v3)]]; float4 v4 [[user(v4)]]; float4 v5 [[user(v5)]];\n"
"    float4 v6 [[user(v6)]]; float4 v7 [[user(v7)]]; float4 v8 [[user(v8)]];\n"
"    float4 v9 [[user(v9)]];\n"
"};\n"
"struct R300FSUniforms {\n"
"    float4 consts[32];\n"
"    float4 blend_color;\n"
"    uint4 tex_swz[16];\n"
"    uint4 tex_info[16];\n"
"    uint cblend, ablend, chanmask, alpha_func;\n"
"    uint4 out_sel;\n"
"    uint rt_swap32; uint clip_rule; uint pad1, pad2;\n"
"    int4 cliprect[4];\n"
"    uint4 zinfo;\n"
"    uint zpass_count, pad3, pad4, pad5;\n"
"};\n"
"\n"
"vertex R300VOut r300_vs(uint vid [[vertex_id]],\n"
"                        const device R300Vertex *vb [[buffer(0)]])\n"
"{\n"
"    R300Vertex x = vb[vid];\n"
"    R300VOut o;\n"
"    o.pos = x.pos;\n"
"    o.v0 = x.v[0]; o.v1 = x.v[1]; o.v2 = x.v[2]; o.v3 = x.v[3];\n"
"    o.v4 = x.v[4]; o.v5 = x.v[5]; o.v6 = x.v[6]; o.v7 = x.v[7];\n"
"    o.v8 = x.v[8]; o.v9 = x.v[9];\n"
"    return o;\n"
"}\n"
"\n"
"static float r300_swz1(float4 v, uint s)\n"
"{\n"
"    return s < 4 ? v[s] : (s == 5 ? 1.0 : 0.0);\n"
"}\n"
"\n"
"/* Sample a texture unit and apply its TX_FORMAT1 swizzle.  VRAM holds\n"
"   what the guest CPU wrote (big-endian words), so a 32-bit texel reads\n"
"   byte-reversed compared with the card's own little-endian view. */\n"
"static float4 r300_tex(texture2d<float> tx, sampler sm,\n"
"                       constant R300FSUniforms &u, uint unit,\n"
"                       float2 uv, float bias)\n"
"{\n"
"    if (u.tex_info[unit].x == 0) return float4(0.0, 0.0, 0.0, 1.0);\n"
"    float4 raw = tx.sample(sm, uv, metal::bias(bias));\n"
"    uint k = u.tex_info[unit].y;\n"
"    float4 hw = k == 1 ? raw.abgr : k == 2 ? raw.grba : raw;\n"
"    uint4 s = u.tex_swz[unit];\n"
"    return float4(r300_swz1(hw, s.x), r300_swz1(hw, s.y),\n"
"                  r300_swz1(hw, s.z), r300_swz1(hw, s.w));\n"
"}\n"
"\n"
"static float r300_bfactor(uint f, float4 src, float4 dst, float4 k,\n"
"                         uint c)\n"
"{\n"
"    switch (f) {\n"
"    case 32: return 0.0;\n"
"    case 33: return 1.0;\n"
"    case 34: return src[c];\n"
"    case 35: return 1.0 - src[c];\n"
"    case 36: return dst[c];\n"
"    case 37: return 1.0 - dst[c];\n"
"    case 38: return src.a;\n"
"    case 39: return 1.0 - src.a;\n"
"    case 40: return dst.a;\n"
"    case 41: return 1.0 - dst.a;\n"
"    case 42: return c == 3 ? 1.0 : min(src.a, 1.0 - dst.a);\n"
"    case 43: return k[c];\n"
"    case 44: return 1.0 - k[c];\n"
"    case 45: return k.a;\n"
"    case 46: return 1.0 - k.a;\n"
"    default: return 1.0;\n"
"    }\n"
"}\n"
"\n"
"static float r300_combine(uint ctl, float s, float sf, float d, float df)\n"
"{\n"
"    float r;\n"
"    switch ((ctl >> 12) & 7) {\n"
"    case 0: return saturate(s * sf + d * df);\n"
"    case 1: return s * sf + d * df;\n"
"    case 2: return saturate(s * sf - d * df);\n"
"    case 3: return s * sf - d * df;\n"
"    case 4: return min(s, d);\n"
"    case 5: return max(s, d);\n"
"    case 6: return saturate(d * df - s * sf);\n"
"    default: r = d * df - s * sf; return r;\n"
"    }\n"
"}\n"
"\n"
"/* Colour (r,g,b,a) <-> the raw RGBA8 view of the colour buffer.  The\n"
"   card writes channel Cn = out_sel[n] of the colour (0 A, 1 R, 2 G, 3 B)\n"
"   to byte n of a little-endian word; with rt_swap32 the word is stored\n"
"   byte-reversed. */\n"
"static float r300_chan(float4 c, uint s)\n"
"{\n"
"    return s == 0 ? c.a : s == 1 ? c.r : s == 2 ? c.g : c.b;\n"
"}\n"
"static float4 r300_pack(float4 c, constant R300FSUniforms &u)\n"
"{\n"
"    float4 hw = float4(r300_chan(c, u.out_sel.x), r300_chan(c, u.out_sel.y),\n"
"                       r300_chan(c, u.out_sel.z), r300_chan(c, u.out_sel.w));\n"
"    return u.rt_swap32 != 0 ? hw.abgr : hw;\n"
"}\n"
"static float4 r300_unpack(float4 raw, constant R300FSUniforms &u)\n"
"{\n"
"    float4 hw = u.rt_swap32 != 0 ? raw.abgr : raw;\n"
"    float4 c = float4(0.0);\n"
"    for (uint n = 0; n < 4; n++) {\n"
"        uint s = u.out_sel[n];\n"
"        if (s == 0) c.a = hw[n]; else if (s == 1) c.r = hw[n];\n"
"        else if (s == 2) c.g = hw[n]; else c.b = hw[n];\n"
"    }\n"
"    return c;\n"
"}\n"
"\n"
"/* Depth/stencil (ZB_*).  Attachment 1 is the guest buffer as uint words\n"
"   as they lie in memory; DEPTHENDIAN says how the card swapped them.\n"
"   Z24S8 words hold Z in bits 31:8 and stencil in 7:0. */\n"
"static uint r300_zswap(uint v, uint e, bool z16)\n"
"{\n"
"    if (z16) return e == 1 || e == 2 ? ((v >> 8) & 0xffu) | ((v & 0xffu) << 8) : v;\n"
"    switch (e) {\n"
"    case 1: return ((v & 0x00ff00ffu) << 8) | ((v >> 8) & 0x00ff00ffu);\n"
"    case 2: return (v >> 24) | ((v >> 8) & 0xff00u) | ((v << 8) & 0xff0000u) | (v << 24);\n"
"    case 3: return (v << 16) | (v >> 16);\n"
"    default: return v;\n"
"    }\n"
"}\n"
"static bool r300_zcmp(uint f, uint a, uint b)\n"
"{\n"
"    switch (f & 7u) {\n"
"    case 0: return false;  case 1: return a < b;   case 2: return a <= b;\n"
"    case 3: return a == b; case 4: return a >= b;  case 5: return a > b;\n"
"    case 6: return a != b; default: return true;\n"
"    }\n"
"}\n"
"static uint r300_sop(uint op, uint s, uint ref)\n"
"{\n"
"    switch (op & 7u) {\n"
"    case 1: return 0u;                  case 2: return ref;\n"
"    case 3: return min(s + 1u, 255u);   case 4: return s > 0u ? s - 1u : 0u;\n"
"    case 5: return ~s & 0xffu;          case 6: return (s + 1u) & 0xffu;\n"
"    case 7: return (s - 1u) & 0xffu;    default: return s;\n"
"    }\n"
"}\n"
"struct R300ZOut { float4 c [[color(0)]]; uint z [[color(1)]]; };\n"
"/* ZB_CNTL: 0 stencil, 1 Z test, 2 Z write, 4 separate back-face stencil.\n"
"   ZB_ZSTENCILCNTL: Z func 2:0; front stencil func/sfail/zpass/zfail at\n"
"   3, 6, 9, 12; back at 15, 18, 21, 24.  STENCILREFMASK: ref, mask,\n"
"   write mask.  Failing fragments leave the colour buffer as it was. */\n"
"static R300ZOut r300_ztest(float4 col, float4 fb, uint zb, float fz, bool front,\n"
"                           constant R300FSUniforms &u, device atomic_uint *zp)\n"
"{\n"
"    uint cntl = u.zinfo.x, zs = u.zinfo.y, rm = u.zinfo.z, fmt = u.zinfo.w;\n"
"    bool z16 = (fmt & 4u) != 0u;\n"
"    uint w = r300_zswap(z16 ? (zb & 0xffffu) : zb, fmt & 3u, z16);\n"
"    uint zmax = z16 ? 0xffffu : 0xffffffu;\n"
"    uint zold = z16 ? w : (w >> 8), sold = z16 ? 0u : (w & 0xffu);\n"
"    /* rint + clamp: 1.0 * 16777215 + 0.5 rounds to 2^24 in float */\n"
"    uint znew = min(uint(rint(saturate(fz) * float(zmax))), zmax);\n"
"    bool sten = (cntl & 1u) != 0u, zen = (cntl & 2u) != 0u;\n"
"    uint sf = zs >> ((!front && (cntl & 16u) != 0u) ? 15u : 3u);\n"
"    uint ref = rm & 0xffu, mask = (rm >> 8) & 0xffu, wmask = (rm >> 16) & 0xffu;\n"
"    R300ZOut o; o.c = col;\n"
"    uint z = zold, sn = sold;\n"
"    if (sten && !r300_zcmp(sf, ref & mask, sold & mask)) {\n"
"        sn = r300_sop(sf >> 3, sold, ref); o.c = fb;\n"
"    } else if (!zen || r300_zcmp(zs, znew, zold)) {\n"
"        if (zen && (cntl & 4u) != 0u) z = znew;\n"
"        if (sten) sn = r300_sop(sf >> 6, sold, ref);\n"
"        if (u.zpass_count != 0u) atomic_fetch_add_explicit(zp, 1u, memory_order_relaxed);\n"
"    } else {\n"
"        if (sten) sn = r300_sop(sf >> 9, sold, ref);\n"
"        o.c = fb;\n"
"    }\n"
"    sn = (sold & ~wmask) | (sn & wmask);\n"
"    o.z = r300_zswap(z16 ? z : ((z << 8) | sn), fmt & 3u, z16);\n"
"    return o;\n"
"}\n"
"\n"
"static bool r300_alpha_pass(uint af, float a)\n"
"{\n"
"    if (!(af & (1u << 11))) return true;\n"
"    float ref = float(af & 0xff) / 255.0;\n"
"    switch ((af >> 8) & 7) {\n"
"    case 0: return false;\n"
"    case 1: return a < ref;\n"
"    case 2: return a == ref;\n"
"    case 3: return a <= ref;\n"
"    case 4: return a > ref;\n"
"    case 5: return a != ref;\n"
"    case 6: return a >= ref;\n"
"    default: return true;\n"
"    }\n"
"}\n"
"\n";

char *r300_us_to_msl(const R300State *st, const R300FSDesc *desc,
                     const char **err)
{
    USNode nodes[4];
    unsigned n = us_nodes(st, nodes);
    uint32_t units = 0;
    R300Sb body, sb;

    *err = NULL;
    r300_sb_init(&body);
    for (unsigned i = 0; i < n; i++) {
        const USNode *nd = &nodes[i];

        if (nd->alu_start + nd->alu_count > R300_US_MAX_ALU ||
            nd->tex_start + nd->tex_count > R300_US_MAX_TEX) {
            *err = "US program runs past instruction memory";
            r300_sb_free(&body);
            return NULL;
        }
        r300_sb_printf(&body, "    // node %u: tex %u+%u alu %u+%u\n", i,
                       nd->tex_start, nd->tex_count, nd->alu_start, nd->alu_count);
        for (unsigned k = 0; k < nd->tex_count; k++) {
            if (!us_emit_tex(&body, r300_reg(st, US_TEX_INST_0 +
                                                 4 * (nd->tex_start + k)),
                             &units, err)) {
                r300_sb_free(&body);
                return NULL;
            }
        }
        for (unsigned k = 0; k < nd->alu_count; k++) {
            us_emit_alu(&body, st, nd->alu_start + k);
        }
    }

    r300_sb_init(&sb);
    r300_sb_printf(&sb, "%s", us_prelude);

    /* Texture parameters of the shading function and the arguments that
     * pass them on (entry points bind unit k at texture/sampler k). */
    R300Sb tparm, targ, tent;
    r300_sb_init(&tparm);
    r300_sb_init(&targ);
    r300_sb_init(&tent);
    for (unsigned k = 0; k < R300_NUM_TEX_UNITS; k++) {
        if (units & (1u << k)) {
            r300_sb_printf(&tparm, ", texture2d<float> tex%u, sampler smp%u", k, k);
            r300_sb_printf(&targ, ", tex%u, smp%u", k, k);
            r300_sb_printf(&tent,
                "                        texture2d<float> tex%u [[texture(%u)]],\n"
                "                        sampler smp%u [[sampler(%u)]],\n",
                k, k, k, k);
        }
    }
    r300_sb_printf(&sb,
        "static float4 r300_shade(R300VOut in, constant R300FSUniforms &u,\n"
        "                         float4 fb%s)\n"
        "{\n"
        "    /* SC_CLIP_RULE: a 16-entry truth table over which of the four\n"
        "       clip rectangles contain the pixel. */\n"
        "    {\n"
        "        int2 p = int2(in.pos.xy);\n"
        "        uint idx = 0;\n"
        "        for (uint i = 0; i < 4; i++) {\n"
        "            int4 r = u.cliprect[i];\n"
        "            if (p.x >= r.x && p.y >= r.y && p.x <= r.z && p.y <= r.w) idx |= 1u << i;\n"
        "        }\n"
        "        if (!((u.clip_rule >> idx) & 1u)) discard_fragment();\n"
        "    }\n"
        "    float4 t[32];\n"
        "    for (int i = 0; i < 32; i++) t[i] = float4(0.0);\n"
        "    float4 vin[10] = { in.v0, in.v1, in.v2, in.v3, in.v4,\n"
        "                       in.v5, in.v6, in.v7, in.v8, in.v9 };\n",
        tparm.buf ? tparm.buf : "");
    for (unsigned k = 0; k < R300_US_NUM_TEMPS; k++) {
        if (desc->route[k] >= 0) {
            r300_sb_printf(&sb, "    t[%u] = vin[%d];\n", k, desc->route[k]);
        }
    }
    r300_sb_printf(&sb, "    float4 oc = float4(0.0, 0.0, 0.0, 1.0);\n%s", body.buf);
    r300_sb_printf(&sb,
        "    if (!r300_alpha_pass(u.alpha_func, saturate(oc.a))) discard_fragment();\n"
        "    float4 d = r300_unpack(fb, u);\n"
        "    float4 s = oc;\n"
        "    float4 res = saturate(s);\n"
        "    if (u.cblend & 1) {\n"
        "        uint ab = (u.cblend & 2) ? u.ablend : u.cblend;\n"
        "        uint cs = (u.cblend >> 16) & 63, cd = (u.cblend >> 24) & 63;\n"
        "        uint as_ = (ab >> 16) & 63, ad = (ab >> 24) & 63;\n"
        "        for (uint c = 0; c < 3; c++)\n"
        "            res[c] = r300_combine(u.cblend, s[c], r300_bfactor(cs, s, d, u.blend_color, c),\n"
        "                                  d[c], r300_bfactor(cd, s, d, u.blend_color, c));\n"
        "        res.a = r300_combine(ab, s.a, r300_bfactor(as_, s, d, u.blend_color, 3),\n"
        "                             d.a, r300_bfactor(ad, s, d, u.blend_color, 3));\n"
        "    }\n"
        "    /* RB3D_COLOR_CHANNEL_MASK: B G R A in bits 0..3 */\n"
        "    uint cm = u.chanmask;\n"
        "    res = float4((cm & 4) ? res.r : d.r, (cm & 2) ? res.g : d.g,\n"
        "                 (cm & 1) ? res.b : d.b, (cm & 8) ? res.a : d.a);\n"
        "    return r300_pack(res, u);\n"
        "}\n"
        "\n"
        "fragment float4 r300_fs(R300VOut in [[stage_in]],\n"
        "                        constant R300FSUniforms &u [[buffer(0)]],\n"
        "                        device atomic_uint *zp [[buffer(1)]],\n"
        "%s"
        "                        float4 fb [[color(0)]])\n"
        "{\n"
        "    float4 c = r300_shade(in, u, fb%s);\n"
        "    if (u.zpass_count != 0u) atomic_fetch_add_explicit(zp, 1u, memory_order_relaxed);\n"
        "    return c;\n"
        "}\n"
        "\n"
        "fragment R300ZOut r300_fs_z(R300VOut in [[stage_in]],\n"
        "                            constant R300FSUniforms &u [[buffer(0)]],\n"
        "                            device atomic_uint *zp [[buffer(1)]],\n"
        "%s"
        "                            float4 fb [[color(0)]], uint zb [[color(1)]],\n"
        "                            bool front [[front_facing]])\n"
        "{\n"
        "    float4 c = r300_shade(in, u, fb%s);\n"
        "    return r300_ztest(c, fb, zb, in.pos.z, front, u, zp);\n"
        "}\n",
        tent.buf ? tent.buf : "", targ.buf ? targ.buf : "",
        tent.buf ? tent.buf : "", targ.buf ? targ.buf : "");
    r300_sb_free(&tparm);
    r300_sb_free(&targ);
    r300_sb_free(&tent);
    r300_sb_free(&body);
    return r300_sb_steal(&sb);
}

char *r300_us_disasm(const R300State *st)
{
    USNode nodes[4];
    unsigned n = us_nodes(st, nodes);
    R300Sb sb;

    r300_sb_init(&sb);
    r300_sb_printf(&sb, "US config %08x offset %08x, %u node(s)\n",
                   r300_reg(st, US_CONFIG), r300_reg(st, US_CODE_OFFSET), n);
    for (unsigned i = 0; i < n; i++) {
        const USNode *nd = &nodes[i];
        r300_sb_printf(&sb, " node %u\n", i);
        for (unsigned k = 0; k < nd->tex_count; k++) {
            uint32_t t = r300_reg(st, US_TEX_INST_0 + 4 * (nd->tex_start + k));
            r300_sb_printf(&sb, "  tex%-2u op%u t%u <- unit%u t%u\n",
                           nd->tex_start + k, (t >> 15) & 7, (t >> 6) & 0x1F,
                           (t >> 11) & 0xF, t & 0x1F);
        }
        for (unsigned k = 0; k < nd->alu_count; k++) {
            unsigned a = nd->alu_start + k;
            r300_sb_printf(&sb, "  alu%-2u rgb %08x %08x  alpha %08x %08x\n", a,
                           r300_reg(st, US_ALU_RGB_ADDR_0 + 4 * a),
                           r300_reg(st, US_ALU_RGB_INST_0 + 4 * a),
                           r300_reg(st, US_ALU_ALPHA_ADDR_0 + 4 * a),
                           r300_reg(st, US_ALU_ALPHA_INST_0 + 4 * a));
        }
    }
    return r300_sb_steal(&sb);
}
