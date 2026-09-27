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
#define US_OUT_FMT_0        0x46A4
#define TX_FILTER0_0        0x4400
#define TX_FORMAT0_0        0x4480
#define TX_FORMAT1_0        0x44C0
#define TX_OFFSET_0         0x4540
#define TX_ENABLE           0x4104
#define RB3D_CCTL           0x4E00
#define RB3D_COLORPITCH0    0x4E38
#define VAP_CLIP_CNTL       0x221C

uint32_t r300_tex_raw_bpp(uint32_t txformat)
{
    switch (txformat & 0x1F) {
    case 0x02: case 0x05:                           /* Y4X4, Z3Y3X2 */
        return 1;
    case 0x12: case 0x14: case 0x15:                /* CxV8U8, VYUY, YVYU */
    case 0x16: case 0x18:                           /* 16 MPEG, 16F */
        return 2;
    case 0x04: case 0x08: case 0x09: case 0x0D:     /* Y16X16, 11/11/10, 2/10/10/10 */
    case 0x13: case 0x17: case 0x19: case 0x1B:     /* AVYU, 16_16 MPEG, 16F x2, 32F */
        return 4;
    case 0x0E: case 0x1A: case 0x1C:                /* W16Z16Y16X16, 16F x4, 32F x2 */
        return 8;
    case 0x1D:                                      /* 32F x4 */
        return 16;
    default:
        return 0;
    }
}

uint32_t r300_cb_view(uint32_t cf, uint32_t of, uint32_t *bpp)
{
    of &= 0x1F;
    switch (cf & 0xF) {
    case 9:                                         /* I8 */
        *bpp = 1;
        return R300_RTV_R8U;
    case 3: case 4: case 13: case 15:               /* 1555, 565, UV88, 4444 */
        *bpp = 2;
        return R300_RTV_R16U;
    case 6:                                         /* ARGB8888 */
        /* C4_10 (Tiger's desktop) still stores 8 bits a channel here:
         * the buffer format decides the layout, and 10-bit storage is
         * the R5xx ARGB2101010.  Other OUT_FMTs are Mesa's aliases
         * (RG16, R32F, RG16F) of a 32-bit pixel. */
        *bpp = 4;
        return of <= 2 ? R300_RTV_RGBA8 : R300_RTV_R32U;
    case 10:                                        /* ARGB16161616 */
        *bpp = 8;
        return R300_RTV_RG32U;
    case 7:                                         /* ARGB32323232 */
        *bpp = 16;
        return R300_RTV_RGBA32U;
    default:                                        /* reserved, YUV 4:2:2 */
        *bpp = 4;
        return R300_RTV_NONE;
    }
}

/* TX_FORMAT1.TXFORMAT 16F..32F x4: the card cannot filter them. */
static bool tex_is_float(uint32_t fmt)
{
    return fmt >= 0x18 && fmt <= 0x1D;
}

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

/* The r300_texu() arguments for a unit whose texels the shader decodes. */
static void us_texu_args(char *buf, size_t len, const R300State *st,
                         unsigned unit)
{
    uint32_t f1 = r300_reg(st, TX_FORMAT1_0 + 4 * unit);
    uint32_t f0 = r300_reg(st, TX_FILTER0_0 + 4 * unit);
    uint32_t fmt = f1 & 0x1F;
    /* filt: bit 0 magnify linear, bit 1 minify linear, S wrap 6:4,
     * T wrap 10:8, R wrap 14:12 (TX_FILTER0 CLAMP_S/T/R) */
    uint32_t filt = ((f0 & 7) << 4) | (((f0 >> 3) & 7) << 8) | (((f0 >> 6) & 7) << 12);

    if (!tex_is_float(fmt)) {
        filt |= (((f0 >> 9) & 3) != 1 ? 1 : 0) | (((f0 >> 11) & 3) != 1 ? 2 : 0);
    }
    snprintf(buf, len, "0x%xu, %uu, %uu, 0x%xu, %s", fmt, r300_tex_raw_bpp(fmt),
             r300_reg(st, TX_OFFSET_0 + 4 * unit) & 3, filt,
             (f1 >> 22) & 1 ? "true" : "false");
}

static bool unit_is_raw(const R300State *st, unsigned unit)
{
    return (r300_reg(st, TX_ENABLE) >> unit) & 1 &&
           r300_tex_raw_bpp(r300_reg(st, TX_FORMAT1_0 + 4 * unit));
}

/* TX_FORMAT1.TEX_COORD_TYPE of an enabled unit: 0 2D, 1 3D, 2 cube. */
static unsigned unit_dim(const R300State *st, unsigned unit)
{
    unsigned d = (r300_reg(st, TX_FORMAT1_0 + 4 * unit) >> 25) & 3;
    return (r300_reg(st, TX_ENABLE) >> unit) & 1 && d <= 2 ? d : 0;
}

/*
 * Border handling of a unit sampled through a float texture: per axis
 * 1 = clamp to border, 2 = mirror once to border (S bits 1:0, T 3:2,
 * R 5:4), bit 6 linear filtering.  The Metal sampler clamps to
 * transparent black and r300_tex adds TX_BORDER_COLOR for the missing
 * coverage (Metal has no arbitrary border colours).
 */
static unsigned unit_border(const R300State *st, unsigned unit)
{
    uint32_t f0 = r300_reg(st, TX_FILTER0_0 + 4 * unit);
    unsigned bm = 0, axes = unit_dim(st, unit) == 1 ? 3 : 2;

    if (unit_dim(st, unit) == 2) {
        return 0;                       /* cube maps never reach a border */
    }
    for (unsigned a = 0; a < axes; a++) {
        unsigned m = (f0 >> (3 * a)) & 7;
        bm |= (m == 6 ? 1u : m == 7 ? 2u : 0u) << (2 * a);
    }
    if (bm && (((f0 >> 9) & 3) != 1 || ((f0 >> 11) & 3) != 1)) {
        bm |= 64;
    }
    return bm;
}

static bool us_emit_tex(R300Sb *sb, const R300State *st, uint32_t inst,
                        uint32_t *units_used, const char **err)
{
    unsigned src = inst & 0x1F, dst = (inst >> 6) & 0x1F;
    unsigned unit = (inst >> 11) & 0xF, op = (inst >> 15) & 7;
    const char *proj, *bias;

    switch (op) {
    case 0:         /* NOP */
        return true;
    case 2:         /* KIL */
        r300_sb_printf(sb, "    if (any(t[%u] < 0.0)) discard_fragment();\n", src);
        return true;
    case 1:         /* LD */
        proj = "false"; bias = "0.0";
        break;
    case 3:         /* TXP */
        proj = "true"; bias = "0.0";
        break;
    case 4:         /* TXB: bias in the source's w */
        proj = "false"; bias = "0.0";
        break;
    default:
        *err = "unknown texture instruction";
        return false;
    }
    char bb[16];
    if (op == 4) {
        snprintf(bb, sizeof(bb), "t[%u].w", src);
        bias = bb;
    }
    if (unit_is_raw(st, unit)) {
        char a[128];
        us_texu_args(a, sizeof(a), st, unit);
        r300_sb_printf(sb, "    t[%u] = r300_texu(tex%u, u, %u, t[%u], %s, %s, %s);\n",
                       dst, unit, unit, src, proj, bias, a);
    } else {
        r300_sb_printf(sb, "    t[%u] = r300_tex(tex%u, smp%u, u, %u, t[%u], %s, %s, %uu);\n",
                       dst, unit, unit, unit, src, proj, bias, unit_border(st, unit));
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

    /* Writes: temporaries, the output FIFO of render target TARGET, and
     * (alpha OMASK_W) the fragment's depth. */
    us_mask(m, (ra >> 23) & 7);
    if (m[0]) {
        r300_sb_printf(sb, "        t[%u].%s = rr.%s;\n", (ra >> 18) & 0x1F, m, m);
    }
    us_mask(m, (ra >> 26) & 7);
    if (m[0]) {
        r300_sb_printf(sb, "        oc[%u].%s = rr.%s;\n", (ra >> 29) & 3, m, m);
    }
    if ((aa >> 23) & 1) {
        r300_sb_printf(sb, "        t[%u].a = ar;\n", (aa >> 18) & 0x1F);
    }
    if ((aa >> 24) & 1) {
        r300_sb_printf(sb, "        oc[%u].a = ar;\n", (aa >> 25) & 3);
    }
    if ((aa >> 27) & 1) {
        r300_sb_printf(sb, "        ow = ar;\n");
    }
    r300_sb_printf(sb, "    }\n");
}

/* Render targets the active program writes (bit n: target n), and
 * whether it writes depth (OMASK_W). */
static uint32_t us_targets_written(const R300State *st, bool *writes_w)
{
    USNode nodes[4];
    unsigned n = us_nodes(st, nodes);
    uint32_t mask = 0;

    *writes_w = false;
    for (unsigned i = 0; i < n; i++) {
        for (unsigned k = 0; k < nodes[i].alu_count; k++) {
            unsigned a = nodes[i].alu_start + k;
            uint32_t ra, aa;
            if (a >= R300_US_MAX_ALU) {
                break;
            }
            ra = r300_reg(st, US_ALU_RGB_ADDR_0 + 4 * a);
            aa = r300_reg(st, US_ALU_ALPHA_ADDR_0 + 4 * a);
            if ((ra >> 26) & 7) {
                mask |= 1u << ((ra >> 29) & 3);
            }
            if ((aa >> 24) & 1) {
                mask |= 1u << ((aa >> 25) & 3);
            }
            if ((aa >> 27) & 1) {
                *writes_w = true;
            }
        }
    }
    return mask;
}

uint32_t r300_us_out_fmt(const R300State *st, unsigned k)
{
    uint32_t f = r300_reg(st, US_OUT_FMT_0 + 4 * k);
    /* multiwrite targets without their own format use target A's */
    return k && (f & 0x1F) == 15 ? r300_reg(st, US_OUT_FMT_0) : f;
}

uint32_t r300_us_num_targets(const R300State *st)
{
    bool w;
    uint32_t written = us_targets_written(st, &w);
    uint32_t n = 1, mw = ((r300_reg(st, RB3D_CCTL) >> 5) & 3) + 1;

    for (unsigned k = 1; k < R300_US_MAX_TARGETS; k++) {
        if ((written >> k) & 1 && (r300_reg(st, US_OUT_FMT_0 + 4 * k) & 0x1F) != 15) {
            n = k + 1;
        }
    }
    return n > mw ? n : mw;
}

/* ---- library -------------------------------------------------------- */

static const char us_prelude[] =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct R300Vertex { float4 pos; float4 v[10]; float4 aux; float ucp[8]; };\n"
"struct R300VOut {\n"
"    float4 pos [[position]];\n"
"    float4 v0 [[user(v0)]]; float4 v1 [[user(v1)]]; float4 v2 [[user(v2)]];\n"
"    float4 v3 [[user(v3)]]; float4 v4 [[user(v4)]]; float4 v5 [[user(v5)]];\n"
"    float4 v6 [[user(v6)]]; float4 v7 [[user(v7)]]; float4 v8 [[user(v8)]];\n"
"    float4 v9 [[user(v9)]]; float4 aux [[user(aux)]];\n"
"};\n"
"struct R300FSUniforms {\n"
"    float4 consts[32];\n"
"    float4 blend_color;\n"
"    uint4 tex_swz[16];\n"
"    uint4 tex_info[16];\n"
"    uint cblend, ablend, chanmask, alpha_func;\n"
"    uint4 out_sel;\n"
"    uint rt_swap32; uint clip_rule; uint rt_endian, rop;\n"
"    int4 cliprect[4];\n"
"    uint4 zinfo;\n"
"    uint zpass_count, poly_en, pad4, pad5;\n"
"    float4 tex_border[16];\n"
"    float4 tex_lod[16];\n"
"    uint4 tex_dim[16];\n"
"    float4 fog_color;\n"
"    uint fog_blend, depth_src, pad6, pad7;\n"
"    float4 poly_offset;\n"
"};\n"
"\n"
"static float r300_swz1(float4 v, uint s)\n"
"{\n"
"    return s < 4 ? v[s] : (s == 5 ? 1.0 : 0.0);\n"
"}\n"
"static float4 r300_tswz(float4 hw, constant R300FSUniforms &u, uint unit)\n"
"{\n"
"    uint4 s = u.tex_swz[unit];\n"
"    return float4(r300_swz1(hw, s.x), r300_swz1(hw, s.y),\n"
"                  r300_swz1(hw, s.z), r300_swz1(hw, s.w));\n"
"}\n"
"\n"
"/* TX_FORMAT1.SIGNED_* (8-bit components as two's complement) and GAMMA\n"
"   (sRGB) for units sampled through a float texture. */\n"
"static float4 r300_tfix(float4 hw, constant R300FSUniforms &u, uint unit)\n"
"{\n"
"    uint f = u.tex_dim[unit].w;\n"
"    for (uint i = 0; i < 4; i++) {\n"
"        if ((f >> i) & 1u) {\n"
"            float b = rint(hw[i] * 255.0);\n"
"            hw[i] = max((b > 127.0 ? b - 256.0 : b) / 127.0, -1.0);\n"
"        }\n"
"    }\n"
"    if ((f & 16u) != 0u) {\n"
"        float3 c = saturate(hw.xyz);\n"
"        hw.xyz = select(pow((c + 0.055) / 1.055, 2.4), c / 12.92, c <= 0.04045);\n"
"    }\n"
"    return hw;\n"
"}\n"
"/* VRAM holds what the guest CPU wrote (big-endian words), so a 32-bit\n"
"   texel reads byte-reversed compared with the card's own view. */\n"
"static float4 r300_tpost(float4 raw, constant R300FSUniforms &u, uint unit)\n"
"{\n"
"    uint k = u.tex_info[unit].y;\n"
"    return r300_tfix(k == 1 ? raw.abgr : k == 2 ? raw.grba : raw, u, unit);\n"
"}\n"
"/* Share of a bilinear (or nearest) footprint inside [0, n) texels along\n"
"   one axis; m 2 = mirror once first. */\n"
"static float r300_bcov(float c, float n, uint m, bool lin)\n"
"{\n"
"    if (m == 0u) return 1.0;\n"
"    if (m == 2u) c = abs(c);\n"
"    if (!lin) { float i = floor(c * n); return i >= 0.0 && i < n ? 1.0 : 0.0; }\n"
"    float p = c * n - 0.5, i0 = floor(p), f = p - i0;\n"
"    return (i0 >= 0.0 && i0 < n ? 1.0 - f : 0.0) + (i0 + 1.0 >= 0.0 && i0 + 1.0 < n ? f : 0.0);\n"
"}\n"
"static float4 r300_border(float4 hw, float3 c, float3 n, uint bm,\n"
"                          constant R300FSUniforms &u, uint unit)\n"
"{\n"
"    bool lin = (bm & 64u) != 0u;\n"
"    float cov = r300_bcov(c.x, n.x, bm & 3u, lin) * r300_bcov(c.y, n.y, (bm >> 2) & 3u, lin) *\n"
"                r300_bcov(c.z, n.z, (bm >> 4) & 3u, lin);\n"
"    /* clamp to border sampled transparent black outside; mirror once\n"
"       sampled the edge there */\n"
"    return (bm & 42u) != 0u ? mix(u.tex_border[unit], hw, cov)\n"
"                            : hw + u.tex_border[unit] * (1.0 - cov);\n"
"}\n"
"/* Units sampled through a float texture: c is the source temporary\n"
"   (TXP divides by w), bias the TXB bias on top of TX_FILTER1.LOD_BIAS. */\n"
"static float4 r300_tex(texture2d<float> tx, sampler sm, constant R300FSUniforms &u,\n"
"                       uint unit, float4 c, bool proj, float bias, uint bm)\n"
"{\n"
"    if (u.tex_info[unit].x == 0) return float4(0.0, 0.0, 0.0, 1.0);\n"
"    float2 uv = proj ? c.xy / c.w : c.xy;\n"
"    float4 hw = r300_tpost(tx.sample(sm, uv, metal::bias(u.tex_lod[unit].x + bias)), u, unit);\n"
"    if (bm != 0u) {\n"
"        float l = floor(max(tx.calculate_clamped_lod(sm, uv), 0.0));\n"
"        float2 n = max(floor(float2(u.tex_info[unit].zw) / exp2(l)), float2(1.0));\n"
"        hw = r300_border(hw, float3(uv, 0.5), float3(n, 1.0), bm, u, unit);\n"
"    }\n"
"    return r300_tswz(hw, u, unit);\n"
"}\n"
"static float4 r300_tex(texture3d<float> tx, sampler sm, constant R300FSUniforms &u,\n"
"                       uint unit, float4 c, bool proj, float bias, uint bm)\n"
"{\n"
"    if (u.tex_info[unit].x == 0) return float4(0.0, 0.0, 0.0, 1.0);\n"
"    float3 uv = proj ? c.xyz / c.w : c.xyz;\n"
"    float4 hw = r300_tpost(tx.sample(sm, uv, metal::bias(u.tex_lod[unit].x + bias)), u, unit);\n"
"    if (bm != 0u) {\n"
"        float l = floor(max(tx.calculate_clamped_lod(sm, uv), 0.0));\n"
"        float3 n = max(floor(float3(float2(u.tex_info[unit].zw), float(u.tex_dim[unit].y)) / exp2(l)),\n"
"                       float3(1.0));\n"
"        hw = r300_border(hw, uv, n, bm, u, unit);\n"
"    }\n"
"    return r300_tswz(hw, u, unit);\n"
"}\n"
"static float4 r300_tex(texturecube<float> tx, sampler sm, constant R300FSUniforms &u,\n"
"                       uint unit, float4 c, bool proj, float bias, uint bm)\n"
"{\n"
"    if (u.tex_info[unit].x == 0) return float4(0.0, 0.0, 0.0, 1.0);\n"
"    float4 hw = r300_tpost(tx.sample(sm, c.xyz, metal::bias(u.tex_lod[unit].x + bias)), u, unit);\n"
"    return r300_tswz(hw, u, unit);\n"
"}\n"
"\n"
"/* Texels the shader decodes itself (r300_tex_raw_bpp): a uint view of\n"
"   guest memory, 4-byte elements for texels of up to 4 bytes.  The card's\n"
"   dword is the TXO_ENDIAN swap of the big-endian dword in VRAM. */\n"
"static uint r300_bswap32(uint v)\n"
"{\n"
"    return (v >> 24) | ((v >> 8) & 0xff00u) | ((v << 8) & 0xff0000u) | (v << 24);\n"
"}\n"
"static uint r300_cswap(uint v, uint e)\n"
"{\n"
"    switch (e & 3u) {\n"
"    case 1: return ((v & 0x00ff00ffu) << 8) | ((v >> 8) & 0x00ff00ffu);\n"
"    case 2: return r300_bswap32(v);\n"
"    case 3: return (v << 16) | (v >> 16);\n"
"    default: return v;\n"
"    }\n"
"}\n"
"static float r300_h16(uint v) { return float(as_type<half>(ushort(v & 0xffffu))); }\n"
"static float r300_u8(uint v, uint sh) { return float((v >> sh) & 0xffu) / 255.0; }\n"
"static float r300_s8(uint v) { return max(float(int(v << 24) >> 24) / 127.0, -1.0); }\n"
"static float r300_s16(uint v) { return max(float(int(v << 16) >> 16) / 32767.0, -1.0); }\n"
"static float r300_un(uint v, uint sh, uint bits)\n"
"{\n"
"    uint m = (1u << bits) - 1u;\n"
"    return float((v >> sh) & m) / float(m);\n"
"}\n"
"/* YUV_TO_RGB: BT.601 video range; the result is B, G, R in X, Y, Z (the\n"
"   raw texel has Cr, Y, Cb there, as Mesa's swizzles for UYVY imply). */\n"
"static float4 r300_yuv(float4 c)\n"
"{\n"
"    float y = 1.164 * (c.y - 16.0 / 255.0), cb = c.z - 128.0 / 255.0, cr = c.x - 128.0 / 255.0;\n"
"    return float4(saturate(y + 2.018 * cb), saturate(y - 0.813 * cr - 0.391 * cb),\n"
"                  saturate(y + 1.596 * cr), c.w);\n"
"}\n"
"static float4 r300_tdecode(uint fmt, uint4 w, uint odd, bool yuv)\n"
"{\n"
"    float4 c = float4(0.0, 0.0, 0.0, 1.0);\n"
"    uint v = w.x;\n"
"    switch (fmt) {\n"
"    case 0x02: c.x = r300_un(v, 0, 4); c.y = r300_un(v, 4, 4); break;\n"
"    case 0x04: c.x = r300_un(v, 0, 16); c.y = r300_un(v, 16, 16); break;\n"
"    case 0x05: c.x = r300_un(v, 0, 2); c.y = r300_un(v, 2, 3); c.z = r300_un(v, 5, 3); break;\n"
"    case 0x08: c.x = r300_un(v, 0, 10); c.y = r300_un(v, 10, 11); c.z = r300_un(v, 21, 11); break;\n"
"    case 0x09: c.x = r300_un(v, 0, 11); c.y = r300_un(v, 11, 11); c.z = r300_un(v, 22, 10); break;\n"
"    case 0x0D: c = float4(r300_un(v, 0, 10), r300_un(v, 10, 10), r300_un(v, 20, 10),\n"
"                          r300_un(v, 30, 2)); break;\n"
"    case 0x0E: c = float4(r300_un(w.x, 0, 16), r300_un(w.x, 16, 16),\n"
"                          r300_un(w.y, 0, 16), r300_un(w.y, 16, 16)); break;\n"
"    case 0x12: c.x = r300_s8(v); c.y = r300_s8(v >> 8);\n"
"               c.z = sqrt(saturate(1.0 - c.x * c.x - c.y * c.y)); break;\n"
"    case 0x13: c = float4(r300_u8(v, 16), r300_u8(v, 8), r300_u8(v, 0), r300_u8(v, 24)); break;\n"
"    /* 4:2:2, the dword of a pixel pair: VYUY = Y0 U Y1 V, YVYU = U Y0 V Y1\n"
"       from the low byte up (Mesa: YUYV and UYVY) */\n"
"    case 0x14: c = float4(r300_u8(v, 24), r300_u8(v, odd ? 16 : 0), r300_u8(v, 8), 1.0); break;\n"
"    case 0x15: c = float4(r300_u8(v, 16), r300_u8(v, odd ? 24 : 8), r300_u8(v, 0), 1.0); break;\n"
"    case 0x16: c.x = r300_s16(v); break;\n"
"    case 0x17: c.x = r300_s16(v); c.y = r300_s16(v >> 16); break;\n"
"    case 0x18: c.x = r300_h16(v); break;\n"
"    case 0x19: c.x = r300_h16(v); c.y = r300_h16(v >> 16); break;\n"
"    case 0x1A: c = float4(r300_h16(w.x), r300_h16(w.x >> 16), r300_h16(w.y), r300_h16(w.y >> 16)); break;\n"
"    case 0x1B: c.x = as_type<float>(v); break;\n"
"    case 0x1C: c.x = as_type<float>(w.x); c.y = as_type<float>(w.y); break;\n"
"    case 0x1D: c = as_type<float4>(w); break;\n"
"    default: break;\n"
"    }\n"
"    return yuv && (fmt == 0x13 || fmt == 0x14 || fmt == 0x15) ? r300_yuv(c) : c;\n"
"}\n"
"/* TX_FILTER0 CLAMP_*: wrap, mirror, clamp to last, mirror once to last,\n"
"   clamp to half-border (as last), mirror once to half-border (as\n"
"   last), clamp to border, mirror once to border. */\n"
"static int r300_wrap1(int p, int n, uint m, thread bool &border)\n"
"{\n"
"    switch (m & 7u) {\n"
"    case 0: return ((p % n) + n) % n;\n"
"    case 1: { int q = ((p % (2 * n)) + 2 * n) % (2 * n); return q < n ? q : 2 * n - 1 - q; }\n"
"    case 3: case 5: case 7: {\n"
"        int q = p < 0 ? -1 - p : p;\n"
"        if (q >= n && (m & 7u) == 7u) border = true;\n"
"        return min(q, n - 1);\n"
"    }\n"
"    case 6: if (p < 0 || p >= n) border = true; return clamp(p, 0, n - 1);\n"
"    default: return clamp(p, 0, n - 1);\n"
"    }\n"
"}\n"
"/* Level l of a unit's chain as the card lays it out (r300_tex_layout):\n"
"   byte offset, row pitch and face/slice size. */\n"
"static uint r300_lvl(constant R300FSUniforms &u, uint unit, uint l, uint bpp,\n"
"                     thread uint &pitch, thread uint &fbytes)\n"
"{\n"
"    uint4 dm = u.tex_dim[unit];\n"
"    uint off = 0u;\n"
"    for (uint i = 0u; i <= l; i++) {\n"
"        uint w = max(u.tex_info[unit].z >> i, 1u), h = max(u.tex_info[unit].w >> i, 1u);\n"
"        uint d = dm.x == 2u ? 6u : dm.x == 1u ? max(dm.y >> i, 1u) : 1u;\n"
"        if ((dm.w & 32u) != 0u && h > 1u) h = 1u << (32u - clz(h - 1u));\n"
"        uint p = i == 0u ? dm.z : (w * bpp + 31u) & ~31u;\n"
"        if (i == l) { pitch = p; fbytes = p * h; break; }\n"
"        off += p * h * d;\n"
"    }\n"
"    return off;\n"
"}\n"
"/* One texel: p = (x, y, slice or cube face), sz the level's size. */\n"
"static float4 r300_texel(texture2d<uint> tx, constant R300FSUniforms &u, uint unit,\n"
"                         uint fmt, uint bpp, uint e, uint filt, int3 p, int3 sz,\n"
"                         uint base, uint pitch, uint fbytes, bool yuv, bool cube)\n"
"{\n"
"    bool border = false;\n"
"    p.x = r300_wrap1(p.x, sz.x, cube ? 2u : filt >> 4, border);\n"
"    p.y = r300_wrap1(p.y, sz.y, cube ? 2u : filt >> 8, border);\n"
"    if (!cube) p.z = r300_wrap1(p.z, sz.z, filt >> 12, border);\n"
"    if (border) return u.tex_border[unit];\n"
"    uint rowel = max(u.tex_info[unit].y, 1u);\n"
"    uint row0 = base + uint(p.z) * fbytes + uint(p.y) * pitch;\n"
"    uint4 w;\n"
"    uint odd = 0u;\n"
"    if (bpp <= 4u) {\n"
"        uint x = uint(p.x);\n"
"        if (fmt == 0x14u || fmt == 0x15u) { odd = x & 1u; x &= ~1u; }\n"
"        uint off = row0 + x * bpp, el = off >> 2;\n"
"        uint d = r300_cswap(r300_bswap32(tx.read(uint2(el % rowel, el / rowel)).x), e);\n"
"        w = uint4(d >> ((off & 3u) * 8u), 0u, 0u, 0u);\n"
"    } else {\n"
"        uint el = (row0 + uint(p.x) * bpp) / bpp;\n"
"        uint4 r = tx.read(uint2(el % rowel, el / rowel));\n"
"        w = uint4(r300_cswap(r300_bswap32(r.x), e), r300_cswap(r300_bswap32(r.y), e),\n"
"                  r300_cswap(r300_bswap32(r.z), e), r300_cswap(r300_bswap32(r.w), e));\n"
"    }\n"
"    return r300_tdecode(fmt, w, odd, yuv);\n"
"}\n"
"/* Nearest or (bi/tri)linear lookup in mip level l; q in [0, 1] units. */\n"
"static float4 r300_texl(texture2d<uint> tx, constant R300FSUniforms &u, uint unit,\n"
"                        uint fmt, uint bpp, uint e, uint filt, float3 q, int face,\n"
"                        uint l, bool lin, bool yuv)\n"
"{\n"
"    uint pitch = 0u, fbytes = 0u;\n"
"    uint base = r300_lvl(u, unit, l, bpp, pitch, fbytes);\n"
"    uint dim = u.tex_dim[unit].x;\n"
"    bool cube = dim == 2u, vol = dim == 1u;\n"
"    int3 sz = int3(max(int(u.tex_info[unit].z >> l), 1), max(int(u.tex_info[unit].w >> l), 1),\n"
"                   vol ? max(int(u.tex_dim[unit].y >> l), 1) : 1);\n"
"    float3 st = q * float3(sz);\n"
"    if (!lin) {\n"
"        int3 p = int3(floor(st));\n"
"        if (cube) p.z = face; else if (!vol) p.z = 0;\n"
"        return r300_texel(tx, u, unit, fmt, bpp, e, filt, p, sz, base, pitch, fbytes, yuv, cube);\n"
"    }\n"
"    float3 f = st - 0.5;\n"
"    int3 p = int3(floor(f));\n"
"    float3 a = f - floor(f);\n"
"    if (cube) { p.z = face; a.z = 0.0; } else if (!vol) { p.z = 0; a.z = 0.0; }\n"
"    float4 r = float4(0.0);\n"
"    for (int k = 0; k < (vol ? 2 : 1); k++) {\n"
"        int3 b = p + int3(0, 0, k);\n"
"        float4 c00 = r300_texel(tx, u, unit, fmt, bpp, e, filt, b, sz, base, pitch, fbytes, yuv, cube);\n"
"        float4 c10 = r300_texel(tx, u, unit, fmt, bpp, e, filt, b + int3(1, 0, 0), sz, base, pitch, fbytes, yuv, cube);\n"
"        float4 c01 = r300_texel(tx, u, unit, fmt, bpp, e, filt, b + int3(0, 1, 0), sz, base, pitch, fbytes, yuv, cube);\n"
"        float4 c11 = r300_texel(tx, u, unit, fmt, bpp, e, filt, b + int3(1, 1, 0), sz, base, pitch, fbytes, yuv, cube);\n"
"        float4 m = mix(mix(c00, c10, a.x), mix(c01, c11, a.x), a.y);\n"
"        r += vol ? m * (k == 0 ? 1.0 - a.z : a.z) : m;\n"
"    }\n"
"    return r;\n"
"}\n"
"/* Cube map face and (s, t) of direction r, as GL and the card pick them\n"
"   (faces +X, -X, +Y, -Y, +Z, -Z). */\n"
"static float2 r300_cubeface(float3 r, thread int &face)\n"
"{\n"
"    float3 a = abs(r);\n"
"    float sc, tc, ma;\n"
"    if (a.x >= a.y && a.x >= a.z) {\n"
"        face = r.x >= 0.0 ? 0 : 1; ma = a.x; sc = r.x >= 0.0 ? -r.z : r.z; tc = -r.y;\n"
"    } else if (a.y >= a.z) {\n"
"        face = r.y >= 0.0 ? 2 : 3; ma = a.y; sc = r.x; tc = r.y >= 0.0 ? r.z : -r.z;\n"
"    } else {\n"
"        face = r.z >= 0.0 ? 4 : 5; ma = a.z; sc = r.z >= 0.0 ? r.x : -r.x; tc = -r.y;\n"
"    }\n"
"    ma = max(ma, 1e-20);\n"
"    return float2(sc / ma, tc / ma) * 0.5 + 0.5;\n"
"}\n"
"/* filt: bit 0 magnify linear, bit 1 minify linear, wrap S 6:4, T 10:8,\n"
"   R 14:12.  Mip level from the texel-space derivatives, TX_FILTER1\n"
"   LOD_BIAS, MAX_MIP_LEVEL..NUM_LEVELS and the mip filter. */\n"
"static float4 r300_texu(texture2d<uint> tx, constant R300FSUniforms &u, uint unit,\n"
"                        float4 c, bool proj, float bias, uint fmt, uint bpp, uint e,\n"
"                        uint filt, bool yuv)\n"
"{\n"
"    uint dim = u.tex_dim[unit].x;\n"
"    float3 q = proj && dim != 2u ? c.xyz / c.w : c.xyz;\n"
"    int face = 0;\n"
"    if (dim == 2u) q = float3(r300_cubeface(c.xyz, face), 0.0);\n"
"    float2 st = q.xy * float2(max(u.tex_info[unit].zw, uint2(1u)));\n"
"    float rho = max(length(dfdx(st)), length(dfdy(st)));\n"
"    if (u.tex_info[unit].x == 0) return float4(0.0, 0.0, 0.0, 1.0);\n"
"    float4 lo = u.tex_lod[unit];\n"
"    float lod = log2(max(rho, 1e-8)) + lo.x + bias;\n"
"    bool lin = (filt & (lod > 0.0 ? 2u : 1u)) != 0u;\n"
"    float4 hw;\n"
"    if (lo.w == 0.0) {\n"
"        hw = r300_texl(tx, u, unit, fmt, bpp, e, filt, q, face, uint(lo.y), lin, yuv);\n"
"    } else {\n"
"        float l = clamp(lod, lo.y, lo.z);\n"
"        if (lo.w == 1.0) {\n"
"            hw = r300_texl(tx, u, unit, fmt, bpp, e, filt, q, face, uint(rint(l)), lin, yuv);\n"
"        } else {\n"
"            uint l0 = uint(floor(l)), l1 = min(l0 + 1u, uint(lo.z));\n"
"            hw = mix(r300_texl(tx, u, unit, fmt, bpp, e, filt, q, face, l0, lin, yuv),\n"
"                     r300_texl(tx, u, unit, fmt, bpp, e, filt, q, face, l1, lin, yuv), fract(l));\n"
"        }\n"
"    }\n"
"    return r300_tswz(hw, u, unit);\n"
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
"   card writes channel Cn = sel[n] of the colour (0 A, 1 R, 2 G, 3 B;\n"
"   US_OUT_FMT C0..C3_SEL) to byte n of a little-endian word; with swap\n"
"   (COLOR_ENDIAN 2) the word is stored byte-reversed. */\n"
"static float r300_chan(float4 c, uint s)\n"
"{\n"
"    return s == 0 ? c.a : s == 1 ? c.r : s == 2 ? c.g : c.b;\n"
"}\n"
"static float4 r300_to_hw(float4 c, uint4 sel)\n"
"{\n"
"    return float4(r300_chan(c, sel.x), r300_chan(c, sel.y),\n"
"                  r300_chan(c, sel.z), r300_chan(c, sel.w));\n"
"}\n"
"static float4 r300_from_hw(float4 hw, uint4 sel)\n"
"{\n"
"    float4 c = float4(0.0);\n"
"    for (uint n = 0; n < 4; n++) {\n"
"        uint s = sel[n];\n"
"        if (s == 0) c.a = hw[n]; else if (s == 1) c.r = hw[n];\n"
"        else if (s == 2) c.g = hw[n]; else c.b = hw[n];\n"
"    }\n"
"    return c;\n"
"}\n"
"static float4 r300_pack(float4 c, uint4 sel, uint swap)\n"
"{\n"
"    float4 hw = r300_to_hw(c, sel);\n"
"    return swap != 0 ? hw.abgr : hw;\n"
"}\n"
"static float4 r300_unpack(float4 raw, uint4 sel, uint swap)\n"
"{\n"
"    return r300_from_hw(swap != 0 ? raw.abgr : raw, sel);\n"
"}\n"
"/* RB3D_ROPCNTL.ROP: a GDI ROP2 code as a truth table over (src, dst)\n"
"   bits: bit 3 both set, 2 src only, 1 dst only, 0 neither. */\n"
"static uint r300_rop1(uint s, uint d, uint c)\n"
"{\n"
"    return ((c & 8u) != 0u ? s & d : 0u) | ((c & 4u) != 0u ? s & ~d : 0u) |\n"
"           ((c & 2u) != 0u ? ~s & d : 0u) | ((c & 1u) != 0u ? ~s & ~d : 0u);\n"
"}\n"
"static uint r300_rop(uint s, uint d, uint c) { return r300_rop1(s, d, c); }\n"
"static uint2 r300_rop(uint2 s, uint2 d, uint c)\n"
"{\n"
"    return uint2(r300_rop1(s.x, d.x, c), r300_rop1(s.y, d.y, c));\n"
"}\n"
"static uint4 r300_rop(uint4 s, uint4 d, uint c)\n"
"{\n"
"    return uint4(r300_rop1(s.x, d.x, c), r300_rop1(s.y, d.y, c),\n"
"                 r300_rop1(s.z, d.z, c), r300_rop1(s.w, d.w, c));\n"
"}\n"
"static float4 r300_rop(float4 s, float4 d, uint c)\n"
"{\n"
"    uint4 a = uint4(rint(saturate(s) * 255.0)), b = uint4(rint(saturate(d) * 255.0));\n"
"    return float4(r300_rop(a, b, c) & 0xffu) / 255.0;\n"
"}\n"
"/* FG_FOG_BLEND.FN: linear, exp, exp2 of the interpolated fog value, or\n"
"   FG_FOG_FACTOR; the result weights the colour against the fog colour. */\n"
"static float r300_fogf(uint fb, float f, float k)\n"
"{\n"
"    switch ((fb >> 1) & 3u) {\n"
"    case 0: return saturate(f);\n"
"    case 1: return saturate(exp(-f));\n"
"    case 2: return saturate(exp(-f * f));\n"
"    default: return k;\n"
"    }\n"
"}\n"
"/* RB3D_CBLEND.DISCARD_SRC_PIXELS: skip blending (keep the destination)\n"
"   for source pixels whose alpha and/or colour is all 0 or all 1. */\n"
"static bool r300_discard_src(uint cb, float4 s)\n"
"{\n"
"    uint m = (cb >> 3) & 7u;\n"
"    bool a0 = s.a == 0.0, c0 = all(s.rgb == 0.0), a1 = s.a == 1.0, c1 = all(s.rgb == 1.0);\n"
"    switch (m) {\n"
"    case 1: return a0;        case 2: return c0;        case 3: return a0 && c0;\n"
"    case 4: return a1;        case 5: return c1;        case 6: return a1 && c1;\n"
"    default: return false;\n"
"    }\n"
"}\n"
"static float r300_upk(uint v, uint sh, uint bits, bool sgn)\n"
"{\n"
"    uint m = (1u << bits) - 1u, x = (v >> sh) & m;\n"
"    if (!sgn) return float(x) / float(m);\n"
"    int sx = int(x << (32u - bits)) >> (32u - bits);\n"
"    return max(float(sx) / float(m >> 1), -1.0);\n"
"}\n"
"static uint r300_pk(float f, uint sh, uint bits, bool sgn)\n"
"{\n"
"    uint m = (1u << bits) - 1u;\n"
"    if (!sgn) return uint(rint(saturate(f) * float(m))) << sh;\n"
"    return (uint(int(rint(clamp(f, -1.0, 1.0) * float(m >> 1)))) & m) << sh;\n"
"}\n"
"static uint r300_ph16(float f) { return uint(as_type<ushort>(half(f))); }\n"
"\n"
"";

static const char us_prelude_z[] =
"/* Depth/stencil (ZB_*).  The attachment after the colour buffers is the\n"
"   guest buffer as uint words\n"
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
"/* ZB_CNTL: 0 stencil, 1 Z test, 2 Z write, 4 separate back-face stencil.\n"
"   ZB_ZSTENCILCNTL: Z func 2:0; front stencil func/sfail/zpass/zfail at\n"
"   3, 6, 9, 12; back at 15, 18, 21, 24.  STENCILREFMASK: ref, mask,\n"
"   write mask.  Returns the new buffer word; pass says whether the\n"
"   colour buffers take the fragment. */\n"
"static uint r300_ztest(uint zb, float fz, bool front, constant R300FSUniforms &u,\n"
"                       device atomic_uint *zp, thread bool &pass)\n"
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
"    uint z = zold, sn = sold;\n"
"    pass = false;\n"
"    if (sten && !r300_zcmp(sf, ref & mask, sold & mask)) {\n"
"        sn = r300_sop(sf >> 3, sold, ref);\n"
"    } else if (!zen || r300_zcmp(zs, znew, zold)) {\n"
"        if (zen && (cntl & 4u) != 0u) z = znew;\n"
"        if (sten) sn = r300_sop(sf >> 6, sold, ref);\n"
"        if (u.zpass_count != 0u) atomic_fetch_add_explicit(zp, 1u, memory_order_relaxed);\n"
"        pass = true;\n"
"    } else {\n"
"        if (sten) sn = r300_sop(sf >> 9, sold, ref);\n"
"    }\n"
"    sn = (sold & ~wmask) | (sn & wmask);\n"
"    return r300_zswap(z16 ? z : ((z << 8) | sn), fmt & 3u, z16);\n"
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

/*
 * Colour buffer k's pixel type (r300_cbK_t) and its conversions to and
 * from the blender's colour (r300_cb_view).  Target A reads its channel
 * selects and swap from the uniforms; B-D have them folded into the text.
 */
static void us_emit_cb(R300Sb *sb, const R300State *st, unsigned k)
{
    uint32_t pitch = r300_reg(st, RB3D_COLORPITCH0 + 4 * k);
    uint32_t cf = (pitch >> 21) & 0xF, e = (pitch >> 19) & 3;
    uint32_t ofr = r300_us_out_fmt(st, k), of = ofr & 0x1F;
    uint32_t bpp, view = r300_cb_view(cf, of, &bpp);
    bool sg[4];
    const char *type;
    char sel[64], swap[16];
    /* The R300 blends and clamps only its 8-bit-per-channel formats. */
    bool fp = of >= 16 && of <= 21;
    bool blend = view == R300_RTV_RGBA8 || cf == 3 || cf == 4 || cf == 15 ||
                 cf == 9 || (cf == 13 && of == 0);

    if (k == 0) {
        snprintf(sel, sizeof(sel), "u.out_sel");
        snprintf(swap, sizeof(swap), "u.rt_swap32");
    } else {
        snprintf(sel, sizeof(sel), "uint4(%uu, %uu, %uu, %uu)", (ofr >> 8) & 3,
                 (ofr >> 10) & 3, (ofr >> 12) & 3, (ofr >> 14) & 3);
        snprintf(swap, sizeof(swap), "%uu", e == 2);
    }
    for (int n = 0; n < 4; n++) {
        sg[n] = (ofr >> (16 + n)) & 1;
    }
    if (view == R300_RTV_RGBA8 || view == R300_RTV_NONE) {
        r300_sb_printf(sb,
            "typedef float4 r300_cb%u_t;\n"
            "#define R300_CB%u_CLAMP 1\n#define R300_CB%u_BLEND 1\n"
            "static float4 r300_cb%u_unpack(float4 fb, constant R300FSUniforms &u) { return r300_unpack(fb, %s, %s); }\n"
            "static float4 r300_cb%u_pack(float4 c, constant R300FSUniforms &u) { return r300_pack(c, %s, %s); }\n\n",
            k, k, k, k, sel, swap, k, sel, swap);
        return;
    }
    type = view == R300_RTV_RG32U ? "uint2" : view == R300_RTV_RGBA32U ? "uint4" : "uint";
    r300_sb_printf(sb, "// colour buffer %u: format %u, US_OUT_FMT %08x, endian %u, %u bytes\n"
                   "typedef %s r300_cb%u_t;\n#define R300_CB%u_CLAMP %d\n#define R300_CB%u_BLEND %d\n",
                   k, cf, ofr, e, bpp, type, k, k, !fp, k, blend);

    /* The card's word(s): swaps are their own inverse. */
    char swb[200];
    if (bpp == 1) {
        snprintf(swb, sizeof(swb), "uint w = fb;");
    } else if (bpp == 2) {
        snprintf(swb, sizeof(swb), "%s", e == 1 || e == 2 ?
                 "uint w = ((fb >> 8) & 0xffu) | ((fb & 0xffu) << 8);" : "uint w = fb & 0xffffu;");
    } else if (bpp == 4) {
        snprintf(swb, sizeof(swb), "uint w = r300_cswap(fb, %uu);", e);
    } else if (bpp == 8) {
        snprintf(swb, sizeof(swb), "uint2 w = uint2(r300_cswap(fb.x, %uu), r300_cswap(fb.y, %uu));", e, e);
    } else {
        snprintf(swb, sizeof(swb), "uint4 w = uint4(r300_cswap(fb.x, %uu), r300_cswap(fb.y, %uu), "
                 "r300_cswap(fb.z, %uu), r300_cswap(fb.w, %uu));", e, e, e, e);
    }

    char ub[640], pb[640];
#define S(n) (sg[n] ? "true" : "false")
    switch (cf) {
    case 9:     /* I8 stores C2 (Mesa r300_translate_out_fmt) */
        snprintf(ub, sizeof(ub), "h.z = r300_upk(w, 0, 8, %s);", S(2));
        snprintf(pb, sizeof(pb), "uint w = r300_pk(h.z, 0, 8, %s);", S(2));
        break;
    case 4:     /* C_5_6_5 */
        snprintf(ub, sizeof(ub), "h = float4(r300_upk(w, 0, 5, false), r300_upk(w, 5, 6, false), r300_upk(w, 11, 5, false), 1.0);");
        snprintf(pb, sizeof(pb), "uint w = r300_pk(h.x, 0, 5, false) | r300_pk(h.y, 5, 6, false) | r300_pk(h.z, 11, 5, false);");
        break;
    case 3:     /* C_1_5_5_5 */
        snprintf(ub, sizeof(ub), "h = float4(r300_upk(w, 0, 5, false), r300_upk(w, 5, 5, false), r300_upk(w, 10, 5, false), r300_upk(w, 15, 1, false));");
        snprintf(pb, sizeof(pb), "uint w = r300_pk(h.x, 0, 5, false) | r300_pk(h.y, 5, 5, false) | r300_pk(h.z, 10, 5, false) | r300_pk(h.w, 15, 1, false);");
        break;
    case 15:    /* C4_4 */
        snprintf(ub, sizeof(ub), "h = float4(r300_upk(w, 0, 4, false), r300_upk(w, 4, 4, false), r300_upk(w, 8, 4, false), r300_upk(w, 12, 4, false));");
        snprintf(pb, sizeof(pb), "uint w = r300_pk(h.x, 0, 4, false) | r300_pk(h.y, 4, 4, false) | r300_pk(h.z, 8, 4, false) | r300_pk(h.w, 12, 4, false);");
        break;
    case 13:    /* UV88: C2 low byte, C0 high (Mesa); C_16 / C_16_FP: C0 */
        if (of == 3) {
            snprintf(ub, sizeof(ub), "h.x = r300_upk(w, 0, 16, %s);", S(0));
            snprintf(pb, sizeof(pb), "uint w = r300_pk(h.x, 0, 16, %s);", S(0));
        } else if (of == 16) {
            snprintf(ub, sizeof(ub), "h.x = r300_h16(w);");
            snprintf(pb, sizeof(pb), "uint w = r300_ph16(h.x);");
        } else {
            snprintf(ub, sizeof(ub), "h.z = r300_upk(w, 0, 8, %s); h.x = r300_upk(w, 8, 8, %s);", S(2), S(0));
            snprintf(pb, sizeof(pb), "uint w = r300_pk(h.z, 0, 8, %s) | r300_pk(h.x, 8, 8, %s);", S(2), S(0));
        }
        break;
    case 6:     /* 32-bit pixels other than C4_8 */
        switch (of) {
        case 3:
            snprintf(ub, sizeof(ub), "h.x = r300_upk(w, 0, 16, %s);", S(0));
            snprintf(pb, sizeof(pb), "uint w = r300_pk(h.x, 0, 16, %s);", S(0));
            break;
        case 16:
            snprintf(ub, sizeof(ub), "h.x = r300_h16(w);");
            snprintf(pb, sizeof(pb), "uint w = r300_ph16(h.x);");
            break;
        case 17:
            snprintf(ub, sizeof(ub), "h.x = r300_h16(w); h.y = r300_h16(w >> 16);");
            snprintf(pb, sizeof(pb), "uint w = r300_ph16(h.x) | (r300_ph16(h.y) << 16);");
            break;
        case 19:
            snprintf(ub, sizeof(ub), "h.x = as_type<float>(w);");
            snprintf(pb, sizeof(pb), "uint w = as_type<uint>(h.x);");
            break;
        default:    /* C2_16 */
            snprintf(ub, sizeof(ub), "h.x = r300_upk(w, 0, 16, %s); h.y = r300_upk(w, 16, 16, %s);", S(0), S(1));
            snprintf(pb, sizeof(pb), "uint w = r300_pk(h.x, 0, 16, %s) | r300_pk(h.y, 16, 16, %s);", S(0), S(1));
            break;
        }
        break;
    case 10:    /* 64-bit pixels */
        if (of == 18) {
            snprintf(ub, sizeof(ub), "h = float4(r300_h16(w.x), r300_h16(w.x >> 16), r300_h16(w.y), r300_h16(w.y >> 16));");
            snprintf(pb, sizeof(pb), "uint2 w = uint2(r300_ph16(h.x) | (r300_ph16(h.y) << 16), r300_ph16(h.z) | (r300_ph16(h.w) << 16));");
        } else if (of == 20) {
            snprintf(ub, sizeof(ub), "h.x = as_type<float>(w.x); h.y = as_type<float>(w.y);");
            snprintf(pb, sizeof(pb), "uint2 w = uint2(as_type<uint>(h.x), as_type<uint>(h.y));");
        } else {    /* C4_16 */
            snprintf(ub, sizeof(ub), "h = float4(r300_upk(w.x, 0, 16, %s), r300_upk(w.x, 16, 16, %s), r300_upk(w.y, 0, 16, %s), r300_upk(w.y, 16, 16, %s));",
                     S(0), S(1), S(2), S(3));
            snprintf(pb, sizeof(pb), "uint2 w = uint2(r300_pk(h.x, 0, 16, %s) | r300_pk(h.y, 16, 16, %s), r300_pk(h.z, 0, 16, %s) | r300_pk(h.w, 16, 16, %s));",
                     S(0), S(1), S(2), S(3));
        }
        break;
    default:    /* 7: 128-bit pixels, C4_32_FP (C2_32_FP fills two) */
        if (of == 20) {
            snprintf(ub, sizeof(ub), "h.x = as_type<float>(w.x); h.y = as_type<float>(w.y);");
            snprintf(pb, sizeof(pb), "uint4 w = uint4(as_type<uint>(h.x), as_type<uint>(h.y), 0u, 0u);");
        } else {
            snprintf(ub, sizeof(ub), "h = as_type<float4>(w);");
            snprintf(pb, sizeof(pb), "uint4 w = as_type<uint4>(h);");
        }
        break;
    }
#undef S

    /* Pack: the word(s), then the same swap back. */
    char back[200];
    if (bpp == 1) {
        snprintf(back, sizeof(back), "return w & 0xffu;");
    } else if (bpp == 2) {
        snprintf(back, sizeof(back), "%s", e == 1 || e == 2 ?
                 "return ((w >> 8) & 0xffu) | ((w & 0xffu) << 8);" : "return w & 0xffffu;");
    } else if (bpp == 4) {
        snprintf(back, sizeof(back), "return r300_cswap(w, %uu);", e);
    } else if (bpp == 8) {
        snprintf(back, sizeof(back), "return uint2(r300_cswap(w.x, %uu), r300_cswap(w.y, %uu));", e, e);
    } else {
        snprintf(back, sizeof(back), "return uint4(r300_cswap(w.x, %uu), r300_cswap(w.y, %uu), "
                 "r300_cswap(w.z, %uu), r300_cswap(w.w, %uu));", e, e, e, e);
    }
    r300_sb_printf(sb,
        "static float4 r300_cb%u_unpack(r300_cb%u_t fb, constant R300FSUniforms &u)\n"
        "{\n    %s\n    float4 h = float4(0.0, 0.0, 0.0, 1.0);\n    %s\n"
        "    return r300_from_hw(h, %s);\n}\n"
        "static r300_cb%u_t r300_cb%u_pack(float4 c, constant R300FSUniforms &u)\n"
        "{\n    float4 h = r300_to_hw(c, %s);\n    %s\n    %s\n}\n\n",
        k, k, swb, ub, sel, k, k, sel, pb, back);
}

/* Blend, ROP and write mask for colour buffer k from shader output src. */
static void us_emit_target(R300Sb *sb, unsigned k, int src)
{
    if (src < 0) {
        r300_sb_printf(sb, "    o.c%u = fb%u;      // target %u: not written\n", k, k, k);
        return;
    }
    r300_sb_printf(sb,
        "    {   // target %u\n"
        "        float4 s = oc[%d];\n"
        "        float4 d = r300_cb%u_unpack(fb%u, u);\n"
        "        float4 res = R300_CB%u_CLAMP ? saturate(s) : s;\n"
        "        if (R300_CB%u_BLEND && (u.cblend & 1u) != 0u) {\n"
        "            uint ab = (u.cblend & 2u) ? u.ablend : u.cblend;\n"
        "            uint cs = (u.cblend >> 16) & 63u, cd = (u.cblend >> 24) & 63u;\n"
        "            uint as_ = (ab >> 16) & 63u, ad = (ab >> 24) & 63u;\n"
        "            for (uint c = 0; c < 3; c++)\n"
        "                res[c] = r300_combine(u.cblend, s[c], r300_bfactor(cs, s, d, u.blend_color, c),\n"
        "                                      d[c], r300_bfactor(cd, s, d, u.blend_color, c));\n"
        "            res.a = r300_combine(ab, s.a, r300_bfactor(as_, s, d, u.blend_color, 3),\n"
        "                                 d.a, r300_bfactor(ad, s, d, u.blend_color, 3));\n"
        "            if (keep) res = d;\n"
        "        }\n"
        "        if ((u.rop & 4u) != 0u)\n"
        "            res = r300_cb%u_unpack(r300_rop(r300_cb%u_pack(R300_CB%u_CLAMP ? saturate(s) : s, u),\n"
        "                                            fb%u, (u.rop >> 8) & 15u), u);\n"
        "        /* RB3D_COLOR_CHANNEL_MASK: B G R A in bits 0..3 */\n"
        "        uint cm = u.chanmask;\n"
        "        res = float4((cm & 4u) ? res.r : d.r, (cm & 2u) ? res.g : d.g,\n"
        "                     (cm & 1u) ? res.b : d.b, (cm & 8u) ? res.a : d.a);\n"
        "        o.c%u = r300_cb%u_pack(res, u);\n"
        "    }\n",
        k, src, k, k, k, k, k, k, k, k, k, k);
}

char *r300_us_to_msl(const R300State *st, const R300FSDesc *desc,
                     const char **err)
{
    USNode nodes[4];
    unsigned n = us_nodes(st, nodes);
    uint32_t units = 0;
    R300Sb body, sb;
    bool writes_w;
    uint32_t written = us_targets_written(st, &writes_w);
    uint32_t nt = r300_us_num_targets(st);
    uint32_t mw = ((r300_reg(st, RB3D_CCTL) >> 5) & 3) + 1;
    uint32_t ucp = r300_reg(st, VAP_CLIP_CNTL) & 0x3F;

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
            if (!us_emit_tex(&body, st, r300_reg(st, US_TEX_INST_0 +
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
    for (unsigned k = 0; k < nt; k++) {
        us_emit_cb(&sb, st, k);
    }
    r300_sb_printf(&sb, "%s", us_prelude_z);

    /* Vertex stage: post-transform vertices as they are (shifted by the
     * sample offset when multisampling), plus the user clip plane
     * distances when VAP_CLIP_CNTL enables any. */
    r300_sb_printf(&sb,
        "struct R300VOutC {\n"
        "    float4 pos [[position]];\n"
        "    float4 v0 [[user(v0)]]; float4 v1 [[user(v1)]]; float4 v2 [[user(v2)]];\n"
        "    float4 v3 [[user(v3)]]; float4 v4 [[user(v4)]]; float4 v5 [[user(v5)]];\n"
        "    float4 v6 [[user(v6)]]; float4 v7 [[user(v7)]]; float4 v8 [[user(v8)]];\n"
        "    float4 v9 [[user(v9)]]; float4 aux [[user(aux)]];\n"
        "%s"
        "};\n"
        "vertex R300VOutC r300_vs(uint vid [[vertex_id]],\n"
        "                         const device R300Vertex *vb [[buffer(0)]],\n"
        "                         constant float4 &ms [[buffer(1)]])\n"
        "{\n"
        "    R300Vertex x = vb[vid];\n"
        "    R300VOutC o;\n"
        "    /* ms.xy: the multisample being drawn, as a clip-space shift */\n"
        "    o.pos = float4(x.pos.xy + ms.xy * x.pos.w, x.pos.zw);\n"
        "    o.v0 = x.v[0]; o.v1 = x.v[1]; o.v2 = x.v[2]; o.v3 = x.v[3];\n"
        "    o.v4 = x.v[4]; o.v5 = x.v[5]; o.v6 = x.v[6]; o.v7 = x.v[7];\n"
        "    o.v8 = x.v[8]; o.v9 = x.v[9]; o.aux = x.aux;\n"
        "%s"
        "    return o;\n"
        "}\n\n",
        ucp ? "    float clip [[clip_distance]] [6];\n" : "",
        ucp ? "    for (int i = 0; i < 6; i++) o.clip[i] = x.ucp[i];\n" : "");

    /* Fragment outputs: colour buffers 0..nt-1, then the depth buffer. */
    R300Sb outs, fbp, fbarg, fbin, zsel;
    r300_sb_init(&outs);
    r300_sb_init(&fbp);
    r300_sb_init(&fbarg);
    r300_sb_init(&fbin);
    r300_sb_init(&zsel);
    for (unsigned k = 0; k < nt; k++) {
        r300_sb_printf(&outs, "r300_cb%u_t c%u [[color(%u)]]; ", k, k, k);
        r300_sb_printf(&fbp, ", r300_cb%u_t fb%u", k, k);
        r300_sb_printf(&fbarg, ", fb%u", k);
        r300_sb_printf(&fbin, "                        r300_cb%u_t fb%u [[color(%u)]],\n", k, k, k);
        r300_sb_printf(&zsel, "    o.c%u = pass ? c.c%u : fb%u;\n", k, k, k);
    }
    r300_sb_printf(&sb, "struct R300FOut { %s};\nstruct R300FOutZ { %suint z [[color(%u)]]; };\n\n",
                   outs.buf, outs.buf, nt);

    /* Texture parameters of the shading function and the arguments that
     * pass them on (entry points bind unit k at texture/sampler k). */
    R300Sb tparm, targ, tent;
    r300_sb_init(&tparm);
    r300_sb_init(&targ);
    r300_sb_init(&tent);
    for (unsigned k = 0; k < R300_NUM_TEX_UNITS; k++) {
        if (units & (1u << k)) {
            static const char *dims[3] = { "texture2d", "texture3d", "texturecube" };
            const char *tt = unit_is_raw(st, k) ? "texture2d<uint>" : NULL;
            char tf[32];
            if (!tt) {
                snprintf(tf, sizeof(tf), "%s<float>", dims[unit_dim(st, k)]);
                tt = tf;
            }
            r300_sb_printf(&tparm, ", %s tex%u, sampler smp%u", tt, k, k);
            r300_sb_printf(&targ, ", tex%u, smp%u", k, k);
            r300_sb_printf(&tent,
                "                        %s tex%u [[texture(%u)]],\n"
                "                        sampler smp%u [[sampler(%u)]],\n",
                tt, k, k, k, k);
        }
    }
    r300_sb_printf(&sb,
        "#define R300_WRITES_W %d\n"
        "static R300FOut r300_shade(R300VOut in, constant R300FSUniforms &u%s,\n"
        "                           thread float &ow%s)\n"
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
        writes_w, fbp.buf ? fbp.buf : "", tparm.buf ? tparm.buf : "");
    for (unsigned k = 0; k < R300_US_NUM_TEMPS; k++) {
        if (desc->route[k] >= 0) {
            r300_sb_printf(&sb, "    t[%u] = vin[%d];\n", k, desc->route[k]);
        }
    }
    r300_sb_printf(&sb,
        "    float4 oc[4] = { float4(0.0, 0.0, 0.0, 1.0), float4(0.0, 0.0, 0.0, 1.0),\n"
        "                     float4(0.0, 0.0, 0.0, 1.0), float4(0.0, 0.0, 0.0, 1.0) };\n%s",
        body.buf ? body.buf : "");
    r300_sb_printf(&sb,
        "    /* FG: fog, then the alpha test on render target A. */\n"
        "    if ((u.fog_blend & 1u) != 0u) {\n"
        "        float f = r300_fogf(u.fog_blend, in.aux.x, u.fog_color.w);\n"
        "        for (int k = 0; k < 4; k++) oc[k].rgb = mix(u.fog_color.rgb, oc[k].rgb, f);\n"
        "    }\n"
        "    if (!r300_alpha_pass(u.alpha_func, saturate(oc[0].a))) discard_fragment();\n"
        "    bool keep = r300_discard_src(u.cblend, saturate(oc[0]));\n"
        "    R300FOut o;\n");
    for (unsigned k = 0; k < nt; k++) {
        int src = mw > 1 && k < mw ? 0 : ((written >> k) & 1) || k == 0 ? (int)k : -1;
        us_emit_target(&sb, k, src);
    }
    r300_sb_printf(&sb,
        "    return o;\n"
        "}\n"
        "\n"
        "fragment R300FOut r300_fs(R300VOut in [[stage_in]],\n"
        "                        constant R300FSUniforms &u [[buffer(0)]],\n"
        "                        device atomic_uint *zp [[buffer(1)]],\n"
        "%s%s"
        "                        bool front_ [[front_facing]])\n"
        "{\n"
        "    float ow = in.pos.z;\n"
        "    R300FOut o = r300_shade(in, u%s, ow%s);\n"
        "    if (u.zpass_count != 0u) atomic_fetch_add_explicit(zp, 1u, memory_order_relaxed);\n"
        "    return o;\n"
        "}\n"
        "\n"
        "fragment R300FOutZ r300_fs_z(R300VOut in [[stage_in]],\n"
        "                            constant R300FSUniforms &u [[buffer(0)]],\n"
        "                            device atomic_uint *zp [[buffer(1)]],\n"
        "%s%s"
        "                            uint zb [[color(%u)]],\n"
        "                            bool front [[front_facing]])\n"
        "{\n"
        "    /* SU_POLY_OFFSET_*: slope in depth per pixel */\n"
        "    float slope = max(abs(dfdx(in.pos.z)), abs(dfdy(in.pos.z)));\n"
        "    float ow = in.pos.z;\n"
        "    R300FOut c = r300_shade(in, u%s, ow%s);\n"
        "    /* FG_DEPTH_SRC: the program's OMASK_W output replaces the depth */\n"
        "    float fz = (u.depth_src != 0u && R300_WRITES_W) ? ow : in.pos.z;\n"
        "    if (front ? (u.poly_en & 1u) != 0u : (u.poly_en & 2u) != 0u)\n"
        "        fz += front ? u.poly_offset.x * slope + u.poly_offset.y\n"
        "                    : u.poly_offset.z * slope + u.poly_offset.w;\n"
        "    bool pass;\n"
        "    R300FOutZ o;\n"
        "    o.z = r300_ztest(zb, fz, front, u, zp, pass);\n"
        "%s"
        "    return o;\n"
        "}\n",
        tent.buf ? tent.buf : "", fbin.buf, fbarg.buf, targ.buf ? targ.buf : "",
        tent.buf ? tent.buf : "", fbin.buf, nt, fbarg.buf, targ.buf ? targ.buf : "",
        zsel.buf);
    r300_sb_free(&outs);
    r300_sb_free(&fbp);
    r300_sb_free(&fbarg);
    r300_sb_free(&fbin);
    r300_sb_free(&zsel);
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
