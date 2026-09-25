/*
 * R300-family fragment shader unit (US) -> Metal Shading Language.
 *
 * Encodings follow AMD's "R3xx 3D Registers" reference (US_* registers).
 * The generated library contains a pass-through vertex function for
 * post-transform vertices and a fragment function that runs the R300
 * program, then does the alpha test, blending and colour-buffer packing in
 * the shader (the guest's big-endian byte order rules out fixed-function
 * blending on the host).
 *
 * Pure C, no QEMU dependencies.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#ifndef HW_DISPLAY_R300_US_H
#define HW_DISPLAY_R300_US_H

#include <stdbool.h>
#include <stdint.h>

#include "r300_state.h"

#define R300_US_NUM_TEMPS       32
#define R300_US_NUM_CONSTS      32
#define R300_US_MAX_ALU         64
#define R300_US_MAX_TEX         32
#define R300_NUM_TEX_UNITS      16
#define R300_NUM_VARYINGS       10

/*
 * Post-transform vertex as the device hands it to Metal: pos is in Metal
 * clip space, v[k] are the values the rasterizer interpolates into
 * fragment temporaries (see R300FSDesc.route).
 */
typedef struct R300Vertex {
    float pos[4];
    float v[R300_NUM_VARYINGS][4];
} R300Vertex;

/* Must match struct R300FSUniforms in the generated MSL. */
typedef struct R300FSUniforms {
    float consts[R300_US_NUM_CONSTS][4];
    float blend_color[4];
    uint32_t tex_swz[R300_NUM_TEX_UNITS][4];  /* R,G,B,A selects: 0-3 XYZW, 4 zero, 5 one */
    uint32_t tex_info[R300_NUM_TEX_UNITS][4]; /* x: 1 bound, y: raw->XYZW (0 as is, 1 .abgr, 2 .grba) */
    uint32_t cblend, ablend, chanmask, alpha_func;
    uint32_t out_sel[4];                      /* US_OUT_FMT_0 C0..C3 (0 A, 1 R, 2 G, 3 B) */
    uint32_t rt_swap32, clip_rule, pad[2];    /* stored bytes are C3,C2,C1,C0 */
    int32_t cliprect[4][4];                   /* SC_CLIPRECT x0, y0, x1, y1 (inclusive) */
    /* Depth/stencil: ZB_CNTL, ZB_ZSTENCILCNTL, ZB_STENCILREFMASK, and
     * R300_ZFMT_* (layout of the buffer bound as colour attachment 1). */
    uint32_t zinfo[4];
    uint32_t zpass_count, pad2[3];            /* count Z-pass samples (ZB_ZPASS_*) */
} R300FSUniforms;

#define R300_ZFMT_ENDIAN_MASK   3u            /* ZB_DEPTHPITCH.DEPTHENDIAN */
#define R300_ZFMT_Z16           (1u << 2)     /* 16-bit Z, no stencil */

/* What the translator needs besides the US registers themselves. */
typedef struct R300FSDesc {
    int8_t route[R300_US_NUM_TEMPS];    /* temp <- varying index, or -1 */
} R300FSDesc;

/*
 * Build the MSL library for the current US program: functions
 * "r300_vs", "r300_fs" (colour only) and "r300_fs_z" (colour plus the
 * depth/stencil buffer as a uint colour attachment 1; R32Uint for Z24S8,
 * R16Uint for Z16).  Both take a Z-pass counter at fragment buffer 1.  Returns a malloc'd string, or NULL with
 * *err set for programs not yet translated.  Every register the source
 * depends on is folded into the text, so the string is its own cache key.
 */
char *r300_us_to_msl(const R300State *st, const R300FSDesc *desc,
                     const char **err);

/* R300 US constants are 24-bit floats (1 sign, 7 exponent, 16 mantissa). */
float r300_float24(uint32_t v);

/* Disassemble the active US program, for traces.  Caller frees. */
char *r300_us_disasm(const R300State *st);

#endif
