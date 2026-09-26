/*
 * R300-family draw assembly: turns a 3D_DRAW_* packet plus the register
 * state into a self-contained draw for the host renderer.
 *
 *   vertex fetch (VAP_PROG_STREAM_CNTL, immediate data or 3D_LOAD_VBPNTR
 *   arrays) -> vertex program (PVS, on the CPU) -> viewport transform
 *   (VAP_VTE_CNTL, SE_VPORT_*) -> primitive assembly into a triangle,
 *   line or point list -> rasterizer routing (RS_IP/RS_INST) of vertex
 *   outputs into fragment temporaries.
 *
 * The fragment program, blending and colour-buffer packing are compiled
 * from r300_us_to_msl(); the renderer only binds and draws.
 *
 * Pure C, no QEMU dependencies.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#ifndef HW_DISPLAY_R300_DRAW_H
#define HW_DISPLAY_R300_DRAW_H

#include <stdbool.h>
#include <stdint.h>

#include "r300_state.h"
#include "r300_us.h"

#define R300_MAX_ARRAYS 16

/* 3D_LOAD_VBPNTR state: one entry per vertex array. */
typedef struct R300Arrays {
    unsigned count;
    struct {
        uint32_t addr;          /* GPU address */
        uint32_t size_dw;       /* dwords per element */
        uint32_t stride_dw;     /* dwords between elements */
    } a[R300_MAX_ARRAYS];
} R300Arrays;

/* How the renderer gets a texture unit's texels. */
enum {
    R300_TEXK_RGBA8 = 0,        /* 32bpp view; shader reverses the bytes */
    R300_TEXK_R8,               /* X8 view */
    R300_TEXK_RG8,              /* Y8X8 view; shader swaps the bytes */
    R300_TEXK_CONVERT16,        /* 16bpp packed: decoded to RGBA8 on the CPU */
    R300_TEXK_DXT1,             /* compressed blocks, copied as they lie */
    R300_TEXK_DXT3,
    R300_TEXK_DXT5,
};

typedef struct R300TexDesc {
    bool bound;
    uint32_t gpu_addr;          /* TX_OFFSET, low bits cleared */
    uint32_t width, height;
    uint32_t pitch_bytes;       /* bytes per row (per row of blocks for DXT) */
    uint32_t format;            /* TX_FORMAT1 & 0x1F */
    uint32_t kind;              /* R300_TEXK_* */
    uint32_t filter0;           /* TX_FILTER0 */
    uint8_t *host_data;         /* texels copied out of the GART (owned), or NULL */
} R300TexDesc;

/* Depth/stencil buffer (ZB_*).  It stays in guest memory in the card's
 * layout; the renderer tests and writes it in the fragment shader. */
typedef struct R300DepthDesc {
    bool attach;                /* ZB_CNTL enables the Z or stencil test */
    uint32_t gpu_addr;          /* ZB_DEPTHOFFSET */
    uint32_t pitch;             /* pixels */
    uint32_t bpp;               /* 2 (Z16, Z13E3) or 4 (Z24S8) */
    uint32_t format;            /* ZB_FORMAT.DEPTHFORMAT */
} R300DepthDesc;

/* SU_CULL_MODE */
#define R300_CULL_FRONT         (1u << 0)
#define R300_CULL_BACK          (1u << 1)
#define R300_FACE_CW            (1u << 2)   /* else counter-clockwise front */

/*
 * The card's winding is the one seen on screen with row 0 at the top (as
 * Mesa's r300 maps gallium's front_ccw straight to FRONT_FACE_CCW), which
 * is also how Metal judges winding for the clip-space positions this
 * module produces (their y is up).  So FACE_CCW is MTLWindingCounterClockwise.
 */
static inline bool r300_front_ccw(uint32_t cull)
{
    return !(cull & R300_FACE_CW);
}

typedef struct R300DrawPacket {
    /* Colour buffer 0 */
    uint32_t rt_gpu_addr;
    uint32_t rt_pitch;          /* pixels */
    uint32_t rt_width, rt_height;
    uint32_t rt_format;         /* RB3D_COLORPITCH0 format field */
    uint32_t scissor[4];        /* x0, y0, x1, y1 (exclusive) */
    R300DepthDesc depth;
    uint32_t cull;              /* SU_CULL_MODE (0 for lines and points) */

    /* Fragment stage */
    char *msl;                  /* library source; owned by the packet */
    R300FSUniforms uniforms;
    R300TexDesc tex[R300_NUM_TEX_UNITS];

    /* Geometry, expanded (no indices) */
    R300Vertex *verts;          /* owned by the packet */
    uint32_t num_verts;
    uint32_t prim_class;        /* 0 triangles, 1 lines, 2 points */

    uint32_t warn;              /* R300_WARN_* bits for things skipped */
} R300DrawPacket;

#define R300_WARN_PVS           (1u << 0)
#define R300_WARN_DEPTH         (1u << 1)
#define R300_WARN_TEXFMT        (1u << 2)
#define R300_WARN_VTXFMT        (1u << 3)
#define R300_WARN_PRIM          (1u << 4)
#define R300_WARN_RTFMT         (1u << 5)
#define R300_WARN_VARYINGS      (1u << 6)

/* Reads guest GPU memory (VRAM or GART) for vertex arrays and indices.
 * Returns false if the range is not mapped. */
typedef bool (*R300ReadFn)(void *opaque, uint32_t gpu_addr, void *dst,
                           uint32_t len);

/* Index buffer from memory (INDX_BUFFER after a DRAW_INDX_2 with no
 * inline indices), already fetched; NULL for other draws. */
typedef struct R300Indices {
    const uint32_t *dw;         /* raw dwords as fetched (VC_SWAP applied here) */
    uint32_t ndw;
} R300Indices;

/* Decode a 3D_LOAD_VBPNTR (type-3 0x2F) body. */
void r300_load_vbpntr(R300Arrays *arr, const uint32_t *d, uint32_t ndw);

/*
 * Build a draw from a 3D_DRAW_VBUF_2 (0x34), 3D_DRAW_IMMD_2 (0x35) or
 * 3D_DRAW_INDX_2 (0x36) body.  Returns false (with *err) when the draw
 * cannot be assembled at all; smaller omissions are flagged in pkt->warn.
 * On success the caller owns pkt->msl and pkt->verts (r300_draw_free).
 */
bool r300_draw_build(const R300State *st, const R300Arrays *arr,
                     uint32_t opcode, const uint32_t *d, uint32_t ndw,
                     R300ReadFn read, void *opaque,
                     R300DrawPacket *pkt, const char **err);

/*
 * The same for DRAW_INDX_2 whose indices come from memory: vf is the
 * packet's VAP_VF_CNTL and idx the dwords INDX_BUFFER pointed at, read
 * as they lie (VC_SWAP is applied here, as the card's vertex cache does
 * for all fetched data).
 */
bool r300_draw_build_indexed(const R300State *st, const R300Arrays *arr,
                             uint32_t vf, const R300Indices *idx,
                             R300ReadFn read, void *opaque,
                             R300DrawPacket *pkt, const char **err);

/* Index i (0-based) of a DRAW_INDX_2 index stream: 16-bit indices are
 * packed two per dword, the first in the low half. */
uint32_t r300_index_at(const uint32_t *dw, uint32_t ndw, bool i32, uint32_t i);

/*
 * Byte offset of sample `sample` of pixel (x, y) in a multisampled R300
 * colour/depth buffer (ns = 2 or 4 samples) as Apple's ATIRadeon9700
 * computes it for CPU access (ATIR300Surface get_offset_of_sample_0):
 * 4x4-pixel micro tiles holding all samples, in 8-row bands, pitch_px
 * pixels wide, bpp 2 or 4.  Returns ~0u for other sample counts.
 */
uint32_t r300_msaa_offset(uint32_t x, uint32_t y, uint32_t ns,
                          uint32_t pitch_px, uint32_t bpp, uint32_t sample);

/* Assemble a primitive list (VAP_VF_CNTL.PRIM_TYPE) of n vertices into
 * triangle/line/point list indices; returns the index count (0 for an
 * unsupported type).  list needs room for 3n + 6 entries. */
uint32_t r300_assemble(unsigned prim, uint32_t n, uint32_t *list,
                       uint32_t *cls);

void r300_draw_free(R300DrawPacket *pkt);

#endif
