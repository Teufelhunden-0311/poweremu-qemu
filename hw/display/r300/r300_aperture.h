/*
 * R300 aperture byte swaps: the CPU's views of VRAM (ppc_mac_gpu.c).
 * Header only, so tests/r300/test_aperture.c checks the same code.
 *
 * A swap mode permutes each aligned group of four bytes: 0 none, 1 the
 * bytes within each 16 bits, 2 all four reversed, 3 the 16-bit halves
 * exchanged (the encoding TX_OFFSET.ENDIAN_SWAP documents; for an aperture
 * only 0 and 2 have been seen used, 1 and 3 are untested).  Each mode is
 * its own inverse.  A mode says nothing about the size of the pixels.
 *
 * VRAM here holds what aperture 0 shows with mode 2 (see r300_ap1_read):
 * the view through mode m is the card's bytes (mode 2 undone), then m.
 * That is right where the data is 32-bit.  Under the experimental
 * byte-order model 16-bit elements are stored big-endian each instead
 * (r300_endian_model), and for them this view is wrong: through mode 1 two
 * neighbouring 16-bit values come out exchanged (test_aperture.c shows it).
 */
#ifndef HW_DISPLAY_R300_APERTURE_H
#define HW_DISPLAY_R300_APERTURE_H

#include <stdint.h>
#include <string.h>

static inline void r300_permute(uint8_t b[4], unsigned mode)
{
    uint8_t t;
    switch (mode & 3) {
    case 1: t = b[0]; b[0] = b[1]; b[1] = t; t = b[2]; b[2] = b[3]; b[3] = t; break;
    case 2: t = b[0]; b[0] = b[3]; b[3] = t; t = b[1]; b[1] = b[2]; b[2] = t; break;
    case 3: t = b[0]; b[0] = b[2]; b[2] = t; t = b[1]; b[1] = b[3]; b[3] = t; break;
    default: break;
    }
}

/* The view through mode m of the aligned four bytes at vram + base. */
static inline void r300_ap_group(const uint8_t *vram, uint64_t base, unsigned m,
                                 uint8_t b[4])
{
    memcpy(b, vram + base, 4);
    r300_permute(b, 2);                 /* the card's bytes */
    r300_permute(b, m);                 /* through the swap */
}

/*
 * A big-endian CPU load of size bytes at off, any alignment, through mode
 * m.  The caller keeps every group touched, off & ~3 to (off + size - 1)
 * | 3, inside vram.
 */
static inline uint64_t r300_ap_load(const uint8_t *vram, uint64_t off,
                                    unsigned size, unsigned m)
{
    uint8_t b[4];
    uint64_t v = 0;

    for (unsigned i = 0; i < size; i++) {
        uint64_t o = off + i;
        if (i == 0 || !(o & 3)) {
            r300_ap_group(vram, o & ~3ull, m, b);
        }
        v = v << 8 | b[o & 3];
    }
    return v;
}

/* A big-endian CPU store, the same way; the other bytes of each group
 * touched are kept. */
static inline void r300_ap_store(uint8_t *vram, uint64_t off, unsigned size,
                                 uint64_t val, unsigned m)
{
    uint8_t b[4];

    for (unsigned i = 0; i < size;) {
        uint64_t base = (off + i) & ~3ull;
        r300_ap_group(vram, base, m, b);
        for (; i < size && ((off + i) & ~3ull) == base; i++) {
            b[(off + i) & 3] = val >> (8 * (size - 1 - i));
        }
        r300_permute(b, m);
        r300_permute(b, 2);
        memcpy(vram + base, b, 4);
    }
}

#endif
