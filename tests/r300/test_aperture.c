/*
 * The aperture byte-swap views (hw/display/r300/r300_aperture.h) against
 * tables written out by hand: loads and stores of 1, 2, 4 and 8 bytes at
 * every alignment, the neighbouring bytes kept, and what the views do to
 * 32-bit and to 16-bit data.
 */
#include <stdio.h>
#include <stdlib.h>
#include "../../hw/display/r300/r300_aperture.h"

/*
 * view[m][i]: which byte of a VRAM group the CPU sees at position i
 * through mode m.  VRAM holds aperture 0's mode-2 view, so the card's
 * bytes are its four reversed (card[i] = vram[3 - i]); mode 0 shows them
 * as they are, mode 1 swaps the bytes of each half (card 1 0 3 2), mode 2
 * reverses all four (3 2 1 0), mode 3 exchanges the halves (2 3 0 1).
 */
static const int view[4][4] = {
    { 3, 2, 1, 0 },
    { 2, 3, 0, 1 },
    { 0, 1, 2, 3 },
    { 1, 0, 3, 2 },
};

#define N 24
static int fails;

static void check(int ok, const char *what, int m, int off, int size)
{
    if (!ok && fails++ < 20) {
        printf("FAIL %s: mode %d offset %d size %d\n", what, m, off, size);
    }
}

/* The CPU's byte at off through mode m, from the table. */
static uint8_t cpu_byte(const uint8_t *vram, int off, int m)
{
    return vram[(off & ~3) + view[m][off & 3]];
}

/* A mapping, as ppc_mac_gpu.c's surfaces: one range converted with mode 0,
 * the rest straight (mode 2).  ctx: the mapping now and after a catch-up. */
struct mock_range { int lo, hi, on; };

static void mock_catch_up(void *ctx)
{
    void **mp = ctx;
    *(struct mock_range *)mp[0] = *(struct mock_range *)mp[1];
}

static unsigned mock_mode_at(void *ctx, uint64_t off)
{
    struct mock_range *r = ((void **)ctx)[0];
    return r->on && off >= (uint64_t)r->lo && off <= (uint64_t)r->hi ? 0 : 2;
}

int main(void)
{
    uint8_t vram[N], ref[N];
    unsigned seed = 1;
    int loads = 0, stores = 0;

    /* every load: size 1, 2, 4, 8 at every offset, every mode */
    for (int i = 0; i < N; i++) {
        vram[i] = 0x10 + i * 7;             /* no two bytes alike */
    }
    for (int m = 0; m < 4; m++) {
        for (int size = 1; size <= 8; size *= 2) {
            for (int off = 0; off + size <= N; off++) {
                uint64_t want = 0;
                for (int i = 0; i < size; i++) {
                    want = want << 8 | cpu_byte(vram, off + i, m);
                }
                check(r300_ap_load(vram, off, size, m) == want, "load", m, off, size);
                loads++;
            }
        }
    }

    /*
     * Stores: random sizes and offsets, each made through the view and
     * into a copy by the table; after each the two must hold the same
     * bytes, so a store changes its own bytes and no neighbour's.
     */
    for (int m = 0; m < 4; m++) {
        for (int i = 0; i < N; i++) {
            vram[i] = ref[i] = 0x80 + i * 5;
        }
        for (int k = 0; k < 4000; k++) {
            int size = 1 << ((seed = seed * 1103515245 + 12345) >> 16) % 4;
            int off = ((seed = seed * 1103515245 + 12345) >> 16) % (N - size + 1);
            uint64_t val = (seed = seed * 1103515245 + 12345);
            val = val << 32 ^ (seed = seed * 1103515245 + 12345);
            r300_ap_store(vram, off, size, val, m);
            for (int i = 0; i < size; i++) {
                int o = off + i;
                ref[(o & ~3) + view[m][o & 3]] = val >> (8 * (size - 1 - i));
            }
            check(!memcmp(vram, ref, N), "store", m, off, size);
            check(r300_ap_load(vram, off, size, m) == (size == 8 ? val :
                  val & ((1ull << (8 * size)) - 1)), "store, load back", m, off, size);
            stores++;
        }
    }

    /*
     * 32-bit data, as the byte-order model stores it: the CPU's big-endian
     * value.  Through swap 2 it reads as stored, through swap 0 as the
     * card's little-endian bytes (rbprobe's glGetTexImage after
     * glCopyTexImage2D: 0x01020007 had come back 0x07000201).
     */
    {
        uint8_t t[8] = { 0x01, 0x02, 0x00, 0x07, 0xA1, 0xB2, 0xC3, 0xD4 };
        check(r300_ap_load(t, 0, 4, 2) == 0x01020007, "32-bit, swap 2", 2, 0, 4);
        check(r300_ap_load(t, 0, 4, 0) == 0x07000201, "32-bit, swap 0", 0, 0, 4);
        check(r300_ap_load(t, 4, 4, 0) == 0xD4C3B2A1, "32-bit, swap 0", 0, 4, 4);
    }

    /*
     * 16-bit data as the model stores it, two different neighbouring 1555
     * values, 0x1122 then 0x3344, big-endian each: 11 22 33 44.  The card's
     * bytes would be 22 11 44 33, and a mode-1 view of those is 11 22 33 44
     * again; this view instead gives 33 44 11 22, the two exchanged.  The
     * conversion does not hold for 16-bit data stored that way: shown here
     * so that it is not taken to.
     */
    {
        uint8_t t[4] = { 0x11, 0x22, 0x33, 0x44 };
        check(r300_ap_load(t, 0, 4, 1) == 0x33441122, "16-bit pair, swap 1 exchanges",
              1, 0, 4);
        check(r300_ap_load(t, 0, 2, 1) == 0x3344, "16-bit pair, swap 1 exchanges", 1, 0, 2);
    }

    /*
     * A mapping changed by the catch-up an access starts with (ring
     * commands carried out before it): the access goes through the mapping
     * as it is after, at the CPU's own offset, group by group across a
     * boundary.  A swap-0 surface over 16..31: kept, turned to swap 2,
     * moved to 24..39, removed; reads and writes at 16 and across 24.
     */
    {
        enum { M = 48 };
        uint8_t mv[M], mr[M];
        struct mock_range now, after;
        static const struct { int lo, hi, on; const char *what; } to[] = {
            { 16, 31, 1, "kept" },
            { 16, 31, 0, "turned to swap 2" },
            { 24, 39, 1, "moved" },
            { 0, 0, 0, "removed" },
        };
        void *mp[2] = { &now, &after };
        R300ApMap map = {
            .vram = mv,
            .catch_up = mock_catch_up,
            .mode_at = mock_mode_at,
            .ctx = mp,
        };
        int remaps = 0;

        for (int t = 0; t < 4; t++) {
            for (int wr = 0; wr < 2; wr++) {
                for (int off = 16; off <= 22; off += 6) {
                    uint64_t val = 0xC1C2C3C4, got = 0;

                    for (int i = 0; i < M; i++) {
                        mv[i] = mr[i] = 0x40 + i * 3;
                    }
                    now.lo = 16, now.hi = 31, now.on = 1;
                    after.lo = to[t].lo, after.hi = to[t].hi, after.on = to[t].on;
                    if (wr) {
                        r300_ap_write_at(&map, off, 4, val);
                    } else {
                        got = r300_ap_read_at(&map, off, 4);
                    }
                    for (int i = 0; i < 4; i++) {
                        int o = off + i;
                        int m = after.on && o >= after.lo && o <= after.hi ? 0 : 2;
                        if (wr) {
                            mr[(o & ~3) + view[m][o & 3]] = val >> (8 * (3 - i));
                        } else {
                            check(cpu_byte(mr, o, m) == (uint8_t)(got >> (8 * (3 - i))),
                                  to[t].what, m, off, 4);
                        }
                    }
                    if (wr) {
                        check(!memcmp(mv, mr, M), to[t].what, -1, off, 4);
                    }
                    remaps++;
                }
            }
        }
        printf("aperture views: %d accesses with the mapping changed under them\n",
               remaps);
    }

    printf("aperture views: %d loads, %d stores, %d failed\n", loads, stores, fails);
    return fails != 0;
}
