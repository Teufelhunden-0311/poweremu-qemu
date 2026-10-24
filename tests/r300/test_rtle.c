/*
 * The record of card-order render targets (rt_note, r300_rt_forget,
 * rt_le_holds in r300_draw.c): its rules, and that it keeps every
 * surface no writer has touched, however many there are.
 */
#include <assert.h>
#include <stdio.h>
#include "../../hw/display/r300/r300_draw.c"

/* The same rules on an unsorted list: what the record must agree with. */
#define REF_MAX 20000
static struct { uint32_t lo, hi; } ref[REF_MAX];
static unsigned ref_n;

static void ref_forget(uint64_t lo, uint64_t hi)
{
    unsigned k = 0;
    for (unsigned i = 0; i < ref_n; i++) {
        if (!(ref[i].lo < hi && ref[i].hi > lo)) {
            ref[k++] = ref[i];
        }
    }
    ref_n = k;
}

static void ref_note(uint32_t lo, uint32_t hi, bool le)
{
    if (hi <= lo) {
        return;
    }
    for (unsigned i = 0; i < ref_n && le; i++) {
        if (ref[i].lo == lo && ref[i].hi == hi) {
            return;
        }
    }
    ref_forget(lo, hi);
    if (le) {
        assert(ref_n < REF_MAX);
        ref[ref_n].lo = lo;
        ref[ref_n].hi = hi;
        ref_n++;
    }
}

static bool ref_holds(uint32_t a)
{
    for (unsigned i = 0; i < ref_n; i++) {
        if (ref[i].lo <= a && a < ref[i].hi) {
            return true;
        }
    }
    return false;
}

static void check_invariant(void)
{
    assert(rt_le_n <= rt_le_cap);
    for (unsigned i = 0; i < rt_le_n; i++) {
        assert(rt_le[i].lo < rt_le[i].hi);
        assert(i == 0 || rt_le[i - 1].hi <= rt_le[i].lo);
    }
    assert(rt_le_n == ref_n);
}

static void both_note(uint32_t lo, uint32_t hi, bool le)
{
    rt_note(lo, hi, le);
    ref_note(lo, hi, le);
}

static void both_forget(uint64_t lo, uint64_t hi)
{
    r300_rt_forget(lo, hi);
    ref_forget(lo, hi);
}

static uint32_t rng = 12345;
static uint32_t rnd(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}

static void clear(void)
{
    both_forget(0, 1ull << 33);
    assert(rt_le_n == 0);
}

int main(void)
{
    /* The record runs only with the byte-order model off. */
    unsetenv("R300_RT_LE");
    setenv("R300_ENDIAN_MODEL", "0", 1);

    /* Endpoints are half-open. */
    both_note(0x1000, 0x2000, true);
    assert(rt_le_holds(0x1000) && rt_le_holds(0x1FFF));
    assert(!rt_le_holds(0x0FFF) && !rt_le_holds(0x2000));
    both_forget(0x2000, 0x3000);        /* touches the end: kept */
    both_forget(0x0800, 0x1000);        /* touches the start: kept */
    assert(rt_le_holds(0x1000));
    both_forget(0x1FFF, 0x2000);        /* one byte inside: the whole entry */
    assert(!rt_le_holds(0x1000) && rt_le_n == 0);

    /* Re-noting the same range keeps it; a swapped render of it removes it. */
    both_note(0x1000, 0x2000, true);
    both_note(0x1000, 0x2000, true);
    assert(rt_le_n == 1);
    both_note(0x1000, 0x2000, false);
    assert(rt_le_n == 0 && !rt_le_holds(0x1000));

    /* A note replaces every entry it overlaps, and only those. */
    both_note(0x1000, 0x2000, true);
    both_note(0x3000, 0x4000, true);
    both_note(0x5000, 0x6000, true);
    both_note(0x1800, 0x3800, true);
    assert(rt_le_n == 2);
    assert(!rt_le_holds(0x1000) && rt_le_holds(0x1800) && rt_le_holds(0x37FF));
    assert(!rt_le_holds(0x3800) && rt_le_holds(0x5000));
    both_note(0x0000, 0x8000, false);   /* swapped render over all of it */
    assert(rt_le_n == 0);

    /* Empty and inverted ranges are ignored. */
    both_note(0x1000, 0x1000, true);
    both_note(0x2000, 0x1000, true);
    assert(rt_le_n == 0);

    /* Forget ranges past 4 GiB (vram_mark takes 64-bit offsets). */
    both_note(0xFFFFF000u, 0xFFFFFFFFu, true);
    both_forget(0xFFFFFFF0u, 0x100000010ull);
    assert(rt_le_n == 0);

    /*
     * The capacity hold: an early surface survives far more disjoint
     * notes than the old fixed record held (16, then 4096).
     */
    const unsigned N = 10000;
    for (unsigned i = 0; i < N; i++) {
        both_note(0x100000 + i * 0x1000, 0x100000 + i * 0x1000 + 0x800, true);
    }
    check_invariant();
    assert(rt_le_n == N);
    for (unsigned i = 0; i < N; i++) {
        assert(rt_le_holds(0x100000 + i * 0x1000));
        assert(!rt_le_holds(0x100000 + i * 0x1000 + 0x800));
    }
    /* In reverse order too (every insert at the front). */
    clear();
    for (unsigned i = N; i-- > 0;) {
        both_note(0x100000 + i * 0x1000, 0x100000 + i * 0x1000 + 0x800, true);
    }
    check_invariant();
    assert(rt_le_holds(0x100000) && rt_le_holds(0x100000 + (N - 1) * 0x1000));
    clear();

    /* Random operations against the list, holding thousands at once. */
    unsigned ops = 0, peak = 0;
    for (int round = 0; round < 4; round++) {
        uint32_t span = round < 2 ? 1u << 28 : 1u << 24;
        for (int k = 0; k < 6000; k++, ops++) {
            uint32_t lo = rnd() % span & ~3u;
            uint32_t len = 4 + (rnd() % (round & 1 ? 0x400 : 0x10000) & ~3u);
            switch (rnd() % 8) {
            case 0:
                both_forget(lo, (uint64_t)lo + (rnd() % 0x100));
                break;
            case 1:
                both_note(lo, lo + len, false);
                break;
            default:
                both_note(lo, lo + len, true);
                break;
            }
            check_invariant();
            if (rt_le_n > peak) {
                peak = rt_le_n;
            }
            for (int q = 0; q < 4; q++) {
                uint32_t a = rnd() % span;
                assert(rt_le_holds(a) == ref_holds(a));
            }
            if (ref_n) {                /* the edges of a random entry */
                unsigned e = rnd() % ref_n;
                assert(rt_le_holds(ref[e].lo) && rt_le_holds(ref[e].hi - 1));
                assert(rt_le_holds(ref[e].hi) == ref_holds(ref[e].hi));
                assert(rt_le_holds(ref[e].lo - 1) == ref_holds(ref[e].lo - 1));
            }
        }
    }
    assert(peak > 4096);
    printf("rt_le: %u random operations, up to %u surfaces held: OK\n", ops, peak);
    return 0;
}
