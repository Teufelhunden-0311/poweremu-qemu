/*
 * ppc-mmu harness: tlbie / TLB invalidation scenarios on a 7400 (32-bit
 * hash MMU), bare metal under qemu-system-ppc -M mac99 -cpu 7400.
 *
 * Runs supervisor-only.  BATs map the ROM (code, constants) and the low
 * 256 MB of RAM identity (stack, page table, results); the tests use
 * segments 2-6 through the hashed page table.  Every test builds its own
 * mappings from fresh VSIDs and physical pages, which are filled before any
 * translation is made: the tested event is always a PTE change (and the
 * invalidation that follows), never a write to code.
 *
 * Results at RES (physical = effective): magic, test count, then one word
 * per test, 0 = pass, else a code; "info" tests record what was seen.
 * tests/ppc-mmu/run.py reads them through the monitor.  AI-generated
 * (Claude).
 */
#include <stdint.h>

#define RES         0x00004000u
#define HTAB        0x00100000u     /* 64 KB, HTABMASK 0 */
#define PAGES       0x00200000u     /* test pages from here */
#define MAGIC_RUN   0x4d4d5552u     /* "MMUR" */
#define MAGIC_DONE  0x4d4d5544u     /* "MMUD" */
#define FAULT       0x00003f00u

#define MSR_IR 0x20
#define MSR_DR 0x10
#define SR_KS  0x40000000u
#define SR_KP  0x20000000u

#define PP_RW 2
#define PP_RO 3

uint32_t try_store(uint32_t ea, uint32_t val);
uint32_t try_load(uint32_t ea, uint32_t *out);
uint32_t call_ctr(uint32_t ea);
uint32_t call_lr(uint32_t ea);

static volatile uint32_t *const res = (volatile uint32_t *)RES;
static volatile uint32_t *const fault = (volatile uint32_t *)FAULT;

#define mtspr(n, v) __asm__ volatile("mtspr %0, %1" :: "i"(n), "r"(v) : "memory")
static inline void sync(void) { __asm__ volatile("sync" ::: "memory"); }
static inline void isync(void) { __asm__ volatile("isync" ::: "memory"); }
static inline void tlbie(uint32_t ea)
{
    __asm__ volatile("tlbie %0" :: "r"(ea) : "memory");
}
static inline void mtsr(int n, uint32_t v)
{
    __asm__ volatile("mtsrin %0, %1" :: "r"(v), "r"((uint32_t)n << 28) : "memory");
}
static inline uint32_t ld(uint32_t ea)
{
    uint32_t v;
    __asm__ volatile("lwz %0, 0(%1)" : "=r"(v) : "b"(ea) : "memory");
    return v;
}
static inline void st(uint32_t ea, uint32_t v)
{
    __asm__ volatile("stw %0, 0(%1)" :: "r"(v), "b"(ea) : "memory");
}
static inline void mtmsr(uint32_t v)
{
    __asm__ volatile("mtmsr %0; isync" :: "r"(v) : "memory");
}
static inline uint32_t mfmsr(void)
{
    uint32_t v;
    __asm__ volatile("mfmsr %0" : "=r"(v));
    return v;
}

/* Physical memory, used with translation on through the identity DBAT. */
static inline void pw(uint32_t pa, uint32_t v) { *(volatile uint32_t *)pa = v; }
static inline uint32_t pr(uint32_t pa) { return *(volatile uint32_t *)pa; }

/* Allocators (the image is ROM: no writable statics) */
static volatile uint32_t *const alloc = (volatile uint32_t *)0x00003e00;
#define next_page (alloc[0])
#define next_vsid (alloc[1])

/* A fresh physical page whose first word is @tag (and @tag + i after). */
static uint32_t page(uint32_t tag)
{
    uint32_t pa = next_page;

    next_page += 0x1000;
    for (uint32_t i = 0; i < 1024; i++) {
        pw(pa + 4 * i, tag + i);
    }
    return pa;
}

/* A code page: @n instruction words at byte offsets @off[i]. */
static uint32_t code_page(const uint32_t *off, const uint32_t *insn, int n)
{
    uint32_t pa = next_page;

    next_page += 0x1000;
    for (uint32_t i = 0; i < 1024; i++) {
        pw(pa + 4 * i, 0x7fe00008);         /* trap */
    }
    for (int i = 0; i < n; i++) {
        pw(pa + off[i], insn[i]);
    }
    for (uint32_t a = pa; a < pa + 0x1000; a += 32) {
        __asm__ volatile("dcbst 0, %0; sync; icbi 0, %0" :: "r"(a) : "memory");
    }
    sync();
    isync();
    return pa;
}

static uint32_t vsid(void) { return next_vsid++; }

/* The primary-hash PTE for (vsid, ea), made valid; returns its address. */
static uint32_t map(uint32_t vs, uint32_t ea, uint32_t pa, uint32_t pp)
{
    uint32_t hash = (vs & 0x7ffff) ^ ((ea >> 12) & 0xffff);
    uint32_t pteg = HTAB | ((hash & 0x3ff) << 6);

    for (int k = 0; k < 8; k++) {
        uint32_t pte = pteg + 8 * k;
        if (!(pr(pte) & 0x80000000u)) {
            pw(pte + 4, (pa & 0xfffff000u) | pp);
            sync();
            pw(pte, 0x80000000u | ((vs & 0xffffff) << 7) | ((ea >> 22) & 0x3f));
            sync();
            return pte;
        }
    }
    res[3] = 0xbad0000 | (hash & 0x3ff);    /* PTEG full: harness bug */
    for (;;) {
    }
}

/* Point an existing PTE elsewhere / change its PP (R and C cleared). */
static void remap(uint32_t pte, uint32_t pa, uint32_t pp)
{
    pw(pte + 4, (pa & 0xfffff000u) | pp);
    sync();
}

static void sr_set(int n, uint32_t v)
{
    mtsr(n, v);
    isync();
}

/* li r3,imm / addi r3,r3,imm / mtlr r4 / blr / b +off */
#define LI3(i)   (0x38600000u | ((i) & 0xffff))
#define ADDI3(i) (0x38630000u | ((i) & 0xffff))
#define MTLR4    0x7c8803a6u
#define BLR      0x4e800020u
#define B(off)   (0x48000000u | ((off) & 0x03fffffcu))
#define NOP      0x60000000u

/* ---------------------------------------------------------------- tests */

/*
 * 1. Two pages of one 64-class (address bits 12-17).  Both cached; V2's PTE
 * changes; tlbie names V1 only.  The 7400 drops both ways of the set, so V2
 * must be walked again.
 */
static uint32_t t_class_pair(void)
{
    uint32_t vs = vsid(), v1 = 0x20001000, v2 = 0x20041000;  /* class 1 both */
    uint32_t a = page(0x1a000000), b = page(0x1b000000), c = page(0x1c000000);
    uint32_t p2;

    sr_set(2, vs);
    map(vs, v1, a, PP_RW);
    p2 = map(vs, v2, b, PP_RW);
    if (ld(v1) != 0x1a000000 || ld(v2) != 0x1b000000) {
        return 1;
    }
    remap(p2, c, PP_RW);
    tlbie(v1);
    sync();
    return ld(v2) == 0x1c000000 ? 0 : 2;
}

/*
 * 2. Info (implementation check, not an architectural promise): a page of
 * another class, its PTE changed without a tlbie of its own, after a tlbie
 * of class 1.  1 = still the old mapping (selective), 2 = the new one.
 */
static uint32_t t_other_class(void)
{
    uint32_t vs = vsid(), v1 = 0x20101000, v3 = 0x20103000;    /* classes 1, 3 */
    uint32_t a = page(0x2a000000), d = page(0x2d000000), e = page(0x2e000000);
    uint32_t p3;

    sr_set(2, vs);
    map(vs, v1, a, PP_RW);
    p3 = map(vs, v3, d, PP_RW);
    (void)ld(v1);
    if (ld(v3) != 0x2d000000) {
        return 0x10;
    }
    remap(p3, e, PP_RW);
    tlbie(v1);
    sync();
    return ld(v3) == 0x2d000000 ? 1 : ld(v3) == 0x2e000000 ? 2 : 0x11;
}

/* 3. One VSID through two segment registers; tlbie by one EA. */
static uint32_t t_alias_segments(void)
{
    uint32_t vs = vsid(), ea_a = 0x20205000, ea_b = 0x40205000;
    uint32_t a = page(0x3a000000), b = page(0x3b000000);
    uint32_t p;

    sr_set(2, vs);
    sr_set(4, vs);
    p = map(vs, ea_a, a, PP_RW);
    if (ld(ea_a) != 0x3a000000 || ld(ea_b) != 0x3a000000) {
        return 1;
    }
    remap(p, b, PP_RW);
    tlbie(ea_a);
    sync();
    return ld(ea_b) == 0x3b000000 ? 0 : 2;
}

/*
 * 4. One VSID in two SR sets (with PPC_SR_SLOTS=1, two slots).  Cached in
 * set B through segment 5; back in set A the PTE changes and tlbie names
 * the segment-2 EA; set B must see the new mapping.
 */
static uint32_t t_alias_sets(void)
{
    uint32_t x = vsid(), y = vsid();
    uint32_t a = page(0x4a000000), b = page(0x4b000000);
    uint32_t p;

    sr_set(2, x);                   /* set A */
    sr_set(5, y);
    p = map(x, 0x20306000, a, PP_RW);   /* (0x50306000 in VSID x: same PTE) */
    if (ld(0x20306000) != 0x4a000000) {
        return 1;
    }
    sr_set(5, x);                   /* set B: VSID x also in segment 5 */
    sr_set(2, y);
    if (ld(0x50306000) != 0x4a000000) {
        return 2;
    }
    sr_set(2, x);                   /* back to set A */
    sr_set(5, y);
    remap(p, b, PP_RW);
    tlbie(0x20306000);
    sync();
    sr_set(5, x);                   /* set B */
    sr_set(2, y);
    return ld(0x50306000) == 0x4b000000 ? 0 : 3;
}

/* 5. Read-write to read-only: the store after tlbie must take a DSI. */
static uint32_t t_perm_store(void)
{
    uint32_t vs = vsid(), v = 0x20407000;
    uint32_t a = page(0x5a000000);
    uint32_t p;

    sr_set(2, vs);
    p = map(vs, v, a, PP_RW);
    if (try_store(v, 0x55555555)) {
        return 1;
    }
    remap(p, a, PP_RO);
    tlbie(v);
    sync();
    if (!try_store(v, 0x66666666)) {
        return 2;
    }
    return fault[0] == 0x300 && (fault[2] & 0x08000000u) ? 0 : 3;
}

/*
 * 6. Instruction remap, entered three ways, all warmed first (several calls,
 * so TBs are chained and the jump cache holds them): bctr to offset 0,
 * which branches within the page to 0x100; blr to 0x200.  Then only the
 * PTE moves to a second, already-filled code page.
 */
static uint32_t t_code_remap(void)
{
    static const uint32_t off[] = { 0x000, 0x100, 0x104, 0x200, 0x204, 0x208 };
    const uint32_t i1[] = { B(0x100), LI3(1), BLR, LI3(11), MTLR4, BLR };
    const uint32_t i2[] = { B(0x100), LI3(2), BLR, LI3(12), MTLR4, BLR };
    uint32_t vs = vsid(), v = 0x30008000;
    uint32_t c1 = code_page(off, i1, 6), c2 = code_page(off, i2, 6);
    uint32_t p;

    sr_set(3, vs);
    p = map(vs, v, c1, PP_RW);
    for (int i = 0; i < 4; i++) {
        if (call_ctr(v) != 1 || call_lr(v + 0x200) != 11) {
            return 1;
        }
    }
    remap(p, c2, PP_RW);
    tlbie(v);
    sync();
    isync();
    if (call_ctr(v) != 2) {
        return 2;
    }
    return call_lr(v + 0x200) == 12 ? 0 : 3;
}

/*
 * 7. Code that runs off the end of one page into the next (a 7400 TB ends at
 * the page boundary: two TBs, the second reached by a lookup).  The second
 * page is remapped.
 */
static uint32_t t_code_fallthrough(void)
{
    static const uint32_t offa[] = { 0xff8, 0xffc };
    const uint32_t ia[] = { LI3(1), NOP };
    static const uint32_t offb[] = { 0x000, 0x004 };
    const uint32_t ib1[] = { ADDI3(10), BLR }, ib2[] = { ADDI3(20), BLR };
    uint32_t vs = vsid(), v = 0x30109000;
    uint32_t ca = code_page(offa, ia, 2), cb1 = code_page(offb, ib1, 2);
    uint32_t cb2 = code_page(offb, ib2, 2);
    uint32_t p;

    sr_set(3, vs);
    map(vs, v, ca, PP_RW);
    p = map(vs, v + 0x1000, cb1, PP_RW);
    for (int i = 0; i < 4; i++) {
        if (call_ctr(v + 0xff8) != 11) {
            return 1;
        }
    }
    remap(p, cb2, PP_RW);
    tlbie(v + 0x1000);
    sync();
    isync();
    return call_ctr(v + 0xff8) == 21 ? 0 : 2;
}

/*
 * 8. Instruction access withdrawn: the code page's PTE becomes invalid; the
 * fetch after tlbie must take an ISI (no translation).  (QEMU's hash32
 * grants execute whatever PP says unless the segment is no-execute, so a PP
 * change cannot deny a fetch there: mmu-hash32.h ppc_hash32_prot.)
 */
static uint32_t t_perm_fetch(void)
{
    static const uint32_t off[] = { 0x000, 0x004 };
    const uint32_t ins[] = { LI3(7), BLR };
    uint32_t vs = vsid(), v = 0x3020b000;
    uint32_t c = code_page(off, ins, 2);
    uint32_t p;

    sr_set(3, vs);
    p = map(vs, v, c, PP_RW);
    for (int i = 0; i < 4; i++) {
        if (call_ctr(v) != 7) {
            return 1;
        }
    }
    pw(p, pr(p) & ~0x80000000u);    /* V = 0 */
    sync();
    tlbie(v);
    sync();
    isync();
    if (call_ctr(v) != (0xfa000000u | 0x400)) {
        return 2;
    }
    return fault[4] & 0x40000000u ? 0 : 3;     /* SRR1: no translation */
}

/*
 * 9. The C bit: cleared in the PTE and invalidated, the next store must
 * set it again (it cannot reuse a TLB entry that still allows writes).
 */
static uint32_t t_change_bit(void)
{
    uint32_t vs = vsid(), v = 0x2050d000;
    uint32_t a = page(0x6a000000);
    uint32_t p;

    sr_set(2, vs);
    p = map(vs, v, a, PP_RW);
    (void)ld(v);
    st(v, 1);
    if (!(pr(p + 4) & 0x80)) {
        return 1;
    }
    remap(p, a, PP_RW);             /* R and C cleared */
    tlbie(v);
    sync();
    st(v, 2);
    return pr(p + 4) & 0x80 ? 0 : 2;
}

/*
 * 10. A DBAT over a page that also has a PTE: tlbie leaves the BAT
 * translation alone; invalidating the BAT exposes the page table.
 */
static uint32_t t_bat(void)
{
    uint32_t vs = vsid(), v = 0x60000000;
    uint32_t bat, pt;

    next_page = (next_page + 0x1ffff) & ~0x1ffffu;
    bat = next_page;

    for (uint32_t i = 0; i < 32; i++) {
        page(0x7a000000 + 0x10000 * i);     /* 128 KB for the BAT */
    }
    pt = page(0x7b000000);
    sr_set(6, vs);
    map(vs, v, pt, PP_RW);
    mtspr(541, bat | 0x2);                  /* DBAT2L: BRPN, PP RW */
    mtspr(540, v | 0x2);                    /* DBAT2U: 128 KB, Vs */
    isync();
    if (ld(v) != 0x7a000000) {
        return 1;
    }
    tlbie(v);
    sync();
    if (ld(v) != 0x7a000000) {
        return 2;
    }
    mtspr(540, 0);
    isync();
    return ld(v) == 0x7b000000 ? 0 : 3;
}

/*
 * 11. tlbie with an SR change pending: mtsr (no isync), tlbie, sync -- the
 * sync handles GLOBAL and, with slots, the SR switch; without slots the SR
 * change's LOCAL flush comes with it.  Both sets must see current mappings.
 */
static uint32_t t_pending_sr(void)
{
    uint32_t xa = vsid(), xb = vsid(), v = 0x2060f000;
    uint32_t a = page(0x8a000000), b = page(0x8b000000), c = page(0x8c000000);
    uint32_t pa;

    sr_set(2, xa);
    pa = map(xa, v, a, PP_RW);
    map(xb, v, b, PP_RW);
    if (ld(v) != 0x8a000000) {
        return 1;
    }
    remap(pa, c, PP_RW);
    mtsr(2, xb);                    /* pending until the sync */
    tlbie(v);
    sync();
    isync();
    if (ld(v) != 0x8b000000) {
        return 2;
    }
    sr_set(2, xa);
    return ld(v) == 0x8c000000 ? 0 : 3;
}

/* 12. A local context check (isync) between tlbie and the sync. */
static uint32_t t_local_first(void)
{
    uint32_t vs = vsid(), v = 0x20711000;
    uint32_t a = page(0x9a000000), b = page(0x9b000000);
    uint32_t p;

    sr_set(2, vs);
    p = map(vs, v, a, PP_RW);
    (void)ld(v);
    remap(p, b, PP_RW);
    tlbie(v);
    isync();
    if (ld(v) != 0x9b000000) {
        return 1;
    }
    sync();
    return ld(v) == 0x9b000000 ? 0 : 2;
}

/* 13. Several tlbies of different classes, one sync. */
static uint32_t t_many_classes(void)
{
    uint32_t vs = vsid(), v[3] = { 0x20813000, 0x20814000, 0x20815000 };
    uint32_t p[3], nw[3];

    sr_set(2, vs);
    for (int i = 0; i < 3; i++) {
        p[i] = map(vs, v[i], page(0xa0000000 + 0x1000000 * i), PP_RW);
        nw[i] = page(0xa8000000 + 0x1000000 * i);
        (void)ld(v[i]);
    }
    for (int i = 0; i < 3; i++) {
        remap(p[i], nw[i], PP_RW);
    }
    for (int i = 0; i < 3; i++) {
        tlbie(v[i]);
    }
    sync();
    for (int i = 0; i < 3; i++) {
        if (ld(v[i]) != 0xa8000000 + 0x1000000 * i) {
            return 1 + i;
        }
    }
    return 0;
}

/*
 * 14. A fill between an SR write and its isync (the slot guard): after
 * switching back, set A must see its own mapping, not the fill made while
 * the switch was pending.
 */
static uint32_t t_fill_pending(void)
{
    uint32_t xa = vsid(), xb = vsid(), v = 0x20917000;
    uint32_t a = page(0xb1000000), b = page(0xb2000000);

    sr_set(2, xa);
    map(xa, v, a, PP_RW);
    map(xb, v, b, PP_RW);
    if (ld(v) != 0xb1000000) {
        return 1;
    }
    mtsr(2, xb);
    (void)ld(v);                    /* either mapping, architecturally */
    isync();
    if (ld(v) != 0xb2000000) {
        return 2;
    }
    sr_set(2, xa);
    return ld(v) == 0xb1000000 ? 0 : 3;
}

/*
 * 15. The victim TLB: V cached, then W at the same TLB index (32768 pages
 * on: the same slot for any table of up to 32768 entries, and the same
 * class) pushes V into the victim TLB.  V's PTE changes; tlbie of V's
 * class; V must be walked again, not found among the victims.
 */
static uint32_t t_victim(void)
{
    uint32_t vs = vsid(), v = 0x20a19000, w = 0x20a19000 + 0x08000000;
    uint32_t a = page(0xc1000000), b = page(0xc2000000), c = page(0xc3000000);
    uint32_t pv;

    sr_set(2, vs);
    pv = map(vs, v, a, PP_RW);
    map(vs, w, b, PP_RW);
    if (ld(v) != 0xc1000000 || ld(w) != 0xc2000000) {     /* V now a victim */
        return 1;
    }
    remap(pv, c, PP_RW);
    tlbie(w);                       /* same class; names W, not V */
    sync();
    return ld(v) == 0xc3000000 ? 0 : 2;
}

static uint32_t (*const tests[])(void) = {
    t_class_pair, t_other_class, t_alias_segments, t_alias_sets,
    t_perm_store, t_code_remap, t_code_fallthrough, t_perm_fetch,
    t_change_bit, t_bat, t_pending_sr, t_local_first, t_many_classes,
    t_fill_pending, t_victim,
};
#define NTESTS (sizeof(tests) / sizeof(tests[0]))

static void bats(void)
{
    /* IBAT0 / DBAT1: ROM 0xf0000000, 256 MB; DBAT0: RAM 0, 256 MB */
    for (int i = 528; i <= 543; i++) {
        switch (i) {
#define Z(n) case n: mtspr(n, 0); break;
        Z(528) Z(529) Z(530) Z(531) Z(532) Z(533) Z(534) Z(535)
        Z(536) Z(537) Z(538) Z(539) Z(540) Z(541) Z(542) Z(543)
        }
    }
    mtspr(529, 0xf0000000u | 0x1);          /* IBAT0L: BRPN, PP RO */
    mtspr(528, 0xf0000000u | 0x1ffc | 0x2); /* IBAT0U: 256 MB, Vs */
    mtspr(537, 0x00000000u | 0x2);          /* DBAT0L: RW */
    mtspr(536, 0x00000000u | 0x1ffc | 0x2); /* DBAT0U */
    mtspr(539, 0xf0000000u | 0x1);          /* DBAT1L: RO */
    mtspr(538, 0xf0000000u | 0x1ffc | 0x2); /* DBAT1U */
    isync();
}

void main(void)
{
    uint32_t pass = 0;

    for (uint32_t a = HTAB; a < HTAB + 0x10000; a += 4) {
        pw(a, 0);
    }
    for (int i = 0; i < 64; i++) {
        res[i] = 0;
    }
    fault[5] = 0;
    next_page = PAGES;
    next_vsid = 0x100;
    res[0] = MAGIC_RUN;
    res[1] = NTESTS;
    __asm__ volatile("mtsdr1 %0" :: "r"(HTAB) : "memory");
    for (int n = 0; n < 16; n++) {
        mtsr(n, n);
    }
    bats();
    mtmsr(mfmsr() | MSR_IR | MSR_DR);
    for (unsigned i = 0; i < NTESTS; i++) {
        res[4 + i] = 0xffffffffu;           /* running */
        res[4 + i] = tests[i]();
        pass += res[4 + i] == 0;
    }
    res[2] = pass;
    res[0] = MAGIC_DONE;
    for (;;) {
    }
}
