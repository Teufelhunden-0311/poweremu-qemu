/* Host unit test of the R300 address helpers, compiled from the real source
 * (extract.sh) against a mock device: two apertures, a mock page table and
 * mock guest RAM. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#define MIN(a, b) ((a) < (b) ? (a) : (b))
typedef uint64_t hwaddr;
typedef int MemTxResult;
#define MEMTX_OK 0
#define MEMTXATTRS_UNSPECIFIED 0
typedef struct { int dummy; } MemoryRegion;
typedef struct {
    struct { uint32_t mc_fb_location, mc_agp_location, aic_ctrl, aic_lo_addr, aic_hi_addr, aic_pt_base; } regs;
    uint32_t r300_aic_pt_base;
    uint64_t vram_size;
    MemoryRegion vram;
} PPCMacGPUState;

static uint8_t vram[1 << 20];
static int agp_pte[4096], aic_tab[2][4096];   /* page index -> guest page + 1, 0 invalid */
#define aic_pte aic_tab[0]                    /* table 1 is used when r300_aic_pt_base is set */
static uint32_t bridge_gen;
static MemoryRegion ram_mr;
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#define RCU_READ_LOCK_GUARD() (void)0
static uint32_t uninorth_get_agp_gart_gen(void) { return bridge_gen; }
static int agp_calls, gart_calls;
static void *address_space_memory;

static uint8_t ram[64 << 12];                 /* 64 guest pages */
static void *memory_region_get_ram_ptr(MemoryRegion *mr) { return mr == &ram_mr ? (void *)ram : (void *)vram; }
static bool memory_region_is_ram(MemoryRegion *mr) { return mr == &ram_mr; }
static MemoryRegion *address_space_translate(void *as, hwaddr addr, hwaddr *xlat, hwaddr *plen, bool w, int at)
{ (void)as; (void)w; (void)at; if (addr + *plen > sizeof(ram)) { *plen = 0; return NULL; } *xlat = addr; return &ram_mr; }
static bool xl(int *pte, uint32_t base, uint32_t a, hwaddr *phys)
{
    int e = pte[(a - base) >> 12];
    if (!e) return false;
    *phys = (hwaddr)(e - 1) * 4096 + (a & 0xFFF);
    return true;
}
static bool ppc_mac_gpu_agp_translate(PPCMacGPUState *s, uint32_t a, hwaddr *p)
{
    uint32_t lo = (s->regs.mc_agp_location & 0xFFFF) << 16;
    uint32_t hi = (s->regs.mc_agp_location >> 16) << 16 | 0xFFFF;
    agp_calls++;
    return s->regs.mc_agp_location && a >= lo && a <= hi && xl(agp_pte, lo, a, p);
}
static bool ppc_mac_gpu_gart_translate(PPCMacGPUState *s, uint32_t a, hwaddr *p)
{
    gart_calls++;
    return (s->regs.aic_ctrl & 1) && a >= s->regs.aic_lo_addr && a <= s->regs.aic_hi_addr &&
           xl(aic_tab[s->r300_aic_pt_base ? 1 : 0], s->regs.aic_lo_addr, a, p);
}
#include "cache.inc"
static int address_space_read(void *as, hwaddr p, int at, void *buf, uint32_t n)
{ (void)as; (void)at; if (p + n > sizeof(ram)) return 1; memcpy(buf, ram + p, n); return 0; }
static int address_space_rw(void *as, hwaddr p, int at, uint8_t *buf, uint32_t n, bool w)
{ (void)as; (void)at; if (p + n > sizeof(ram)) return 1;
  if (w) memcpy(ram + p, buf, n); else memcpy(buf, ram + p, n); return 0; }

#include "helpers.inc"

static int fails, checks;
#define CHECK(c) do { checks++; if (!(c)) { fails++; printf("FAIL line %d: %s\n", __LINE__, #c); } } while (0)

static void setup(PPCMacGPUState *s)
{
    memset(s, 0, sizeof(*s));
    s->vram_size = 128u << 20;
    s->regs.mc_fb_location = 0x07bf0000;          /* VRAM 0 .. 0x07bfffff */
    s->regs.mc_agp_location = 0x17ff1000;         /* AGP 0x10000000 .. 0x17ffffff */
    s->regs.aic_ctrl = 1;
    s->regs.aic_lo_addr = 0x07c00000;
    s->regs.aic_hi_addr = 0x0fbfffff;
    memset(agp_pte, 0, sizeof(agp_pte));
    memset(aic_tab, 0, sizeof(aic_tab));
    r200_agp_tc_flush();
}

int main(int argc, char **argv)
{
    PPCMacGPUState s;
    uint8_t buf[0x3000];
    hwaddr p;

    setup(&s);
    if (argc > 1 && !strcmp(argv[1], "bench")) {      /* ns per 12-byte AGP read, cache warm */
        struct timespec t0, t1; volatile uint32_t sink = 0; const int N = 50000000;
        for (int i = 0; i < 16; i++) agp_pte[i] = i + 1;
        for (int r = 0; r < 3; r++) {
            clock_gettime(CLOCK_MONOTONIC, &t0);
            for (int i = 0; i < N; i++) {
                r300_read_raw(&s, 0x10000000u + (uint32_t)(i * 12 & 0xfff0), buf, 12);
                sink += buf[0];
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            printf("%.2f ns/read\n", ((t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9) / N * 1e9);
        }
        return sink == 12345;
    }
    /* r300_span: inclusive end, no wrap */
    CHECK(r300_span(10, 1, 10, 10));
    CHECK(!r300_span(10, 2, 10, 10));
    CHECK(!r300_span(10, 0, 0, 100));
    CHECK(!r300_span(0xFFFFFFF0u, 0x20, 0, 0xFFFFFFFFu));
    CHECK(r300_span(0xFFFFFFF0u, 0x10, 0, 0xFFFFFFFFu));
    CHECK(!r300_span(5, 1, 10, 9));                   /* reversed range */

    /* r300_in_fb: reported range, not vram_size */
    CHECK(r300_in_fb(&s, 0x07bffffc, 4));
    CHECK(!r300_in_fb(&s, 0x07bffffc, 5));
    CHECK(!r300_in_fb(&s, 0x07c00000, 1));            /* the 4 MB tail is not VRAM */
    CHECK(!r300_in_fb(&s, 0x07d1c000, 0x1000));
    CHECK(!r300_in_fb(&s, 0xFFFFFFF0u, 0x20));
    s.regs.mc_fb_location = 0xffff0000;               /* transient at driver start: 0 .. 4 GB */
    CHECK(r300_in_fb(&s, 0x07fffffc, 4));             /* ... capped at the VRAM that exists */
    CHECK(!r300_in_fb(&s, 0x08000000, 4));
    s.regs.mc_fb_location = 0x07ff0000;               /* reports more than exists */
    s.vram_size = 64u << 20;
    CHECK(r300_in_fb(&s, 0x03fffffc, 4));
    CHECK(!r300_in_fb(&s, 0x04000000, 4));
    s.regs.mc_fb_location = 0x00000010;               /* top < base: none */
    CHECK(!r300_in_fb(&s, 0x00100000, 4));
    setup(&s);

    /* classification */
    CHECK(r300_in_sysmem(&s, 0x07c00000));
    CHECK(r300_in_sysmem(&s, 0x10054000));
    CHECK(!r300_in_sysmem(&s, 0x00300000));
    CHECK(!r300_in_sysmem(&s, 0x0fc00000));           /* between AIC end and AGP: neither */
    s.regs.aic_ctrl = 0;
    CHECK(!r300_in_sysmem(&s, 0x07c00000));           /* PCI GART off */
    setup(&s);

    /* chunks: whole chunk in one aperture, translated there only */
    aic_pte[0] = 3;                                   /* 0x07c00000 -> guest page 2 */
    CHECK(r300_sys_chunk(&s, 0x07c00010, 0x10, &p) && p == 2 * 4096 + 0x10);
    s.regs.aic_hi_addr = 0x07c007ff;                  /* AIC ends mid-page */
    CHECK(r300_sys_chunk(&s, 0x07c00000, 0x800, &p));
    CHECK(!r300_sys_chunk(&s, 0x07c00000, 0x801, &p));
    CHECK(!r300_sys_chunk(&s, 0x07c00700, 0x200, &p));
    setup(&s);
    s.regs.mc_fb_location = 0x07c00000;               /* FB top 0x07c0ffff overlaps AIC start */
    aic_pte[0] = 3;
    CHECK(!r300_sys_chunk(&s, 0x07c00000, 0x10, &p));
    setup(&s);
    /* overlapping apertures: AGP wins, and an invalid AGP entry does not fall to the GART */
    s.regs.mc_agp_location = 0x0fff07c0;              /* AGP 0x07c00000 .. 0x0fffffff */
    aic_pte[0] = 3;
    agp_calls = gart_calls = 0;
    CHECK(!r300_sys_chunk(&s, 0x07c00000, 4, &p) && gart_calls == 0);
    agp_pte[0] = 5;
    CHECK(r300_sys_chunk(&s, 0x07c00000, 4, &p) && p == 4 * 4096);
    setup(&s);

    /* reads and writes over non-contiguous pages, unaligned start */
    aic_pte[0x11c] = 8; aic_pte[0x11d] = 3;           /* 0x07d1c000 -> page 7, 0x07d1d000 -> page 2 */
    for (int i = 0; i < 0x2000; i++) buf[i] = (uint8_t)(i * 7 + 1);
    CHECK(r300_sysmem_rw(&s, 0x07d1c040, buf, 0x1fc0, true));
    CHECK(ram[7 * 4096 + 0x40] == buf[0] && ram[2 * 4096] == buf[0xfc0] &&
          ram[2 * 4096 + 0xfff] == buf[0x1fbf]);
    uint8_t back[0x2000];
    memset(back, 0, sizeof(back));
    CHECK(r300_read_raw(&s, 0x07d1c040, back, 0x1fc0) && !memcmp(back, buf, 0x1fc0));
    CHECK(!r300_sysmem_rw(&s, 0x07d1c040, buf, 0x2fc0, true));   /* third page unmapped */
    CHECK(!r300_read_raw(&s, 0x07d1c040, back, 0x2fc0));
    /* spans: VRAM -> GART boundary, wrap, empty */
    CHECK(!r300_read_raw(&s, 0x07bff000, back, 0x2000));
    CHECK(!r300_sysmem_rw(&s, 0x07bff000, buf, 0x2000, true));
    CHECK(!r300_read_raw(&s, 0xfffff000u, back, 0x2000));
    CHECK(!r300_sysmem_rw(&s, 0xfffff000u, buf, 0x2000, true));
    CHECK(!r300_sysmem_rw(&s, 0x07d1c000, buf, 0, true));
    CHECK(r300_read_raw(&s, 0x00001000, back, 16));               /* plain VRAM read */


    /* the real page cache: an entry does not survive a configuration change */
    setup(&s);
    aic_tab[0][0x11c] = 8;                            /* table 0: -> guest page 7 */
    aic_tab[1][0x11c] = 3;                            /* table 1: -> guest page 2 */
    memset(ram + 7 * 4096, 0x77, 4096);
    memset(ram + 2 * 4096, 0x22, 4096);
    CHECK(r300_read_raw(&s, 0x07d1c010, back, 8) && back[0] == 0x77);   /* populates the cache */
    s.r300_aic_pt_base = 0x00836000;                  /* the driver points the GART at another table */
    CHECK(r300_read_raw(&s, 0x07d1c010, back, 8) && back[0] == 0x22);
    s.r300_aic_pt_base = 0;
    CHECK(r300_read_raw(&s, 0x07d1c010, back, 8) && back[0] == 0x77);
    s.regs.aic_ctrl = 0;                              /* PCI GART switched off: no cached page */
    CHECK(!r300_read_raw(&s, 0x07d1c010, back, 8));
    s.regs.aic_ctrl = 1;
    CHECK(r300_read_raw(&s, 0x07d1c010, back, 8) && back[0] == 0x77);
    s.regs.aic_lo_addr = 0x07d00000;                  /* aperture moved: same address, other page */
    aic_tab[0][0x1c] = 3;
    CHECK(r300_read_raw(&s, 0x07d1c010, back, 8) && back[0] == 0x22);
    s.regs.aic_lo_addr = 0x07c00000;
    s.regs.mc_agp_location = 0x0fff07c0;              /* AGP now covers the address: its table decides */
    CHECK(!r300_read_raw(&s, 0x07d1c010, back, 8));   /* AGP entry invalid: no fall-back to the GART */
    agp_pte[0x11c] = 6;
    memset(ram + 5 * 4096, 0x55, 4096);
    CHECK(r300_read_raw(&s, 0x07d1c010, back, 8) && back[0] == 0x55);
    s.regs.mc_agp_location = 0x17ff1000;
    CHECK(r300_read_raw(&s, 0x07d1c010, back, 8) && back[0] == 0x77);
    bridge_gen++;                                     /* bridge GART rewritten */
    aic_tab[0][0x11c] = 3;
    CHECK(r300_read_raw(&s, 0x07d1c010, back, 8) && back[0] == 0x22);
    /* cached path, apertures apart: an invalid AGP entry is not served by the PCI GART */
    setup(&s);
    CHECK(!r300_read_raw(&s, 0x10054000, back, 8));
    agp_pte[0x54] = 6;
    CHECK(r300_read_raw(&s, 0x10054000, back, 8) && back[0] == 0x55);
    /* an unaligned PCI GART base is not system memory */
    setup(&s);
    aic_tab[0][0] = 3;
    s.regs.aic_lo_addr = 0x07c00800;
    CHECK(!r300_in_sysmem(&s, 0x07c00800));
    CHECK(!r300_read_raw(&s, 0x07c00800, back, 8));
    CHECK(!r300_sysmem_rw(&s, 0x07c00800, buf, 8, true));

    printf("%d checks, %d failed\n", checks, fails);
    return fails != 0;
}
