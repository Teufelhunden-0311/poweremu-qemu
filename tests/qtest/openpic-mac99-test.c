/*
 * QTest for the KeyLargo OpenPIC of the mac99 machine
 *
 * - An edge masked by CTPR survives EOI and a masked IACK, and is
 *   delivered once when CTPR drops (upstream 2c87618).
 * - TBCR writes on a model without timer objects leave QEMU running
 *   (KeyLargo maps the timer registers but never creates the timers).
 *
 * No firmware runs, so the test places the MacIO BAR itself.  KeyLargo
 * registers are little-endian; they are accessed byte-wise so that the
 * values do not depend on the target's byte order.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "libqtest.h"

#define UNIN_CFG_ADDR   0xf2800000u
#define UNIN_CFG_DATA   0xf2c00000u
#define MACIO_BAR       0x80000000u     /* in the UniNorth PCI hole */
#define PIC_BASE        (MACIO_BAR + 0x40000)

#define SPVE            0x010e0
#define TBCR0           0x01110
#define IVPR(i)         (0x10000 + 0x20 * (i))
#define IDR(i)          (0x10010 + 0x20 * (i))
#define CTPR0           0x20080
#define IACK0           0x200a0
#define EOI0            0x200b0

#define TBCR_CI         0x80000000u
#define VEC_SPURIOUS    0xff

static void wr_le(QTestState *qts, uint64_t addr, uint32_t v)
{
    uint8_t b[4] = { v, v >> 8, v >> 16, v >> 24 };
    qtest_memwrite(qts, addr, b, 4);
}

static uint32_t rd_le(QTestState *qts, uint64_t addr)
{
    uint8_t b[4];
    qtest_memread(qts, addr, b, 4);
    return b[0] | b[1] << 8 | b[2] << 16 | (uint32_t)b[3] << 24;
}

static void pic_wr(QTestState *qts, uint32_t off, uint32_t v)
{
    wr_le(qts, PIC_BASE + off, v);
}

static uint32_t pic_rd(QTestState *qts, uint32_t off)
{
    return rd_le(qts, PIC_BASE + off);
}

/*
 * Config space of the UniNorth main bus.  An address with bit 31 set is
 * passed through in the x86 layout (bus 23:16, devfn 15:8, register 7:0),
 * the form OpenBIOS uses.
 */
static uint32_t cfg_rd(QTestState *qts, int devfn, int reg)
{
    wr_le(qts, UNIN_CFG_ADDR, 0x80000000u | devfn << 8 | reg);
    return rd_le(qts, UNIN_CFG_DATA);
}

static void cfg_wr(QTestState *qts, int devfn, int reg, uint32_t v)
{
    wr_le(qts, UNIN_CFG_ADDR, 0x80000000u | devfn << 8 | reg);
    wr_le(qts, UNIN_CFG_DATA, v);
}

static QTestState *start(void)
{
    QTestState *qts = qtest_init("-machine mac99 -display none");
    int devfn;

    /* MacIO: Apple vendor, class "other" (0xff00) */
    for (devfn = 0; devfn < 256; devfn++) {
        uint32_t id = cfg_rd(qts, devfn, 0x00);
        if ((id & 0xffff) == 0x106b && cfg_rd(qts, devfn, 0x08) >> 16 == 0xff00) {
            break;
        }
    }
    g_assert_cmpint(devfn, <, 256);
    cfg_wr(qts, devfn, 0x10, MACIO_BAR);
    cfg_wr(qts, devfn, 0x04, cfg_rd(qts, devfn, 0x04) | 0x2);   /* memory */
    g_assert_cmphex(cfg_rd(qts, devfn, 0x10) & ~0xfu, ==, MACIO_BAR);

    /* the byte-wise accessors reach the PIC with the right byte order */
    pic_wr(qts, SPVE, VEC_SPURIOUS);
    g_assert_cmphex(pic_rd(qts, SPVE) & 0xff, ==, VEC_SPURIOUS);
    pic_wr(qts, CTPR0, 8);
    g_assert_cmphex(pic_rd(qts, CTPR0), ==, 8);
    return qts;
}

static void pulse(QTestState *qts, int src)
{
    qtest_set_irq_in(qts, "pic", NULL, src, 1);
    qtest_set_irq_in(qts, "pic", NULL, src, 0);
}

/* CPU0 INT is the first output of the PIC's first destination. */
#define INT0(qts) qtest_get_irq(qts, 0)

static void test_ctpr_masked_edge(void)
{
    QTestState *qts = start();

    qtest_irq_intercept_out_named(qts, "pic", "sysbus-irq");
    pic_wr(qts, CTPR0, 0);
    pic_wr(qts, IVPR(2), 10 << 16 | 0x2a);      /* edge, unmasked, prio 10 */
    pic_wr(qts, IDR(2), 1);
    pic_wr(qts, IVPR(1), 5 << 16 | 0x29);       /* edge, unmasked, prio 5 */
    pic_wr(qts, IDR(1), 1);

    pulse(qts, 2);
    g_assert_true(INT0(qts));
    g_assert_cmphex(pic_rd(qts, IACK0), ==, 0x2a);

    pulse(qts, 1);                              /* pending behind prio 10 */
    pic_wr(qts, CTPR0, 8);
    g_assert_false(INT0(qts));

    pic_wr(qts, EOI0, 0);                       /* prio 5 is still masked */
    g_assert_false(INT0(qts));
    g_assert_cmphex(pic_rd(qts, IACK0), ==, VEC_SPURIOUS);
    g_assert_cmphex(pic_rd(qts, IACK0), ==, VEC_SPURIOUS);
    g_assert_false(INT0(qts));

    pic_wr(qts, CTPR0, 5);                      /* equal priority: masked */
    g_assert_false(INT0(qts));
    g_assert_cmphex(pic_rd(qts, IACK0), ==, VEC_SPURIOUS);

    pic_wr(qts, CTPR0, 0);                      /* the edge was kept */
    g_assert_true(INT0(qts));
    g_assert_cmphex(pic_rd(qts, IACK0), ==, 0x29);
    pic_wr(qts, EOI0, 0);
    g_assert_false(INT0(qts));
    g_assert_cmphex(pic_rd(qts, IACK0), ==, VEC_SPURIOUS);

    /* positive control: an edge above CTPR is delivered directly */
    pic_wr(qts, CTPR0, 4);
    pulse(qts, 1);
    g_assert_true(INT0(qts));
    g_assert_cmphex(pic_rd(qts, IACK0), ==, 0x29);
    pic_wr(qts, EOI0, 0);
    qtest_quit(qts);
}

static void tbcr_write(uint32_t val)
{
    QTestState *qts = start();

    g_assert_cmphex(pic_rd(qts, TBCR0) & TBCR_CI, ==, TBCR_CI);
    pic_wr(qts, TBCR0, val);                    /* clears count inhibit */
    g_assert_cmphex(pic_rd(qts, TBCR0), ==, val);
    qtest_clock_step(qts, 1000000000);          /* 1 s: nothing to fire */
    pic_wr(qts, TBCR0, TBCR_CI);
    g_assert_cmphex(pic_rd(qts, TBCR0), ==, TBCR_CI);
    qtest_system_reset(qts);
    qtest_quit(qts);
}

static void test_tbcr_zero(void)
{
    tbcr_write(0);              /* reached timer_del(NULL) */
}

static void test_tbcr_count(void)
{
    tbcr_write(1);              /* reached timer_mod(NULL, ...) */
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);

    /* qemu-system-ppc64's mac99 is a G5 (U3, other addresses) */
    if (strcmp(qtest_get_arch(), "ppc") || !qtest_has_machine("mac99")) {
        g_test_skip("needs qemu-system-ppc with mac99");
        return g_test_run();
    }
    qtest_add_func("openpic-mac99/ctpr-masked-edge", test_ctpr_masked_edge);
    qtest_add_func("openpic-mac99/tbcr-zero", test_tbcr_zero);
    qtest_add_func("openpic-mac99/tbcr-count", test_tbcr_count);
    return g_test_run();
}
