/* Run the Quartz Extreme vertex program captured from Tiger 10.4.11. */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../../hw/display/r300/r300_pvs.h"

static const uint32_t qe_code[] = {
    0x00100201, 0x00D10002, 0x00D10001, 0x00D10005,
    0x00200201, 0x00D10022, 0x00D10001, 0x00D10005,
    0x00400201, 0x00D10042, 0x00D10001, 0x00D10005,
    0x00800201, 0x00D10062, 0x00D10001, 0x00D10005,
    0x00104201, 0x00D10082, 0x00D10041, 0x00D10045,
    0x00204201, 0x00D100A2, 0x00D10041, 0x00D10045,
    0x00404201, 0x00D100C2, 0x00D10041, 0x00D10045,
    0x00804201, 0x00D100E2, 0x00D10041, 0x00D10045,
    0x00F02202, 0x00D10021, 0x016DA021, 0x016DA025,
};

/* PVS instruction words (R5xx guide 7.5): VE_ADD dst <- a + b. */
#define SRC(type, idx, mode2) ((type) | ((idx) << 5) | 0x00D10000u | ((mode2) ? 1u << 31 : 0))
#define DST(type, idx, mask) (3u | ((type) << 8) | ((idx) << 13) | ((mask) << 20))

static void flow_control(void)
{
    /* c0 = (1,0,0,0), c2 = 0, c3..c5 distinct */
    static const float k[6][4] = { { 1, 0, 0, 0 }, { 0 }, { 0 }, { 1, 2, 3, 4 },
                                   { 10, 20, 30, 40 }, { 100, 200, 300, 400 } };
    float in[R300_PVS_NUM_INPUTS][4], out[R300_PVS_NUM_OUTPUTS][4];
    memset(in, 0, sizeof(in));

    /* LOOP 3 times over 1..2 with aL = 0, 1, 2: t0.x += 1, t1 += c[3 + aL] */
    static const uint32_t loop[] = {
        DST(0, 0, 15), SRC(2, 2, 0), SRC(2, 2, 0), 0,           /* 0: t0 = 0 */
        DST(0, 0, 1), SRC(0, 0, 0), SRC(2, 0, 0), 0,            /* 1: t0.x += c0 */
        DST(0, 1, 15), SRC(0, 1, 0), SRC(2, 3, 1), 0,           /* 2: t1 += c[3 + aL] */
        DST(2, 0, 15), SRC(0, 0, 0), SRC(2, 2, 0), 0,           /* 3: o0 = t0 */
        DST(2, 1, 15), SRC(0, 1, 0), SRC(2, 2, 0), 0,           /* 4: o1 = t1 */
    };
    R300PVSProgram p = { loop, 0, 4, k, 5 };
    p.fc_opc = 2;                                               /* FC0 = LOOP */
    p.fc_addrs[0] = 0 | (3u << 8) | (2u << 16) | (1u << 24);
    p.fc_loop[0] = 0 | (1u << 8);                               /* aL from 0 by 1 */
    memset(out, 0, sizeof(out));
    uint32_t u = r300_pvs_run(&p, in, out);
    printf("loop: o0.x %g o1 %g %g %g %g\n", out[0][0], out[1][0], out[1][1], out[1][2], out[1][3]);
    assert(!u && out[0][0] == 3 && out[1][0] == 111 && out[1][3] == 444);

    /* JSR to 3 (returns to 1), then JUMP from 1 over the subroutine to 4 */
    static const uint32_t jsr[] = {
        DST(0, 0, 15), SRC(2, 2, 0), SRC(2, 2, 0), 0,           /* 0: t0 = 0; JSR 3 */
        DST(2, 0, 15), SRC(0, 0, 0), SRC(2, 2, 0), 0,           /* 1: o0 = t0; JUMP 4 */
        DST(2, 2, 15), SRC(2, 5, 0), SRC(2, 2, 0), 0,           /* 2: skipped */
        DST(0, 0, 1), SRC(0, 0, 0), SRC(2, 0, 0), 0,            /* 3: t0.x += 1; RET */
        DST(2, 1, 15), SRC(2, 3, 0), SRC(2, 2, 0), 0,           /* 4: o1 = c3 */
    };
    R300PVSProgram q = { jsr, 0, 4, k, 5 };
    q.fc_opc = 3 | (1u << 2);                                   /* FC0 JSR, FC1 JUMP */
    q.fc_addrs[0] = 0 | (3u << 8) | (3u << 16) | (1u << 24);
    q.fc_addrs[1] = 1 | (4u << 8);
    memset(out, 0, sizeof(out));
    u = r300_pvs_run(&q, in, out);
    printf("jsr: o0.x %g o1.x %g o2.x %g\n", out[0][0], out[1][0], out[2][0]);
    assert(!u && out[0][0] == 1 && out[1][0] == 1 && out[2][0] == 0);

    /* A loop that never ends is cut off and reported. */
    R300PVSProgram r = { loop, 0, 4, k, 5 };
    r.fc_opc = 1;                                               /* JUMP 2 -> 1 forever */
    r.fc_addrs[0] = 2 | (1u << 8);
    assert(r300_pvs_run(&r, in, out) & R300_PVS_UNSUP_FLOW);
}

int main(void)
{
    const float consts[8][4] = {
        { 0.0025f, 0, 0, -1 }, { 0, -1.0f / 300, 0, 1 }, { 0, 0, -1, 0 },
        { 0, 0, 0, 1 }, { 1.0f / 256, 0, 0, 0 }, { 0, 1.0f / 256, 0, 0 },
        { 0, 0, 1, 0 }, { 0, 0, 0, 1 },
    };
    R300PVSProgram p = { qe_code, 0, 8, consts, 7 };
    float in[R300_PVS_NUM_INPUTS][4] = { { 256, 256, 0, 1 }, { 1, 1, 1, 1 },
                                         { 256, 256, 0, 1 } };
    float out[R300_PVS_NUM_OUTPUTS][4];
    char buf[160];

    memset(out, 0, sizeof(out));
    for (int i = 0; i < 9; i++) {
        r300_pvs_disasm_inst(&qe_code[i * 4], buf, sizeof(buf));
        printf("%d: %s\n", i, buf);
    }
    uint32_t u = r300_pvs_run(&p, in, out);
    for (int i = 0; i < 3; i++) {
        printf("o%d = %g %g %g %g\n", i, out[i][0], out[i][1], out[i][2], out[i][3]);
    }
    assert(!u);
    assert(fabsf(out[0][0] - (-0.36f)) < 1e-5 && fabsf(out[0][1] - (1 - 256.0f / 300)) < 1e-5);
    assert(out[0][3] == 1.0f);
    assert(out[1][0] == 1 && out[1][3] == 1);          /* colour passes through */
    assert(fabsf(out[2][0] - 1) < 1e-6 && fabsf(out[2][1] - 1) < 1e-6);
    flow_control();
    puts("PASS");
    return 0;
}
