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
    puts("PASS");
    return 0;
}
