/* Translate the Quartz Extreme fragment program captured from Tiger. */
#include <stdio.h>
#include <stdlib.h>
#include "../../hw/display/r300/r300_us.h"

int main(int argc, char **argv)
{
    static R300State st;
    FILE *f = fopen(argc > 1 ? argv[1] : "qe_state_draw1.txt", "r");
    unsigned a, v;
    R300FSDesc d;
    const char *err;

    r300_state_reset(&st);
    while (f && fscanf(f, "%x %x", &a, &v) == 2) {
        r300_state_write(&st, a, v);
    }
    for (int i = 0; i < 32; i++) d.route[i] = -1;
    d.route[0] = 0;     /* colour 0 -> t0 (RS_INST_0 COL_ADDR 0) */
    d.route[1] = 1;     /* texcoord 0 -> t1 (RS_INST_0 TEX_ADDR 1) */
    char *dis = r300_us_disasm(&st);
    fprintf(stderr, "%s", dis);
    char *msl = r300_us_to_msl(&st, &d, &err);
    if (!msl) { fprintf(stderr, "error: %s\n", err); return 1; }
    printf("%s", msl);
    fprintf(stderr, "PVS const0 float24(0x003F0000) = %g\n", r300_float24(0x003F0000));
    return 0;
}
