/* GLSL from test_us -> SPIR-V -> MSL for each stage, as the Metal backend
 * does: glsl2msl in.glsl out-prefix (writes prefix.{vs,fs,fs_z}.metal). */
#include <stdio.h>
#include <stdlib.h>
#include "../../hw/display/r300/r300_spirv.h"

int main(int argc, char **argv)
{
    FILE *f = fopen(argv[1], "rb");
    static char src[1 << 20];
    size_t n = f ? fread(src, 1, sizeof(src) - 1, f) : 0;
    static const char *sfx[] = { "vs", "fs", "fs_z" };

    if (!n) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    src[n] = 0;
    for (int s = R300_STAGE_VS; s <= R300_STAGE_FS_Z; s++) {
        char *err, path[1024];
        char *msl = r300_glsl_to_msl(src, s, &err);
        if (!msl) { fprintf(stderr, "%s: %s\n", sfx[s], err); return 1; }
        snprintf(path, sizeof(path), "%s.%s.metal", argv[2], sfx[s]);
        FILE *o = fopen(path, "w");
        fputs(msl, o);
        fclose(o);
        free(msl);
    }
    return 0;
}
