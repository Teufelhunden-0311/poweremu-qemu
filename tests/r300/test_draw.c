/* Assemble the first Quartz Extreme draw captured from Tiger. */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "../../hw/display/r300/r300_draw.h"

static bool no_read(void *o, uint32_t a, void *d, uint32_t l) { return false; }

int main(void)
{
    static R300State st;
    FILE *f = fopen("qe_draw1_full.txt", "r");
    char line[256];
    int mode = 0;

    r300_state_reset(&st);
    while (fgets(line, sizeof(line), f)) {
        unsigned a, v, i, d[4];
        float c[4];
        if (strstr(line, "PVS code")) { mode = 1; continue; }
        if (strstr(line, "PVS constants")) { mode = 2; continue; }
        if (mode == 0 && sscanf(line, " %x %x", &a, &v) == 2) r300_state_write(&st, a, v);
        if (mode == 1 && sscanf(line, " %u: %x %x %x %x", &i, &d[0], &d[1], &d[2], &d[3]) == 5)
            memcpy(&st.pvs_mem[i * 4], d, 16);
        if (mode == 2 && sscanf(line, " c%u %f %f %f %f", &i, &c[0], &c[1], &c[2], &c[3]) == 5)
            memcpy(&st.pvs_mem[(512 + i) * 4], c, 16);
    }
    /* First draw: VF_CNTL + 4 vertices x 12 dwords (pos, colour, texcoord). */
    uint32_t pkt3[49] = { 0x0004003D };
    float verts[4][12] = {
        { 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1 },
        { 256, 0, 0, 1, 1, 1, 1, 1, 256, 0, 0, 1 },
        { 256, 256, 0, 1, 1, 1, 1, 1, 256, 256, 0, 1 },
        { 0, 256, 0, 1, 1, 1, 1, 1, 0, 256, 0, 1 },
    };
    memcpy(&pkt3[1], verts, sizeof(verts));
    R300Arrays arr = { 0 };
    R300DrawPacket p;
    const char *err;
    bool ok = r300_draw_build(&st, &arr, 0x35, pkt3, 49, no_read, NULL, &p, &err);
    if (!ok) { printf("build failed: %s\n", err); return 1; }
    printf("rt %08x pitch %u %ux%u fmt %u scissor %u,%u-%u,%u warn %x verts %u cls %u\n",
           p.rt_gpu_addr, p.rt_pitch, p.rt_width, p.rt_height, p.rt_format,
           p.scissor[0], p.scissor[1], p.scissor[2], p.scissor[3], p.warn,
           p.num_verts, p.prim_class);
    for (unsigned i = 0; i < p.num_verts; i++) {
        R300Vertex *v = &p.verts[i];
        float wx = (v->pos[0] / v->pos[3] + 1) * p.rt_width / 2;
        float wy = (1 - v->pos[1] / v->pos[3]) * p.rt_height / 2;
        printf(" v%u win(%.1f,%.1f) col(%.2f %.2f %.2f %.2f) tc(%.3f %.3f %.3f %.3f)\n", i, wx, wy,
               v->v[0][0], v->v[0][1], v->v[0][2], v->v[0][3],
               v->v[1][0], v->v[1][1], v->v[1][2], v->v[1][3]);
    }
    printf("tex0 bound %d addr %08x %ux%u pitch %u swz %u%u%u%u swap %u out_sel %u%u%u%u rtswap %u\n",
           p.tex[0].bound, p.tex[0].gpu_addr, p.tex[0].width, p.tex[0].height,
           p.tex[0].pitch_bytes, p.uniforms.tex_swz[0][0], p.uniforms.tex_swz[0][1],
           p.uniforms.tex_swz[0][2], p.uniforms.tex_swz[0][3], p.uniforms.tex_info[0][1],
           p.uniforms.out_sel[0], p.uniforms.out_sel[1], p.uniforms.out_sel[2],
           p.uniforms.out_sel[3], p.uniforms.rt_swap32);
    r300_draw_free(&p);

    /* Early depth/stencil rejection through both packet paths (GPU and CPU
     * vertex shading): only single-sampled Z16 / Z24S8 draws get the
     * preview.  AI-generated (Claude). */
    static const struct { uint32_t aa, zf; bool keep; } ex[] = {
        { 0, 0, true }, { 0, 2, true }, { 0, 1, false }, { 0, 3, false },
        { 1, 2, false }, { 3, 0, false }, { 5, 2, false }, { 7, 2, false },
    };
    int bad = 0;
    for (int gpu = 0; gpu < 2; gpu++) {
        for (unsigned i = 0; i < sizeof(ex) / sizeof(ex[0]); i++) {
            r300_state_write(&st, 0x4020, ex[i].aa);        /* GB_AA_CONFIG */
            r300_state_write(&st, 0x4F10, ex[i].zf);        /* ZB_FORMAT */
            st.glsl_flags = R300_GLSL_EARLY_Z | (gpu ? R300_GLSL_GPU_VS : 0);
            memset(&arr, 0, sizeof(arr));
            if (!r300_draw_build(&st, &arr, 0x35, pkt3, 49, no_read, NULL, &p, &err)) {
                printf("FAIL early-Z build (%s, aa %x zf %x): %s\n", gpu ? "GPU VS" : "CPU VS",
                       ex[i].aa, ex[i].zf, err);
                bad++;
                continue;
            }
            bool pre = p.glsl && strstr(p.glsl, "r300_zpreview(zb") != NULL;
            if (pre != ex[i].keep || (p.vs_glsl != NULL) != gpu) {
                printf("FAIL early-Z %s path aa %x zf %x: preview %d vs_glsl %d\n",
                       gpu ? "GPU VS" : "CPU VS", ex[i].aa, ex[i].zf, pre, p.vs_glsl != NULL);
                bad++;
            }
            r300_draw_free(&p);
        }
    }
    st.glsl_flags = 0;
    printf("early-Z packet exclusions: %s\n", bad ? "FAIL" : "ok (GPU and CPU vertex paths)");
    return bad != 0;
}
