/*
 * R300 shader compilation: GLSL -> SPIR-V (shaderc) -> MSL (SPIRV-Cross).
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#include "r300_spirv.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <shaderc/shaderc.h>
#ifdef R300_HAVE_SPIRV_CROSS
#include <spirv_cross_c.h>
#endif

#include "r300_us.h"

/*
 * What shapes the MSL besides the GLSL and the two libraries, in one place
 * so that r300_msl_translation_id() names all of it for the MSL disk
 * cache's key.
 */
#define R300_VULKAN_ENV     shaderc_env_version_vulkan_1_1
#define R300_SPIRV_VERSION  shaderc_spirv_version_1_3
#define R300_MSL_MAJOR      2
#define R300_MSL_MINOR      4

/* Each stage's macro; R300_FS_Z also defines R300_FS. */
static const char *const r300_stage_macro[] = { "R300_VS", "R300_FS", "R300_FS_Z" };

/* SPIR-V binding -> MSL buffer index; texture unit k is texture and
 * sampler k (binding R300_BIND_TEX0 + k). */
static const unsigned r300_vs_buffers[][2] = {
    { R300_BIND_VERTS, 0 }, { R300_BIND_MS, 1 }, { R300_BIND_VSU, 2 },
};
static const unsigned r300_fs_buffers[][2] = {
    { R300_BIND_UNIFORMS, 0 }, { R300_BIND_ZPASS, 1 }, { R300_BIND_VRAM, 2 },
    { R300_BIND_AUX, 3 },
};

const char *r300_stage_entry(R300Stage stage)
{
    static const char *names[] = { "r300_vs", "r300_fs", "r300_fs_z" };
    return names[stage];
}

char *r300_msl_translation_id(void)
{
    char buf[1024];
    unsigned v, r;
    size_t n;

    n = (size_t)snprintf(buf, sizeof(buf),
                         "GLSL -> SPIR-V: Vulkan env 0x%x, SPIR-V 0x%x, entry main, "
                         "macros %s %s %s+%s; SPIR-V -> MSL: macOS, MSL %u.%u.0, "
                         "framebuffer fetch for input attachments, entries %s %s %s; "
                         "VS buffers",
                         (unsigned)R300_VULKAN_ENV, (unsigned)R300_SPIRV_VERSION,
                         r300_stage_macro[0], r300_stage_macro[1], r300_stage_macro[2],
                         r300_stage_macro[1], R300_MSL_MAJOR, R300_MSL_MINOR,
                         r300_stage_entry(R300_STAGE_VS), r300_stage_entry(R300_STAGE_FS),
                         r300_stage_entry(R300_STAGE_FS_Z));
    for (size_t i = 0; i < sizeof(r300_vs_buffers) / sizeof(r300_vs_buffers[0]); i++) {
        n += (size_t)snprintf(buf + n, sizeof(buf) - n, " %u>%u",
                              r300_vs_buffers[i][0], r300_vs_buffers[i][1]);
    }
    n += (size_t)snprintf(buf + n, sizeof(buf) - n, "; FS buffers");
    for (size_t i = 0; i < sizeof(r300_fs_buffers) / sizeof(r300_fs_buffers[0]); i++) {
        n += (size_t)snprintf(buf + n, sizeof(buf) - n, " %u>%u",
                              r300_fs_buffers[i][0], r300_fs_buffers[i][1]);
    }
    n += (size_t)snprintf(buf + n, sizeof(buf) - n, "; FS textures %u+k>k, k < %u",
                          (unsigned)R300_BIND_TEX0, (unsigned)R300_NUM_TEX_UNITS);
    shaderc_get_spv_version(&v, &r);
    n += (size_t)snprintf(buf + n, sizeof(buf) - n, "\nshaderc: SPIR-V 0x%x rev %u", v, r);
#ifdef R300_HAVE_SPIRV_CROSS
    {
        unsigned p;
        spvc_get_version(&v, &r, &p);
        snprintf(buf + n, sizeof(buf) - n, "\nSPIRV-Cross: C API %u.%u.%u, %s", v, r, p,
                 spvc_get_commit_revision_and_timestamp());
    }
#endif
    return strdup(buf);
}

void r300_converter_functions(const void **shaderc, const void **spirv_cross)
{
    *shaderc = (const void *)shaderc_compile_into_spv;
#ifdef R300_HAVE_SPIRV_CROSS
    *spirv_cross = (const void *)spvc_compiler_compile;
#else
    *spirv_cross = NULL;
#endif
}

static char *r300_strdup_err(const char *what, const char *msg)
{
    size_t n = strlen(what) + strlen(msg ? msg : "") + 3;
    char *s = malloc(n);
    snprintf(s, n, "%s: %s", what, msg ? msg : "");
    return s;
}

uint32_t *r300_glsl_to_spirv(const char *glsl, R300Stage stage,
                             size_t *nwords, char **err)
{
    static shaderc_compiler_t compiler;
    const char *macro = r300_stage_macro[stage];
    shaderc_compile_options_t opts;
    shaderc_compilation_result_t res;
    uint32_t *words = NULL;

    *err = NULL;
    if (!compiler) {
        compiler = shaderc_compiler_initialize();
        if (!compiler) {
            *err = r300_strdup_err("shaderc", "cannot initialise the compiler");
            return NULL;
        }
    }
    opts = shaderc_compile_options_initialize();
    shaderc_compile_options_set_target_env(opts, shaderc_target_env_vulkan,
                                           R300_VULKAN_ENV);
    shaderc_compile_options_set_target_spirv(opts, R300_SPIRV_VERSION);
    shaderc_compile_options_add_macro_definition(opts, macro, strlen(macro), "1", 1);
    if (stage == R300_STAGE_FS_Z) {
        macro = r300_stage_macro[R300_STAGE_FS];
        shaderc_compile_options_add_macro_definition(opts, macro, strlen(macro), "1", 1);
    }
    res = shaderc_compile_into_spv(compiler, glsl, strlen(glsl),
                                   stage == R300_STAGE_VS ? shaderc_glsl_vertex_shader
                                                          : shaderc_glsl_fragment_shader,
                                   r300_stage_entry(stage), "main", opts);
    if (shaderc_result_get_compilation_status(res) != shaderc_compilation_status_success) {
        *err = r300_strdup_err("GLSL", shaderc_result_get_error_message(res));
    } else {
        size_t len = shaderc_result_get_length(res);
        words = malloc(len);
        memcpy(words, shaderc_result_get_bytes(res), len);
        *nwords = len / 4;
    }
    shaderc_result_release(res);
    shaderc_compile_options_release(opts);
    return words;
}

#ifdef R300_HAVE_SPIRV_CROSS
/* The pre-"_2" API: SPIRV-Cross in Linux distributions still lacks _2. */
static void msl_bind(spvc_compiler c, SpvExecutionModel model, unsigned binding,
                     unsigned buffer, unsigned texture, unsigned sampler)
{
    spvc_msl_resource_binding b;

    spvc_msl_resource_binding_init(&b);
    b.stage = model;
    b.desc_set = 0;
    b.binding = binding;
    b.msl_buffer = buffer;
    b.msl_texture = texture;
    b.msl_sampler = sampler;
    spvc_compiler_msl_add_resource_binding(c, &b);
}

char *r300_spirv_to_msl(const uint32_t *spv, size_t nwords, R300Stage stage,
                        char **err)
{
    spvc_context ctx = NULL;
    spvc_parsed_ir ir = NULL;
    spvc_compiler c = NULL;
    spvc_compiler_options o = NULL;
    const char *src = NULL;
    char *out = NULL;
    SpvExecutionModel model = stage == R300_STAGE_VS ? SpvExecutionModelVertex
                                                     : SpvExecutionModelFragment;

    *err = NULL;
    if (spvc_context_create(&ctx) != SPVC_SUCCESS) {
        *err = r300_strdup_err("SPIRV-Cross", "cannot create a context");
        return NULL;
    }
    if (spvc_context_parse_spirv(ctx, spv, nwords, &ir) != SPVC_SUCCESS ||
        spvc_context_create_compiler(ctx, SPVC_BACKEND_MSL, ir,
                                     SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &c) != SPVC_SUCCESS ||
        spvc_compiler_create_compiler_options(c, &o) != SPVC_SUCCESS) {
        goto fail;
    }
    spvc_compiler_options_set_uint(o, SPVC_COMPILER_OPTION_MSL_PLATFORM,
                                   SPVC_MSL_PLATFORM_MACOS);
    spvc_compiler_options_set_uint(o, SPVC_COMPILER_OPTION_MSL_VERSION,
                                   SPVC_MAKE_MSL_VERSION(R300_MSL_MAJOR, R300_MSL_MINOR, 0));
    /* input attachment k -> [[color(k)]] */
    spvc_compiler_options_set_bool(o, SPVC_COMPILER_OPTION_MSL_IOS_FRAMEBUFFER_FETCH_SUBPASS,
                                   SPVC_TRUE);
    if (spvc_compiler_install_compiler_options(c, o) != SPVC_SUCCESS) {
        goto fail;
    }
    if (stage == R300_STAGE_VS) {
        for (size_t i = 0; i < sizeof(r300_vs_buffers) / sizeof(r300_vs_buffers[0]); i++) {
            msl_bind(c, model, r300_vs_buffers[i][0], r300_vs_buffers[i][1], 0, 0);
        }
    } else {
        for (size_t i = 0; i < sizeof(r300_fs_buffers) / sizeof(r300_fs_buffers[0]); i++) {
            msl_bind(c, model, r300_fs_buffers[i][0], r300_fs_buffers[i][1], 0, 0);
        }
        for (unsigned k = 0; k < R300_NUM_TEX_UNITS; k++) {
            msl_bind(c, model, R300_BIND_TEX0 + k, 0, k, k);
        }
    }
    if (spvc_compiler_rename_entry_point(c, "main", r300_stage_entry(stage),
                                         model) != SPVC_SUCCESS ||
        spvc_compiler_compile(c, &src) != SPVC_SUCCESS) {
        goto fail;
    }
    out = strdup(src);
    spvc_context_destroy(ctx);
    return out;

fail:
    *err = r300_strdup_err("SPIRV-Cross", spvc_context_get_last_error_string(ctx));
    spvc_context_destroy(ctx);
    return NULL;
}
#else
/* Only the Metal backend (macOS) wants MSL; elsewhere SPIRV-Cross is not
 * linked. */
char *r300_spirv_to_msl(const uint32_t *spv, size_t nwords, R300Stage stage,
                        char **err)
{
    *err = r300_strdup_err("SPIRV-Cross", "not built in (macOS only)");
    return NULL;
}
#endif

char *r300_glsl_to_msl(const char *glsl, R300Stage stage, char **err)
{
    size_t n;
    uint32_t *spv = r300_glsl_to_spirv(glsl, stage, &n, err);
    char *msl;

    if (!spv) {
        return NULL;
    }
    msl = r300_spirv_to_msl(spv, n, stage, err);
    free(spv);
    return msl;
}
