/*
 * Shader source text shared by reference (AI-generated).
 *
 * The fragment and GPU vertex shader memos (r300_us_glsl_cached,
 * gpu_vs_get) keep one immutable text per program.  A draw packet takes
 * either a reference to it (r300_glsl_share, PPCGPU_GLSL_SHARE=1) or its
 * own copy (default); either way it is released with r300_text_unref,
 * never free().  The count is atomic: packets are released on the render
 * thread while the memo, under the BQL, may drop its own reference.
 */
#ifndef R300_TEXT_H
#define R300_TEXT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

extern bool r300_glsl_share;

typedef struct R300Text {
    uint32_t ref;
    char s[];
} R300Text;

static inline R300Text *r300_text_of(char *s)
{
    return (R300Text *)(s - offsetof(R300Text, s));
}

/* A new text holding a copy of s, with one reference. */
static inline char *r300_text_new(const char *s)
{
    size_t n = strlen(s) + 1;
    R300Text *t = malloc(sizeof(*t) + n);
    t->ref = 1;
    memcpy(t->s, s, n);
    return t->s;
}

/*
 * Another reference to s, or (sharing off) a copy of it; NULL stays NULL.
 * The caller must already hold a live reference to s (a memo entry, under
 * the memo's own serialization, or a packet it owns): the relaxed
 * increment cannot revive a text whose count may reach 0 meanwhile.  It
 * does not publish the text to another thread either; the render queue's
 * hand-off does.
 */
static inline char *r300_text_ref(char *s)
{
    if (!s) {
        return NULL;
    }
    if (!r300_glsl_share) {
        return r300_text_new(s);
    }
    __atomic_fetch_add(&r300_text_of(s)->ref, 1, __ATOMIC_RELAXED);
    return s;
}

static inline void r300_text_unref(char *s)
{
    if (s && __atomic_sub_fetch(&r300_text_of(s)->ref, 1, __ATOMIC_ACQ_REL) == 0) {
        free(r300_text_of(s));
    }
}

#endif
