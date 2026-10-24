/*
 * scissorprobe: a guest program (Mac OS X 10.4, PowerPC, GLUT) that checks
 * drawing under changing scissor rectangles pixel for pixel, on a real
 * Radeon and in the emulator, with or without PPCGPU_R300_EXTENT.
 *
 *   gcc -std=gnu99 -O2 -o scissorprobe scissorprobe.c -framework GLUT -framework OpenGL
 *   ./scissorprobe [out=FILE] [hold=SECONDS]
 *
 * A  The window's back buffer with depth: 48 axis-aligned quads, each with
 *    its own colour and depth, under eight scissor rectangles of different
 *    heights (the full window among them) and alternately a full-window
 *    and an offset viewport (non-zero origin).  Read back after draw 24
 *    and at the end, against the same sequence computed on the CPU
 *    (coverage by pixel centres, viewport clipping, scissor, depth LESS).
 * B  Render-to-texture with sentinels: texture P (128x128) is cleared, drawn
 *    at full height once, then three times with small scissors; between
 *    those draws glTexSubImage2D writes a sentinel into rows outside the
 *    scissor.  Texture Q (128x32), made right after P, is cleared to its
 *    own sentinel and never drawn again.  Both read back through their
 *    framebuffer objects.
 * C  80 render targets (32x32 textures), cleared, then drawn round-robin
 *    twice with alternating full and half-height scissors, then each read
 *    back.
 * D  View-cache capacity: the window gets a quad textured with TC (an
 *    ordinary texture) and texture T0 is drawn into; then 34 framebuffer
 *    objects with their own colour texture and depth renderbuffer are each
 *    cleared and drawn with the depth test on (68 views, more than the
 *    emulator's 64-view cache); then the window gets quads textured with TC
 *    again and with T0, without a read-back in between.
 * E  A depth buffer kept for a later draw: colour texture C (fully drawn
 *    first, depth test off), then a depth-tested quad under a 16-row
 *    scissor with a fresh depth renderbuffer, then the whole of C drawn
 *    again with the depth test off.  Twice: scissor at the bottom and at
 *    the top rows (the hardware row order of a framebuffer object is not
 *    assumed).
 * Render targets are power-of-two (the R300 renders into no other
 * GL_TEXTURE_2D) and get their first contents by glClear through their
 * framebuffer object: the Tiger ATI driver did not keep glTexImage2D
 * contents of a texture first used as a render target (PowerBook,
 * 2026-10-22).
 *
 * Output: one line per check (PASS / FAIL with the first wrong pixel, and
 * an FNV-1a hash of the read-back RGB, so two runs can be compared pixel
 * for pixel even where a check fails), TOTAL, CONTROLS (incomplete
 * framebuffer objects, GL errors), RESULT;
 * exit status 1 on any failure.  Colours are exact 8-bit values, no
 * blending, no dithering.
 *
 * AI-generated (Claude, 2026-10-22).
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glext.h>
#include <stdio.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#define WW 400
#define WH 300

static FILE *out;
static const char *out_path = "scissorprobe.txt";
static int hold, nchecks, nfail, nfbo_bad, ngl_err;

static void say(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vfprintf(out, fmt, ap);
    va_end(ap);
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    fflush(out);
}

static void gl_check(const char *where)
{
    GLenum e;
    while ((e = glGetError()) != GL_NO_ERROR) {
        ngl_err++;
        say("GLERR %s 0x%04x\n", where, e);
    }
}

typedef struct { int x0, y0, x1, y1; } Rect;           /* [x0, x1) x [y0, y1) */

static int in_rect(const Rect *r, int x, int y)
{
    return x >= r->x0 && x < r->x1 && y >= r->y0 && y < r->y1;
}

/* A palette of distinct exact colours (RGBA, alpha 255). */
static void colour(int i, uint8_t c[4])
{
    c[0] = (uint8_t)(17 + 37 * i);
    c[1] = (uint8_t)(201 - 23 * i);
    c[2] = (uint8_t)(64 + 91 * (i % 3) + 5 * i);
    c[3] = 255;
}

/* FNV-1a 64 of the RGB bytes of an RGBA area. */
static unsigned long long rgb_hash(const uint8_t *p, int n)
{
    unsigned long long h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) {
        for (int c = 0; c < 3; c++) {
            h = (h ^ p[4 * i + c]) * 1099511628211ULL;
        }
    }
    return h;
}

/* Compare a read-back RGBA area with the expected one; one result line. */
static void check(const char *name, const uint8_t *got, const uint8_t *want, int w, int h)
{
    nchecks++;
    say("%-28s hash %016llx\n", name, rgb_hash(got, w * h));
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            const uint8_t *g = got + 4 * (y * w + x), *e = want + 4 * (y * w + x);
            if (memcmp(g, e, 3)) {
                int bad = 0;
                for (int k = 0; k < w * h; k++) {
                    bad += memcmp(got + 4 * k, want + 4 * k, 3) != 0;
                }
                nfail++;
                say("%-28s FAIL %d/%d pixels wrong, first (%d,%d) got %02x%02x%02x want %02x%02x%02x\n",
                    name, bad, w * h, x, y, g[0], g[1], g[2], e[0], e[1], e[2]);
                return;
            }
        }
    }
    say("%-28s PASS %d pixels\n", name, w * h);
}

/* Draw quad q (window coordinates) at depth z, with colour c, under the
 * current viewport v (glOrtho matching it, so object = window coordinates). */
static void quad(const Rect *v, const Rect *q, float z, const uint8_t c[4])
{
    glViewport(v->x0, v->y0, v->x1 - v->x0, v->y1 - v->y0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(v->x0, v->x1, v->y0, v->y1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glColor4ub(c[0], c[1], c[2], c[3]);
    glBegin(GL_QUADS);
    glVertex3f(q->x0, q->y0, z);
    glVertex3f(q->x1, q->y0, z);
    glVertex3f(q->x1, q->y1, z);
    glVertex3f(q->x0, q->y1, z);
    glEnd();
}

/* ---- A: alternating scissors in the window, depth on ---- */
static void test_a(void)
{
    static const Rect sc[8] = {
        { 0, 0, WW, WH }, { 10, 20, 210, 60 }, { 0, 0, WW, WH - 1 }, { 50, 0, 130, WH },
        { 0, 150, WW, WH }, { 300, 5, 395, 290 }, { 0, 0, WW, 1 }, { 120, 100, 280, 200 },
    };
    static const Rect vp[2] = { { 0, 0, WW, WH }, { 37, 21, 37 + 300, 21 + 230 } };
    static uint8_t want[WW * WH * 4], got[WW * WH * 4];
    static float depth[WW * WH];
    uint8_t bg[4] = { 3, 5, 7, 255 };

    glDisable(GL_DITHER);
    glDisable(GL_BLEND);
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glDisable(GL_SCISSOR_TEST);
    glViewport(0, 0, WW, WH);
    glClearColor(bg[0] / 255.0f, bg[1] / 255.0f, bg[2] / 255.0f, 1);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    for (int i = 0; i < WW * WH; i++) {
        memcpy(want + 4 * i, bg, 4);
        depth[i] = 1.0f;
    }
    glEnable(GL_SCISSOR_TEST);
    for (int i = 0; i < 48; i++) {
        const Rect *s = &sc[(i * 5) % 8], *v = &vp[(i / 3) % 2];
        Rect q = { (i * 29) % 260, (i * 17) % 180, 0, 0 };
        uint8_t c[4];
        /* z in object space -0.9 .. 0.9; window depth (z * -1 + 1) / 2 for
         * glOrtho near -1 far 1 maps object z to -z in eye space */
        float z = -0.9f + 1.8f * (float)((i * 7) % 48) / 47.0f;
        float wz = (1.0f - z) * 0.5f;
        q.x1 = q.x0 + 60 + (i * 13) % 120;
        q.y1 = q.y0 + 40 + (i * 11) % 110;
        colour(i, c);
        glScissor(s->x0, s->y0, s->x1 - s->x0, s->y1 - s->y0);
        quad(v, &q, z, c);
        for (int y = 0; y < WH; y++) {
            for (int x = 0; x < WW; x++) {
                if (in_rect(s, x, y) && in_rect(v, x, y) && in_rect(&q, x, y) &&
                    wz < depth[y * WW + x]) {
                    depth[y * WW + x] = wz;
                    memcpy(want + 4 * (y * WW + x), c, 4);
                }
            }
        }
        if (i == 23 || i == 47) {
            glDisable(GL_SCISSOR_TEST);
            glReadPixels(0, 0, WW, WH, GL_RGBA, GL_UNSIGNED_BYTE, got);
            gl_check("A readback");
            check(i == 23 ? "A window after draw 24" : "A window after draw 48", got, want, WW, WH);
            glEnable(GL_SCISSOR_TEST);
        }
    }
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_DEPTH_TEST);
}

/* ---- framebuffer objects ---- */
static GLuint tex_new(int w, int h, const uint8_t *rgba)
{
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
    return t;
}

static int fbo_bind(GLuint fbo, GLuint t, int w, int h)
{
    GLenum st;
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, t, 0);
    st = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
    if (st != GL_FRAMEBUFFER_COMPLETE_EXT) {
        nfbo_bad++;
        say("FBO incomplete 0x%04x: a failed control\n", st);
        return 0;
    }
    glViewport(0, 0, w, h);
    return 1;
}

/* fbo_bind with a depth renderbuffer as well */
static int fbo_bind_d(GLuint fbo, GLuint t, GLuint rb, int w, int h)
{
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,
                                 GL_RENDERBUFFER_EXT, rb);
    return fbo_bind(fbo, t, w, h);
}

static GLuint depth_rb(int w, int h)
{
    GLuint rb;
    glGenRenderbuffersEXT(1, &rb);
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, rb);
    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT24, w, h);
    return rb;
}

static void fill(uint8_t *p, int w, int h, const uint8_t c[4])
{
    for (int i = 0; i < w * h; i++) {
        memcpy(p + 4 * i, c, 4);
    }
}

/* ---- B: sentinels next to and inside a render target ---- */
static void test_b(void)
{
    enum { PW = 128, PH = 128, QH = 32 };
    static uint8_t p_want[PW * PH * 4], q_want[PW * QH * 4], got[PW * PH * 4], rows[PW * 11 * 4];
    uint8_t p0[4] = { 11, 22, 33, 255 }, qs[4] = { 201, 13, 77, 255 }, s2[4] = { 90, 180, 45, 255 };
    GLuint fbo[2], tp, tq;
    Rect full = { 0, 0, PW, PH };

    fill(p_want, PW, PH, p0);
    fill(q_want, PW, QH, qs);
    tp = tex_new(PW, PH, p_want);
    tq = tex_new(PW, QH, q_want);           /* made right after P */
    glGenFramebuffersEXT(2, fbo);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_DITHER);
    if (!fbo_bind(fbo[1], tq, PW, QH)) {
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
        return;
    }
    glDisable(GL_SCISSOR_TEST);
    glClearColor(qs[0] / 255.0f, qs[1] / 255.0f, qs[2] / 255.0f, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    if (!fbo_bind(fbo[0], tp, PW, PH)) {
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
        return;
    }
    glClearColor(p0[0] / 255.0f, p0[1] / 255.0f, p0[2] / 255.0f, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    gl_check("B setup");
    glEnable(GL_SCISSOR_TEST);
    for (int i = 0; i < 4; i++) {
        /* draw 0: the whole texture; 1-3: rows 0-19 and a strip */
        Rect s = i ? (Rect){ 4 * i, 0, 100 + 4 * i, 20 } : full;
        Rect q = { 0, 0, PW, PH };
        uint8_t c[4];
        colour(60 + i, c);
        glScissor(s.x0, s.y0, s.x1 - s.x0, s.y1 - s.y0);
        quad(&full, &q, 0.0f, c);
        for (int y = 0; y < PH; y++) {
            for (int x = 0; x < PW; x++) {
                if (in_rect(&s, x, y)) {
                    memcpy(p_want + 4 * (y * PW + x), c, 4);
                }
            }
        }
        if (i >= 1 && i <= 2) {
            /* a CPU write between draws into rows 50-60 (outside the scissor) */
            fill(rows, PW, 11, s2);
            glBindTexture(GL_TEXTURE_2D, tp);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 50, PW, 11, GL_RGBA, GL_UNSIGNED_BYTE, rows);
            memcpy(p_want + 4 * 50 * PW, rows, sizeof(rows));
            s2[1] += 7;
        }
    }
    glDisable(GL_SCISSOR_TEST);
    glReadPixels(0, 0, PW, PH, GL_RGBA, GL_UNSIGNED_BYTE, got);
    gl_check("B readback P");
    check("B P drawn + CPU rows", got, p_want, PW, PH);
    if (fbo_bind(fbo[1], tq, PW, QH)) {
        glReadPixels(0, 0, PW, QH, GL_RGBA, GL_UNSIGNED_BYTE, got);
        gl_check("B readback Q");
        check("B Q sentinel untouched", got, q_want, PW, QH);
    }
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    glDeleteFramebuffersEXT(2, fbo);
    glDeleteTextures(1, &tp);
    glDeleteTextures(1, &tq);
}

/* ---- C: more render targets than the emulator's 64-view cache ---- */
static void test_c(void)
{
    enum { N = 80, S = 32 };
    static GLuint tex[N], fbo[N];
    static uint8_t want[N][S * S * 4], got[S * S * 4];
    uint8_t c0[4] = { 0, 0, 0, 255 };
    Rect full = { 0, 0, S, S };
    int ok = 0;

    glGenFramebuffersEXT(N, fbo);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_SCISSOR_TEST);
    glClearColor(0, 0, 0, 1);
    for (int k = 0; k < N; k++) {
        fill(want[k], S, S, c0);
        tex[k] = tex_new(S, S, want[k]);
        if (fbo_bind(fbo[k], tex[k], S, S)) {
            glClear(GL_COLOR_BUFFER_BIT);
        }
    }
    gl_check("C setup");
    glEnable(GL_SCISSOR_TEST);
    for (int round = 0; round < 2; round++) {
        for (int k = 0; k < N; k++) {
            Rect s = (k + round) % 2 ? (Rect){ 0, 0, S, S / 2 } : full;
            Rect q = { 2 + round, 2 + round, S - 2, S - 2 };
            uint8_t c[4];
            if (!fbo_bind(fbo[k], tex[k], S, S)) {
                continue;
            }
            colour(k * 3 + round, c);
            glScissor(s.x0, s.y0, s.x1 - s.x0, s.y1 - s.y0);
            quad(&full, &q, 0.0f, c);
            for (int y = 0; y < S; y++) {
                for (int x = 0; x < S; x++) {
                    if (in_rect(&s, x, y) && in_rect(&q, x, y)) {
                        memcpy(want[k] + 4 * (y * S + x), c, 4);
                    }
                }
            }
        }
    }
    glDisable(GL_SCISSOR_TEST);
    gl_check("C drawing");
    for (int k = 0; k < N; k++) {
        char name[32];
        if (!fbo_bind(fbo[k], tex[k], S, S)) {
            continue;
        }
        glReadPixels(0, 0, S, S, GL_RGBA, GL_UNSIGNED_BYTE, got);
        snprintf(name, sizeof(name), "C target %02d", k);
        int before = nfail;
        check(name, got, want[k], S, S);
        ok += nfail == before;
    }
    gl_check("C");
    say("C %d of %d targets right\n", ok, N);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    glDeleteFramebuffersEXT(N, fbo);
    glDeleteTextures(N, tex);
}

/* A texture-replaced quad in the window at (x, y), size w x h */
static void tex_quad(GLuint t, int x, int y, int w, int h)
{
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    glViewport(0, 0, WW, WH);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WW, 0, WH, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(x, y);
    glTexCoord2f(1, 0); glVertex2f(x + w, y);
    glTexCoord2f(1, 1); glVertex2f(x + w, y + h);
    glTexCoord2f(0, 1); glVertex2f(x, y + h);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

/* ---- D: more pinned render-target views than the cache holds ---- */
static void test_d(void)
{
    enum { N = 34, S = 64, T = 64 };
    static GLuint tex[N], rb[N], fbo[N + 1];
    static uint8_t want[N][S * S * 4], got[WW * WH * 4], t0want[T * T * 4];
    uint8_t c0[4] = { 0, 0, 0, 255 }, ct[4] = { 17, 199, 83, 255 };
    Rect full = { 0, 0, S, S }, tfull = { 0, 0, T, T };
    static uint8_t tcwant[T * T * 4], wgot[T * T * 4];
    GLuint t0, tc;

    glGenFramebuffersEXT(N + 1, fbo);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_DITHER);
    /* TC: a pattern from the CPU, drawn once into the window (cleared) */
    for (int y = 0; y < T; y++) {
        for (int x = 0; x < T; x++) {
            uint8_t *p = tcwant + 4 * (y * T + x);
            p[0] = 4 * x; p[1] = 4 * y; p[2] = (x ^ y) & 1 ? 200 : 30; p[3] = 255;
        }
    }
    tc = tex_new(T, T, tcwant);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    glViewport(0, 0, WW, WH);
    glClearColor(3 / 255.0f, 5 / 255.0f, 7 / 255.0f, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    tex_quad(tc, 20, 20, T, T);
    /* T0: cleared, then a quad in its middle */
    fill(t0want, T, T, c0);
    t0 = tex_new(T, T, t0want);
    if (!fbo_bind(fbo[N], t0, T, T)) {
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
        return;
    }
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    {
        Rect q = { 8, 8, 56, 56 };
        quad(&tfull, &q, 0.0f, ct);
        for (int y = 0; y < T; y++) {
            for (int x = 0; x < T; x++) {
                if (in_rect(&q, x, y)) {
                    memcpy(t0want + 4 * (y * T + x), ct, 4);
                }
            }
        }
    }
    /* 34 colour + depth targets, depth test on */
    for (int k = 0; k < N; k++) {
        uint8_t c[4];
        Rect q = { k % 7, k % 5, S - 3, S - 1 - k % 3 };
        fill(want[k], S, S, c0);
        tex[k] = tex_new(S, S, want[k]);
        rb[k] = depth_rb(S, S);
        if (!fbo_bind_d(fbo[k], tex[k], rb[k], S, S)) {
            continue;
        }
        glDisable(GL_DEPTH_TEST);
        glClearColor(0, 0, 0, 1);
        glClearDepth(1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        colour(100 + k, c);
        quad(&full, &q, 0.0f, c);
        for (int y = 0; y < S; y++) {
            for (int x = 0; x < S; x++) {
                if (in_rect(&q, x, y)) {
                    memcpy(want[k] + 4 * (y * S + x), c, 4);
                }
            }
        }
    }
    glDisable(GL_DEPTH_TEST);
    gl_check("D targets");
    /* the window again: TC (its view no longer cached) and T0 */
    tex_quad(tc, 200, 20, T, T);
    tex_quad(t0, 100, 100, T, T);
    gl_check("D window");
    glReadPixels(20, 20, T, T, GL_RGBA, GL_UNSIGNED_BYTE, wgot);
    check("D window TC before targets", wgot, tcwant, T, T);
    glReadPixels(200, 20, T, T, GL_RGBA, GL_UNSIGNED_BYTE, wgot);
    check("D window TC after targets", wgot, tcwant, T, T);
    glReadPixels(100, 100, T, T, GL_RGBA, GL_UNSIGNED_BYTE, wgot);
    gl_check("D readback window");
    check("D window textured with rendered T0", wgot, t0want, T, T);
    for (int k = 0; k < N; k++) {
        char name[32];
        if (!fbo_bind_d(fbo[k], tex[k], rb[k], S, S)) {
            continue;
        }
        glReadPixels(0, 0, S, S, GL_RGBA, GL_UNSIGNED_BYTE, got);
        snprintf(name, sizeof(name), "D target %02d", k);
        check(name, got, want[k], S, S);
    }
    gl_check("D");
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    glDeleteFramebuffersEXT(N + 1, fbo);
    glDeleteRenderbuffersEXT(N, rb);
    glDeleteTextures(N, tex);
    glDeleteTextures(1, &t0);
    glDeleteTextures(1, &tc);
}

/* ---- E: a kept depth buffer shorter than a later draw ---- */
static void test_e(void)
{
    enum { S = 128, H = 16 };
    static uint8_t want[S * S * 4], got[S * S * 4];
    Rect full = { 0, 0, S, S };

    for (int v = 0; v < 2; v++) {
        GLuint fbo, t, rb;
        uint8_t c1[4], c2[4], c3[4], c0[4] = { 0, 0, 0, 255 };
        Rect s = v ? (Rect){ 0, S - H, S, S } : (Rect){ 0, 0, S, H };
        colour(150 + 3 * v, c1);
        colour(151 + 3 * v, c2);
        colour(152 + 3 * v, c3);
        fill(want, S, S, c0);
        t = tex_new(S, S, want);
        rb = depth_rb(S, S);
        glGenFramebuffersEXT(1, &fbo);
        if (!fbo_bind_d(fbo, t, rb, S, S)) {
            glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
            continue;
        }
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_DEPTH_TEST);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        quad(&full, &full, 0.0f, c1);           /* all rows, no depth */
        glEnable(GL_SCISSOR_TEST);
        glScissor(s.x0, s.y0, s.x1 - s.x0, s.y1 - s.y0);
        glClearDepth(1.0);
        glClear(GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        quad(&full, &full, 0.0f, c2);           /* 16 rows, depth */
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_SCISSOR_TEST);
        quad(&full, &full, 0.0f, c3);           /* all rows again, no depth */
        fill(want, S, S, c3);
        gl_check("E drawing");
        glReadPixels(0, 0, S, S, GL_RGBA, GL_UNSIGNED_BYTE, got);
        gl_check("E readback");
        check(v ? "E depth kept, top rows" : "E depth kept, bottom rows", got, want, S, S);
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
        glDeleteFramebuffersEXT(1, &fbo);
        glDeleteRenderbuffersEXT(1, &rb);
        glDeleteTextures(1, &t);
    }
}

static void display(void)
{
    static int done;
    if (done) {
        return;
    }
    done = 1;
    say("scissorprobe: renderer %s, version %s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));
    test_a();
    glutSwapBuffers();
    test_b();
    test_c();
    test_d();
    test_e();
    gl_check("end");
    say("TOTAL checks %d failed %d\n", nchecks, nfail);
    say("CONTROLS framebuffer objects incomplete %d, GL errors %d\n", nfbo_bad, ngl_err);
    say("RESULT %s\n", nfail || nfbo_bad || ngl_err ? "FAIL" : "PASS");
    fclose(out);
    if (hold) {
        sleep(hold);
    }
    exit(nfail || nfbo_bad || ngl_err ? 1 : 0);
}

int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "out=", 4)) {
            out_path = argv[i] + 4;
        } else if (!strncmp(argv[i], "hold=", 5)) {
            hold = atoi(argv[i] + 5);
        }
    }
    out = fopen(out_path, "w");
    if (!out) {
        perror(out_path);
        return 2;
    }
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(WW, WH);
    glutInitWindowPosition(40, 60);
    glutCreateWindow("scissorprobe");
    glutDisplayFunc(display);
    glutMainLoop();
    return 0;
}
