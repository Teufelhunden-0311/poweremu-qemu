/*
 * rbprobe: what glReadPixels (and its relatives) return for pixels whose
 * values are known exactly, so a real Radeon and the emulator can be
 * compared.  Build on Tiger:
 *   gcc -std=gnu99 -O2 -o rbprobe rbprobe.c -framework GLUT -framework OpenGL
 * Writes /tmp/rbprobe.txt, then shows the same text on screen as a strip of
 * grey blocks (one byte each) so it can be read from a lossless screenshot
 * when readback itself is what is broken.  readstrip.py decodes it.
 *
 * Scene: clear to CLR, then a 64x64 pattern of single-pixel points at the
 * window's bottom left, pixel (x,y) = pat(x,y).  Blend, dither, texturing,
 * alpha/depth/stencil tests and scissor are off; colour masks full.
 */
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glext.h>
#include <stdio.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

static int WW = 640, WH = 600;           /* "big": 1100 x 1100 */
#define PN 64                           /* pattern size */
#define GUARD 64                        /* sentinel bytes before and after */
#define SENT 0xA5

static const uint8_t CLR[4] = { 0x40, 0x80, 0xC0, 0x60 };
static GLint abits;                      /* alpha bits of the window */
static char txt[16384];
static int ntxt;
static FILE *out;

static void say(const char *fmt, ...)
{
    va_list ap;
    char b[512];
    va_start(ap, fmt);
    int n = vsnprintf(b, sizeof(b), fmt, ap);
    va_end(ap);
    if (n > (int)sizeof(b) - 1) {
        n = sizeof(b) - 1;
    }
    if (out) {
        fputs(b, out);
    }
    if (ntxt + n < (int)sizeof(txt)) {
        memcpy(txt + ntxt, b, n);
        ntxt += n;
    }
}

static int pat2, capmode, maskmode;                        /* second pattern: every byte ^ 0x5A */

static void pat(int x, int y, uint8_t c[4])
{
    c[0] = (uint8_t)(4 * x + 1);
    c[1] = (uint8_t)(4 * y + 2);
    c[2] = (uint8_t)(x * 7 + y * 13);
    c[3] = (uint8_t)(x * 3 + y * 5 + 7);
    if (pat2) {
        for (int k = 0; k < 4; k++) {
            c[k] ^= 0x5A;
        }
    }
}

/* expected RGBA at window pixel (x,y) */
static void expect(int x, int y, uint8_t c[4])
{
    if (x < PN && y < PN) {
        pat(x, y, c);
    } else {
        memcpy(c, CLR, 4);
    }
    if (abits == 0) {
        c[3] = 0xFF;
    }
}

static double now(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static void ortho(void)
{
    glViewport(0, 0, WW, WH);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WW, 0, WH, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

static void plain_state(void)
{
    glDisable(GL_BLEND);
    glDisable(GL_DITHER);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_ALPHA_TEST);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_LIGHTING);
    glDisable(GL_FOG);
    glShadeModel(GL_FLAT);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    glPixelStorei(GL_PACK_SKIP_ROWS, 0);
    glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_PACK_SWAP_BYTES, GL_FALSE);
}

static void draw_scene(void)
{
    ortho();
    plain_state();
    glDrawBuffer(GL_BACK);
    glClearColor(CLR[0] / 255.0f, CLR[1] / 255.0f, CLR[2] / 255.0f, CLR[3] / 255.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POINTS);
    for (int y = 0; y < PN; y++) {
        for (int x = 0; x < PN; x++) {
            uint8_t c[4];
            pat(x, y, c);
            glColor4ub(c[0], c[1], c[2], c[3]);
            glVertex2f(x + 0.5f, y + 0.5f);
        }
    }
    glEnd();
}

static uint8_t *sentbuf(int n)
{
    uint8_t *b = malloc(n + 2 * GUARD);
    memset(b, SENT, n + 2 * GUARD);
    return b;
}

static int guards_ok(const uint8_t *b, int n)
{
    for (int i = 0; i < GUARD; i++) {
        if (b[i] != SENT || b[GUARD + n + i] != SENT) {
            return 0;
        }
    }
    return 1;
}

/* RGBA/UNSIGNED_BYTE read of w x h at (x0,y0) from the current read buffer,
   compared with expect() */
static void read_rgba(const char *name, int x0, int y0, int w, int h, int finish)
{
    int n = w * h * 4;
    uint8_t *b = sentbuf(n), *d = b + GUARD;
    if (finish) {
        glFinish();
    }
    GLenum e0 = glGetError();
    double t = now();
    glReadPixels(x0, y0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, d);
    t = now() - t;
    GLenum e1 = glGetError();
    int untouched = 0, zero = 0, exact = 0, near = 0, first = -1;
    for (int i = 0; i < w * h; i++) {
        uint8_t ex[4], *p = d + 4 * i;
        expect(x0 + i % w, y0 + i / w, ex);
        if (p[0] == SENT && p[1] == SENT && p[2] == SENT && p[3] == SENT) {
            untouched++;
        }
        if (!(p[0] | p[1] | p[2] | p[3])) {
            zero++;
        }
        int dmax = 0;
        for (int k = 0; k < 4; k++) {
            int dd = abs(p[k] - ex[k]);
            dmax = dd > dmax ? dd : dmax;
        }
        if (dmax == 0) {
            exact++;
        } else if (first < 0) {
            first = i;
        }
        if (dmax <= 4) {
            near++;
        }
    }
    say("%s: %dx%d@%d,%d err %#x/%#x %.2fms px %d exact %d near %d zero %d untouched %d guards %s\n",
        name, w, h, x0, y0, e0, e1, t * 1e3, w * h, exact, near, zero, untouched,
        guards_ok(b, n) ? "ok" : "HIT");
    say("  first px %02x%02x%02x%02x", d[0], d[1], d[2], d[3]);
    if (w * h > 1) {
        say(" %02x%02x%02x%02x", d[4], d[5], d[6], d[7]);
    }
    if (first >= 0) {
        uint8_t ex[4], *p = d + 4 * first;
        expect(x0 + first % w, y0 + first / w, ex);
        say("  1st miss (%d,%d) got %02x%02x%02x%02x want %02x%02x%02x%02x",
            x0 + first % w, y0 + first / w, p[0], p[1], p[2], p[3], ex[0], ex[1], ex[2], ex[3]);
    }
    say("\n");
    free(b);
}

/* BGRA as packed 32-bit words: word = A<<24 | R<<16 | G<<8 | B */
static void read_bgra_rev(void)
{
    int w = PN, h = PN, n = w * h * 4;
    uint8_t *b = sentbuf(n);
    uint32_t *d = (uint32_t *)(b + GUARD);
    glFinish();
    GLenum e0 = glGetError();
    glReadPixels(0, 0, w, h, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, d);
    GLenum e1 = glGetError();
    int exact = 0, zero = 0;
    for (int i = 0; i < w * h; i++) {
        uint8_t ex[4];
        expect(i % w, i / w, ex);
        uint32_t want = (uint32_t)ex[3] << 24 | ex[0] << 16 | ex[1] << 8 | ex[2];
        exact += d[i] == want;
        zero += d[i] == 0;
    }
    GLint sw = -1;
    glGetIntegerv(GL_PACK_SWAP_BYTES, &sw);
    say("BGRA/8888_REV words: 64x64 err %#x/%#x exact %d zero %d guards %s swap_bytes %d first %08x %08x\n",
        e0, e1, exact, zero, guards_ok(b, n) ? "ok" : "HIT", sw, d[0], d[1]);
    free(b);
}

/* RGB width 3: rows of 9 bytes, padded to 12 under PACK_ALIGNMENT 4 */
static void read_rgb3(int align)
{
    int w = 3, h = 4, stride = align == 4 ? 12 : 9, n = stride * h;
    uint8_t *b = sentbuf(n), *d = b + GUARD;
    glPixelStorei(GL_PACK_ALIGNMENT, align);
    glFinish();
    GLenum e0 = glGetError();
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, d);
    GLenum e1 = glGetError();
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    int exact = 0, pad_sent = 0, pad = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uint8_t ex[4], *p = d + y * stride + 3 * x;
            expect(x, y, ex);
            exact += !memcmp(p, ex, 3);
        }
        for (int k = 9; k < stride; k++) {
            pad++;
            pad_sent += d[y * stride + k] == SENT;
        }
    }
    say("RGB 3x4 align %d: err %#x/%#x exact %d/12 pad bytes %d still sentinel %d guards %s\n",
        align, e0, e1, exact, pad, pad_sent, guards_ok(b, n) ? "ok" : "HIT");
    free(b);
}

static GLuint copytex;                  /* shown on the results screen */

/* glCopyTexImage2D from BACK, then glGetTexImage; and a plain upload control */
static void tex_paths(void)
{
    int n = PN * PN * 4;
    uint8_t *b = sentbuf(n), *d = b + GUARD;

    glGenTextures(1, &copytex);
    glBindTexture(GL_TEXTURE_2D, copytex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glReadBuffer(GL_BACK);
    glFinish();
    GLenum e0 = glGetError();
    glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 0, 0, PN, PN, 0);
    GLenum e1 = glGetError();
    glFinish();
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, d);
    GLenum e2 = glGetError();
    int exact = 0, zero = 0;
    for (int i = 0; i < PN * PN; i++) {
        uint8_t ex[4];
        expect(i % PN, i / PN, ex);
        exact += !memcmp(d + 4 * i, ex, 4);
        zero += !(d[4 * i] | d[4 * i + 1] | d[4 * i + 2] | d[4 * i + 3]);
    }
    say("CopyTexImage(BACK)+GetTexImage: err %#x/%#x/%#x exact %d/4096 zero %d guards %s first %02x%02x%02x%02x\n",
        e0, e1, e2, exact, zero, guards_ok(b, n) ? "ok" : "HIT", d[0], d[1], d[2], d[3]);

    /* control: upload the pattern itself, read it back */
    uint8_t *src = malloc(n);
    for (int i = 0; i < PN * PN; i++) {
        expect(i % PN, i / PN, src + 4 * i);
    }
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, PN, PN, 0, GL_RGBA, GL_UNSIGNED_BYTE, src);
    memset(b, SENT, n + 2 * GUARD);
    glFinish();
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, d);
    GLenum e3 = glGetError();
    say("Upload+GetTexImage control: err %#x exact %d/4096 guards %s\n",
        e3, memcmp(d, src, n) ? -1 : 4096, guards_ok(b, n) ? "ok" : "HIT");
    glDeleteTextures(1, &t);
    free(src);
    free(b);
}

static void depth_read(void)
{
    GLint db = 0;
    glGetIntegerv(GL_DEPTH_BITS, &db);
    if (!db) {
        say("depth: no depth buffer\n");
        return;
    }
    ortho();
    plain_state();
    glClearDepth(0.25);
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_ALWAYS);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glBegin(GL_QUADS);                  /* z 0 in ortho(-1,1) -> depth 0.5 */
    glVertex3f(0, 0, 0); glVertex3f(32, 0, 0); glVertex3f(32, 32, 0); glVertex3f(0, 32, 0);
    glEnd();
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDisable(GL_DEPTH_TEST);
    glFinish();
    float z[2] = { -1, -1 };
    GLenum e0 = glGetError();
    glReadPixels(4, 4, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &z[0]);
    glReadPixels(48, 48, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &z[1]);
    GLenum e1 = glGetError();
    say("depth (%d bits): err %#x/%#x inside quad %.4f (want 0.5) outside %.4f (want 0.25)\n",
        db, e0, e1, z[0], z[1]);
}

/* ---- results as grey blocks: magenta 8x8 anchor at the window's top left,
   then 128 blocks of 4x4 px per row starting 16 px below the top: bytes
   0..255 (calibration), 4-byte big-endian length, the text, 4-byte BE sum.
   The CopyTexImage texture is drawn 64x64 at x 528, 16 px below the top. */
#define SB 4
#define SROW 128

static uint8_t strip[20000];
static int nstrip;

static void build_strip(void)
{
    nstrip = 0;
    for (int i = 0; i < 256; i++) {
        strip[nstrip++] = (uint8_t)i;
    }
    uint32_t sum = 0;
    strip[nstrip++] = ntxt >> 24; strip[nstrip++] = ntxt >> 16;
    strip[nstrip++] = ntxt >> 8; strip[nstrip++] = ntxt;
    for (int i = 0; i < ntxt && nstrip < (int)sizeof(strip) - 4; i++) {
        strip[nstrip++] = (uint8_t)txt[i];
        sum += (uint8_t)txt[i];
    }
    strip[nstrip++] = sum >> 24; strip[nstrip++] = sum >> 16;
    strip[nstrip++] = sum >> 8; strip[nstrip++] = sum;
}

static void show_strip(void)
{
    ortho();
    plain_state();
    glDrawBuffer(GL_BACK);
    glClearColor(10 / 255.0f, 250 / 255.0f, 10 / 255.0f, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3ub(255, 0, 255);
    glRecti(0, WH - 8, 8, WH);
    for (int i = 0; i < nstrip; i++) {
        int x = (i % SROW) * SB, top = WH - 16 - (i / SROW) * SB;
        glColor3ub(strip[i], strip[i], strip[i]);
        glRecti(x, top - SB, x + SB, top);
    }
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, copytex);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glBegin(GL_QUADS);                  /* texel (0,0) at the bottom left, as read */
    glTexCoord2f(0, 0); glVertex2i(528, WH - 16 - PN);
    glTexCoord2f(1, 0); glVertex2i(528 + PN, WH - 16 - PN);
    glTexCoord2f(1, 1); glVertex2i(528 + PN, WH - 16);
    glTexCoord2f(0, 1); glVertex2i(528, WH - 16);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glutSwapBuffers();
    glFinish();
}

#ifndef GL_TEXTURE_RECTANGLE_EXT
#define GL_TEXTURE_RECTANGLE_EXT 0x84F5
#endif
#ifndef GL_UNPACK_CLIENT_STORAGE_APPLE
#define GL_UNPACK_CLIENT_STORAGE_APPLE 0x85B2
#endif
#ifndef GL_TEXTURE_STORAGE_HINT_APPLE
#define GL_TEXTURE_STORAGE_HINT_APPLE 0x85BC
#define GL_STORAGE_SHARED_APPLE 0x85BF
#endif

/* pattern 3, for the client texture: (255-4x, 4y+1, x^y, 255) */
static void pat3(int x, int y, uint8_t c[4])
{
    c[0] = (uint8_t)(255 - 4 * x);
    c[1] = (uint8_t)(4 * y + 1);
    c[2] = (uint8_t)(x ^ y);
    c[3] = 255;
}

static void client_tex(void)
{
    const char *ext = (const char *)glGetString(GL_EXTENSIONS);
    int range = ext && strstr(ext, "GL_APPLE_texture_range") != NULL;
    int cs = ext && strstr(ext, "GL_APPLE_client_storage") != NULL;
    uint32_t *px = valloc(PN * PN * 4);
    for (int y = 0; y < PN; y++) {
        for (int x = 0; x < PN; x++) {
            uint8_t c[4];
            pat3(x, y, c);
            px[y * PN + x] = (uint32_t)c[3] << 24 | c[0] << 16 | c[1] << 8 | c[2];
        }
    }
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_RECTANGLE_EXT, t);
    if (range) {
        glTextureRangeAPPLE(GL_TEXTURE_RECTANGLE_EXT, PN * PN * 4, px);
        glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_STORAGE_HINT_APPLE,
                        GL_STORAGE_SHARED_APPLE);
    }
    glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_CLIENT_STORAGE_APPLE, cs ? GL_TRUE : GL_FALSE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_RECTANGLE_EXT, 0, GL_RGBA8, PN, PN, 0, GL_BGRA,
                 GL_UNSIGNED_INT_8_8_8_8_REV, px);
    glPixelStorei(GL_UNPACK_CLIENT_STORAGE_APPLE, GL_FALSE);
    GLenum e0 = glGetError();

    ortho();
    plain_state();
    glClearColor(CLR[0] / 255.0f, CLR[1] / 255.0f, CLR[2] / 255.0f, CLR[3] / 255.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_RECTANGLE_EXT);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2i(0, 0);
    glTexCoord2f(PN, 0); glVertex2i(PN, 0);
    glTexCoord2f(PN, PN); glVertex2i(PN, PN);
    glTexCoord2f(0, PN); glVertex2i(0, PN);
    glEnd();
    glDisable(GL_TEXTURE_RECTANGLE_EXT);
    glFinish();
    uint8_t *b = sentbuf(PN * PN * 4), *d = b + GUARD;
    glReadBuffer(GL_BACK);
    glReadPixels(0, 0, PN, PN, GL_RGBA, GL_UNSIGNED_BYTE, d);
    GLenum e1 = glGetError();
    int exact = 0, first = -1;
    for (int i = 0; i < PN * PN; i++) {
        uint8_t ex[4];
        pat3(i % PN, i / PN, ex);
        if (!memcmp(d + 4 * i, ex, 4)) {
            exact++;
        } else if (first < 0) {
            first = i;
        }
    }
    say("CLIENT texture (range %d, client storage %d): err %#x/%#x exact %d/4096",
        range, cs, e0, e1, exact);
    if (first >= 0) {
        uint8_t ex[4], *q = d + 4 * first;
        pat3(first % PN, first / PN, ex);
        say("  1st miss (%d,%d) got %02x%02x%02x%02x want %02x%02x%02x%02x", first % PN,
            first / PN, q[0], q[1], q[2], q[3], ex[0], ex[1], ex[2], ex[3]);
    }
    say("\n");
    glDeleteTextures(1, &t);
    free(b);
    free(px);
}

static void run_tests(void)
{
    GLint r = 0, g = 0, bl = 0, db = 0, sb = 0, ms = 0;
    glGetIntegerv(GL_RED_BITS, &r);
    glGetIntegerv(GL_GREEN_BITS, &g);
    glGetIntegerv(GL_BLUE_BITS, &bl);
    glGetIntegerv(GL_ALPHA_BITS, &abits);
    glGetIntegerv(GL_DEPTH_BITS, &db);
    glGetIntegerv(GL_STENCIL_BITS, &sb);
    glGetIntegerv(GL_SAMPLE_BUFFERS_ARB, &ms);
    say("rbprobe 2\nGL_RENDERER %s\nGL_VERSION %s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));
    say("bits R%d G%d B%d A%d depth %d stencil %d samplebuffers %d\n", r, g, bl, abits, db, sb, ms);
    say("CLR %02x%02x%02x%02x; pat(x,y) = (4x+1, 4y+2, 7x+13y, 3x+5y+7) for x,y < 64\n",
        CLR[0], CLR[1], CLR[2], CLR[3]);

    /* 1. BACK before swap, RGBA/UB, several sizes */
    draw_scene();
    glReadBuffer(GL_BACK);
    read_rgba("BACK 1x1 finish", 10, 20, 1, 1, 1);
    read_rgba("BACK 4x4 finish", 8, 8, 4, 4, 1);
    read_rgba("BACK 64x64 finish", 0, 0, PN, PN, 1);
    read_rgba("BACK 64x64 again", 0, 0, PN, PN, 1);
    read_rgba("BACK 96x80 finish", 0, 0, 96, 80, 1);
    read_rgba("BACK 256x128 finish", 0, 0, 256, 128, 1);
    draw_scene();
    glReadBuffer(GL_BACK);
    read_rgba("BACK 64x64 nofinish", 0, 0, PN, PN, 0);
    draw_scene();
    glReadBuffer(GL_BACK);
    glFlush();
    read_rgba("BACK 4x4 flush-only", 8, 8, 4, 4, 0);

    /* 2. other formats, same scene */
    draw_scene();
    glReadBuffer(GL_BACK);
    read_bgra_rev();
    read_rgb3(4);
    read_rgb3(1);

    /* 3. FRONT after swap, freshly rendered */
    draw_scene();
    glutSwapBuffers();
    glFinish();
    glReadBuffer(GL_FRONT);
    read_rgba("FRONT 64x64 after swap", 0, 0, PN, PN, 1);
    read_rgba("FRONT 96x80 after swap", 0, 0, 96, 80, 1);

    /* 4. texture paths */
    draw_scene();
    tex_paths();

    /* 5. depth */
    depth_read();

    /* 6. consecutive readbacks, different patterns and sizes, odd origins */
    pat2 = 1;
    draw_scene();
    glReadBuffer(GL_BACK);
    read_rgba("SEQ pattern2 64x64", 0, 0, PN, PN, 1);
    pat2 = 0;
    draw_scene();
    glReadBuffer(GL_BACK);
    read_rgba("SEQ pattern1 37x45@5,3", 5, 3, 37, 45, 1);
    read_rgba("SEQ pattern1 100x70@3,1", 3, 1, 100, 70, 1);

    /* 7. a client-storage texture (Apple: drawn from the program's memory) */
    client_tex();

    /* 8. "big": staging capacity, exactly 4 MiB and just above */
    if (maskmode) {     /* emulator hook: readback draws cover the left half, R only */
        static uint8_t ra[PN * PN * 4], rb[PN * PN * 4];
        pat2 = 1;
        draw_scene();
        glReadBuffer(GL_BACK);
        glFinish();
        glReadPixels(0, 0, PN, PN, GL_RGBA, GL_UNSIGNED_BYTE, ra);
        pat2 = 0;
        draw_scene();
        glReadBuffer(GL_BACK);
        glFinish();
        glReadPixels(0, 0, PN, PN, GL_RGBA, GL_UNSIGNED_BYTE, rb);
        int lr = 0, lgba = 0, rsame = 0, rr1 = 0, n = 0;
        for (int y = 0; y < PN; y++) {
            for (int x = 0; x < PN; x++, n++) {
                uint8_t ex[4], *a = ra + 4 * n, *b = rb + 4 * n;
                expect(x, y, ex);
                if (x < PN / 2) {
                    lr += b[0] == ex[0];
                    lgba += b[1] == a[1] && b[2] == a[2] && b[3] == a[3];
                } else {
                    rsame += !memcmp(a, b, 4);
                    rr1 += b[0] == ex[0];
                }
            }
        }
        say("MASK left half (2048 px): R = pattern1 %d, G/B/A kept from read 1 %d; "
            "right half (2048 px): unchanged %d, R = pattern1 %d\n", lr, lgba, rsame, rr1);
        say("  read1 px0 %02x%02x%02x%02x read2 px0 %02x%02x%02x%02x px40 %02x%02x%02x%02x -> %02x%02x%02x%02x\n",
            ra[0], ra[1], ra[2], ra[3], rb[0], rb[1], rb[2], rb[3],
            ra[160], ra[161], ra[162], ra[163], rb[160], rb[161], rb[162], rb[163]);
    }
    if (capmode) {                      /* emulator staging limited to 256 x 101 x 4 */
        draw_scene();
        glReadBuffer(GL_BACK);
        read_rgba("CAP 256x100 (target = limit)", 0, 0, 256, 100, 1);
        read_rgba("CAP 256x101 (target = limit + 1 row)", 0, 0, 256, 101, 1);
        read_rgba("CAP 64x64 after the drop", 0, 0, PN, PN, 1);
    }
    if (WW >= 1100) {
        draw_scene();
        glReadBuffer(GL_BACK);
        read_rgba("BIG 1024x1023 (target 4 MiB)", 0, 0, 1024, 1023, 1);
        read_rgba("BIG 1024x1024 (target 4 MiB + 4 KiB)", 0, 0, 1024, 1024, 1);
    }
    say("DONE\n");
}

static int phase;
static void quit_cb(int v) { exit(0); }

static void display(void)
{
    if (phase == 0) {
        phase = 1;
        out = fopen("/tmp/rbprobe.txt", "w");
        run_tests();
        if (out) {
            fclose(out);
        }
        build_strip();
        glutTimerFunc(40000, quit_cb, 0);       /* ~40 s on screen for a screenshot */
    }
    show_strip();
}

int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    if (argc > 1 && !strcmp(argv[1], "big")) {
        WW = WH = 1100;
    }
    capmode = argc > 1 && !strcmp(argv[1], "cap");
    maskmode = argc > 1 && !strcmp(argv[1], "mask");
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(WW, WH);
    glutInitWindowPosition(40, 60);
    glutCreateWindow("rbprobe");
    glutDisplayFunc(display);
    glutMainLoop();
    return 0;
}
