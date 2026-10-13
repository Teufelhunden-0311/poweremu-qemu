/*
 * texprobe: mip-mapped textures whose every texel names its level, face
 * and place, drawn one level at a time at one texel per pixel and read
 * back, so a real Radeon and the emulator can be compared texel by texel
 * (and so each level can be found again in a dump of the card's memory).
 * Build on Tiger:
 *   gcc -std=gnu99 -O2 -o texprobe texprobe.c -framework GLUT -framework OpenGL
 * Run: texprobe [stage=N] [only=ID] [hold=SECONDS] [wait=SECONDS] [list]
 * Writes /tmp/texprobe.txt.  tags.py computes the same texels.
 *
 * A level is drawn with GL_NEAREST_MIPMAP_NEAREST at one texel per pixel,
 * so the card chooses the level itself.  When what comes back is another
 * level's texels the draw is repeated with a LOD bias, and the bias that
 * was needed is reported.
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

#define WW 560
#define WH 560
#define OX 16
#define OY 16
#define TILE 256
#define SUBSALT 77

enum { K_RGBA8, K_1555, K_565, K_4444, K_DXT1, K_DXT3, K_DXT5,
       K_A8, K_L8, K_I8, K_AL88 };
static const char *const kname[] = { "rgba8", "1555", "565", "4444", "dxt1",
                                     "dxt3", "dxt5", "a8", "l8", "i8", "al88" };

typedef struct {
    int id, kind, w, h, cube, sub, stage;
    /* filled in */
    GLuint name;
    int levels;
    int nsub;
    struct { int lv, x, y, w, h; } subr[2];
} Tex;

static Tex cases[] = {
    /* id kind     w     h  cube sub stage */
    {  0, K_RGBA8,  256,  256, 0, 1, 1 },
    {  1, K_RGBA8,   16,   16, 0, 0, 1 },
    {  2, K_RGBA8,    4,    4, 0, 0, 1 },
    {  3, K_RGBA8, 1024,    4, 0, 1, 1 },   /* one-row levels many microtiles wide */
    {  4, K_RGBA8,    4, 1024, 0, 0, 1 },
    {  5, K_RGBA8, 2048,    8, 0, 0, 1 },   /* wider than a macrotile, not as tall */
    {  6, K_RGBA8,    8, 2048, 0, 0, 1 },
    {  7, K_RGBA8,   64,   64, 1, 0, 1 },
    {  8, K_RGBA8,    4,    4, 1, 0, 1 },
    {  9, K_1555,   256,  256, 0, 1, 1 },
    { 10, K_1555,  1024,    4, 0, 0, 1 },
    { 11, K_1555,     4, 1024, 0, 0, 1 },
    { 12, K_1555,    64,   64, 1, 0, 1 },
    { 13, K_DXT1,   256,  256, 0, 1, 1 },
    { 14, K_DXT1,  1024,  256, 0, 0, 1 },
    { 15, K_DXT1,   256, 1024, 0, 0, 1 },
    { 16, K_DXT1,    64,   64, 1, 0, 1 },
    { 17, K_DXT3,   512,  256, 0, 0, 1 },
    { 18, K_DXT5,   256,  256, 0, 0, 1 },
    { 19, K_DXT5,    64,   64, 1, 0, 1 },
    { 20, K_565,    256,  256, 0, 0, 1 },
    { 21, K_4444,   256,  256, 0, 0, 1 },
    { 22, K_RGBA8,    1,    1, 0, 0, 1 },
    { 23, K_RGBA8,    2,    2, 0, 0, 1 },
    { 24, K_RGBA8,  128,    2, 0, 0, 1 },
    { 25, K_1555,  2048,    8, 0, 0, 1 },
    { 26, K_1555,     8, 2048, 0, 0, 1 },
    { 27, K_DXT1,  2048,    8, 0, 0, 1 },
    { 28, K_DXT3,   256, 1024, 0, 0, 1 },
    { 29, K_DXT3,    64,   64, 1, 0, 1 },
    /* stage 2: eight-bit texels and two-byte pairs */
    { 40, K_A8,     256,  256, 0, 1, 2 },
    { 41, K_A8,      16,   16, 0, 0, 2 },
    { 42, K_A8,    1024,    4, 0, 0, 2 },
    { 43, K_A8,       4, 1024, 0, 0, 2 },
    { 44, K_L8,     256,  256, 0, 0, 2 },
    { 45, K_I8,      64,   64, 1, 0, 2 },
    { 46, K_AL88,   256,  256, 0, 0, 2 },
    { 47, K_AL88,  1024,    4, 0, 0, 2 },
    { 48, K_A8,    2048,   32, 0, 0, 2 },
    { 49, K_A8,      32, 2048, 0, 0, 2 },
    { 50, K_L8,       4,    4, 0, 0, 2 },
    { 51, K_A8,       4,    4, 1, 0, 2 },
};
#define NCASES ((int)(sizeof(cases) / sizeof(cases[0])))

static FILE *out;
static GLint abits, maxsize;
static int stage = 1, only = -1, hold = 4, waits;
static int n_levels, n_ok, n_bad, n_biased;

static void say(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    if (out) {
        vfprintf(out, fmt, ap);
        fflush(out);
    }
    va_end(ap);
}

static int dim(int d, int lv)
{
    d >>= lv;
    return d < 1 ? 1 : d;
}

static int is_dxt(int k)
{
    return k == K_DXT1 || k == K_DXT3 || k == K_DXT5;
}

/* Eight bits that look unrelated from one texel, level, face or salt to
 * the next, so that a row of them is found only where it was put. */
static int hash8(int lv, int face, int x, int y, int salt)
{
    uint32_t h = (uint32_t)(x + 1) * 2654435761u ^ (uint32_t)(y + 1) * 2246822519u ^
                 (uint32_t)(lv * 6 + face + 1) * 3266489917u ^ (uint32_t)salt * 668265263u;

    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    return h & 0xFF;
}

/*
 * The texel (for DXT: the block) at x, y of a level: the bits to upload
 * (v[0], and v[1..3] for the longer DXT blocks), what a read of the
 * frame buffer should give, and which of its bits to compare.
 */
static void texel(int kind, int salt, int face, int lv, int x, int y,
                  uint32_t v[4], uint8_t want[4], uint8_t mask[4])
{
    int r, g, b, a;

    v[0] = v[1] = v[2] = v[3] = 0;
    switch (kind) {
    case K_RGBA8:
        r = (lv << 4) | (face << 1) | 1;
        g = (x + 7 * salt) & 255;
        b = (y + 13 * salt) & 255;
        a = 0x40 | (((x >> 8) & 7) << 3) | ((y >> 8) & 7);
        v[0] = (uint32_t)a << 24 | r << 16 | g << 8 | b;
        want[0] = r; want[1] = g; want[2] = b; want[3] = a;
        mask[0] = mask[1] = mask[2] = mask[3] = 0xFF;
        break;
    case K_1555:
        r = (lv + 1) & 31;
        g = (x + 5 * face + 7 * salt) & 31;
        b = (y + 11 * face + 13 * salt) & 31;
        a = ((x >> 5) ^ (y >> 5) ^ 1) & 1;
        v[0] = a << 15 | r << 10 | g << 5 | b;
        want[0] = r << 3; want[1] = g << 3; want[2] = b << 3; want[3] = a ? 255 : 0;
        mask[0] = mask[1] = mask[2] = 0xF8; mask[3] = 0x80;
        break;
    case K_565:
        r = (lv + 1) & 31;
        g = ((x + 7 * salt) & 31) << 1;         /* five bits: may be kept as 1555 */
        b = (y + 13 * salt) & 31;
        v[0] = r << 11 | g << 5 | b;
        want[0] = r << 3; want[1] = g << 2; want[2] = b << 3; want[3] = 255;
        mask[0] = mask[1] = mask[2] = 0xF8; mask[3] = 0x80;
        break;
    case K_4444:
        r = (lv + 1) & 15;
        g = (x + 7 * salt) & 15;
        b = (y + 13 * salt) & 15;
        a = 8 | ((x >> 4) & 7);
        v[0] = a << 12 | r << 8 | g << 4 | b;
        want[0] = r << 4; want[1] = g << 4; want[2] = b << 4; want[3] = a << 4;
        mask[0] = mask[1] = mask[2] = mask[3] = 0xF0;
        break;
    case K_DXT1: case K_DXT3: case K_DXT5:
        r = (lv + 1) & 31;
        g = (x + 5 * face + 7 * salt) & 63;
        b = (y + 11 * face + 13 * salt) & 31;
        v[0] = r << 11 | g << 5 | b;            /* both colours of the block */
        want[0] = r << 3; want[1] = g << 2; want[2] = b << 3; want[3] = 255;
        mask[0] = 0xF8; mask[1] = 0xFC; mask[2] = 0xF8; mask[3] = 0x80;
        if (kind == K_DXT3) {
            a = 8 | (x & 7);                    /* four bits a texel */
            v[1] = a * 0x1111;
            want[3] = a * 17; mask[3] = 0xF0;
        } else if (kind == K_DXT5) {
            a = 0x80 | (x & 0x3F);              /* both ends of the alpha ramp */
            v[1] = a;
            want[3] = a; mask[3] = 0xF8;
        }
        break;
    case K_A8:
        a = hash8(lv, face, x, y, salt);
        v[0] = a;
        want[0] = want[1] = want[2] = 255; want[3] = a;
        mask[0] = mask[1] = mask[2] = mask[3] = 0xFF;
        break;
    case K_L8: case K_I8:
        a = hash8(lv, face, x, y, salt);
        v[0] = a;
        want[0] = want[1] = want[2] = a; want[3] = kind == K_I8 ? a : 255;
        mask[0] = mask[1] = mask[2] = mask[3] = 0xFF;
        break;
    default:                                    /* K_AL88 */
        r = hash8(lv, face, x, y, salt);
        a = ((lv << 4) | (y & 15)) ^ 0x80;
        v[0] = r; v[1] = a;
        want[0] = want[1] = want[2] = r; want[3] = a;
        mask[0] = mask[1] = mask[2] = mask[3] = 0xFF;
        break;
    }
}

static int salt_at(const Tex *t, int lv, int x, int y)
{
    for (int i = 0; i < t->nsub; i++) {
        if (t->subr[i].lv == lv && x >= t->subr[i].x && y >= t->subr[i].y &&
            x < t->subr[i].x + t->subr[i].w && y < t->subr[i].y + t->subr[i].h) {
            return t->id + SUBSALT;
        }
    }
    return t->id;
}

/* What a read should give for the pixel at px, py of a level. */
static void expect(const Tex *t, int face, int lv, int px, int py,
                   uint8_t want[4], uint8_t mask[4])
{
    uint32_t v[4];
    int x = is_dxt(t->kind) ? px / 4 : px, y = is_dxt(t->kind) ? py / 4 : py;

    texel(t->kind, salt_at(t, lv, px, py), face, lv, x, y, v, want, mask);
    if (!abits) {
        mask[3] = 0;
    }
}

/* The bytes of the rectangle x, y, w, h (texels) of a level, as uploaded. */
static uint8_t *level_bytes(const Tex *t, int salt, int face, int lv,
                            int x0, int y0, int w, int h, int *size)
{
    uint8_t want[4], mask[4], *buf, *p;
    uint32_t v[4];

    if (is_dxt(t->kind)) {
        int bw = (w + 3) / 4, bh = (h + 3) / 4, bs = t->kind == K_DXT1 ? 8 : 16;
        p = buf = malloc(bw * bh * bs);
        for (int y = 0; y < bh; y++) {
            for (int x = 0; x < bw; x++) {
                texel(t->kind, salt, face, lv, x0 / 4 + x, y0 / 4 + y, v, want, mask);
                if (t->kind == K_DXT3) {
                    for (int i = 0; i < 4; i++) {
                        *p++ = v[1] & 0xFF; *p++ = v[1] >> 8;
                    }
                } else if (t->kind == K_DXT5) {
                    *p++ = v[1]; *p++ = v[1];
                    memset(p, 0, 6); p += 6;
                }
                *p++ = v[0] & 0xFF; *p++ = v[0] >> 8;
                *p++ = v[0] & 0xFF; *p++ = v[0] >> 8;
                memset(p, 0, 4); p += 4;
            }
        }
        *size = bw * bh * bs;
        return buf;
    }
    p = buf = malloc(w * h * 4);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            texel(t->kind, salt, face, lv, x0 + x, y0 + y, v, want, mask);
            switch (t->kind) {
            case K_RGBA8:
                memcpy(p, &v[0], 4); p += 4;            /* host order, 8_8_8_8_REV */
                break;
            case K_1555: case K_565: case K_4444: {
                uint16_t s = v[0];
                memcpy(p, &s, 2); p += 2;
                break;
            }
            case K_AL88:
                *p++ = v[0]; *p++ = v[1];
                break;
            default:
                *p++ = v[0];
                break;
            }
        }
    }
    *size = p - buf;
    return buf;
}

static void formats(int kind, GLenum *internal, GLenum *format, GLenum *type)
{
    switch (kind) {
    case K_RGBA8: *internal = GL_RGBA8; *format = GL_BGRA; *type = GL_UNSIGNED_INT_8_8_8_8_REV; break;
    case K_1555: *internal = GL_RGB5_A1; *format = GL_BGRA; *type = GL_UNSIGNED_SHORT_1_5_5_5_REV; break;
    case K_565: *internal = GL_RGB5; *format = GL_RGB; *type = GL_UNSIGNED_SHORT_5_6_5; break;
    case K_4444: *internal = GL_RGBA4; *format = GL_BGRA; *type = GL_UNSIGNED_SHORT_4_4_4_4_REV; break;
    case K_DXT1: *internal = GL_COMPRESSED_RGB_S3TC_DXT1_EXT; *format = *type = 0; break;
    case K_DXT3: *internal = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT; *format = *type = 0; break;
    case K_DXT5: *internal = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT; *format = *type = 0; break;
    case K_A8: *internal = GL_ALPHA8; *format = GL_ALPHA; *type = GL_UNSIGNED_BYTE; break;
    case K_L8: *internal = GL_LUMINANCE8; *format = GL_LUMINANCE; *type = GL_UNSIGNED_BYTE; break;
    case K_I8: *internal = GL_INTENSITY8; *format = GL_LUMINANCE; *type = GL_UNSIGNED_BYTE; break;
    default: *internal = GL_LUMINANCE8_ALPHA8; *format = GL_LUMINANCE_ALPHA; *type = GL_UNSIGNED_BYTE; break;
    }
}

static GLenum target_of(const Tex *t)
{
    return t->cube ? GL_TEXTURE_CUBE_MAP : GL_TEXTURE_2D;
}

static void upload(Tex *t)
{
    GLenum internal, format, type, tgt = target_of(t);
    int m = t->w > t->h ? t->w : t->h;

    for (t->levels = 1; m > 1; m >>= 1) {
        t->levels++;
    }
    formats(t->kind, &internal, &format, &type);
    glGenTextures(1, &t->name);
    glBindTexture(tgt, t->name);
    glTexParameteri(tgt, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(tgt, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(tgt, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(tgt, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    if (t->cube) {
        glTexParameteri(tgt, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    for (int face = 0; face < (t->cube ? 6 : 1); face++) {
        GLenum ft = t->cube ? GL_TEXTURE_CUBE_MAP_POSITIVE_X + face : GL_TEXTURE_2D;
        for (int lv = 0; lv < t->levels; lv++) {
            int lw = dim(t->w, lv), lh = dim(t->h, lv), size;
            uint8_t *buf = level_bytes(t, t->id, face, lv, 0, 0, lw, lh, &size);
            if (is_dxt(t->kind)) {
                glCompressedTexImage2D(ft, lv, internal, lw, lh, 0, size, buf);
            } else {
                glTexImage2D(ft, lv, internal, lw, lh, 0, format, type, buf);
            }
            free(buf);
        }
    }
}

/* Replace part of two levels with texels of another salt. */
static void subimage(Tex *t)
{
    GLenum internal, format, type;
    int step = is_dxt(t->kind) ? 4 : 1, small = -1;

    formats(t->kind, &internal, &format, &type);
    glBindTexture(GL_TEXTURE_2D, t->name);
    for (int lv = 0; lv < t->levels; lv++) {
        if (dim(t->w, lv) == 8 * step || (small < 0 && dim(t->w, lv) <= 8 * step &&
                                          dim(t->w, lv) >= 2 * step)) {
            small = lv;
        }
    }
    t->nsub = 0;
    for (int i = 0; i < 2; i++) {
        int lv = i ? small : 0, lw, lh, x, y, w, h, size;
        uint8_t *buf;
        if (lv < 0 || (i && lv == 0)) {
            continue;
        }
        lw = dim(t->w, lv); lh = dim(t->h, lv);
        if (i == 0) {
            x = 3 * step; y = 1 * step; w = 5 * step; h = 3 * step;
        } else {
            x = 1 * step; y = 0; w = 3 * step; h = 2 * step;
        }
        if (x >= lw) { x = 0; }
        if (y >= lh) { y = 0; }
        if (x + w > lw) { w = lw - x; }
        if (y + h > lh) { h = lh - y; }
        buf = level_bytes(t, t->id + SUBSALT, 0, lv, x, y, w, h, &size);
        if (is_dxt(t->kind)) {
            glCompressedTexSubImage2D(GL_TEXTURE_2D, lv, x, y, w, h, internal, size, buf);
        } else {
            glTexSubImage2D(GL_TEXTURE_2D, lv, x, y, w, h, format, type, buf);
        }
        free(buf);
        t->subr[t->nsub].lv = lv; t->subr[t->nsub].x = x; t->subr[t->nsub].y = y;
        t->subr[t->nsub].w = w; t->subr[t->nsub].h = h;
        t->nsub++;
    }
}

static void coord(const Tex *t, int face, float s, float u)
{
    float a = 2 * s - 1, b = 2 * u - 1;

    if (!t->cube) {
        glTexCoord2f(s, u);
        return;
    }
    switch (face) {
    case 0: glTexCoord3f(1, -b, -a); break;
    case 1: glTexCoord3f(-1, -b, a); break;
    case 2: glTexCoord3f(a, 1, b); break;
    case 3: glTexCoord3f(a, -1, -b); break;
    case 4: glTexCoord3f(a, -b, 1); break;
    default: glTexCoord3f(-a, -b, -1); break;
    }
}

static uint8_t px[TILE * TILE * 4];

/* Draw texels x0.., y0.. of a level at the window's corner and read them. */
static void draw_read(const Tex *t, int face, int lv, int x0, int y0, int tw, int th,
                      float bias)
{
    int lw = dim(t->w, lv), lh = dim(t->h, lv);
    float s0 = (float)x0 / lw, s1 = (float)(x0 + tw) / lw;
    float t0 = (float)y0 / lh, t1 = (float)(y0 + th) / lh;

    glClearColor(0.07f, 0.13f, 0.19f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(target_of(t));
    glBindTexture(target_of(t), t->name);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexEnvf(GL_TEXTURE_FILTER_CONTROL, GL_TEXTURE_LOD_BIAS, bias);
    glColor4f(1, 1, 1, 1);
    glBegin(GL_QUADS);
    coord(t, face, s0, t0); glVertex2f(OX, OY);
    coord(t, face, s1, t0); glVertex2f(OX + tw, OY);
    coord(t, face, s1, t1); glVertex2f(OX + tw, OY + th);
    coord(t, face, s0, t1); glVertex2f(OX, OY + th);
    glEnd();
    glDisable(target_of(t));
    glReadBuffer(GL_BACK);
    glReadPixels(OX, OY, tw, th, GL_RGBA, GL_UNSIGNED_BYTE, px);
}

static int same(const uint8_t *got, const uint8_t *want, const uint8_t *mask)
{
    return !(((got[0] ^ want[0]) & mask[0]) | ((got[1] ^ want[1]) & mask[1]) |
             ((got[2] ^ want[2]) & mask[2]) | ((got[3] ^ want[3]) & mask[3]));
}

/* How many of the pixels just read are level m's texels (of face f)? */
static int score(const Tex *t, int f, int lv, int m, int x0, int y0, int tw, int th)
{
    int lw = dim(t->w, lv), lh = dim(t->h, lv), mw = dim(t->w, m), mh = dim(t->h, m), n = 0;
    uint8_t want[4], mask[4];

    for (int j = 0; j < th; j++) {
        for (int i = 0; i < tw; i++) {
            int x = (int)(((int64_t)(2 * (x0 + i) + 1) * mw) / (2 * lw));
            int y = (int)(((int64_t)(2 * (y0 + j) + 1) * mh) / (2 * lh));
            expect(t, f, m, x, y, want, mask);
            n += same(px + (j * tw + i) * 4, want, mask);
        }
    }
    return n;
}

static uint32_t crc(const uint8_t *p, int n)
{
    uint32_t h = 2166136261u;
    while (n--) {
        h = (h ^ *p++) * 16777619u;
    }
    return h;
}

static void check_level(const Tex *t, int face, int lv, const char *pass)
{
    int lw = dim(t->w, lv), lh = dim(t->h, lv), ok = 0, total = 0, tries = 0;
    float bias = 0;
    uint32_t h = 0;
    char note[256] = "";

    for (int y0 = 0; y0 < lh; y0 += TILE) {
        for (int x0 = 0; x0 < lw; x0 += TILE) {
            int tw = lw - x0 < TILE ? lw - x0 : TILE, th = lh - y0 < TILE ? lh - y0 : TILE, n;
            for (;;) {
                draw_read(t, face, lv, x0, y0, tw, th, bias);
                n = score(t, face, lv, lv, x0, y0, tw, th);
                if (n == tw * th || tries >= 3) {
                    break;
                }
                /* another level's texels?  then lean the LOD the other way */
                int best = -1, bn = n;
                for (int m = 0; m < t->levels; m++) {
                    int k = m == lv ? n : score(t, face, lv, m, x0, y0, tw, th);
                    if (k > bn) {
                        bn = k; best = m;
                    }
                }
                if (best < 0) {
                    break;
                }
                bias += lv - best;
                tries++;
            }
            if (n != tw * th && !note[0]) {
                int best = lv, bf = face, bn = n, shown = 0, o = 0;
                for (int f = 0; f < (t->cube ? 6 : 1); f++) {
                    for (int m = 0; m < t->levels; m++) {
                        int k = score(t, f, lv, m, x0, y0, tw, th);
                        if (k > bn) {
                            bn = k; best = m; bf = f;
                        }
                    }
                }
                o += snprintf(note + o, sizeof(note) - o, " tile %d,%d %d/%d", x0, y0, n, tw * th);
                if (best != lv || bf != face) {
                    o += snprintf(note + o, sizeof(note) - o, " (f%d lv%d fits %d)", bf, best, bn);
                }
                for (int j = 0; j < th && shown < 3; j++) {
                    for (int i = 0; i < tw && shown < 3; i++) {
                        uint8_t want[4], mask[4], *g = px + (j * tw + i) * 4;
                        expect(t, face, lv, x0 + i, y0 + j, want, mask);
                        if (!same(g, want, mask)) {
                            o += snprintf(note + o, sizeof(note) - o,
                                          " (%d,%d) got %02x%02x%02x%02x want %02x%02x%02x%02x",
                                          x0 + i, y0 + j, g[0], g[1], g[2], g[3],
                                          want[0], want[1], want[2], want[3]);
                            shown++;
                        }
                    }
                }
            }
            ok += n; total += tw * th;
            h = h * 31 + crc(px, tw * th * 4);
        }
    }
    n_levels++;
    if (ok == total) {
        n_ok++;
    } else {
        n_bad++;
    }
    if (bias != 0) {
        n_biased++;
    }
    say("T%02d %s%s %dx%d f%d lv%d %dx%d %s bias %+.0f px %d ok %d crc %08x %s%s\n",
        t->id, kname[t->kind], t->cube ? " cube" : "", t->w, t->h, face, lv, lw, lh, pass,
        bias, total, ok, h, ok == total ? "PASS" : "FAIL", note);
}

static void check_tex(const Tex *t, const char *pass)
{
    for (int face = 0; face < (t->cube ? 6 : 1); face++) {
        for (int lv = 0; lv < t->levels; lv++) {
            check_level(t, face, lv, pass);
        }
    }
}

static int has_ext(const char *name)
{
    const char *e = (const char *)glGetString(GL_EXTENSIONS);
    return e && strstr(e, name) != NULL;
}

static void run_tests(void)
{
    GLint v[4];
    int s3tc;

    glViewport(0, 0, WW, WH);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WW, 0, WH, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glDisable(GL_BLEND);
    glDisable(GL_DITHER);
    glDisable(GL_ALPHA_TEST);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_LIGHTING);
    glDisable(GL_FOG);
    glShadeModel(GL_FLAT);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glDrawBuffer(GL_BACK);

    say("texprobe 1 stage %d only %d\n", stage, only);
    say("GL_RENDERER %s\nGL_VERSION %s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));
    glGetIntegerv(GL_RED_BITS, &v[0]); glGetIntegerv(GL_GREEN_BITS, &v[1]);
    glGetIntegerv(GL_BLUE_BITS, &v[2]); glGetIntegerv(GL_ALPHA_BITS, &v[3]);
    abits = v[3];
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxsize);
    s3tc = has_ext("GL_EXT_texture_compression_s3tc");
    say("bits R%d G%d B%d A%d max texture %d s3tc %d lod_bias %d cube %d\n", v[0], v[1], v[2],
        v[3], maxsize, s3tc, has_ext("GL_EXT_texture_lod_bias"),
        has_ext("GL_ARB_texture_cube_map"));

    /* everything is uploaded before anything is drawn, as a game would */
    for (int i = 0; i < NCASES; i++) {
        Tex *t = &cases[i];
        if (t->stage != stage || (only >= 0 && t->id != only)) {
            continue;
        }
        if (t->w > maxsize || t->h > maxsize || (is_dxt(t->kind) && !s3tc)) {
            say("T%02d skipped\n", t->id);
            t->stage = -1;
            continue;
        }
        upload(t);
    }
    say("uploaded, GL error %04x\n", glGetError());
    if (waits) {
        glFinish();
        sleep(waits);
    }
    for (int i = 0; i < NCASES; i++) {
        Tex *t = &cases[i];
        if (t->stage != stage || (only >= 0 && t->id != only)) {
            continue;
        }
        check_tex(t, "first");
    }
    for (int i = 0; i < NCASES; i++) {
        Tex *t = &cases[i];
        if (t->stage != stage || (only >= 0 && t->id != only) || !t->sub) {
            continue;
        }
        subimage(t);
        check_tex(t, "sub");
    }
    say("GL error %04x\n", glGetError());
    say("TOTAL levels %d ok %d bad %d biased %d\nDONE\n", n_levels, n_ok, n_bad, n_biased);
}

static int phase;
static void quit_cb(int v) { exit(0); }

static void display(void)
{
    if (phase == 0) {
        phase = 1;
        out = fopen("/tmp/texprobe.txt", "w");
        run_tests();
        if (out) {
            fclose(out);
        }
        glutTimerFunc(hold * 1000, quit_cb, 0);
    }
    glDrawBuffer(GL_BACK);
    if (n_bad) {
        glClearColor(0.8f, 0.1f, 0.1f, 1);
    } else {
        glClearColor(0.1f, 0.7f, 0.1f, 1);
    }
    glClear(GL_COLOR_BUFFER_BIT);
    glutSwapBuffers();
}

int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "stage=", 6)) {
            stage = atoi(argv[i] + 6);
        } else if (!strncmp(argv[i], "only=", 5)) {
            only = atoi(argv[i] + 5);
        } else if (!strncmp(argv[i], "hold=", 5)) {
            hold = atoi(argv[i] + 5);
        } else if (!strncmp(argv[i], "wait=", 5)) {
            waits = atoi(argv[i] + 5);
        } else if (!strcmp(argv[i], "list")) {
            for (int k = 0; k < NCASES; k++) {
                printf("%d %s %d %d cube %d sub %d stage %d\n", cases[k].id,
                       kname[cases[k].kind], cases[k].w, cases[k].h, cases[k].cube,
                       cases[k].sub, cases[k].stage);
            }
            return 0;
        }
    }
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE);
    glutInitWindowSize(WW, WH);
    glutInitWindowPosition(40, 60);
    glutCreateWindow("texprobe");
    glutDisplayFunc(display);
    glutMainLoop();
    return 0;
}
