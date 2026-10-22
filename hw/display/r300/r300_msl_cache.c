/*
 * R300 MSL disk cache (see r300_msl_cache.h).
 *
 * An entry file is one header line and the MSL compressed with zlib:
 *
 *   r300-msl <schema> <stage> <MSL bytes> <zlib bytes> <key> <MSL sha256>\n
 *
 * where <key> is the SHA-256 of the stage (one byte) and the whole GLSL,
 * also the file's name.  Every field is checked against the request and
 * the data, and the header must be exactly the one put would write.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#include "r300_msl_cache.h"

#include <CommonCrypto/CommonDigest.h>
#include <errno.h>
#include <fcntl.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>

#define HEX_LEN (2 * CC_SHA256_DIGEST_LENGTH)
#define IDENTITY_MAX (64 * 1024)

struct R300MslCache {
    char *dir;
    _Atomic uint64_t hits, misses, rejected, writes, write_failures, dropped,
                     bytes_written;
};

static void count(_Atomic uint64_t *n, uint64_t k)
{
    atomic_fetch_add_explicit(n, k, memory_order_relaxed);
}

/* CC_SHA256_Update takes a 32-bit length. */
static void sha_update(CC_SHA256_CTX *x, const void *p, size_t n)
{
    const unsigned char *b = p;

    while (n) {
        CC_LONG k = n > (1u << 30) ? (1u << 30) : (CC_LONG)n;
        CC_SHA256_Update(x, b, k);
        b += k;
        n -= k;
    }
}

static void sha_hex(CC_SHA256_CTX *x, char out[HEX_LEN + 1])
{
    unsigned char d[CC_SHA256_DIGEST_LENGTH];

    CC_SHA256_Final(d, x);
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) {
        snprintf(out + 2 * i, 3, "%02x", d[i]);
    }
}

static void bytes_hex(const void *p, size_t n, char out[HEX_LEN + 1])
{
    CC_SHA256_CTX x;

    CC_SHA256_Init(&x);
    sha_update(&x, p, n);
    sha_hex(&x, out);
}

static void key_hex(const char *glsl, R300Stage stage, char out[HEX_LEN + 1])
{
    CC_SHA256_CTX x;
    unsigned char s = (unsigned char)stage;

    CC_SHA256_Init(&x);
    sha_update(&x, &s, 1);
    sha_update(&x, glsl, strlen(glsl));
    sha_hex(&x, out);
}

static char *entry_path(const R300MslCache *c, const char *key)
{
    size_t n = strlen(c->dir) + HEX_LEN + 8;
    char *p = malloc(n);

    snprintf(p, n, "%s/%s.msl", c->dir, key);
    return p;
}

static int header(char *buf, size_t size, R300Stage stage, size_t mlen,
                  size_t zlen, const char *key, const char *mhex)
{
    return snprintf(buf, size, "r300-msl %u %u %zu %zu %s %s\n",
                    R300_MSL_CACHE_SCHEMA, (unsigned)stage, mlen, zlen, key, mhex);
}

static char *dup_err(const char *what, const char *path, int e)
{
    size_t n = strlen(what) + strlen(path) + 64;
    char *s = malloc(n);

    snprintf(s, n, "%s %s: %s", what, path, strerror(e));
    return s;
}

static bool write_all(int fd, const void *p, size_t n)
{
    const char *b = p;

    while (n) {
        ssize_t k = write(fd, b, n);
        if (k < 0 && errno == EINTR) {
            continue;
        }
        if (k <= 0) {
            return false;
        }
        b += k;
        n -= (size_t)k;
    }
    return true;
}

/*
 * Write the two parts to a new file with a unique name next to path,
 * then rename it over path: concurrent writers each publish a whole file,
 * and the last rename wins.  On failure nothing is left behind.
 */
static bool publish(const char *path, const void *a, size_t na,
                    const void *b, size_t nb)
{
    size_t n = strlen(path) + 16;
    char *tmp = malloc(n);
    bool ok;
    int fd;

    snprintf(tmp, n, "%s.tmp-XXXXXX", path);
    fd = mkostemp(tmp, O_CLOEXEC);
    if (fd < 0) {
        free(tmp);
        return false;
    }
    ok = write_all(fd, a, na) && write_all(fd, b, nb);
    ok = close(fd) == 0 && ok;
    ok = ok && rename(tmp, path) == 0;
    if (!ok) {
        int e = errno;
        unlink(tmp);
        errno = e;
    }
    free(tmp);
    return ok;
}

/*
 * Read a whole regular file of at most max bytes.  Returns 1 with a
 * malloc'd buffer, 0 when there is no such file, -1 otherwise (*why).
 */
static int read_file(const char *path, size_t max, char **buf, size_t *size,
                     const char **why)
{
    struct stat sb;
    size_t got = 0;
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);

    if (fd < 0) {
        if (errno == ENOENT) {
            return 0;
        }
        *why = "cannot open";
        return -1;
    }
    if (fstat(fd, &sb) < 0 || !S_ISREG(sb.st_mode)) {
        close(fd);
        *why = "not a regular file";
        return -1;
    }
    if (sb.st_size < 0 || (uint64_t)sb.st_size > max) {
        close(fd);
        *why = "file too large";
        return -1;
    }
    *buf = malloc((size_t)sb.st_size + 1);
    while (got < (size_t)sb.st_size) {
        ssize_t k = read(fd, *buf + got, (size_t)sb.st_size - got);
        if (k < 0 && errno == EINTR) {
            continue;
        }
        if (k <= 0) {
            break;
        }
        got += (size_t)k;
    }
    close(fd);
    if (got != (size_t)sb.st_size) {
        free(*buf);
        *buf = NULL;
        *why = "short read";
        return -1;
    }
    (*buf)[got] = 0;
    *size = got;
    return 1;
}

R300MslCache *r300_msl_cache_open(const char *root, const char *identity,
                                  char **err)
{
    size_t n = strlen(identity) + 64;
    char *full = malloc(n), *idpath = NULL, *old = NULL;
    char hex[HEX_LEN + 1];
    const char *why = NULL;
    R300MslCache *c = calloc(1, sizeof(*c));
    struct stat sb;
    size_t oldn = 0;
    int r;

    *err = NULL;
    snprintf(full, n, "r300-msl schema %u\n%s", R300_MSL_CACHE_SCHEMA, identity);
    bytes_hex(full, strlen(full), hex);
    n = strlen(root) + HEX_LEN + 2;
    c->dir = malloc(n);
    snprintf(c->dir, n, "%s/%s", root, hex);

    if (mkdir(root, 0700) < 0 && errno != EEXIST) {
        *err = dup_err("cannot create", root, errno);
        goto fail;
    }
    if (mkdir(c->dir, 0700) < 0 && errno != EEXIST) {
        *err = dup_err("cannot create", c->dir, errno);
        goto fail;
    }
    if (stat(c->dir, &sb) < 0 || !S_ISDIR(sb.st_mode)) {
        *err = dup_err("cannot use", c->dir, ENOTDIR);
        goto fail;
    }
    n = strlen(c->dir) + 16;
    idpath = malloc(n);
    snprintf(idpath, n, "%s/identity.txt", c->dir);
    r = read_file(idpath, IDENTITY_MAX, &old, &oldn, &why);
    if (r > 0 && (oldn != strlen(full) || memcmp(old, full, oldn))) {
        r = -1;
        why = "content differs";
    }
    if (r < 0) {
        n = strlen(idpath) + strlen(why) + 64;
        *err = malloc(n);
        snprintf(*err, n, "%s: %s (delete the directory)", idpath, why);
        goto fail;
    }
    if (r == 0 && !publish(idpath, full, strlen(full), "", 0)) {
        *err = dup_err("cannot write", idpath, errno);
        goto fail;
    }
    free(old);
    free(idpath);
    free(full);
    return c;

fail:
    free(old);
    free(idpath);
    free(full);
    r300_msl_cache_close(c);
    return NULL;
}

void r300_msl_cache_close(R300MslCache *c)
{
    if (c) {
        free(c->dir);
        free(c);
    }
}

const char *r300_msl_cache_dir(const R300MslCache *c)
{
    return c->dir;
}

char *r300_msl_cache_get(R300MslCache *c, const char *glsl, R300Stage stage,
                         const char **why)
{
    char key[HEX_LEN + 1], kh[HEX_LEN + 1], mhex[HEX_LEN + 1];
    char line[256], want[256];
    char *path, *buf = NULL, *msl = NULL;
    const char *bad = NULL, *nl;
    unsigned schema, st;
    size_t size = 0, hl, mlen, zlen;
    uLongf dlen;
    int r;

    key_hex(glsl, stage, key);
    path = entry_path(c, key);
    r = read_file(path, R300_MSL_CACHE_MAX_FILE, &buf, &size, &bad);
    free(path);
    if (r == 0) {
        count(&c->misses, 1);
        if (why) {
            *why = NULL;
        }
        return NULL;
    }
    if (r < 0) {
        goto reject;
    }
    nl = memchr(buf, '\n', size < sizeof(line) ? size : sizeof(line));
    if (!nl) {
        bad = "no header";
        goto reject;
    }
    hl = (size_t)(nl - buf) + 1;
    memcpy(line, buf, hl - 1);
    line[hl - 1] = 0;
    if (sscanf(line, "r300-msl %u %u %zu %zu %64s", &schema, &st, &mlen, &zlen,
               kh) != 5) {
        bad = "bad header";
        goto reject;
    }
    if (schema != R300_MSL_CACHE_SCHEMA) {
        bad = "other schema";
        goto reject;
    }
    if (st != (unsigned)stage) {
        bad = "other stage";
        goto reject;
    }
    if (strcmp(kh, key)) {
        bad = "other key";
        goto reject;
    }
    if (!mlen || mlen > R300_MSL_CACHE_MAX_MSL || zlen != size - hl) {
        bad = "lengths do not match the file";
        goto reject;
    }
    msl = malloc(mlen + 1);
    dlen = mlen;
    if (uncompress((Bytef *)msl, &dlen, (const Bytef *)buf + hl, zlen) != Z_OK ||
        dlen != mlen) {
        bad = "bad compressed data";
        goto reject;
    }
    msl[mlen] = 0;
    if (memchr(msl, 0, mlen)) {
        bad = "NUL in MSL";
        goto reject;
    }
    bytes_hex(msl, mlen, mhex);
    /* the MSL hash, and nothing else differing from what put writes */
    if (header(want, sizeof(want), stage, mlen, zlen, key, mhex) != (int)hl ||
        memcmp(want, buf, hl)) {
        bad = "header does not match the data";
        goto reject;
    }
    free(buf);
    count(&c->hits, 1);
    if (why) {
        *why = NULL;
    }
    return msl;

reject:
    free(buf);
    free(msl);
    count(&c->rejected, 1);
    if (why) {
        *why = bad;
    }
    return NULL;
}

bool r300_msl_cache_put(R300MslCache *c, const char *glsl, R300Stage stage,
                        const char *msl, const char **why)
{
    char key[HEX_LEN + 1], mhex[HEX_LEN + 1], hdr[256];
    size_t mlen = strlen(msl);
    const char *bad;
    uLongf zlen;
    Bytef *z = NULL;
    char *path;
    int hl;

    if (!mlen || mlen > R300_MSL_CACHE_MAX_MSL) {
        bad = mlen ? "MSL too large" : "empty MSL";
        goto fail;
    }
    zlen = compressBound(mlen);
    z = malloc(zlen);
    if (compress2(z, &zlen, (const Bytef *)msl, mlen, Z_DEFAULT_COMPRESSION) != Z_OK) {
        bad = "compression failed";
        goto fail;
    }
    key_hex(glsl, stage, key);
    bytes_hex(msl, mlen, mhex);
    hl = header(hdr, sizeof(hdr), stage, mlen, zlen, key, mhex);
    if ((size_t)hl + zlen > R300_MSL_CACHE_MAX_FILE) {
        bad = "entry too large";
        goto fail;
    }
    path = entry_path(c, key);
    if (!publish(path, hdr, (size_t)hl, z, zlen)) {
        free(path);
        bad = "cannot write the entry";
        goto fail;
    }
    free(path);
    free(z);
    count(&c->writes, 1);
    count(&c->bytes_written, (uint64_t)hl + zlen);
    return true;

fail:
    free(z);
    count(&c->write_failures, 1);
    if (why) {
        *why = bad;
    }
    return false;
}

void r300_msl_cache_drop(R300MslCache *c, const char *glsl, R300Stage stage)
{
    char key[HEX_LEN + 1];
    char *path;

    key_hex(glsl, stage, key);
    path = entry_path(c, key);
    if (unlink(path) == 0) {
        count(&c->dropped, 1);
    }
    free(path);
}

void r300_msl_cache_stats(R300MslCache *c, R300MslCacheStats *s)
{
    s->hits = atomic_load_explicit(&c->hits, memory_order_relaxed);
    s->misses = atomic_load_explicit(&c->misses, memory_order_relaxed);
    s->rejected = atomic_load_explicit(&c->rejected, memory_order_relaxed);
    s->writes = atomic_load_explicit(&c->writes, memory_order_relaxed);
    s->write_failures = atomic_load_explicit(&c->write_failures, memory_order_relaxed);
    s->dropped = atomic_load_explicit(&c->dropped, memory_order_relaxed);
    s->bytes_written = atomic_load_explicit(&c->bytes_written, memory_order_relaxed);
}
