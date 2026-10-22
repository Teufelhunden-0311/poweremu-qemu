/*
 * The MSL disk cache (hw/display/r300/r300_msl_cache.c), on real MSL from
 * r300_glsl_to_msl() for each stage of one program:
 *   test_mslcache PROGRAM.glsl SCRATCHDIR
 * miss, write, hit (same handle, a new one, another identity); damaged,
 * foreign and odd entries refused and then replaced; unwritable storage;
 * drop; writers and readers in several processes at once.
 * Includes the cache's source, to write entries in its own format.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#include "../../hw/display/r300/r300_msl_cache.c"

#include <dirent.h>
#include <sys/wait.h>

static int fails;

#define CHECK(c, ...) do {                                              \
        if (!(c)) {                                                     \
            fails++;                                                    \
            printf("FAIL line %d: ", __LINE__);                         \
            printf(__VA_ARGS__);                                        \
            printf("\n");                                               \
        }                                                               \
    } while (0)

static char *slurp(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    char *b;
    long k;

    if (!f) {
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    k = ftell(f);
    fseek(f, 0, SEEK_SET);
    b = malloc((size_t)k + 1);
    *n = fread(b, 1, (size_t)k, f);
    b[*n] = 0;
    fclose(f);
    return b;
}

static void spit(const char *path, const void *b, size_t n)
{
    FILE *f;

    unlink(path);
    f = fopen(path, "wb");
    fwrite(b, 1, n, f);
    fclose(f);
}

/* files in dir ending in suffix, and temporary files */
static int files_in(const char *dir, const char *suffix, int *tmp)
{
    DIR *d = opendir(dir);
    struct dirent *e;
    int n = 0;

    *tmp = 0;
    while (d && (e = readdir(d))) {
        size_t l = strlen(e->d_name);
        if (strstr(e->d_name, ".tmp-")) {
            (*tmp)++;
        } else if (l > strlen(suffix) && !strcmp(e->d_name + l - strlen(suffix), suffix)) {
            n++;
        }
    }
    if (d) {
        closedir(d);
    }
    return n;
}

/* An entry in put's format from parts that may disagree. */
static size_t forge(char *out, size_t size, R300Stage stage, const char *key,
                    const char *msl, size_t mlen, const char *hash_of, size_t hlen)
{
    char mhex[HEX_LEN + 1];
    uLongf zlen = compressBound(mlen);
    Bytef *z = malloc(zlen);
    int hl;

    compress2(z, &zlen, (const Bytef *)msl, mlen, Z_DEFAULT_COMPRESSION);
    bytes_hex(hash_of, hlen, mhex);
    hl = header(out, size, stage, mlen, zlen, key, mhex);
    memcpy(out + hl, z, zlen);
    free(z);
    return (size_t)hl + zlen;
}

static R300MslCacheStats st(R300MslCache *c)
{
    R300MslCacheStats s;

    r300_msl_cache_stats(c, &s);
    return s;
}

/* get refuses the entry now at path (why names the check), then a put
 * replaces it and get returns msl again */
static void refused(R300MslCache *c, const char *glsl, R300Stage stage,
                    const char *msl, const char *path, const char *expect,
                    const char *what)
{
    uint64_t rej = st(c).rejected;
    const char *why = NULL, *reason;
    char *m = r300_msl_cache_get(c, glsl, stage, &why);
    bool ok;

    reason = why;

    CHECK(!m, "%s: entry used", what);
    CHECK(why && (!expect || !strcmp(why, expect)), "%s: why \"%s\", want \"%s\"",
          what, why ? why : "(miss)", expect ? expect : "any");
    CHECK(st(c).rejected == rej + 1, "%s: not counted as rejected", what);
    free(m);
    if (!expect || strcmp(expect, "not a regular file")) {
        ok = r300_msl_cache_put(c, glsl, stage, msl, &why);
        CHECK(ok, "%s: put after refusal failed: %s", what, why);
        m = r300_msl_cache_get(c, glsl, stage, &why);
        CHECK(m && !strcmp(m, msl), "%s: not replaced", what);
        free(m);
    }
    printf("  refused %-28s %s\n", what, reason ? reason : "-");
    (void)path;
}

int main(int argc, char **argv)
{
    char root[1024], cmd[1200], path[1200], idp[1200], *glsl, *err = NULL;
    static char buf[R300_MSL_CACHE_MAX_FILE + 4096];
    char *msl[3], *m, *tid, *glsl2, *glsl3;
    const char *why;
    size_t n;
    int tmp;
    R300MslCache *c, *c2;
    bool root_user = geteuid() == 0;

    if (argc != 3 || !(glsl = slurp(argv[1], &n))) {
        fprintf(stderr, "usage: test_mslcache PROGRAM.glsl SCRATCHDIR\n");
        return 2;
    }
    snprintf(root, sizeof(root), "%s/mslcache", argv[2]);
    snprintf(cmd, sizeof(cmd), "chmod -R u+rwx '%s' 2>/dev/null; rm -rf '%s'", root, root);
    system(cmd);

    for (int s = 0; s < 3; s++) {
        msl[s] = r300_glsl_to_msl(glsl, (R300Stage)s, &err);
        if (!msl[s]) {
            printf("FAIL: conversion of stage %d: %s\n", s, err);
            return 1;
        }
    }
    CHECK(strcmp(msl[0], msl[1]) && strcmp(msl[1], msl[2]), "stages give the same MSL");
    tid = r300_msl_translation_id();
    printf("translation id:\n%s\n", tid);
    CHECK(strstr(tid, "MSL 2.4.0") && strstr(tid, "SPIRV-Cross: C API") &&
          strstr(tid, "shaderc: SPIR-V"), "translation id incomplete");
    free(tid);

    /* miss, write, hit */
    c = r300_msl_cache_open(root, "identity A", &err);
    CHECK(c, "open: %s", err);
    if (!c) {
        return 1;
    }
    for (int s = 0; s < 3; s++) {
        why = "x";
        m = r300_msl_cache_get(c, glsl, (R300Stage)s, &why);
        CHECK(!m && !why, "stage %d: not a plain miss", s);
        CHECK(r300_msl_cache_put(c, glsl, (R300Stage)s, msl[s], &why), "put %d: %s", s, why);
        m = r300_msl_cache_get(c, glsl, (R300Stage)s, &why);
        CHECK(m && !strcmp(m, msl[s]), "stage %d: hit differs from the conversion", s);
        free(m);
    }
    CHECK(st(c).misses == 3 && st(c).hits == 3 && st(c).writes == 3 && !st(c).rejected,
          "counts after miss/write/hit");
    CHECK(files_in(r300_msl_cache_dir(c), ".msl", &tmp) == 3 && !tmp, "3 entries, no temporaries");
    snprintf(idp, sizeof(idp), "%s/identity.txt", r300_msl_cache_dir(c));
    m = slurp(idp, &n);
    CHECK(m && !strcmp(m, "r300-msl schema 1\nidentity A"), "identity.txt");
    free(m);
    printf("  entry sizes: MSL %zu/%zu/%zu bytes, written %llu bytes in all\n",
           strlen(msl[0]), strlen(msl[1]), strlen(msl[2]),
           (unsigned long long)st(c).bytes_written);

    /* a later session; another identity */
    c2 = r300_msl_cache_open(root, "identity A", &err);
    CHECK(c2 && !strcmp(r300_msl_cache_dir(c2), r300_msl_cache_dir(c)), "reopen");
    for (int s = 0; c2 && s < 3; s++) {
        m = r300_msl_cache_get(c2, glsl, (R300Stage)s, &why);
        CHECK(m && !strcmp(m, msl[s]), "stage %d: not kept across handles", s);
        free(m);
    }
    r300_msl_cache_close(c2);
    c2 = r300_msl_cache_open(root, "identity B", &err);
    CHECK(c2 && strcmp(r300_msl_cache_dir(c2), r300_msl_cache_dir(c)), "identity B dir");
    m = c2 ? r300_msl_cache_get(c2, glsl, R300_STAGE_FS, &why) : NULL;
    CHECK(c2 && !m && !why, "identity B sees identity A's entry");
    free(m);
    r300_msl_cache_close(c2);

    /* damaged, foreign and odd entries at the FS entry's path */
    {
        R300Stage fs = R300_STAGE_FS;
        char key[HEX_LEN + 1], *orig, *p, *nl, *nm;
        size_t on, hl;

        key_hex(glsl, fs, key);
        p = entry_path(c, key);
        snprintf(path, sizeof(path), "%s", p);
        free(p);
        orig = slurp(path, &on);
        nl = memchr(orig, '\n', on);
        hl = (size_t)(nl - orig) + 1;

        spit(path, "", 0);
        refused(c, glsl, fs, msl[1], path, "no header", "empty file");
        spit(path, orig, (size_t)(nl - orig) + 1);
        refused(c, glsl, fs, msl[1], path, "lengths do not match the file", "header only");
        spit(path, orig, on - 1);
        refused(c, glsl, fs, msl[1], path, "lengths do not match the file", "one byte short");
        memcpy(buf, orig, on);
        buf[on] = 'x';
        spit(path, buf, on + 1);
        refused(c, glsl, fs, msl[1], path, "lengths do not match the file", "one byte extra");
        memcpy(buf, orig, on);
        buf[(nl - orig) + 1 + (on - (nl - orig)) / 2] ^= 0x10;
        spit(path, buf, on);
        refused(c, glsl, fs, msl[1], path, NULL, "flipped compressed bit");
        memcpy(buf, orig, on);
        buf[9] = '2';                                   /* "r300-msl 2 ..." */
        spit(path, buf, on);
        refused(c, glsl, fs, msl[1], path, "other schema", "schema 2");
        memcpy(buf, orig, on);
        buf[11] = '2';                                  /* stage FS_Z */
        spit(path, buf, on);
        refused(c, glsl, fs, msl[1], path, "other stage", "stage in header");
        /* the right header shape with another program's key */
        glsl2 = malloc(strlen(glsl) + 16);
        sprintf(glsl2, "%s\n// other\n", glsl);
        key_hex(glsl2, fs, key);
        n = forge(buf, sizeof(buf), fs, key, msl[1], strlen(msl[1]), msl[1], strlen(msl[1]));
        spit(path, buf, n);
        refused(c, glsl, fs, msl[1], path, "other key", "another program's entry");
        key_hex(glsl, fs, key);
        n = forge(buf, sizeof(buf), fs, key, msl[2], strlen(msl[2]), msl[1], strlen(msl[1]));
        spit(path, buf, n);
        refused(c, glsl, fs, msl[1], path, "header does not match the data", "MSL hash mismatch");
        nm = strdup(msl[1]);
        nm[5] = 0;                          /* hashed and compressed with the NUL */
        n = forge(buf, sizeof(buf), fs, key, nm, strlen(msl[1]), nm, strlen(msl[1]));
        free(nm);
        spit(path, buf, n);
        refused(c, glsl, fs, msl[1], path, "NUL in MSL", "NUL in MSL");
        n = forge(buf, sizeof(buf), fs, key, msl[1], strlen(msl[1]), msl[1], strlen(msl[1]));
        CHECK(n == on && !memcmp(buf, orig, on), "forge does not reproduce put");
        /* headers sscanf reads as put's, with the lengths still matching
         * the file: only the comparison with put's header refuses them */
        memcpy(buf, "r300-msl  ", 10);                  /* two spaces */
        memcpy(buf + 10, orig + 9, on - 9);
        spit(path, buf, on + 1);
        refused(c, glsl, fs, msl[1], path, "header does not match the data",
                "header with two spaces");
        {
            unsigned sc, sg;
            size_t ml, zl;
            int k;

            sscanf(orig, "r300-msl %u %u %zu %zu", &sc, &sg, &ml, &zl);
            k = snprintf(buf, sizeof(buf), "r300-msl %u %u +%zu %zu %.*s", sc, sg, ml, zl,
                         2 * HEX_LEN + 1, orig + hl - 2 - 2 * HEX_LEN);
            memcpy(buf + k, orig + hl - 1, on - hl + 1);
            spit(path, buf, (size_t)k + on - hl + 1);
        }
        refused(c, glsl, fs, msl[1], path, "header does not match the data",
                "header with a plus sign");
        spit(path, orig, on);
        truncate(path, R300_MSL_CACHE_MAX_FILE + 1);
        refused(c, glsl, fs, msl[1], path, "file too large", "oversized file");
        unlink(path);
        mkdir(path, 0700);
        refused(c, glsl, fs, msl[1], path, "not a regular file", "directory in the way");
        CHECK(!r300_msl_cache_put(c, glsl, fs, msl[1], &why), "put over a directory");
        rmdir(path);
        CHECK(files_in(r300_msl_cache_dir(c), ".msl", &tmp) >= 2 && !tmp, "temporaries left");
        unlink(path);
        {
            char other[1200];
            key_hex(glsl, R300_STAGE_VS, key);
            p = entry_path(c, key);
            snprintf(other, sizeof(other), "%s", p);
            free(p);
            symlink(other, path);
        }
        refused(c, glsl, fs, msl[1], path, "cannot open", "symlink");
        if (!root_user) {
            chmod(path, 0);
            refused(c, glsl, fs, msl[1], path, "cannot open", "unreadable file");
        }
        m = slurp(path, &n);
        CHECK(m && n == on && !memcmp(m, orig, on), "entry not back to put's bytes");
        free(m);
        free(orig);
    }

    /* unwritable storage */
    glsl3 = malloc(strlen(glsl) + 16);
    sprintf(glsl3, "%s\n// third\n", glsl);
    if (!root_user) {
        uint64_t wf = st(c).write_failures;
        chmod(r300_msl_cache_dir(c), 0500);
        why = NULL;
        CHECK(!r300_msl_cache_put(c, glsl3, R300_STAGE_FS, msl[1], &why) &&
              why && !strcmp(why, "cannot write the entry"), "put into a read-only directory");
        CHECK(st(c).write_failures == wf + 1, "write failure not counted");
        m = r300_msl_cache_get(c, glsl, R300_STAGE_VS, &why);
        CHECK(m && !strcmp(m, msl[0]), "read-only directory: entries unreadable");
        free(m);
        m = r300_msl_cache_get(c, glsl3, R300_STAGE_FS, &why);
        CHECK(!m && !why, "read-only directory: a miss is not a plain miss");
        chmod(r300_msl_cache_dir(c), 0700);
        CHECK(files_in(r300_msl_cache_dir(c), ".msl", &tmp) == 3 && !tmp, "read-only: files changed");

        snprintf(path, sizeof(path), "%s/ro", argv[2]);
        snprintf(cmd, sizeof(cmd), "chmod u+w '%s' 2>/dev/null; rm -rf '%s'; mkdir '%s'; chmod 0500 '%s'",
                 path, path, path, path);
        system(cmd);
        err = NULL;
        c2 = r300_msl_cache_open(path, "identity A", &err);
        CHECK(!c2 && err, "open in a read-only root");
        printf("  open in a read-only root: %s\n", err ? err : "(opened)");
        free(err);
        r300_msl_cache_close(c2);
        chmod(path, 0700);
        rmdir(path);
        printf("  read-only directory: put refused, reads fine\n");
    }
    snprintf(path, sizeof(path), "%s/plainfile", argv[2]);
    spit(path, "x", 1);
    err = NULL;
    c2 = r300_msl_cache_open(path, "identity A", &err);
    CHECK(!c2 && err, "open with a file as root");
    printf("  open with a file as root: %s\n", err ? err : "(opened)");
    free(err);
    unlink(path);

    /* identity.txt that does not match its directory */
    m = slurp(idp, &n);
    spit(idp, "something else", 14);
    err = NULL;
    c2 = r300_msl_cache_open(root, "identity A", &err);
    CHECK(!c2 && err && strstr(err, "content differs"), "damaged identity.txt accepted");
    printf("  damaged identity.txt: %s\n", err ? err : "(opened)");
    free(err);
    spit(idp, m, n);
    free(m);

    /* sizes put refuses */
    m = malloc(R300_MSL_CACHE_MAX_MSL + 2);
    memset(m, 'a', R300_MSL_CACHE_MAX_MSL + 1);
    m[R300_MSL_CACHE_MAX_MSL + 1] = 0;
    CHECK(!r300_msl_cache_put(c, glsl3, R300_STAGE_VS, m, &why) && !strcmp(why, "MSL too large"),
          "oversized MSL written");
    CHECK(!r300_msl_cache_put(c, glsl3, R300_STAGE_VS, "", &why) && !strcmp(why, "empty MSL"),
          "empty MSL written");
    free(m);

    /* drop */
    r300_msl_cache_drop(c, glsl, R300_STAGE_FS_Z);
    m = r300_msl_cache_get(c, glsl, R300_STAGE_FS_Z, &why);
    CHECK(!m && !why && st(c).dropped == 1, "drop");
    CHECK(r300_msl_cache_put(c, glsl, R300_STAGE_FS_Z, msl[2], &why), "put after drop");

    /* processes at once: writers republish the same entry, one also drops
     * it; every read is a whole entry or a plain miss */
    {
        enum { NPROC = 8, ROUNDS = 400 };
        pid_t pid[NPROC];
        int bad = 0;

        for (int i = 0; i < NPROC; i++) {
            pid[i] = fork();
            if (pid[i] == 0) {
                R300MslCache *x = r300_msl_cache_open(root, "identity A", &err);
                int wrong = !x;
                for (int k = 0; x && k < ROUNDS; k++) {
                    char *y = r300_msl_cache_get(x, glsl, R300_STAGE_FS, &why);
                    if (y ? strcmp(y, msl[1]) != 0 : why != NULL) {
                        wrong++;
                    }
                    free(y);
                    if (i == 0 && k % 3 == 0) {
                        r300_msl_cache_drop(x, glsl, R300_STAGE_FS);
                    } else if (!r300_msl_cache_put(x, glsl, R300_STAGE_FS, msl[1], &why)) {
                        wrong++;
                    }
                }
                _exit(wrong ? 1 : 0);
            }
        }
        for (int i = 0; i < NPROC; i++) {
            int status;
            waitpid(pid[i], &status, 0);
            bad += !WIFEXITED(status) || WEXITSTATUS(status);
        }
        CHECK(!bad, "%d of %d processes saw a damaged entry or failed to write", bad, NPROC);
        r300_msl_cache_put(c, glsl, R300_STAGE_FS, msl[1], &why);
        m = r300_msl_cache_get(c, glsl, R300_STAGE_FS, &why);
        CHECK(m && !strcmp(m, msl[1]), "entry after the processes");
        free(m);
        CHECK(files_in(r300_msl_cache_dir(c), ".msl", &tmp) == 3 && !tmp,
              "after the processes: %d temporaries", tmp);
        printf("  %d processes x %d rounds of get+put (one dropping): %s\n", NPROC, ROUNDS,
               bad ? "FAILED" : "every read whole or a miss");
    }

    r300_msl_cache_close(c);
    snprintf(cmd, sizeof(cmd), "chmod -R u+rwx '%s' 2>/dev/null; rm -rf '%s'", root, root);
    system(cmd);
    if (fails) {
        printf("test_mslcache: %d FAILED\n", fails);
        return 1;
    }
    printf("test_mslcache: all passed\n");
    return 0;
}
