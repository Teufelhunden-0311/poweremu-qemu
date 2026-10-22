/*
 * R300 MSL disk cache: r300_glsl_to_msl() results kept across sessions,
 * so a later session skips shaderc and SPIRV-Cross (most of the cost of
 * a new shader; Metal caches its own compile of the MSL).
 *
 * One directory per converter identity (the caller's description of
 * everything besides the GLSL that shapes the MSL), named by its SHA-256;
 * one file per GLSL source and stage in it, named by their SHA-256.  An
 * entry is used only when its schema, stage, key hash, lengths and MSL
 * hash all check out; anything else is a miss, and the caller's next put
 * replaces it.  Writers publish by rename(), so readers in this and other
 * processes see a whole entry or none (a writer killed mid-put can leave
 * a *.tmp-* file, which nothing reads).  The cache is disposable: deleting
 * the directory loses nothing but time.  There is no eviction.
 *
 * Pure C (POSIX, zlib, CommonCrypto), no QEMU dependencies: macOS only,
 * like the Metal backend that uses it.  Safe to call from several threads
 * and processes at once.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#ifndef HW_DISPLAY_R300_MSL_CACHE_H
#define HW_DISPLAY_R300_MSL_CACHE_H

#include <stdbool.h>
#include <stdint.h>

#include "r300_spirv.h"

/* Entry layout revision; part of every directory name and entry header. */
#define R300_MSL_CACHE_SCHEMA 1

/* Largest MSL kept, and largest entry file read (Doom 3's largest MSL
 * is about 42 KB). */
#define R300_MSL_CACHE_MAX_MSL   (1u << 20)
#define R300_MSL_CACHE_MAX_FILE  (1u << 20)

typedef struct R300MslCache R300MslCache;

typedef struct R300MslCacheStats {
    uint64_t hits;              /* entries returned by get */
    uint64_t misses;            /* no entry */
    uint64_t rejected;          /* an entry failed a check, or could not be read */
    uint64_t writes;            /* entries published by put */
    uint64_t write_failures;    /* put gave up; nothing published */
    uint64_t dropped;           /* entries removed by drop */
    uint64_t bytes_written;     /* entry file bytes published */
} R300MslCacheStats;

/*
 * Open the cache directory for one converter identity under root
 * (root/<sha256 of schema + identity>/), creating root's last component
 * and that directory as needed, and identity.txt in it (the identity in
 * plain text, for people).  Returns NULL with a malloc'd *err when the
 * cache cannot be used; the caller then converts without it.
 */
R300MslCache *r300_msl_cache_open(const char *root, const char *identity,
                                  char **err);
void r300_msl_cache_close(R300MslCache *c);

/* The identity directory. */
const char *r300_msl_cache_dir(const R300MslCache *c);

/*
 * The cached MSL for glsl and stage, malloc'd, or NULL.  On NULL, *why
 * (if why is not NULL) says why: NULL for a plain miss, else a static
 * string naming the failed check (the entry was counted as rejected).
 */
char *r300_msl_cache_get(R300MslCache *c, const char *glsl, R300Stage stage,
                         const char **why);

/*
 * Publish msl as the entry for glsl and stage, replacing any entry there.
 * Returns false, with a static string in *why, when nothing was published;
 * the cache is unchanged then.
 */
bool r300_msl_cache_put(R300MslCache *c, const char *glsl, R300Stage stage,
                        const char *msl, const char **why);

/* Remove the entry for glsl and stage, if there is one. */
void r300_msl_cache_drop(R300MslCache *c, const char *glsl, R300Stage stage);

void r300_msl_cache_stats(R300MslCache *c, R300MslCacheStats *s);

#endif
