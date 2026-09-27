/*
 * Tests: the Metal functions of a r300_us_to_glsl() source, compiled the
 * way the Metal backend does (GLSL -> SPIR-V -> MSL).  Index the result
 * by R300Stage.  Returns nil after printing the error.
 */
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include "../../hw/display/r300/r300_spirv.h"

static inline NSArray<id<MTLFunction>> *r300_test_lib(id<MTLDevice> dev, const char *glsl)
{
    NSMutableArray<id<MTLFunction>> *fns = [NSMutableArray array];
    for (int s = R300_STAGE_VS; s <= R300_STAGE_FS_Z; s++) {
        char *err = NULL;
        char *msl = r300_glsl_to_msl(glsl, (R300Stage)s, &err);
        if (!msl) {
            printf("%s: %s\n%s\n", r300_stage_entry(s), err, glsl);
            free(err);
            return nil;
        }
        NSError *e = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:@(msl) options:nil error:&e];
        if (!lib) {
            printf("compile %s: %s\n%s\n", r300_stage_entry(s),
                   e.localizedDescription.UTF8String, msl);
            free(msl);
            return nil;
        }
        free(msl);
        [fns addObject:[lib newFunctionWithName:@(r300_stage_entry(s))]];
    }
    return fns;
}
