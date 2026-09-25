/* Compile an MSL file with the runtime compiler, as QEMU will. */
#import <Metal/Metal.h>
#include <stdio.h>
int main(int argc, char **argv)
{
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSString *src = [NSString stringWithContentsOfFile:@(argv[1])
                                                  encoding:NSUTF8StringEncoding error:nil];
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { printf("FAIL: %s\n", err.localizedDescription.UTF8String); return 1; }
        printf("OK: %s\n", [[lib functionNames] componentsJoinedByString:@", "].UTF8String);
    }
    return 0;
}
