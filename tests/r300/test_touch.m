/*
 * R300DrawPacket.touch_y1 (tris_touch_y1 in r300_draw.c) against Metal's
 * own rasterizer: point squares built as emit_point builds them, each in
 * its own column band, drawn, and the lowest row each one covered
 * compared with its bound.  The bound must never be below it.
 */
#import <Metal/Metal.h>
#include <stdio.h>
#include "../../hw/display/r300/r300_draw.c"

#define BANDS 64
#define BAND 32

static const char *src =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "vertex float4 vs(uint i [[vertex_id]], constant float4 *p [[buffer(0)]]) { return p[i]; }\n"
    "fragment half4 fs() { return half4(1); }\n";

static uint32_t rng = 777;
static float frand(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return (rng >> 8) / 16777216.0f;
}

int main(void)
{
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        if (!dev) {
            printf("FAIL: no Metal device\n");
            return 1;
        }
        id<MTLLibrary> lib = [dev newLibraryWithSource:@(src) options:nil error:&err];
        if (!lib) {
            printf("FAIL: library: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
        pd.vertexFunction = [lib newFunctionWithName:@"vs"];
        pd.fragmentFunction = [lib newFunctionWithName:@"fs"];
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatR8Unorm;
        id<MTLRenderPipelineState> ps = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
        id<MTLCommandQueue> q = [dev newCommandQueue];
        if (!ps || !q) {
            printf("FAIL: pipeline or queue: %s\n", err ? err.localizedDescription.UTF8String : "no queue");
            return 1;
        }
        static const uint32_t heights[] = { 32, 33, 129, 300, 480, 481 };
        unsigned checked = 0, exact = 0, slack = 0, unknown = 0, fails = 0, drawn = 0;

        for (unsigned hi = 0; hi < sizeof(heights) / sizeof(heights[0]); hi++) {
            uint32_t H = heights[hi];
            R300DrawPacket pkt = { 0 };
            pkt.rt_width = BANDS * BAND;
            pkt.rt_height = H;
            MTLTextureDescriptor *td = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                width:pkt.rt_width height:H mipmapped:NO];
            td.usage = MTLTextureUsageRenderTarget;
            td.storageMode = MTLStorageModeShared;
            id<MTLTexture> t = [dev newTextureWithDescriptor:td];
            uint8_t *px = malloc((size_t)pkt.rt_width * H);
            if (!t || !px) {
                printf("FAIL: render target %ux%u\n", pkt.rt_width, H);
                return 1;
            }

            for (int round = 0; round < 40; round++) {
                R300Vertex sq[BANDS][6];
                uint32_t want[BANDS];
                float pos[BANDS * 6][4];
                for (int k = 0; k < BANDS; k++) {
                    /* GA_POINT_SIZE halves are twelfths of a pixel */
                    float hw = (1 + (int)(frand() * 120)) / 12.0f, hh = (1 + (int)(frand() * 120)) / 12.0f;
                    if (hw > BAND / 2 - 1) {
                        hw = BAND / 2 - 1;
                    }
                    float cx = k * BAND + BAND / 2 + (frand() - 0.5f) * (BAND - 2 * hw - 2);
                    float cy = frand() * (H + 4) - 2;
                    switch (round % 4) {        /* bottom edges on and near row boundaries */
                    case 1: cy = floorf(cy + hh) - hh; break;
                    case 2: cy = floorf(cy + hh) + 0.5f - hh; break;
                    case 3: cy = floorf(cy + hh) + (frand() - 0.5f) * 1e-3f - hh; break;
                    }
                    float x[4] = { cx - hw, cx + hw, cx + hw, cx - hw };
                    float y[4] = { cy - hh, cy - hh, cy + hh, cy + hh };
                    static const int corner[6] = { 0, 1, 2, 0, 2, 3 };
                    Build b = { .pkt = &pkt };
                    for (int v = 0; v < 6; v++) {
                        memset(&sq[k][v], 0, sizeof(sq[k][v]));
                        sq[k][v].pos[2] = 0.5f;
                        sq[k][v].pos[3] = 1.0f;
                        px_ndc(&b, x[corner[v]], y[corner[v]], sq[k][v].pos);
                        memcpy(pos[k * 6 + v], sq[k][v].pos, sizeof(pos[0]));
                    }
                    want[k] = tris_touch_y1(&pkt, sq[k], 6);
                }
                MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
                rp.colorAttachments[0].texture = t;
                rp.colorAttachments[0].loadAction = MTLLoadActionClear;
                rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
                rp.colorAttachments[0].storeAction = MTLStoreActionStore;
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
                [e setRenderPipelineState:ps];
                [e setViewport:(MTLViewport){ 0, 0, pkt.rt_width, H, 0, 1 }];
                id<MTLBuffer> vb = [dev newBufferWithBytes:pos length:sizeof(pos)
                                                   options:MTLResourceStorageModeShared];
                if (!vb) {
                    printf("FAIL: vertex buffer\n");
                    return 1;
                }
                [e setVertexBuffer:vb offset:0 atIndex:0];
                [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:BANDS * 6];
                [e endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.status != MTLCommandBufferStatusCompleted) {
                    printf("FAIL: command buffer status %ld: %s\n", (long)cb.status,
                           cb.error ? cb.error.localizedDescription.UTF8String : "");
                    return 1;
                }
                [t getBytes:px bytesPerRow:pkt.rt_width
                 fromRegion:MTLRegionMake2D(0, 0, pkt.rt_width, H) mipmapLevel:0];
                for (int k = 0; k < BANDS; k++) {
                    uint32_t got = 0;           /* rows [0, got) reach the lowest covered */
                    for (uint32_t r = 0; r < H; r++) {
                        for (int c = k * BAND; c < (k + 1) * BAND; c++) {
                            if (px[r * pkt.rt_width + c]) {
                                got = r + 1;
                                break;
                            }
                        }
                    }
                    checked++;
                    drawn += got != 0;
                    if (!want[k]) {
                        unknown++;              /* 0: not known, so rt_height */
                    } else if (got > want[k]) {
                        if (fails++ < 10) {
                            printf("FAIL H=%u band %d: covered up to row %u, bound %u (bottom NDC y %.6f)\n",
                                   H, k, got, want[k], sq[k][2].pos[1]);
                        }
                    } else if (got == want[k]) {
                        exact++;
                    } else {
                        slack++;
                    }
                }
            }
            free(px);
        }
        printf("touch_y1: %u squares (%u drew pixels), bound exact %u, above %u, not known %u, below (FAIL) %u\n",
               checked, drawn, exact, slack, unknown, fails);
        /* a draw that silently did nothing would leave every bound looking safe */
        if (drawn < checked / 2) {
            printf("FAIL: only %u of %u squares drew anything\n", drawn, checked);
            return 1;
        }
        return fails != 0;
    }
}
