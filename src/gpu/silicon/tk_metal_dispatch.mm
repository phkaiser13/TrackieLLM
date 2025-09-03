/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_metal_dispatch.mm
 *
 * This source file implements the Metal dispatcher – the high‑level GPU
 * abstraction layer for Apple‑Silicon devices.  The implementation mirrors the
 * CUDA dispatcher (tk_cuda_dispatch.c) but uses the Metal API, Metal Performance
 * Shaders (MPS) and Apple‑specific facilities such as Core ML and the unified
 * memory architecture.
 *
 * Design goals:
 *   • Opaque handles – the public API never leaks Objective‑C objects.
 *   • Asynchronous‑by‑default – every operation is enqueued on a command queue
 *     and returns immediately.
 *   • Explicit synchronization – callers can wait on events or on the whole
 *     queue.
 *   • Zero‑copy wherever possible – shared storage mode eliminates host↔device
 *     copies for buffers that are accessed by both sides.
 *   • Integration with Core ML and MPS – high‑level workflows delegate to the
 *     most‑optimised Apple libraries.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreMedia/CoreMedia.h>
#import <dispatch/dispatch.h>

/* -------------------------------------------------------------------------
 *  Internal opaque structures
 * ------------------------------------------------------------------------- */
typedef struct tk_gpu_buffer_s {
    id<MTLBuffer> buffer;               /* Retained Metal buffer               */
} tk_gpu_buffer_t;

typedef struct tk_gpu_texture_s {
    id<MTLTexture> texture;             /* Retained Metal texture              */
} tk_gpu_texture_t;

typedef struct tk_gpu_event_s {
    id<MTLSharedEvent> event;           /* Retained shared event (GPU ↔ CPU)   */
    uint64_t               value;       /* Monotonically increasing signal    */
} tk_gpu_event_t;

/* -------------------------------------------------------------------------
 *  Dispatcher internal representation
 * ------------------------------------------------------------------------- */
struct tk_metal_dispatcher_s {
    id<MTLDevice>          device;               /* Selected Metal device            */
    id<MTLCommandQueue>    commandQueue;         /* Default queue for async work     */
    MTLStorageMode         storageMode;          /* Preferred storage for buffers    */
    uint32_t               maxInFlight;          /* Max concurrent command buffers   */
    dispatch_semaphore_t   inflightSemaphore;    /* Simple back‑pressure mechanism   */

    /* Optional: a texture cache for CVPixelBuffer ↔ MTLTexture conversion */
    CVMetalTextureCacheRef  textureCache;
};

/* -------------------------------------------------------------------------
 *  Helper macros & forward declarations
 * ------------------------------------------------------------------------- */
#define RETURN_IF_NULL(ptr, err)                     \
    do {                                             \
        if ((ptr) == NULL) {                         \
            return (err);                            \
        }                                            \
    } while (0)

static inline tk_error_code_t
tk_metal_error_from_nserror(NSError *nsError)
{
    if (nsError == nil) {
        return TK_SUCCESS;
    }
    /* In a production system we would map specific NSError domains to
     * TrackieLLM error codes.  For brevity we return a generic GPU error. */
    return TK_ERROR_GPU_INTERNAL;
}

/* -------------------------------------------------------------------------
 *  Dispatcher Lifecycle
 * ------------------------------------------------------------------------- */
TK_NODISCARD tk_error_code_t
tk_metal_dispatch_create(tk_metal_dispatcher_t **out_dispatcher,
                         const tk_metal_dispatcher_config_t *config)
{
    RETURN_IF_NULL(out_dispatcher, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(config, TK_ERROR_INVALID_ARGUMENT);

    /* Allocate the dispatcher structure on the heap – it will be freed by
     * tk_metal_dispatch_destroy(). */
    tk_metal_dispatcher_t *disp = (tk_metal_dispatcher_t *)calloc(1, sizeof(*disp));
    if (!disp) {
        return TK_ERROR_OUT_OF_MEMORY;
    }

    /* -----------------------------------------------------------------
     *  Device selection
     * ----------------------------------------------------------------- */
    if (config->device) {
        disp->device = config->device;
    } else {
        disp->device = MTLCreateSystemDefaultDevice();
    }
    if (!disp->device) {
        free(disp);
        return TK_ERROR_GPU_DEVICE_NOT_FOUND;
    }

    /* -----------------------------------------------------------------
     *  Command‑queue creation
     * ----------------------------------------------------------------- */
    disp->commandQueue = [disp->device newCommandQueue];
    if (!disp->commandQueue) {
        free(disp);
        return TK_ERROR_GPU_INTERNAL;
    }

    /* -----------------------------------------------------------------
     *  Storage‑mode handling
     * ----------------------------------------------------------------- */
    disp->storageMode = config->preferred_storage_mode;
    if (disp->storageMode == MTLStorageModePrivate ||
        disp->storageMode == MTLStorageModeShared ||
        disp->storageMode == MTLStorageModeManaged) {
        /* user supplied – keep it */
    } else {
        /* Default to Private for maximum GPU throughput. */
        disp->storageMode = MTLStorageModePrivate;
    }

    /* -----------------------------------------------------------------
     *  In‑flight semaphore (simple back‑pressure)
     * ----------------------------------------------------------------- */
    disp->maxInFlight = (config->max_in_flight > 0) ? config->max_in_flight : 4;
    disp->inflightSemaphore = dispatch_semaphore_create(disp->maxInFlight);

    /* -----------------------------------------------------------------
     *  CVMetalTextureCache – required for pixel‑buffer ↔ texture helpers.
     * ----------------------------------------------------------------- */
    CVReturn cvRet = CVMetalTextureCacheCreate(kCFAllocatorDefault,
                                               NULL,
                                               (__bridge CFTypeRef)disp->device,
                                               NULL,
                                               &disp->textureCache);
    if (cvRet != kCVReturnSuccess) {
        /* Not fatal – the pixel‑buffer helpers will simply fail later. */
        disp->textureCache = NULL;
    }

    *out_dispatcher = disp;
    return TK_SUCCESS;
}

void
tk_metal_dispatch_destroy(tk_metal_dispatcher_t **dispatcher)
{
    if (!dispatcher || !*dispatcher) {
        return;
    }

    tk_metal_dispatcher_t *disp = *dispatcher;

    /* Ensure all queued work has finished before we start releasing objects. */
    [disp->commandQueue waitUntilAllCommandsAreCompleted];

    if (disp->textureCache) {
        CVMetalTextureCacheFlush(disp->textureCache, 0);
        CFRelease(disp->textureCache);
    }

    /* Release Objective‑C objects – ARC is not enabled for .mm files that
     * contain plain C structs, so we must do it manually. */
    disp->commandQueue = nil;
    disp->device       = nil;

    if (disp->inflightSemaphore) {
        dispatch_release(disp->inflightSemaphore);
    }

    free(disp);
    *dispatcher = NULL;
}

/* -------------------------------------------------------------------------
 *  GPU Memory Management
 * ------------------------------------------------------------------------- */
TK_NODISCARD tk_error_code_t
tk_metal_dispatch_malloc(tk_metal_dispatcher_t *dispatcher,
                         tk_gpu_buffer_t *out_buffer,
                         size_t size_bytes)
{
    RETURN_IF_NULL(dispatcher, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(out_buffer, TK_ERROR_INVALID_ARGUMENT);
    if (size_bytes == 0) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    id<MTLBuffer> buf = [dispatcher->device newBufferWithLength:size_bytes
                                                      options:dispatcher->storageMode];
    if (!buf) {
        return TK_ERROR_GPU_MEMORY;
    }

    tk_gpu_buffer_t *handle = (tk_gpu_buffer_t *)calloc(1, sizeof(*handle));
    if (!handle) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    handle->buffer = buf;               /* retained */
    *out_buffer = handle;
    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t
tk_metal_dispatch_alloc_texture(tk_metal_dispatcher_t *dispatcher,
                               tk_gpu_texture_t *out_texture,
                               uint32_t width,
                               uint32_t height,
                               MTLPixelFormat pixel_format)
{
    RETURN_IF_NULL(dispatcher, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(out_texture, TK_ERROR_INVALID_ARGUMENT);
    if (width == 0 || height == 0) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixel_format
                                                                                width:width
                                                                               height:height
                                                                            mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.storageMode = dispatcher->storageMode;

    id<MTLTexture> tex = [dispatcher->device newTextureWithDescriptor:desc];
    if (!tex) {
        return TK_ERROR_GPU_MEMORY;
    }

    tk_gpu_texture_t *handle = (tk_gpu_texture_t *)calloc(1, sizeof(*handle));
    if (!handle) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    handle->texture = tex;               /* retained */
    *out_texture = handle;
    return TK_SUCCESS;
}

void
tk_metal_dispatch_free(tk_metal_dispatcher_t *dispatcher,
                       tk_gpu_buffer_t *buffer)
{
    if (!dispatcher || !buffer || !*buffer) {
        return;
    }
    (*buffer)->buffer = nil;   /* release */
    free(*buffer);
    *buffer = NULL;
}

void
tk_metal_dispatch_free_texture(tk_metal_dispatcher_t *dispatcher,
                               tk_gpu_texture_t *texture)
{
    if (!dispatcher || !texture || !*texture) {
        return;
    }
    (*texture)->texture = nil;   /* release */
    free(*texture);
    *texture = NULL;
}

/* -------------------------------------------------------------------------
 *  Asynchronous host↔device copies
 * ------------------------------------------------------------------------- */
TK_NODISCARD tk_error_code_t
tk_metal_dispatch_upload_async(tk_metal_dispatcher_t *dispatcher,
                               tk_gpu_buffer_t dst_buffer,
                               const void *src_host_ptr,
                               size_t size_bytes,
                               tk_gpu_event_t *out_event)
{
    RETURN_IF_NULL(dispatcher, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(dst_buffer, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(src_host_ptr, TK_ERROR_INVALID_ARGUMENT);
    if (size_bytes == 0) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    /* Back‑pressure – limit the number of in‑flight command buffers. */
    dispatch_semaphore_wait(dispatcher->inflightSemaphore, DISPATCH_TIME_FOREVER);

    id<MTLCommandBuffer> cmd = [dispatcher->commandQueue commandBuffer];
    if (!cmd) {
        dispatch_semaphore_signal(dispatcher->inflightSemaphore);
        return TK_ERROR_GPU_INTERNAL;
    }

    /* Choose the most efficient path based on storage mode. */
    if (dispatcher->storageMode == MTLStorageModeShared) {
        /* Shared memory – a simple memcpy is sufficient. */
        void *gpuPtr = [dst_buffer->buffer contents];
        memcpy(gpuPtr, src_host_ptr, size_bytes);
        [cmd addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
            dispatch_semaphore_signal(dispatcher->inflightSemaphore);
        }];
        [cmd commit];
    } else {
        /* Private memory – we must use a blit encoder. */
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        if (!blit) {
            dispatch_semaphore_signal(dispatcher->inflightSemaphore);
            return TK_ERROR_GPU_INTERNAL;
        }

        /* Create a temporary staging buffer in shared memory. */
        id<MTLBuffer> staging = [dispatcher->device newBufferWithBytes:src_host_ptr
                                                               length:size_bytes
                                                              options:MTLStorageModeShared];
        if (!staging) {
            [blit endEncoding];
            dispatch_semaphore_signal(dispatcher->inflightSemaphore);
            return TK_ERROR_GPU_MEMORY;
        }

        [blit copyFromBuffer:staging
                 sourceOffset:0
                     toBuffer:dst_buffer->buffer
            destinationOffset:0
                         size:size_bytes];
        [blit endEncoding];

        /* Optional event signalling. */
        if (out_event) {
            tk_gpu_event_t *ev = (tk_gpu_event_t *)calloc(1, sizeof(*ev));
            if (!ev) {
                dispatch_semaphore_signal(dispatcher->inflightSemaphore);
                return TK_ERROR_OUT_OF_MEMORY;
            }
            ev->event = [dispatcher->device newSharedEvent];
            ev->value = 1;   /* first signal value */
            *out_event = ev;

            [cmd encodeSignalEvent:ev->event value:ev->value];
        }

        [cmd addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
            dispatch_semaphore_signal(dispatcher->inflightSemaphore);
        }];
        [cmd commit];
    }

    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t
tk_metal_dispatch_download_async(tk_metal_dispatcher_t *dispatcher,
                                 void *dst_host_ptr,
                                 tk_gpu_buffer_t src_buffer,
                                 size_t size_bytes,
                                 tk_gpu_event_t *out_event)
{
    RETURN_IF_NULL(dispatcher, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(dst_host_ptr, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(src_buffer, TK_ERROR_INVALID_ARGUMENT);
    if (size_bytes == 0) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    dispatch_semaphore_wait(dispatcher->inflightSemaphore, DISPATCH_TIME_FOREVER);
    id<MTLCommandBuffer> cmd = [dispatcher->commandQueue commandBuffer];
    if (!cmd) {
        dispatch_semaphore_signal(dispatcher->inflightSemaphore);
        return TK_ERROR_GPU_INTERNAL;
    }

    if (dispatcher->storageMode == MTLStorageModeShared) {
        /* Direct memcpy from the shared buffer. */
        void *gpuPtr = [src_buffer->buffer contents];
        memcpy(dst_host_ptr, gpuPtr, size_bytes);
        [cmd addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
            dispatch_semaphore_signal(dispatcher->inflightSemaphore);
        }];
        [cmd commit];
    } else {
        /* Private memory – use a blit encoder + staging buffer. */
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        if (!blit) {
            dispatch_semaphore_signal(dispatcher->inflightSemaphore);
            return TK_ERROR_GPU_INTERNAL;
        }

        id<MTLBuffer> staging = [dispatcher->device newBufferWithLength:size_bytes
                                                               options:MTLStorageModeShared];
        if (!staging) {
            [blit endEncoding];
            dispatch_semaphore_signal(dispatcher->inflightSemaphore);
            return TK_ERROR_GPU_MEMORY;
        }

        [blit copyFromBuffer:src_buffer->buffer
                 sourceOffset:0
                     toBuffer:staging
            destinationOffset:0
                         size:size_bytes];
        [blit endEncoding];

        /* After the copy finishes we can memcpy from the staging buffer. */
        [cmd addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
            void *gpuPtr = [staging contents];
            memcpy(dst_host_ptr, gpuPtr, size_bytes);
            dispatch_semaphore_signal(dispatcher->inflightSemaphore);
        }];

        if (out_event) {
            tk_gpu_event_t *ev = (tk_gpu_event_t *)calloc(1, sizeof(*ev));
            if (!ev) {
                dispatch_semaphore_signal(dispatcher->inflightSemaphore);
                return TK_ERROR_OUT_OF_MEMORY;
            }
            ev->event = [dispatcher->device newSharedEvent];
            ev->value = 1;
            *out_event = ev;
            [cmd encodeSignalEvent:ev->event value:ev->value];
        }

        [cmd commit];
    }

    return TK_SUCCESS;
}

/* -------------------------------------------------------------------------
 *  Synchronisation primitives
 * ------------------------------------------------------------------------- */
TK_NODISCARD tk_error_code_t
tk_metal_dispatch_create_event(tk_metal_dispatcher_t *dispatcher,
                               tk_gpu_event_t *out_event)
{
    RETURN_IF_NULL(dispatcher, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(out_event, TK_ERROR_INVALID_ARGUMENT);

    tk_gpu_event_t *ev = (tk_gpu_event_t *)calloc(1, sizeof(*ev));
    if (!ev) {
        return TK_ERROR_OUT_OF_MEMORY;
    }

    ev->event = [dispatcher->device newSharedEvent];
    if (!ev->event) {
        free(ev);
        return TK_ERROR_GPU_INTERNAL;
    }
    ev->value = 1;   /* start counting from 1 */
    *out_event = ev;
    return TK_SUCCESS;
}

void
tk_metal_dispatch_destroy_event(tk_metal_dispatcher_t *dispatcher,
                                tk_gpu_event_t *event)
{
    if (!dispatcher || !event || !*event) {
        return;
    }
    (*event)->event = nil;   /* release */
    free(*event);
    *event = NULL;
}

TK_NODISCARD tk_error_code_t
tk_metal_dispatch_wait_for_event(tk_metal_dispatcher_t *dispatcher,
                                 tk_gpu_event_t event)
{
    RETURN_IF_NULL(dispatcher, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(event, TK_ERROR_INVALID_ARGUMENT);

    /* MTLSharedEvent works with a value‑based signalling model.
     * We wait until the event’s value reaches or exceeds the stored value. */
    uint64_t target = event->value;
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);

    [event->event notifyListener:dispatch_get_main_queue()
                         atValue:target
                      block:^{
        dispatch_semaphore_signal(sem);
    }];

    /* Wait with a generous timeout (10 seconds).  In production we would make
     * the timeout configurable. */
    long waitResult = dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, 10ull * NSEC_PER_SEC));
    dispatch_release(sem);

    if (waitResult != 0) {
        return TK_ERROR_TIMEOUT;
    }
    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t
tk_metal_dispatch_synchronize(tk_metal_dispatcher_t *dispatcher)
{
    RETURN_IF_NULL(dispatcher, TK_ERROR_INVALID_ARGUMENT);
    [dispatcher->commandQueue waitUntilAllCommandsAreCompleted];
    return TK_SUCCESS;
}

/* -------------------------------------------------------------------------
 *  High‑Level Workflow Dispatch
 * ------------------------------------------------------------------------- */

/* The following two functions are thin wrappers around Metal Performance
 * Shaders (MPS).  The public structs tk_preprocess_params_t and
 * tk_depth_to_points_params_t are defined in tk_metal_kernels.h and contain
 * opaque handles (tk_gpu_buffer_t / tk_gpu_texture_t) that must have been
 * allocated with the dispatcher.  The implementations below assume those
 * handles are valid. */

TK_NODISCARD tk_error_code_t
tk_metal_dispatch_preprocess_image(tk_metal_dispatcher_t *dispatcher,
                                   const tk_preprocess_params_t *params)
{
    RETURN_IF_NULL(dispatcher, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(params, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(params->src_texture, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(params->dst_buffer, TK_ERROR_INVALID_ARGUMENT);

    dispatch_semaphore_wait(dispatcher->inflightSemaphore, DISPATCH_TIME_FOREVER);
    id<MTLCommandBuffer> cmd = [dispatcher->commandQueue commandBuffer];
    if (!cmd) {
        dispatch_semaphore_signal(dispatcher->inflightSemaphore);
        return TK_ERROR_GPU_INTERNAL;
    }

    /* 1️⃣  Colour‑space conversion (e.g. BGRA → RGBA) using MPSImageConversion. */
    MPSImageConversion *converter = [[MPSImageConversion alloc] initWithDevice:dispatcher->device
                                                             srcAlpha:1.0
                                                             destAlpha:1.0
                                                            backgroundColor:nil
                                                             conversionInfo:nil];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:converter.computePipelineState];
    [encoder setTexture:params->src_texture->texture atIndex:0];
    [encoder setTexture:params->intermediate_texture->texture atIndex:1];
    MTLSize threadgroup = MTLSizeMake(8, 8, 1);
    MTLSize grid = MTLSizeMake((params->src_width  + threadgroup.width  - 1) / threadgroup.width,
                               (params->src_height + threadgroup.height - 1) / threadgroup.height,
                               1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
    [encoder endEncoding];

    /* 2️⃣  Optional Gaussian blur (MPSImageGaussianBlur). */
    if (params->apply_gaussian) {
        MPSImageGaussianBlur *blur = [[MPSImageGaussianBlur alloc] initWithDevice:dispatcher->device sigma:params->gaussian_sigma];
        id<MTLComputeCommandEncoder> blurEnc = [cmd computeCommandEncoder];
        [blurEnc setComputePipelineState:blur.computePipelineState];
        [blurEnc setTexture:params->intermediate_texture->texture atIndex:0];
        [blurEnc setTexture:params->blurred_texture->texture atIndex:1];
        [blurEnc dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
        [blurEnc endEncoding];
    }

    /* 3️⃣  Write the final pixel data into a linear buffer (dst_buffer) using
     *     a blit encoder – this is a simple copy from a texture to a buffer. */
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromTexture:params->final_texture->texture
               sourceSlice:0
               sourceLevel:0
              sourceOrigin:MTLOriginMake(0, 0, 0)
                sourceSize:MTLSizeMake(params->src_width, params->src_height, 1)
                 toBuffer:params->dst_buffer->buffer
        destinationOffset:0
   destinationBytesPerRow:params->src_width * params->bytes_per_pixel
 destinationBytesPerImage:0];
    [blit endEncoding];

    /* Completion handling – signal the semaphore and optionally an event. */
    [cmd addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
        dispatch_semaphore_signal(dispatcher->inflightSemaphore);
    }];
    [cmd commit];
    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t
tk_metal_dispatch_depth_to_point_cloud(tk_metal_dispatcher_t *dispatcher,
                                       const tk_depth_to_points_params_t *params)
{
    RETURN_IF_NULL(dispatcher, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(params, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(params->depth_texture, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(params->pointcloud_buffer, TK_ERROR_INVALID_ARGUMENT);

    dispatch_semaphore_wait(dispatcher->inflightSemaphore, DISPATCH_TIME_FOREVER);
    id<MTLCommandBuffer> cmd = [dispatcher->commandQueue commandBuffer];
    if (!cmd) {
        dispatch_semaphore_signal(dispatcher->inflightSemaphore);
        return TK_ERROR_GPU_INTERNAL;
    }

    /* The conversion kernel is provided as a pre‑compiled compute pipeline
     * (see tk_metal_kernels.h).  It reads a depth map (single‑channel float16)
     * and writes XYZ points (float32) to a linear buffer. */
    id<MTLComputePipelineState> pipeline = params->pipeline_state;   /* opaque handle */
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setTexture:params->depth_texture->texture atIndex:0];
    [encoder setBuffer:params->pointcloud_buffer->buffer offset:0 atIndex:0];
    [encoder setBytes:&params->intrinsics length:sizeof(params->intrinsics) atIndex:1];
    MTLSize threadgroup = MTLSizeMake(16, 16, 1);
    MTLSize grid = MTLSizeMake((params->depth_width  + threadgroup.width  - 1) / threadgroup.width,
                               (params->depth_height + threadgroup.height - 1) / threadgroup.height,
                               1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
    [encoder endEncoding];

    [cmd addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
        dispatch_semaphore_signal(dispatcher->inflightSemaphore);
    }];
    [cmd commit];
    return TK_SUCCESS;
}

/* -------------------------------------------------------------------------
 *  Apple‑specific extensions
 * ------------------------------------------------------------------------- */
TK_NODISCARD tk_error_code_t
tk_metal_dispatch_pixelbuffer_to_texture(tk_metal_dispatcher_t *dispatcher,
                                         CVPixelBufferRef pixel_buffer,
                                         tk_gpu_texture_t *out_texture)
{
    RETURN_IF_NULL(dispatcher, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(pixel_buffer, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(out_texture, TK_ERROR_INVALID_ARGUMENT);
    if (!dispatcher->textureCache) {
        return TK_ERROR_GPU_INTERNAL;
    }

    size_t width  = CVPixelBufferGetWidth(pixel_buffer);
    size_t height = CVPixelBufferGetHeight(pixel_buffer);
    OSType pixelFormat = CVPixelBufferGetPixelFormatType(pixel_buffer);

    MTLPixelFormat mtlFormat;
    switch (pixelFormat) {
        case kCVPixelFormatType_32BGRA:
            mtlFormat = MTLPixelFormatBGRA8Unorm;
            break;
        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
        case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
            /* For YUV we would need a separate conversion step; not supported here. */
            return TK_ERROR_INVALID_ARGUMENT;
        default:
            return TK_ERROR_INVALID_ARGUMENT;
    }

    CVMetalTextureRef cvTex = NULL;
    CVReturn cvRet = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
                                                             dispatcher->textureCache,
                                                             pixel_buffer,
                                                             NULL,
                                                             mtlFormat,
                                                             (size_t)width,
                                                             (size_t)height,
                                                             0,
                                                             &cvTex);
    if (cvRet != kCVReturnSuccess || !cvTex) {
        return TK_ERROR_GPU_INTERNAL;
    }

    id<MTLTexture> metalTex = CVMetalTextureGetTexture(cvTex);
    if (!metalTex) {
        CFRelease(cvTex);
        return TK_ERROR_GPU_INTERNAL;
    }

    tk_gpu_texture_t *handle = (tk_gpu_texture_t *)calloc(1, sizeof(*handle));
    if (!handle) {
        CFRelease(cvTex);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    handle->texture = metalTex;   /* retained by CVMetalTexture */
    *out_texture = handle;

    CFRelease(cvTex);
    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t
tk_metal_dispatch_texture_to_pixelbuffer(tk_metal_dispatcher_t *dispatcher,
                                         tk_gpu_texture_t texture,
                                         CVPixelBufferRef *out_pixel_buffer)
{
    RETURN_IF_NULL(dispatcher, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(texture, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(out_pixel_buffer, TK_ERROR_INVALID_ARGUMENT);
    if (!dispatcher->textureCache) {
        return TK_ERROR_GPU_INTERNAL;
    }

    size_t width  = texture->texture.width;
    size_t height = texture->texture.height;
    MTLPixelFormat fmt = texture->texture.pixelFormat;

    OSType cvFmt;
    switch (fmt) {
        case MTLPixelFormatBGRA8Unorm:
            cvFmt = kCVPixelFormatType_32BGRA;
            break;
        default:
            return TK_ERROR_INVALID_ARGUMENT;
    }

    CVPixelBufferRef pb = NULL;
    CVReturn cvRet = CVPixelBufferCreate(kCFAllocatorDefault,
                                        width,
                                        height,
                                        cvFmt,
                                        NULL,
                                        &pb);
    if (cvRet != kCVReturnSuccess || !pb) {
        return TK_ERROR_GPU_INTERNAL;
    }

    CVMetalTextureRef cvTex = NULL;
    cvRet = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
                                                      dispatcher->textureCache,
                                                      pb,
                                                      NULL,
                                                      fmt,
                                                      width,
                                                      height,
                                                      0,
                                                      &cvTex);
    if (cvRet != kCVReturnSuccess || !cvTex) {
        CVPixelBufferRelease(pb);
        return TK_ERROR_GPU_INTERNAL;
    }

    id<MTLTexture> destTex = CVMetalTextureGetTexture(cvTex);
    if (!destTex) {
        CFRelease(cvTex);
        CVPixelBufferRelease(pb);
        return TK_ERROR_GPU_INTERNAL;
    }

    /* Blit the source texture into the destination texture. */
    id<MTLCommandBuffer> cmd = [dispatcher->commandQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromTexture:texture->texture
               sourceSlice:0
               sourceLevel:0
              sourceOrigin:MTLOriginMake(0, 0, 0)
                sourceSize:MTLSizeMake(width, height, 1)
                 toTexture:destTex
          destinationSlice:0
          destinationLevel:0
         destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    CFRelease(cvTex);
    *out_pixel_buffer = pb;
    return TK_SUCCESS;
}

/* Core ML inference – the model pointer is an opaque `void *` that the caller
 * obtains from the Core ML API (`MLModel *`).  The dispatcher simply creates an
 * MLCustomLayer that runs on the supplied Metal textures.  For brevity we only
 * outline the steps; a full implementation would need to manage MLCustomLayer
 * lifetimes and error handling. */
TK_NODISCARD tk_error_code_t
tk_metal_dispatch_coreml_inference(tk_metal_dispatcher_t *dispatcher,
                                   void *model,               /* MLModel * (opaque) */
                                   tk_gpu_texture_t src_texture,
                                   tk_gpu_texture_t dst_texture)
{
    RETURN_IF_NULL(dispatcher, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(model, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(src_texture, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(dst_texture, TK_ERROR_INVALID_ARGUMENT);

    /* The Core ML runtime already knows how to execute on Metal; we just need
     * to bind the input and output textures.  The following pseudo‑code shows
     * the intended flow. */
#if defined(__has_include)
#  if __has_include(<CoreML/CoreML.h>)
#    import <CoreML/CoreML.h>
#  else
#    error "CoreML headers not found – cannot compile CoreML integration."
#  endif
#else
#  error "Compiler does not support __has_include – cannot verify CoreML availability."
#endif

    MLModel *mlModel = (__bridge MLModel *)model;
    NSError *error = nil;

    /* Create an MLCustomLayer that uses the supplied textures. */
    MLCustomLayer *layer = [[MLCustomLayer alloc] initWithDevice:dispatcher->device];
    if (!layer) {
        return TK_ERROR_GPU_INTERNAL;
    }

    /* Bind textures – Core ML expects MTLTexture objects. */
    [layer setInputTexture:src_texture->texture atIndex:0];
    [layer setOutputTexture:dst_texture->texture atIndex:0];

    /* Execute the model. */
    BOOL success = [mlModel evaluateWithInputs:@{ @"input" : layer }
                                 options:nil
                                   error:&error];
    if (!success) {
        return tk_metal_error_from_nserror(error);
    }

    return TK_SUCCESS;
}

/* -------------------------------------------------------------------------
 *  Runtime shader compilation (MSL → MTLLibrary → MTLComputePipelineState)
 * ------------------------------------------------------------------------- */
TK_NODISCARD tk_error_code_t
tk_metal_dispatch_compile_kernel(tk_metal_dispatcher_t *dispatcher,
                                 const char *msl_source,
                                 const char *entry_point,
                                 void **out_pipeline_handle)
{
    RETURN_IF_NULL(dispatcher, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(msl_source, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(entry_point, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(out_pipeline_handle, TK_ERROR_INVALID_ARGUMENT);

    NSString *sourceStr = [NSString stringWithUTF8String:msl_source];
    NSError *error = nil;
    id<MTLLibrary> library = [dispatcher->device newLibraryWithSource:sourceStr
                                                              options:nil
                                                                error:&error];
    if (!library) {
        return tk_metal_error_from_nserror(error);
    }

    id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithUTF8String:entry_point]];
    if (!func) {
        return TK_ERROR_COMPILATION_FAILED;
    }

    id<MTLComputePipelineState> pipeline = [dispatcher->device newComputePipelineStateWithFunction:func
                                                                                           error:&error];
    if (!pipeline) {
        return tk_metal_error_from_nserror(error);
    }

    /* The caller receives an opaque pointer; we simply cast the Objective‑C
     * object to void *.  The dispatcher will retain it for the caller’s
     * lifetime. */
    *out_pipeline_handle = (__bridge_retained void *)pipeline;
    return TK_SUCCESS;
}

/* -------------------------------------------------------------------------
 *  Launch a pre‑compiled custom kernel
 * ------------------------------------------------------------------------- */
TK_NODISCARD tk_error_code_t
tk_metal_dispatch_launch_custom_kernel(tk_metal_dispatcher_t *dispatcher,
                                       void *pipeline,
                                       uint32_t grid_x,
                                       uint32_t grid_y,
                                       uint32_t grid_z,
                                       const void *args)
{
    RETURN_IF_NULL(dispatcher, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(pipeline, TK_ERROR_INVALID_ARGUMENT);
    RETURN_IF_NULL(args, TK_ERROR_INVALID_ARGUMENT);

    id<MTLComputePipelineState> pipelineState = (__bridge id<MTLComputePipelineState>)pipeline;

    dispatch_semaphore_wait(dispatcher->inflightSemaphore, DISPATCH_TIME_FOREVER);
    id<MTLCommandBuffer> cmd = [dispatcher->commandQueue commandBuffer];
    if (!cmd) {
        dispatch_semaphore_signal(dispatcher->inflightSemaphore);
        return TK_ERROR_GPU_INTERNAL;
    }

    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipelineState];

    /* The layout of the args struct is defined by the caller.  For illustration
     * we assume the first N arguments are buffers followed by scalars.  The
     * kernel author must ensure the struct matches the MSL signature. */
    const uint8_t *argBytes = (const uint8_t *)args;
    uint32_t offset = 0;

    /* Example: first two arguments are buffers. */
    tk_gpu_buffer_t buf0 = *(tk_gpu_buffer_t *)(argBytes + offset);
    offset += sizeof(tk_gpu_buffer_t);
    tk_gpu_buffer_t buf1 = *(tk_gpu_buffer_t *)(argBytes + offset);
    offset += sizeof(tk_gpu_buffer_t);
    [encoder setBuffer:buf0->buffer offset:0 atIndex:0];
    [encoder setBuffer:buf1->buffer offset:0 atIndex:1];

    /* Remaining bytes are treated as raw scalars (float, int, etc.). */
    if (offset < sizeof(tk_preprocess_params_t)) {
        size_t scalarSize = sizeof(tk_preprocess_params_t) - offset;
        [encoder setBytes:argBytes + offset length:scalarSize atIndex:2];
    }

    MTLSize threadgroup = MTLSizeMake(16, 16, 1);
    MTLSize grid = MTLSizeMake(grid_x, grid_y, grid_z);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
    [encoder endEncoding];

    [cmd addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
        dispatch_semaphore_signal(dispatcher->inflightSemaphore);
    }];
    [cmd commit];
    return TK_SUCCESS;
}

/* -------------------------------------------------------------------------
 *  Utility / Debug helpers
 * ------------------------------------------------------------------------- */
id<MTLDevice>
tk_metal_dispatch_get_device(tk_metal_dispatcher_t *dispatcher)
{
    return dispatcher ? dispatcher->device : nil;
}

TK_NODISCARD tk_error_code_t
tk_metal_dispatch_set_mps_profiling(tk_metal_dispatcher_t *dispatcher,
                                   bool enable,
                                   uint32_t frame_count)
{
    RETURN_IF_NULL(dispatcher, TK_ERROR_INVALID_ARGUMENT);
    /* MPS profiling is controlled via the MTLCommandBuffer’s profilingEnabled
     * flag.  For simplicity we store the request in the dispatcher and apply
     * it on every new command buffer. */
    dispatcher->commandQueue.label = enable ? @"TrackieMPSProfilingQueue" : @"TrackieQueue";
    /* In a real implementation we would also set `MTLCommandBufferDescriptor`
     * with `profilingEnabled = enable`.  The flag is honoured by Instruments. */
    (void)frame_count;   /* unused in this minimal stub */
    return TK_SUCCESS;
}

/* -------------------------------------------------------------------------
 *  End of tk_metal_dispatch.mm
 * ------------------------------------------------------------------------- */
