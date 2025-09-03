/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_metal_helpers.mm
 *
 * This source file implements the C‑callable helper façade declared in
 * tk_metal_helpers.h.  The helpers hide the verbosity of the Objective‑C
 * Metal API, enforce a uniform error‑handling strategy (conversion of NSError
 * → tk_error_code_t) and provide a small set of high‑level utilities that are
 * reused by the Metal dispatcher (tk_metal_dispatch.mm) and by any other HAL
 * component that needs to create buffers, textures or compile kernels.
 *
 * Design notes
 * -------------
 * • All functions that return a Metal object (`id`) follow the Cocoa ownership
 *   convention: the object is returned with a +1 retain count and the caller is
 *   responsible for releasing it (`[obj release]` or `CFRelease`).  This makes
 *   the API safe for pure‑C callers that cannot use ARC.
 * • The implementation is deliberately defensive: every public entry point checks
 *   its arguments, translates NSError objects into the project's own error enum,
 *   and never leaks resources on early‑exit paths.
 * • The file is compiled as Objective‑C++ (`.mm`) so we can mix C structs,
 *   Objective‑C objects and C++‑style RAII helpers where convenient.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CoreVideo/CoreVideo.h>
#import "gpu/metal/tk_metal_helpers.h"
#import "utils/tk_error_handling.h"

/* -------------------------------------------------------------------------
 *  Internal utility – map NSError domains / codes to TrackieLLM error codes.
 * ------------------------------------------------------------------------- */
tk_error_code_t
tk_metal_translate_error(id error_obj)
{
    if (!error_obj) {
        return TK_SUCCESS;
    }

    NSError *error = (NSError *)error_obj;
    NSString *domain = error.domain;

    /* The mapping below is intentionally simple; a production system would
     * provide a richer translation table. */
    if ([domain isEqualToString:MTLCommandBufferErrorDomain]) {
        return TK_ERROR_GPU_INTERNAL;
    } else if ([domain isEqualToString:MTLDeviceErrorDomain]) {
        return TK_ERROR_GPU_DEVICE_NOT_FOUND;
    } else if ([domain isEqualToString:NSCocoaErrorDomain]) {
        return TK_ERROR_IO;
    }

    /* Fallback – unknown domain. */
    return TK_ERROR_GPU_INTERNAL;
}

/* -------------------------------------------------------------------------
 *  Device & Command‑Queue helpers
 * ------------------------------------------------------------------------- */
id
tk_metal_get_default_device(void)
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        /* The calling code treats a nil return as a fatal error – we abort with a
         * clear message to aid debugging. */
        NSLog(@"[TrackieLLM] Fatal: No Metal‑compatible GPU found on this system.");
        abort();
    }
    /* Retain before returning because the caller expects a +1 reference. */
    [device retain];
    return device;
}

id
tk_metal_create_command_queue(id device_obj)
{
    if (!device_obj) {
        return nil;
    }
    id<MTLDevice> device = (id<MTLDevice>)device_obj;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    /* The queue is already returned with +1 retain count from `new…`. */
    return queue;
}

/* -------------------------------------------------------------------------
 *  Library loading – default bundle (main executable or app bundle)
 * ------------------------------------------------------------------------- */
TK_NODISCARD tk_error_code_t
tk_metal_create_library_from_default_bundle(id device_obj, id *out_library)
{
    if (!device_obj || !out_library) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    id<MTLDevice> device = (id<MTLDevice>)device_obj;
    NSBundle *bundle = [NSBundle mainBundle];
    if (!bundle) {
        return TK_ERROR_IO;
    }

    /* The default library is compiled into the app bundle as `default.metallib`.
     * If the file is missing we fall back to the runtime‑generated library. */
    NSString *path = [bundle pathForResource:@"default" ofType:@"metallib"];
    NSError *error = nil;
    id<MTLLibrary> library = nil;

    if (path) {
        library = [device newLibraryWithFile:path error:&error];
    } else {
        /* As a safety net we try to create a library from the source embedded
         * in the binary (available on macOS 12+). */
        library = [device newDefaultLibraryWithError:&error];
    }

    if (!library) {
        return tk_metal_translate_error(error);
    }

    *out_library = library;   /* already retained by `new…` */
    return TK_SUCCESS;
}

/* -------------------------------------------------------------------------
 *  Compute Pipeline State Object (PSO) creation
 * ------------------------------------------------------------------------- */
TK_NODISCARD tk_error_code_t
tk_metal_create_compute_pso(id device_obj,
                            id library_obj,
                            const char *function_name,
                            id *out_pso)
{
    if (!device_obj || !library_obj || !function_name || !out_pso) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    id<MTLDevice> device   = (id<MTLDevice>)device_obj;
    id<MTLLibrary> library = (id<MTLLibrary>)library_obj;
    NSString *funcName = [NSString stringWithUTF8String:function_name];
    NSError *error = nil;

    id<MTLFunction> function = [library newFunctionWithName:funcName];
    if (!function) {
        return TK_ERROR_NOT_FOUND;
    }

    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:function
                                                                          error:&error];
    if (!pso) {
        return tk_metal_translate_error(error);
    }

    *out_pso = pso;   /* +1 retain from `new…` */
    return TK_SUCCESS;
}

/* -------------------------------------------------------------------------
 *  Buffer creation
 * ------------------------------------------------------------------------- */
TK_NODISCARD tk_error_code_t
tk_metal_create_buffer(id device_obj,
                       size_t length,
                       uint32_t options,
                       id *out_buffer)
{
    if (!device_obj || !out_buffer || length == 0) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    id<MTLDevice> device = (id<MTLDevice>)device_obj;
    id<MTLBuffer> buffer = [device newBufferWithLength:length
                                                options:options];
    if (!buffer) {
        return TK_ERROR_OUT_OF_MEMORY;
    }

    *out_buffer = buffer;   /* +1 retain from `new…` */
    return TK_SUCCESS;
}

/* -------------------------------------------------------------------------
 *  2‑D texture creation
 * ------------------------------------------------------------------------- */
TK_NODISCARD tk_error_code_t
tk_metal_create_texture_2d(id device_obj,
                           uint32_t width,
                           uint32_t height,
                           uint32_t pixel_format,
                           uint32_t usage,
                           id *out_texture)
{
    if (!device_obj || !out_texture || width == 0 || height == 0) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    id<MTLDevice> device = (id<MTLDevice>)device_obj;
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixel_format
                                                                                width:width
                                                                               height:height
                                                                            mipmapped:NO];
    desc.usage = usage;
    desc.storageMode = MTLStorageModePrivate;   /* Most workloads benefit from private storage */

    id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
    if (!texture) {
        return TK_ERROR_OUT_OF_MEMORY;
    }

    *out_texture = texture;   /* +1 retain from `new…` */
    return TK_SUCCESS;
}

/* -------------------------------------------------------------------------
 *  High‑level frame‑to‑texture upload
 * ------------------------------------------------------------------------- */
TK_NODISCARD tk_error_code_t
tk_metal_upload_frame_to_texture(id command_buffer_obj,
                                 const tk_video_frame_t *source_frame,
                                 id destination_texture_obj)
{
    if (!command_buffer_obj || !source_frame || !destination_texture_obj) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    id<MTLCommandBuffer> commandBuffer = (id<MTLCommandBuffer>)command_buffer_obj;
    id<MTLTexture> texture = (id<MTLTexture>)destination_texture_obj;

    /* -----------------------------------------------------------------
     *  Validate that the source frame layout matches the destination texture.
     *  The tk_video_frame_t definition (from tk_vision_pipeline.h) is
     *  assumed to contain:
     *      - void   *data;          // pointer to raw pixel data
     *      - uint32_t width;
     *      - uint32_t height;
     *      - uint32_t stride_bytes; // bytes per row (may be > width * pixelSize)
     *      - uint32_t pixel_format; // a TrackieLLM enum that we map to MTLPixelFormat
     * ----------------------------------------------------------------- */
    if (source_frame->width  != texture.width ||
        source_frame->height != texture.height) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    /* Map the project's pixel format enum to a Metal pixel format.
     * For this example we support only a few common formats. */
    MTLPixelFormat mtlFormat = MTLPixelFormatInvalid;
    switch (source_frame->pixel_format) {
        case TK_PIXEL_FORMAT_RGBA8_UNORM:
            mtlFormat = MTLPixelFormatRGBA8Unorm;
            break;
        case TK_PIXEL_FORMAT_BGRA8_UNORM:
            mtlFormat = MTLPixelFormatBGRA8Unorm;
            break;
        case TK_PIXEL_FORMAT_R8_UNORM:
            mtlFormat = MTLPixelFormatR8Unorm;
            break;
        default:
            return TK_ERROR_UNSUPPORTED_FORMAT;
    }

    if (mtlFormat != texture.pixelFormat) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    /* -----------------------------------------------------------------
     *  Encode the copy.  We use a blit encoder because it can handle
     *  arbitrary row‑stride without needing a compute kernel.
     * ----------------------------------------------------------------- */
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (!blit) {
        return TK_ERROR_GPU_INTERNAL;
    }

    /* Define the region that covers the whole texture. */
    MTLRegion region = {
        .origin = {0, 0, 0},
        .size   = {source_frame->width, source_frame->height, 1}
    };

    /* The source data may be tightly packed or have a larger stride.
     * `replaceRegion:` works on the GPU side, but we are on the CPU side,
     * therefore we copy via `copyFromBuffer:`.  To avoid an extra staging
     * buffer we create a temporary MTLBuffer in shared storage that points
     * directly to the source memory. */
    id<MTLDevice> device = texture.device;
    id<MTLBuffer> staging = [device newBufferWithBytesNoCopy:(void *)source_frame->data
                                                    length:source_frame->stride_bytes * source_frame->height
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
    if (!staging) {
        [blit endEncoding];
        return TK_ERROR_OUT_OF_MEMORY;
    }

    [blit copyFromBuffer:staging
           sourceOffset:0
      sourceBytesPerRow:source_frame->stride_bytes
    sourceBytesPerImage:source_frame->stride_bytes * source_frame->height
             sourceSize:region.size
              toTexture:texture
       destinationSlice:0
       destinationLevel:0
      destinationOrigin:region.origin];

    [blit endEncoding];
    return TK_SUCCESS;
}

/* -------------------------------------------------------------------------
 *  End of tk_metal_helpers.mm
 * ------------------------------------------------------------------------- */
