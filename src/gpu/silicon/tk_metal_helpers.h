/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_metal_helpers.h
*
* This header file provides a C-callable API facade for common Apple Metal
* framework operations. Its purpose is to abstract away the verbosity of the
* Objective-C API, enforce consistent error handling, and provide high-level
* utilities for resource creation and data transfer.
*
* This is a foundational component of the Hardware Abstraction Layer (HAL) for
* Apple Silicon. It is intended for internal use by the Metal dispatch layer
* (`tk_metal_dispatch.mm`) and should not be exposed to higher-level modules.
*
* All functions returning Metal objects (`id`) follow the Cocoa ownership policy:
* they return objects with a +1 retain count. The caller is responsible for
* releasing these objects.
*
* SPDX-License-Identifier:
*/

#ifndef TRACKIELLM_GPU_SILICON_TK_METAL_HELPERS_H
#define TRACKIELLM_GPU_SILICON_TK_METAL_HELPERS_H

#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "vision/tk_vision_pipeline.h" // For tk_video_frame_t

// Forward-declare Objective-C types using @class to avoid including the full
// Metal header here. This is a critical best practice for C/Obj-C interop.
#ifdef __OBJC__
@class MTLDevice;
@class MTLLibrary;
@class MTLCommandQueue;
@class MTLComputePipelineState;
@class MTLBuffer;
@class MTLTextureDescriptor;
@class MTLTexture;
@class MTLCommandBuffer;
@class NSError;
#else
// For pure C/C++ compilers, define `id` as a generic pointer.
#include <objc/objc.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Error Handling
//------------------------------------------------------------------------------

/**
 * @brief Translates an `NSError` object from a Metal API call into a
 *        TrackieLLM standard `tk_error_code_t`.
 *
 * @param[in] error The `NSError` object. Can be nil.
 * @return The corresponding `tk_error_code_t`. Returns TK_SUCCESS if error is nil.
 */
tk_error_code_t tk_metal_translate_error(id error);

//------------------------------------------------------------------------------
// Device and Library Management
//------------------------------------------------------------------------------

/**
 * @brief Retrieves the default system Metal device.
 *
 * This function is the entry point for all Metal operations. It will cause the
 * program to terminate if no suitable Metal device is found.
 *
 * @return A retained (`+1`) pointer to the default `id<MTLDevice>`. The caller
 *         is responsible for releasing this object.
 */
id tk_metal_get_default_device(void);

/**
 * @brief Creates a new command queue for a given device.
 *
 * @param[in] device The `id<MTLDevice>` to create the queue on.
 * @return A retained (`+1`) pointer to the `id<MTLCommandQueue>`.
 */
id tk_metal_create_command_queue(id device);

/**
 * @brief Loads the default Metal shader library (`.metallib`) from the
 *        application's main bundle.
 *
 * @param[in] device The `id<MTLDevice>` to associate the library with.
 * @param[out] out_library On success, a retained (`+1`) pointer to the
 *                         `id<MTLLibrary>`.
 *
 * @return TK_SUCCESS on success, or an error code on failure.
 */
TK_NODISCARD tk_error_code_t tk_metal_create_library_from_default_bundle(id device, id* out_library);

//------------------------------------------------------------------------------
// Pipeline State Object (PSO) Creation
//------------------------------------------------------------------------------

/**
 * @brief Creates a compute pipeline state object for a given kernel function.
 *
 * This is a high-level wrapper that encapsulates the complex PSO creation process.
 *
 * @param[in] device The `id<MTLDevice>`.
 * @param[in] library The `id<MTLLibrary>` containing the kernel function.
 * @param[in] function_name The name of the kernel function in the shader library.
 * @param[out] out_pso On success, a retained (`+1`) pointer to the
 *                     `id<MTLComputePipelineState>`.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_NOT_FOUND if the function name does not exist in the library.
 * @return TK_ERROR_GPU_ERROR for other PSO creation failures.
 */
TK_NODISCARD tk_error_code_t tk_metal_create_compute_pso(id device, id library, const char* function_name, id* out_pso);

//------------------------------------------------------------------------------
// Resource Creation (Buffers and Textures)
//------------------------------------------------------------------------------

/**
 * @brief Creates a new, uninitialized Metal buffer.
 *
 * @param[in] device The `id<MTLDevice>`.
 * @param[in] length The size of the buffer in bytes.
 * @param[in] options The `MTLResourceOptions` for the buffer.
 * @param[out] out_buffer On success, a retained (`+1`) pointer to the `id<MTLBuffer>`.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_OUT_OF_MEMORY if the buffer could not be allocated.
 */
TK_NODISCARD tk_error_code_t tk_metal_create_buffer(id device, size_t length, uint32_t options, id* out_buffer);

/**
 * @brief Creates a new 2D Metal texture.
 *
 * @param[in] device The `id<MTLDevice>`.
 * @param[in] width The width of the texture in pixels.
 * @param[in] height The height of the texture in pixels.
 * @param[in] pixel_format The `MTLPixelFormat` of the texture.
 * @param[in] usage The `MTLTextureUsage` flags.
 * @param[out] out_texture On success, a retained (`+1`) pointer to the `id<MTLTexture>`.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_OUT_OF_MEMORY if the texture could not be allocated.
 */
TK_NODISCARD tk_error_code_t tk_metal_create_texture_2d(id device, uint32_t width, uint32_t height, uint32_t pixel_format, uint32_t usage, id* out_texture);

//------------------------------------------------------------------------------
// High-Level Data Transfer
//------------------------------------------------------------------------------

/**
 * @brief Uploads a CPU-side image buffer to a GPU texture.
 *
 * This helper function encapsulates the necessary steps to efficiently transfer
 * image data to the GPU, including the use of a blit command encoder.
 *
 * @param[in] command_buffer The `id<MTLCommandBuffer>` to encode the operation into.
 * @param[in] source_frame A pointer to the CPU-side video frame.
 * @param[in] destination_texture The `id<MTLTexture>` to upload the data to.
 *                                The texture dimensions and format must be compatible.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_INVALID_ARGUMENT if parameters are invalid or incompatible.
 */
TK_NODISCARD tk_error_code_t tk_metal_upload_frame_to_texture(id command_buffer, const tk_video_frame_t* source_frame, id destination_texture);


#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_GPU_SILICON_TK_METAL_HELPERS_H