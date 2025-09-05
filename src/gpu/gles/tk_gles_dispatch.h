/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_gles_dispatch.h
*
* This header defines the public API for the OpenGL ES (GLES) Dispatcher.
* It provides a C-style, opaque interface for GPU compute tasks, specifically
* targeting embedded systems and mobile devices like Android where GLES is
* the primary graphics and compute API.
*
* This backend uses EGL to create a headless, off-screen rendering context
* to ensure that compute operations can run without interfering with any
* visible UI.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TRACKIELLM_GPU_GLES_TK_GLES_DISPATCH_H
#define TRACKIELLM_GPU_GLES_TK_GLES_DISPATCH_H

#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "gpu/tk_gpu_helper.h" // For kernel parameter structs

// Forward-declare the primary dispatcher object as an opaque type.
typedef struct tk_gles_dispatcher_s tk_gles_dispatcher_t;

// Opaque handle for a GPU buffer (wraps a GLuint buffer ID).
typedef struct tk_gpu_buffer_s* tk_gpu_buffer_t;

/**
 * @struct tk_gles_dispatcher_config_t
 * @brief Configuration for initializing the GLES Dispatcher.
 * (Currently empty, but reserved for future settings like EGL config choices).
 */
typedef struct {
    void* reserved;
} tk_gles_dispatcher_config_t;

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Dispatcher Lifecycle
//------------------------------------------------------------------------------

/**
 * @brief Creates and initializes a new GLES Dispatcher instance.
 *
 * This function initializes EGL, finds a suitable display and configuration,
 * creates a headless Pbuffer surface, and establishes a GLES context.
 *
 * @param[out] out_dispatcher Pointer to receive the address of the new dispatcher.
 * @param[in] config The configuration for the dispatcher.
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_GPU_INITIALIZATION_FAILED if EGL or GLES context creation fails.
 */
TK_NODISCARD tk_error_code_t tk_gles_dispatch_create(tk_gles_dispatcher_t** out_dispatcher, const tk_gles_dispatcher_config_t* config);

/**
 * @brief Destroys a GLES Dispatcher instance, freeing all associated EGL/GLES resources.
 *
 * @param[in,out] dispatcher Pointer to the dispatcher instance to be destroyed.
 */
void tk_gles_dispatch_destroy(tk_gles_dispatcher_t** dispatcher);

//------------------------------------------------------------------------------
// GPU Memory Management
//------------------------------------------------------------------------------

/**
 * @brief Allocates a buffer on the GPU using OpenGL.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[out] out_buffer Pointer to receive the handle to the new GPU buffer.
 * @param[in] size_bytes The size of the allocation in bytes.
 * @param[in] usage_hint The OpenGL usage hint (e.g., GL_STATIC_DRAW).
 * @return TK_SUCCESS on success.
 */
TK_NODISCARD tk_error_code_t tk_gles_dispatch_malloc(tk_gles_dispatcher_t* dispatcher, tk_gpu_buffer_t* out_buffer, size_t size_bytes, uint32_t usage_hint);

/**
 * @brief Frees a GPU buffer.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[in,out] buffer Handle to the GPU buffer to be freed.
 */
void tk_gles_dispatch_free(tk_gles_dispatcher_t* dispatcher, tk_gpu_buffer_t* buffer);

/**
 * @brief Asynchronously copies data from host (CPU) to device (GPU).
 * Note: True async uploads are tricky in GLES. This might map the buffer and memcpy.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[in] dst_buffer The destination buffer handle on the GPU.
 * @param[in] src_host_ptr A pointer to the source data on the CPU.
 * @param[in] size_bytes The number of bytes to copy.
 * @return TK_SUCCESS if the copy was successful.
 */
TK_NODISCARD tk_error_code_t tk_gles_dispatch_upload_async(tk_gles_dispatcher_t* dispatcher, tk_gpu_buffer_t dst_buffer, const void* src_host_ptr, size_t size_bytes);

/**
 * @brief Asynchronously copies data from device (GPU) to host (CPU).
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[in] dst_host_ptr A pointer to the destination buffer on the CPU.
 * @param[in] src_buffer The source buffer handle on the GPU.
 * @param[in] size_bytes The number of bytes to copy.
 * @return TK_SUCCESS if the copy was successful.
 */
TK_NODISCARD tk_error_code_t tk_gles_dispatch_download_async(tk_gles_dispatcher_t* dispatcher, void* dst_host_ptr, tk_gpu_buffer_t src_buffer, size_t size_bytes);

//------------------------------------------------------------------------------
// Synchronization Primitives
//------------------------------------------------------------------------------

/**
 * @brief Blocks the CPU until all previously issued GLES commands are complete.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @return TK_SUCCESS on success.
 */
TK_NODISCARD tk_error_code_t tk_gles_dispatch_synchronize(tk_gles_dispatcher_t* dispatcher);

//------------------------------------------------------------------------------
// High-Level Workflow Dispatch
//------------------------------------------------------------------------------

/**
 * @brief Dispatches the vision pre-processing workflow to the GPU.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[in] params The parameters for the pre-processing kernel.
 * @return TK_SUCCESS if the workflow was successfully dispatched.
 */
TK_NODISCARD tk_error_code_t tk_gles_dispatch_preprocess_image(tk_gles_dispatcher_t* dispatcher, const tk_preprocess_params_t* params);

/**
 * @brief Dispatches the depth-to-point-cloud conversion workflow to the GPU.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[in] params The parameters for the conversion kernel.
 * @return TK_SUCCESS if the workflow was successfully dispatched.
 */
TK_NODISCARD tk_error_code_t tk_gles_dispatch_depth_to_point_cloud(tk_gles_dispatcher_t* dispatcher, const tk_depth_to_points_params_t* params);

#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_GPU_GLES_TK_GLES_DISPATCH_H
