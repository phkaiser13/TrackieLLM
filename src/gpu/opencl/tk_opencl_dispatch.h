/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_opencl_dispatch.h
*
* This header defines the public API for the OpenCL Dispatcher. It provides
* a C-style, opaque interface for GPU compute tasks on platforms that support
* OpenCL, which is a wide range of devices including CPUs, GPUs, and other
* accelerators.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TRACKIELLM_GPU_OPENCL_TK_OPENCL_DISPATCH_H
#define TRACKIELLM_GPU_OPENCL_TK_OPENCL_DISPATCH_H

#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "gpu/tk_gpu_helper.h" // For kernel parameter structs

// Forward-declare the primary dispatcher object as an opaque type.
typedef struct tk_opencl_dispatcher_s tk_opencl_dispatcher_t;

// Opaque handle for a GPU buffer (wraps a cl_mem object).
typedef struct tk_gpu_buffer_s* tk_gpu_buffer_t;

/**
 * @struct tk_opencl_dispatcher_config_t
 * @brief Configuration for initializing the OpenCL Dispatcher.
 */
typedef struct {
    uint32_t platform_id; /**< The index of the OpenCL platform to use. */
    uint32_t device_id;   /**< The index of the OpenCL device to use on that platform. */
} tk_opencl_dispatcher_config_t;

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Dispatcher Lifecycle
//------------------------------------------------------------------------------

/**
 * @brief Creates and initializes a new OpenCL Dispatcher instance.
 *
 * @param[out] out_dispatcher Pointer to receive the address of the new dispatcher.
 * @param[in] config The configuration specifying which platform and device to use.
 * @return TK_SUCCESS on success.
 */
TK_NODISCARD tk_error_code_t tk_opencl_dispatch_create(tk_opencl_dispatcher_t** out_dispatcher, const tk_opencl_dispatcher_config_t* config);

/**
 * @brief Destroys an OpenCL Dispatcher instance, freeing all associated OpenCL resources.
 *
 * @param[in,out] dispatcher Pointer to the dispatcher instance to be destroyed.
 */
void tk_opencl_dispatch_destroy(tk_opencl_dispatcher_t** dispatcher);

//------------------------------------------------------------------------------
// GPU Memory Management
//------------------------------------------------------------------------------

/**
 * @brief Allocates a memory buffer on the OpenCL device.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[out] out_buffer Pointer to receive the handle to the new GPU buffer.
 * @param[in] size_bytes The size of the allocation in bytes.
 * @param[in] flags OpenCL memory flags (e.g., CL_MEM_READ_WRITE).
 * @return TK_SUCCESS on success.
 */
TK_NODISCARD tk_error_code_t tk_opencl_dispatch_malloc(tk_opencl_dispatcher_t* dispatcher, tk_gpu_buffer_t* out_buffer, size_t size_bytes, uint64_t flags);

/**
 * @brief Frees a device memory buffer.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[in,out] buffer Handle to the GPU buffer to be freed.
 */
void tk_opencl_dispatch_free(tk_opencl_dispatcher_t* dispatcher, tk_gpu_buffer_t* buffer);

/**
 * @brief Asynchronously copies data from host to device.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[in] dst_buffer The destination buffer on the device.
 * @param[in] src_host_ptr A pointer to the source data on the host.
 * @param[in] size_bytes The number of bytes to copy.
 * @return TK_SUCCESS if the copy was successfully queued.
 */
TK_NODISCARD tk_error_code_t tk_opencl_dispatch_upload_async(tk_opencl_dispatcher_t* dispatcher, tk_gpu_buffer_t dst_buffer, const void* src_host_ptr, size_t size_bytes);

/**
 * @brief Asynchronously copies data from device to host.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[in] dst_host_ptr A pointer to the destination buffer on the host.
 * @param[in] src_buffer The source buffer on the device.
 * @param[in] size_bytes The number of bytes to copy.
 * @return TK_SUCCESS if the copy was successfully queued.
 */
TK_NODISCARD tk_error_code_t tk_opencl_dispatch_download_async(tk_opencl_dispatcher_t* dispatcher, void* dst_host_ptr, tk_gpu_buffer_t src_buffer, size_t size_bytes);

//------------------------------------------------------------------------------
// Synchronization Primitives
//------------------------------------------------------------------------------

/**
 * @brief Blocks the CPU until all commands in the queue are complete.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @return TK_SUCCESS on success.
 */
TK_NODISCARD tk_error_code_t tk_opencl_dispatch_synchronize(tk_opencl_dispatcher_t* dispatcher);

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
TK_NODISCARD tk_error_code_t tk_opencl_dispatch_preprocess_image(tk_opencl_dispatcher_t* dispatcher, const tk_preprocess_params_t* params);

/**
 * @brief Dispatches the depth-to-point-cloud conversion workflow to the GPU.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[in] params The parameters for the conversion kernel.
 * @return TK_SUCCESS if the workflow was successfully dispatched.
 */
TK_NODISCARD tk_error_code_t tk_opencl_dispatch_depth_to_point_cloud(tk_opencl_dispatcher_t* dispatcher, const tk_depth_to_points_params_t* params);

#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_GPU_OPENCL_TK_OPENCL_DISPATCH_H
