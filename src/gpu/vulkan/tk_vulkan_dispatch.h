/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_vulkan_dispatch.h
*
* This header defines the public API for the Vulkan Dispatcher, the orchestration
* engine for the Vulkan backend. It provides a C-style, opaque interface to
* the underlying Vulkan compute capabilities, ensuring API stability and
* abstraction from the core logic.
*
* This backend is designed for modern, cross-platform GPU acceleration on
* hardware that supports the Vulkan API, including Android, Linux, and Windows.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TRACKIELLM_GPU_VULKAN_TK_VULKAN_DISPATCH_H
#define TRACKIELLM_GPU_VULKAN_TK_VULKAN_DISPATCH_H

#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "gpu/vulkan/tk_vulkan_kernels.h" // For kernel parameter structs

// Forward-declare the primary dispatcher object as an opaque type.
typedef struct tk_vulkan_dispatcher_s tk_vulkan_dispatcher_t;

// Opaque handles for GPU resources to enforce abstraction.
typedef struct tk_gpu_buffer_s* tk_gpu_buffer_t;
typedef struct tk_gpu_event_s*  tk_gpu_event_t;

/**
 * @struct tk_vulkan_dispatcher_config_t
 * @brief Configuration for initializing the Vulkan Dispatcher.
 */
typedef struct {
    uint32_t device_id; /**< The index of the physical device to use from the list of available devices. */
    bool enable_validation_layers; /**< If true, enables Vulkan validation layers for debugging. */
} tk_vulkan_dispatcher_config_t;

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Dispatcher Lifecycle
//------------------------------------------------------------------------------

/**
 * @brief Creates and initializes a new Vulkan Dispatcher instance.
 *
 * @param[out] out_dispatcher Pointer to receive the address of the new dispatcher.
 * @param[in] config The configuration specifying which GPU to use and validation settings.
 * @return TK_SUCCESS on success.
 */
TK_NODISCARD tk_error_code_t tk_vulkan_dispatch_create(tk_vulkan_dispatcher_t** out_dispatcher, const tk_vulkan_dispatcher_config_t* config);

/**
 * @brief Destroys a Vulkan Dispatcher instance, freeing all associated Vulkan resources.
 *
 * @param[in,out] dispatcher Pointer to the dispatcher instance to be destroyed.
 */
void tk_vulkan_dispatch_destroy(tk_vulkan_dispatcher_t** dispatcher);

//------------------------------------------------------------------------------
// GPU Memory Management
//------------------------------------------------------------------------------

/**
 * @brief Allocates a memory buffer on the Vulkan device.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[out] out_buffer Pointer to receive the handle to the new GPU buffer.
 * @param[in] size_bytes The size of the allocation in bytes.
 * @param[in] memory_property_flags Vulkan memory property flags (e.g., VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT).
 * @return TK_SUCCESS on success.
 */
TK_NODISCARD tk_error_code_t tk_vulkan_dispatch_malloc(tk_vulkan_dispatcher_t* dispatcher, tk_gpu_buffer_t* out_buffer, size_t size_bytes, uint32_t memory_property_flags);

/**
 * @brief Frees a device memory buffer.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[in,out] buffer Handle to the GPU buffer to be freed.
 */
void tk_vulkan_dispatch_free(tk_vulkan_dispatcher_t* dispatcher, tk_gpu_buffer_t* buffer);

/**
 * @brief Asynchronously copies data from host to device.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[in] dst_buffer The destination buffer on the device.
 * @param[in] src_host_ptr A pointer to the source data on the host.
 * @param[in] size_bytes The number of bytes to copy.
 * @return TK_SUCCESS if the copy was successfully queued.
 */
TK_NODISCARD tk_error_code_t tk_vulkan_dispatch_upload_async(tk_vulkan_dispatcher_t* dispatcher, tk_gpu_buffer_t dst_buffer, const void* src_host_ptr, size_t size_bytes);

/**
 * @brief Asynchronously copies data from device to host.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[in] dst_host_ptr A pointer to the destination buffer on the host.
 * @param[in] src_buffer The source buffer on the device.
 * @param[in] size_bytes The number of bytes to copy.
 * @return TK_SUCCESS if the copy was successfully queued.
 */
TK_NODISCARD tk_error_code_t tk_vulkan_dispatch_download_async(tk_vulkan_dispatcher_t* dispatcher, void* dst_host_ptr, tk_gpu_buffer_t src_buffer, size_t size_bytes);

//------------------------------------------------------------------------------
// Synchronization Primitives
//------------------------------------------------------------------------------

/**
 * @brief Blocks the CPU until the device queue is idle.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @return TK_SUCCESS on success.
 */
TK_NODISCARD tk_error_code_t tk_vulkan_dispatch_synchronize(tk_vulkan_dispatcher_t* dispatcher);

//------------------------------------------------------------------------------
// High-Level Workflow Dispatch
//------------------------------------------------------------------------------

/**
 * @brief Dispatches the vision pre-processing workflow to the GPU.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[in] params The parameters for the pre-processing kernel.
 * @return TK_SUCCESS if the workflow was successfully queued.
 */
TK_NODISCARD tk_error_code_t tk_vulkan_dispatch_preprocess_image(tk_vulkan_dispatcher_t* dispatcher, const tk_preprocess_params_t* params);

/**
 * @brief Dispatches the depth-to-point-cloud conversion workflow to the GPU.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[in] params The parameters for the conversion kernel.
 * @return TK_SUCCESS if the workflow was successfully queued.
 */
TK_NODISCARD tk_error_code_t tk_vulkan_dispatch_depth_to_point_cloud(tk_vulkan_dispatcher_t* dispatcher, const tk_depth_to_points_params_t* params);


#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_GPU_VULKAN_TK_VULKAN_DISPATCH_H
