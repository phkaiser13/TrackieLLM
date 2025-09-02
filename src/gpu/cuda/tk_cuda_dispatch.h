/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_cuda_dispatch.h
*
* This header file defines the public API for the CUDA Dispatcher, the high-level
* orchestration engine for the NVIDIA GPU backend. This module serves as the
* primary interface for all other system components that require GPU acceleration.
*
* The architecture is built on several core principles of high-performance GPGPU:
*   - Opaque Handles: All GPU resources (device memory, synchronization events)
*     are managed via opaque handles (`tk_gpu_buffer_t`, `tk_gpu_event_t`). This
*     prevents resource leakage and direct pointer manipulation by consumers.
*   - Asynchronous-by-Default: All operations that involve the GPU (memory
*     transfers, kernel launches) are non-blocking and are queued on CUDA streams.
*   - Explicit Synchronization: The caller has explicit control over synchronization
*     via an event-based system, allowing the CPU to perform other work while
*     the GPU is busy.
*   - Workflow-Oriented: The API exposes high-level functions that represent
*     complete computational workflows (e.g., "preprocess this image"), rather
*     than low-level kernel launch primitives.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TRACKIELLM_GPU_CUDA_TK_CUDA_DISPATCH_H
#define TRACKIELLM_GPU_CUDA_TK_CUDA_DISPATCH_H

#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "vision/tk_vision_pipeline.h" // For tk_video_frame_t
#include "gpu/cuda/tk_cuda_kernels.h"   // For kernel parameter structs

// Forward-declare the primary dispatcher object as an opaque type.
typedef struct tk_cuda_dispatcher_s tk_cuda_dispatcher_t;

// Opaque handles for GPU resources to enforce abstraction.
typedef struct tk_gpu_buffer_s* tk_gpu_buffer_t;
typedef struct tk_gpu_event_s*  tk_gpu_event_t;

/**
 * @struct tk_cuda_dispatcher_config_t
 * @brief Configuration for initializing the CUDA Dispatcher.
 */
typedef struct {
    int device_id; /**< The ordinal of the CUDA-enabled GPU to use. */
} tk_cuda_dispatcher_config_t;

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Dispatcher Lifecycle
//------------------------------------------------------------------------------

/**
 * @brief Creates and initializes a new CUDA Dispatcher instance for a specific GPU.
 *
 * This function probes the system for a compatible CUDA device, sets the device,
 * and creates the necessary resources like default streams and event pools.
 *
 * @param[out] out_dispatcher Pointer to receive the address of the new dispatcher instance.
 * @param[in] config The configuration specifying which GPU to use.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_GPU_DEVICE_NOT_FOUND if the specified device_id is invalid or unavailable.
 * @return TK_ERROR_OUT_OF_MEMORY on host memory allocation failure.
 */
TK_NODISCARD tk_error_code_t tk_cuda_dispatch_create(tk_cuda_dispatcher_t** out_dispatcher, const tk_cuda_dispatcher_config_t* config);

/**
 * @brief Destroys a CUDA Dispatcher instance, freeing all associated GPU resources.
 *
 * @param[in,out] dispatcher Pointer to the dispatcher instance to be destroyed.
 */
void tk_cuda_dispatch_destroy(tk_cuda_dispatcher_t** dispatcher);

//------------------------------------------------------------------------------
// GPU Memory Management
//------------------------------------------------------------------------------

/**
 * @brief Allocates a block of memory on the GPU.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[out] out_buffer Pointer to receive the handle to the new GPU buffer.
 * @param[in] size_bytes The size of the allocation in bytes.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_GPU_MEMORY if the allocation fails on the device.
 */
TK_NODISCARD tk_error_code_t tk_cuda_dispatch_malloc(tk_cuda_dispatcher_t* dispatcher, tk_gpu_buffer_t* out_buffer, size_t size_bytes);

/**
 * @brief Frees a block of memory on the GPU.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[in,out] buffer Handle to the GPU buffer to be freed. The handle is invalidated.
 */
void tk_cuda_dispatch_free(tk_cuda_dispatcher_t* dispatcher, tk_gpu_buffer_t* buffer);

/**
 * @brief Asynchronously copies data from host (CPU) to device (GPU).
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[in] dst_buffer The destination buffer handle on the GPU.
 * @param[in] src_host_ptr A pointer to the source data on the CPU.
 * @param[in] size_bytes The number of bytes to copy.
 *
 * @return TK_SUCCESS if the copy was successfully queued.
 */
TK_NODISCARD tk_error_code_t tk_cuda_dispatch_upload_async(tk_cuda_dispatcher_t* dispatcher, tk_gpu_buffer_t dst_buffer, const void* src_host_ptr, size_t size_bytes);

/**
 * @brief Asynchronously copies data from device (GPU) to host (CPU).
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[in] dst_host_ptr A pointer to the destination buffer on the CPU.
 * @param[in] src_buffer The source buffer handle on the GPU.
 * @param[in] size_bytes The number of bytes to copy.
 *
 * @return TK_SUCCESS if the copy was successfully queued.
 */
TK_NODISCARD tk_error_code_t tk_cuda_dispatch_download_async(tk_cuda_dispatcher_t* dispatcher, void* dst_host_ptr, tk_gpu_buffer_t src_buffer, size_t size_bytes);

//------------------------------------------------------------------------------
// Synchronization Primitives
//------------------------------------------------------------------------------

/**
 * @brief Blocks the calling CPU thread until all previously queued work in the
 *        dispatcher's default stream is complete. Use `wait_for_event` for
 *        more efficient, fine-grained synchronization.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @return TK_SUCCESS on success.
 */
TK_NODISCARD tk_error_code_t tk_cuda_dispatch_synchronize(tk_cuda_dispatcher_t* dispatcher);

//------------------------------------------------------------------------------
// High-Level Workflow Dispatch
//------------------------------------------------------------------------------

/**
 * @brief Dispatches the entire vision pre-processing workflow to the GPU.
 *
 * This high-level function encapsulates an asynchronous upload of the source
 * frame and the launch of the pre-processing kernel.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[in] params The parameters for the pre-processing kernel. Note that the
 *                   device pointers within this struct must correspond to valid
 *                   `tk_gpu_buffer_t` handles.
 *
 * @return TK_SUCCESS if the workflow was successfully queued.
 */
TK_NODISCARD tk_error_code_t tk_cuda_dispatch_preprocess_image(tk_cuda_dispatcher_t* dispatcher, const tk_preprocess_params_t* params);

/**
 * @brief Dispatches the depth-to-point-cloud conversion workflow to the GPU.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[in] params The parameters for the conversion kernel. Device pointers
 *                   must correspond to valid `tk_gpu_buffer_t` handles.
 *
 * @return TK_SUCCESS if the workflow was successfully queued.
 */
TK_NODISCARD tk_error_code_t tk_cuda_dispatch_depth_to_point_cloud(tk_cuda_dispatcher_t* dispatcher, const tk_depth_to_points_params_t* params);


#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_GPU_CUDA_TK_CUDA_DISPATCH_H