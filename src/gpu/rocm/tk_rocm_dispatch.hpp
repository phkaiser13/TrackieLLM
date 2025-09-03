/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_rocm_dispatch.hpp
 *
 * This header file defines the public API for the ROCm Dispatcher, the high-level
 * orchestration engine for the AMD GPU backend. This module serves as the
 * primary interface for all other system components that require GPU acceleration
 * via the HIP (Heterogeneous-compute Interface for Portability) API.
 *
 * The architecture is engineered around these core principles for high-performance GPGPU:
 *   - **Opaque Handles**: All GPU resources (device memory, synchronization events)
 *     are managed via opaque handles (`tk_gpu_buffer_t`, `tk_gpu_event_t`). This
 *     enforces a strong abstraction layer, prevents resource leakage, and forbids
 *     direct pointer manipulation by consumers, enhancing system stability.
 *   - **Asynchronous-by-Default**: All operations that involve the GPU (memory
 *     transfers, kernel launches) are non-blocking and are enqueued on HIP streams.
 *     This design maximizes GPU utilization by allowing the CPU to perform other
 *     work while the GPU is busy.
 *   - **Explicit Synchronization**: The caller retains explicit and fine-grained
 *     control over synchronization via an event-based system, which is crucial for
 *     building complex, high-throughput data pipelines.
 *   - **Workflow-Oriented API**: The API exposes high-level functions that represent
 *     complete computational workflows (e.g., "preprocess this image"), abstracting
 *     away the low-level details of kernel launches and memory management.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#ifndef TRACKIELLM_GPU_ROCM_TK_ROCM_DISPATCH_HPP
#define TRACKIELLM_GPU_ROCM_TK_ROCM_DISPATCH_HPP

#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "gpu/rocm/tk_rocm_kernels.hpp"   // For kernel parameter structs

// Forward-declare the primary dispatcher object as an opaque type.
// Consumers of this API interact with a pointer to this struct, but cannot
// see its internal layout, which is a key principle of good API design.
typedef struct tk_rocm_dispatcher_s tk_rocm_dispatcher_t;

// Opaque handles for GPU resources to enforce abstraction and safety.
// These handles are managed internally by the dispatcher.
typedef struct tk_gpu_buffer_s* tk_gpu_buffer_t;
typedef struct tk_gpu_event_s*  tk_gpu_event_t;

/**
 * @struct tk_rocm_dispatcher_config_t
 * @brief Configuration structure for initializing the ROCm Dispatcher.
 */
typedef struct {
    int device_id; /**< The ordinal of the HIP-enabled GPU to be used. */
} tk_rocm_dispatcher_config_t;

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Dispatcher Lifecycle Management
//------------------------------------------------------------------------------

/**
 * @brief Creates and initializes a new ROCm Dispatcher instance for a specific GPU.
 *
 * This is the entry point for using the ROCm backend. The function probes the
 * system for a compatible HIP device, sets the device context, and creates the
 * necessary resources like default streams, event pools, and other internal
 * management structures. It performs rigorous checks to ensure the selected
 * device is available and meets minimum requirements.
 *
 * @param[out] out_dispatcher A pointer to a variable that will receive the address of the new dispatcher instance.
 * @param[in]  config A pointer to the configuration structure specifying which GPU to use.
 *
 * @return TK_SUCCESS on successful initialization.
 * @return TK_ERROR_INVALID_ARGUMENT if out_dispatcher or config are NULL.
 * @return TK_ERROR_GPU_DEVICE_NOT_FOUND if the specified device_id is invalid or unavailable.
 * @return TK_ERROR_OUT_OF_MEMORY on host memory allocation failure.
 * @return TK_ERROR_GPU_DEVICE_SET_FAILED if the device context cannot be set.
 */
TK_NODISCARD tk_error_code_t tk_rocm_dispatch_create(tk_rocm_dispatcher_t** out_dispatcher, const tk_rocm_dispatcher_config_t* config);

/**
 * @brief Destroys a ROCm Dispatcher instance, freeing all associated GPU and host resources.
 *
 * This function performs a graceful shutdown, synchronizing all active streams to
 * ensure pending operations are complete before releasing resources. It frees all
 * memory pools, destroys streams and events, and resets the device context.
 *
 * @param[in,out] dispatcher A pointer to the dispatcher instance to be destroyed. The pointer is set to NULL after destruction.
 */
void tk_rocm_dispatch_destroy(tk_rocm_dispatcher_t** dispatcher);

//------------------------------------------------------------------------------
// GPU Memory Management
//------------------------------------------------------------------------------

/**
 * @brief Allocates a block of memory on the GPU managed by this dispatcher.
 *
 * @param[in]  dispatcher The dispatcher instance.
 * @param[out] out_buffer A pointer to receive the handle to the new GPU buffer.
 * @param[in]  size_bytes The size of the allocation in bytes.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_GPU_MEMORY if the allocation fails on the device (e.g., out of memory).
 * @return TK_ERROR_INVALID_ARGUMENT if any parameter is invalid.
 */
TK_NODISCARD tk_error_code_t tk_rocm_dispatch_malloc(tk_rocm_dispatcher_t* dispatcher, tk_gpu_buffer_t* out_buffer, size_t size_bytes);

/**
 * @brief Frees a block of memory on the GPU.
 *
 * @param[in]     dispatcher The dispatcher instance.
 * @param[in,out] buffer Handle to the GPU buffer to be freed. The handle is invalidated (set to NULL).
 */
void tk_rocm_dispatch_free(tk_rocm_dispatcher_t* dispatcher, tk_gpu_buffer_t* buffer);

/**
 * @brief Asynchronously copies data from host (CPU) memory to device (GPU) memory.
 *
 * The operation is enqueued on a dedicated upload stream to allow for potential
 * overlap with computation on the default stream.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[in] dst_buffer The destination buffer handle on the GPU.
 * @param[in] src_host_ptr A pointer to the source data on the CPU.
 * @param[in] size_bytes The number of bytes to copy.
 *
 * @return TK_SUCCESS if the copy was successfully enqueued.
 * @return TK_ERROR_GPU_MEMORY_COPY_FAILED on failure.
 */
TK_NODISCARD tk_error_code_t tk_rocm_dispatch_upload_async(tk_rocm_dispatcher_t* dispatcher, tk_gpu_buffer_t dst_buffer, const void* src_host_ptr, size_t size_bytes);

/**
 * @brief Asynchronously copies data from device (GPU) memory to host (CPU) memory.
 *
 * The operation is enqueued on a dedicated download stream to allow for potential
 * overlap with computation on the default stream.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[in] dst_host_ptr A pointer to the destination buffer on the CPU.
 * @param[in] src_buffer The source buffer handle on the GPU.
 * @param[in] size_bytes The number of bytes to copy.
 *
 * @return TK_SUCCESS if the copy was successfully enqueued.
 * @return TK_ERROR_GPU_MEMORY_COPY_FAILED on failure.
 */
TK_NODISCARD tk_error_code_t tk_rocm_dispatch_download_async(tk_rocm_dispatcher_t* dispatcher, void* dst_host_ptr, tk_gpu_buffer_t src_buffer, size_t size_bytes);

//------------------------------------------------------------------------------
// Synchronization Primitives
//------------------------------------------------------------------------------

/**
 * @brief Blocks the calling CPU thread until all previously enqueued work in the
 *        dispatcher's default stream is complete.
 *
 * This is a coarse-grained synchronization primitive. For more efficient,
 * fine-grained synchronization, an event-based system should be used.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @return TK_SUCCESS on successful synchronization.
 * @return TK_ERROR_GPU_SYNCHRONIZATION_FAILED on failure.
 */
TK_NODISCARD tk_error_code_t tk_rocm_dispatch_synchronize(tk_rocm_dispatcher_t* dispatcher);

//------------------------------------------------------------------------------
// Utilities
//------------------------------------------------------------------------------

/**
 * @brief Retrieves the default HIP stream from the dispatcher instance.
 *
 * This allows other components to enqueue operations on the same stream, ensuring
 * correct ordering and synchronization.
 *
 * @param[in]  dispatcher The dispatcher instance.
 * @param[out] stream     A pointer to a hipStream_t variable to receive the stream.
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_INVALID_ARGUMENT if any parameter is invalid.
 */
TK_NODISCARD tk_error_code_t tk_rocm_dispatch_get_stream(tk_rocm_dispatcher_t* dispatcher, hipStream_t* stream);


//------------------------------------------------------------------------------
// High-Level Workflow Dispatch
//------------------------------------------------------------------------------

/**
 * @brief Dispatches the entire vision pre-processing workflow to the GPU.
 *
 * This high-level function encapsulates the launch of the pre-processing kernel.
 * It assumes that the necessary data has already been uploaded to the GPU buffers
 * specified in the parameters.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[in] params The parameters for the pre-processing kernel. The device
 *                   pointers within this struct must correspond to valid
 *                   `tk_gpu_buffer_t` handles that have been allocated.
 *
 * @return TK_SUCCESS if the workflow was successfully enqueued.
 */
TK_NODISCARD tk_error_code_t tk_rocm_dispatch_preprocess_image(tk_rocm_dispatcher_t* dispatcher, const tk_preprocess_params_t* params);

/**
 * @brief Dispatches the depth-to-point-cloud conversion workflow to the GPU.
 *
 * @param[in] dispatcher The dispatcher instance.
 * @param[in] params The parameters for the conversion kernel. Device pointers
 *                   must correspond to valid `tk_gpu_buffer_t` handles.
 *
 * @return TK_SUCCESS if the workflow was successfully enqueued.
 */
TK_NODISCARD tk_error_code_t tk_rocm_dispatch_depth_to_point_cloud(tk_rocm_dispatcher_t* dispatcher, const tk_depth_to_points_params_t* params);

#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_GPU_ROCM_TK_ROCM_DISPATCH_HPP
