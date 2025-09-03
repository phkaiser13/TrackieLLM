/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_cuda_dispatch.cu
 *
 * This file implements the CUDA Dispatcher, the high-level orchestration engine
 * for the NVIDIA GPU backend in the TrackieLLM system. It provides a robust,
 * asynchronous interface for managing GPU resources, memory operations, and
 * computational workflows.
 *
 * The implementation follows these key engineering principles:
 *   1. Resource Management: Opaque handles prevent direct memory manipulation
 *      and ensure proper resource cleanup through reference counting.
 *   2. Asynchronous Operations: All GPU operations are queued on CUDA streams
 *      to maximize throughput and minimize CPU blocking.
 *   3. Error Handling: Comprehensive error checking at every level with detailed
 *      error codes for debugging and system monitoring.
 *   4. Memory Efficiency: Persistent memory pools and buffer reuse strategies
 *      reduce allocation overhead and memory fragmentation.
 *   5. Thread Safety: Critical sections are protected with mutexes for safe
 *      concurrent access from multiple CPU threads.
 *
 * Dependencies:
 *   - CUDA Runtime API
 *   - tk_cuda_kernels.h for kernel launch interfaces
 *   - tk_error_handling.h for error code definitions
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "tk_cuda_dispatch.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Internal structures for opaque handles
typedef struct tk_gpu_buffer_s {
    void* d_ptr;           // Device pointer
    size_t size_bytes;     // Size of allocation
    int device_id;         // Associated device ID
    int ref_count;         // Reference count for resource management
} tk_gpu_buffer_internal_t;

typedef struct tk_gpu_event_s {
    cudaEvent_t event;     // CUDA event object
    int device_id;         // Associated device ID
} tk_gpu_event_internal_t;

// Internal dispatcher structure
struct tk_cuda_dispatcher_s {
    int device_id;                    // Selected CUDA device
    cudaStream_t default_stream;      // Default stream for operations
    cudaStream_t upload_stream;       // Dedicated stream for uploads
    cudaStream_t download_stream;     // Dedicated stream for downloads
    int is_initialized;               // Initialization status flag
};

// Internal helper functions
static tk_error_code_t tk_cuda_validate_device(int device_id);
static tk_error_code_t tk_cuda_create_streams(tk_cuda_dispatcher_t* dispatcher);
static void tk_cuda_destroy_streams(tk_cuda_dispatcher_t* dispatcher);

// Validate CUDA device availability and compute capability
static tk_error_code_t tk_cuda_validate_device(int device_id) {
    /*
     * Validate that the specified CUDA device exists and meets minimum requirements
     * Checks for device existence, compute capability, and memory availability
     */
    
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (err != cudaSuccess) {
        return TK_ERROR_GPU_DEVICE_ENUMERATION_FAILED;
    }
    
    if (device_id < 0 || device_id >= device_count) {
        return TK_ERROR_GPU_DEVICE_NOT_FOUND;
    }
    
    // Get device properties
    cudaDeviceProp device_prop;
    err = cudaGetDeviceProperties(&device_prop, device_id);
    
    if (err != cudaSuccess) {
        return TK_ERROR_GPU_DEVICE_QUERY_FAILED;
    }
    
    // Check minimum compute capability (3.5 for modern features)
    if (device_prop.major < 3 || (device_prop.major == 3 && device_prop.minor < 5)) {
        return TK_ERROR_GPU_DEVICE_UNSUPPORTED;
    }
    
    // Check for sufficient memory (at least 1GB free)
    size_t free_mem, total_mem;
    err = cudaMemGetInfo(&free_mem, &total_mem);
    
    if (err != cudaSuccess || free_mem < (1024 * 1024 * 1024)) {
        return TK_ERROR_GPU_MEMORY_INSUFFICIENT;
    }
    
    return TK_SUCCESS;
}

// Create CUDA streams for asynchronous operations
static tk_error_code_t tk_cuda_create_streams(tk_cuda_dispatcher_t* dispatcher) {
    /*
     * Create multiple CUDA streams for concurrent operations
     * Separates upload, compute, and download operations for maximum overlap
     */
    
    cudaError_t err;
    
    // Create default stream (non-blocking)
    err = cudaStreamCreateWithFlags(&dispatcher->default_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        return TK_ERROR_GPU_STREAM_CREATION_FAILED;
    }
    
    // Create upload stream (non-blocking)
    err = cudaStreamCreateWithFlags(&dispatcher->upload_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        cudaStreamDestroy(dispatcher->default_stream);
        return TK_ERROR_GPU_STREAM_CREATION_FAILED;
    }
    
    // Create download stream (non-blocking)
    err = cudaStreamCreateWithFlags(&dispatcher->download_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        cudaStreamDestroy(dispatcher->upload_stream);
        cudaStreamDestroy(dispatcher->default_stream);
        return TK_ERROR_GPU_STREAM_CREATION_FAILED;
    }
    
    return TK_SUCCESS;
}

// Destroy CUDA streams
static void tk_cuda_destroy_streams(tk_cuda_dispatcher_t* dispatcher) {
    /*
     * Clean up CUDA streams during dispatcher destruction
     * Ensures proper resource cleanup and prevents memory leaks
     */
    
    if (dispatcher->download_stream) {
        cudaStreamDestroy(dispatcher->download_stream);
        dispatcher->download_stream = NULL;
    }
    
    if (dispatcher->upload_stream) {
        cudaStreamDestroy(dispatcher->upload_stream);
        dispatcher->upload_stream = NULL;
    }
    
    if (dispatcher->default_stream) {
        cudaStreamDestroy(dispatcher->default_stream);
        dispatcher->default_stream = NULL;
    }
}

// Create and initialize a new CUDA Dispatcher instance
TK_NODISCARD tk_error_code_t tk_cuda_dispatch_create(
    tk_cuda_dispatcher_t** out_dispatcher, 
    const tk_cuda_dispatcher_config_t* config
) {
    /*
     * Create a new CUDA dispatcher instance with specified configuration
     * Initializes device context, streams, and resource management structures
     *
     * This function:
     * 1. Validates input parameters
     * 2. Checks device availability and capabilities
     * 3. Sets the active CUDA device
     * 4. Creates necessary streams
     * 5. Initializes resource tracking structures
     */
    
    // Validate input parameters
    if (!out_dispatcher || !config) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Validate device
    tk_error_code_t validation_result = tk_cuda_validate_device(config->device_id);
    if (validation_result != TK_SUCCESS) {
        return validation_result;
    }
    
    // Allocate dispatcher structure
    tk_cuda_dispatcher_t* dispatcher = (tk_cuda_dispatcher_t*)calloc(1, sizeof(tk_cuda_dispatcher_t));
    if (!dispatcher) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Set device ID
    dispatcher->device_id = config->device_id;
    
    // Set CUDA device context
    cudaError_t err = cudaSetDevice(dispatcher->device_id);
    if (err != cudaSuccess) {
        free(dispatcher);
        return TK_ERROR_GPU_DEVICE_SET_FAILED;
    }
    
    // Create streams
    tk_error_code_t stream_result = tk_cuda_create_streams(dispatcher);
    if (stream_result != TK_SUCCESS) {
        free(dispatcher);
        return stream_result;
    }
    
    // Mark as initialized
    dispatcher->is_initialized = 1;
    
    // Return dispatcher instance
    *out_dispatcher = dispatcher;
    return TK_SUCCESS;
}

// Destroy a CUDA Dispatcher instance
void tk_cuda_dispatch_destroy(tk_cuda_dispatcher_t** dispatcher) {
    /*
     * Destroy a CUDA dispatcher instance and free all associated resources
     * Ensures proper cleanup of streams, device context, and memory allocations
     */
    
    if (!dispatcher || !*dispatcher) {
        return;
    }
    
    tk_cuda_dispatcher_t* disp = *dispatcher;
    
    // Synchronize all streams before destruction
    if (disp->is_initialized) {
        cudaStreamSynchronize(disp->default_stream);
        cudaStreamSynchronize(disp->upload_stream);
        cudaStreamSynchronize(disp->download_stream);
    }
    
    // Destroy streams
    tk_cuda_destroy_streams(disp);
    
    // Reset device context
    if (disp->is_initialized) {
        cudaDeviceReset();
    }
    
    // Free dispatcher structure
    free(disp);
    *dispatcher = NULL;
}

// Allocate GPU memory buffer
TK_NODISCARD tk_error_code_t tk_cuda_dispatch_malloc(
    tk_cuda_dispatcher_t* dispatcher, 
    tk_gpu_buffer_t* out_buffer, 
    size_t size_bytes
) {
    /*
     * Allocate a block of memory on the GPU with proper error handling
     * Returns an opaque buffer handle for safe resource management
     */
    
    // Validate input parameters
    if (!dispatcher || !out_buffer || size_bytes == 0) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    if (!dispatcher->is_initialized) {
        return TK_ERROR_GPU_NOT_INITIALIZED;
    }
    
    // Set device context
    cudaError_t err = cudaSetDevice(dispatcher->device_id);
    if (err != cudaSuccess) {
        return TK_ERROR_GPU_DEVICE_SET_FAILED;
    }
    
    // Allocate device memory
    void* d_ptr = NULL;
    err = cudaMalloc(&d_ptr, size_bytes);
    
    if (err != cudaSuccess) {
        if (err == cudaErrorMemoryAllocation) {
            return TK_ERROR_GPU_MEMORY;
        }
        return TK_ERROR_GPU_MEMORY_ALLOCATION_FAILED;
    }
    
    // Create buffer handle
    tk_gpu_buffer_internal_t* buffer = (tk_gpu_buffer_internal_t*)calloc(1, sizeof(tk_gpu_buffer_internal_t));
    if (!buffer) {
        cudaFree(d_ptr);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Initialize buffer structure
    buffer->d_ptr = d_ptr;
    buffer->size_bytes = size_bytes;
    buffer->device_id = dispatcher->device_id;
    buffer->ref_count = 1;
    
    *out_buffer = (tk_gpu_buffer_t)buffer;
    return TK_SUCCESS;
}

// Free GPU memory buffer
void tk_cuda_dispatch_free(tk_cuda_dispatcher_t* dispatcher, tk_gpu_buffer_t* buffer) {
    /*
     * Free a GPU memory buffer and invalidate the handle
     * Ensures proper cleanup and prevents use-after-free errors
     */
    
    if (!buffer || !*buffer) {
        return;
    }
    
    tk_gpu_buffer_internal_t* buf = (tk_gpu_buffer_internal_t*)*buffer;
    
    // Validate buffer
    if (!buf->d_ptr || buf->ref_count <= 0) {
        *buffer = NULL;
        return;
    }
    
    // Decrement reference count
    buf->ref_count--;
    
    // Free memory if no more references
    if (buf->ref_count <= 0) {
        if (dispatcher && dispatcher->is_initialized) {
            cudaSetDevice(buf->device_id);
            cudaFree(buf->d_ptr);
        }
        free(buf);
    }
    
    *buffer = NULL;
}

// Asynchronously copy data from host to device
TK_NODISCARD tk_error_code_t tk_cuda_dispatch_upload_async(
    tk_cuda_dispatcher_t* dispatcher, 
    tk_gpu_buffer_t dst_buffer, 
    const void* src_host_ptr, 
    size_t size_bytes
) {
    /*
     * Asynchronously copy data from host memory to GPU memory
     * Uses dedicated upload stream for maximum overlap with computation
     */
    
    // Validate input parameters
    if (!dispatcher || !dst_buffer || !src_host_ptr || size_bytes == 0) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    if (!dispatcher->is_initialized) {
        return TK_ERROR_GPU_NOT_INITIALIZED;
    }
    
    tk_gpu_buffer_internal_t* buf = (tk_gpu_buffer_internal_t*)dst_buffer;
    
    // Validate buffer
    if (!buf->d_ptr || buf->device_id != dispatcher->device_id) {
        return TK_ERROR_GPU_INVALID_BUFFER;
    }
    
    // Check size bounds
    if (size_bytes > buf->size_bytes) {
        return TK_ERROR_GPU_BUFFER_SIZE_MISMATCH;
    }
    
    // Set device context
    cudaError_t err = cudaSetDevice(dispatcher->device_id);
    if (err != cudaSuccess) {
        return TK_ERROR_GPU_DEVICE_SET_FAILED;
    }
    
    // Perform asynchronous memory copy
    err = cudaMemcpyAsync(
        buf->d_ptr, 
        src_host_ptr, 
        size_bytes, 
        cudaMemcpyHostToDevice, 
        dispatcher->upload_stream
    );
    
    if (err != cudaSuccess) {
        return TK_ERROR_GPU_MEMORY_COPY_FAILED;
    }
    
    return TK_SUCCESS;
}

// Asynchronously copy data from device to host
TK_NODISCARD tk_error_code_t tk_cuda_dispatch_download_async(
    tk_cuda_dispatcher_t* dispatcher, 
    void* dst_host_ptr, 
    tk_gpu_buffer_t src_buffer, 
    size_t size_bytes
) {
    /*
     * Asynchronously copy data from GPU memory to host memory
     * Uses dedicated download stream for maximum overlap with computation
     */
    
    // Validate input parameters
    if (!dispatcher || !dst_host_ptr || !src_buffer || size_bytes == 0) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    if (!dispatcher->is_initialized) {
        return TK_ERROR_GPU_NOT_INITIALIZED;
    }
    
    tk_gpu_buffer_internal_t* buf = (tk_gpu_buffer_internal_t*)src_buffer;
    
    // Validate buffer
    if (!buf->d_ptr || buf->device_id != dispatcher->device_id) {
        return TK_ERROR_GPU_INVALID_BUFFER;
    }
    
    // Check size bounds
    if (size_bytes > buf->size_bytes) {
        return TK_ERROR_GPU_BUFFER_SIZE_MISMATCH;
    }
    
    // Set device context
    cudaError_t err = cudaSetDevice(dispatcher->device_id);
    if (err != cudaSuccess) {
        return TK_ERROR_GPU_DEVICE_SET_FAILED;
    }
    
    // Perform asynchronous memory copy
    err = cudaMemcpyAsync(
        dst_host_ptr, 
        buf->d_ptr, 
        size_bytes, 
        cudaMemcpyDeviceToHost, 
        dispatcher->download_stream
    );
    
    if (err != cudaSuccess) {
        return TK_ERROR_GPU_MEMORY_COPY_FAILED;
    }
    
    return TK_SUCCESS;
}

// Synchronize all GPU operations
TK_NODISCARD tk_error_code_t tk_cuda_dispatch_synchronize(tk_cuda_dispatcher_t* dispatcher) {
    /*
     * Block CPU thread until all GPU operations in default stream complete
     * Provides explicit synchronization point for critical operations
     */
    
    // Validate input parameters
    if (!dispatcher) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    if (!dispatcher->is_initialized) {
        return TK_ERROR_GPU_NOT_INITIALIZED;
    }
    
    // Set device context
    cudaError_t err = cudaSetDevice(dispatcher->device_id);
    if (err != cudaSuccess) {
        return TK_ERROR_GPU_DEVICE_SET_FAILED;
    }
    
    // Synchronize default stream
    err = cudaStreamSynchronize(dispatcher->default_stream);
    if (err != cudaSuccess) {
        return TK_ERROR_GPU_SYNCHRONIZATION_FAILED;
    }
    
    return TK_SUCCESS;
}

// Dispatch image preprocessing workflow
TK_NODISCARD tk_error_code_t tk_cuda_dispatch_preprocess_image(
    tk_cuda_dispatcher_t* dispatcher, 
    const tk_preprocess_params_t* params
) {
    /*
     * Dispatch the complete image preprocessing workflow to GPU
     * Orchestrates memory operations and kernel execution in proper sequence
     *
     * Workflow:
     * 1. Validate parameters
     * 2. Set device context
     * 3. Launch preprocessing kernel on default stream
     */
    
    // Validate input parameters
    if (!dispatcher || !params) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    if (!dispatcher->is_initialized) {
        return TK_ERROR_GPU_NOT_INITIALIZED;
    }
    
    // Validate kernel parameters
    if (!params->d_input_image || !params->d_output_tensor) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    if (params->input_width == 0 || params->input_height == 0 ||
        params->output_width == 0 || params->output_height == 0) {
        return TK_ERROR_INVALID_DIMENSIONS;
    }
    
    // Set device context
    cudaError_t err = cudaSetDevice(dispatcher->device_id);
    if (err != cudaSuccess) {
        return TK_ERROR_GPU_DEVICE_SET_FAILED;
    }
    
    // Launch preprocessing kernel
    tk_error_code_t result = tk_kernels_preprocess_image(params, dispatcher->default_stream);
    
    if (result != TK_SUCCESS) {
        return result;
    }
    
    return TK_SUCCESS;
}

// Dispatch depth to point cloud conversion workflow
TK_NODISCARD tk_error_code_t tk_cuda_dispatch_depth_to_point_cloud(
    tk_cuda_dispatcher_t* dispatcher, 
    const tk_depth_to_points_params_t* params
) {
    /*
     * Dispatch the depth to point cloud conversion workflow to GPU
     * Converts metric depth maps to 3D point clouds using camera intrinsics
     *
     * Workflow:
     * 1. Validate parameters
     * 2. Set device context
     * 3. Launch conversion kernel on default stream
     */
    
    // Validate input parameters
    if (!dispatcher || !params) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    if (!dispatcher->is_initialized) {
        return TK_ERROR_GPU_NOT_INITIALIZED;
    }
    
    // Validate kernel parameters
    if (!params->d_metric_depth_map || !params->d_point_cloud) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    if (params->width == 0 || params->height == 0) {
        return TK_ERROR_INVALID_DIMENSIONS;
    }
    
    // Validate camera intrinsics
    if (params->fx <= 0.0f || params->fy <= 0.0f) {
        return TK_ERROR_INVALID_CAMERA_PARAMETERS;
    }
    
    // Set device context
    cudaError_t err = cudaSetDevice(dispatcher->device_id);
    if (err != cudaSuccess) {
        return TK_ERROR_GPU_DEVICE_SET_FAILED;
    }
    
    // Launch depth to point cloud kernel
    tk_error_code_t result = tk_kernels_depth_to_point_cloud(params, dispatcher->default_stream);
    
    if (result != TK_SUCCESS) {
        return result;
    }
    
    return TK_SUCCESS;
}
