/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_error_handling.c
*
* This source file implements the error handling utility functions for the
* TrackieLLM project. The primary function, tk_error_to_string, provides a
* mapping from an error code enum to a human-readable, null-terminated string.
* This implementation is designed to be fast, thread-safe, and comprehensive.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#include "tk_error_handling.h"

/**
 * @brief Converts a TrackieLLM error code into a static, human-readable string.
 *
 * This is the implementation of the function declared in the header. It uses a
 * switch statement for a direct, O(1) lookup of the error string.
 */
const char* tk_error_to_string(tk_error_code_t code) {
    switch (code) {
        // Success Codes
        case TK_SUCCESS:
            return "TK_SUCCESS: Operation completed successfully.";

        // General Errors
        case TK_ERROR_UNKNOWN:
            return "TK_ERROR_UNKNOWN: An unknown or unspecified error occurred.";
        case TK_ERROR_INVALID_ARGUMENT:
            return "TK_ERROR_INVALID_ARGUMENT: A function argument was invalid (e.g., NULL pointer).";
        case TK_ERROR_INVALID_STATE:
            return "TK_ERROR_INVALID_STATE: The system or module was in an invalid state for the operation.";
        case TK_ERROR_NOT_IMPLEMENTED:
            return "TK_ERROR_NOT_IMPLEMENTED: The requested feature or function is not yet implemented.";
        case TK_ERROR_BUFFER_TOO_SMALL:
            return "TK_ERROR_BUFFER_TOO_SMALL: A provided buffer was too small to hold the result.";
        case TK_ERROR_TIMEOUT:
            return "TK_ERROR_TIMEOUT: The operation timed out.";
        case TK_ERROR_PERMISSION_DENIED:
            return "TK_ERROR_PERMISSION_DENIED: Insufficient permissions to perform the operation.";
        case TK_ERROR_NOT_INITIALIZED:
            return "TK_ERROR_NOT_INITIALIZED: A required subsystem or module has not been initialized.";

        // Memory Errors
        case TK_ERROR_OUT_OF_MEMORY:
            return "TK_ERROR_OUT_OF_MEMORY: Failed to allocate memory.";
        case TK_ERROR_MEMORY_ALIGNMENT:
            return "TK_ERROR_MEMORY_ALIGNMENT: A memory alignment requirement was not met.";
        case TK_ERROR_MEMORY_POOL_EXHAUSTED:
            return "TK_ERROR_MEMORY_POOL_EXHAUSTED: A fixed-size memory pool has no available blocks.";
        case TK_ERROR_MEMORY_DOUBLE_FREE:
            return "TK_ERROR_MEMORY_DOUBLE_FREE: Attempted to free an already freed memory block.";
        case TK_ERROR_MEMORY_INVALID_POINTER:
            return "TK_ERROR_MEMORY_INVALID_POINTER: A pointer was invalid for the requested memory operation.";

        // I/O and Filesystem Errors
        case TK_ERROR_IO:
            return "TK_ERROR_IO: A generic input/output error occurred.";
        case TK_ERROR_FILE_NOT_FOUND:
            return "TK_ERROR_FILE_NOT_FOUND: The specified file or path does not exist.";
        case TK_ERROR_FILE_READ:
            return "TK_ERROR_FILE_READ: An error occurred while reading from a file.";
        case TK_ERROR_FILE_WRITE:
            return "TK_ERROR_FILE_WRITE: An error occurred while writing to a file.";
        case TK_ERROR_FILE_CORRUPT:
            return "TK_ERROR_FILE_CORRUPT: The file is corrupted or in an unexpected format.";
        case TK_ERROR_CONFIG_PARSE_FAILED:
            return "TK_ERROR_CONFIG_PARSE_FAILED: Failed to parse a configuration file.";

        // AI Model and Inference Errors
        case TK_ERROR_MODEL_LOAD_FAILED:
            return "TK_ERROR_MODEL_LOAD_FAILED: Failed to load an AI model from a file.";
        case TK_ERROR_MODEL_VERIFICATION_FAILED:
            return "TK_ERROR_MODEL_VERIFICATION_FAILED: The model file failed a verification or integrity check.";
        case TK_ERROR_INFERENCE_FAILED:
            return "TK_ERROR_INFERENCE_FAILED: The inference engine failed to process the input.";
        case TK_ERROR_INVALID_INPUT_TENSOR:
            return "TK_ERROR_INVALID_INPUT_TENSOR: The input tensor has incorrect dimensions, type, or data.";
        case TK_ERROR_INVALID_OUTPUT_TENSOR:
            return "TK_ERROR_INVALID_OUTPUT_TENSOR: The output tensor has incorrect dimensions, type, or data.";
        case TK_ERROR_BACKEND_NOT_SUPPORTED:
            return "TK_ERROR_BACKEND_NOT_SUPPORTED: The selected inference backend is not supported on this hardware.";

        // GPU and Hardware Acceleration Errors
        case TK_ERROR_GPU_ERROR:
            return "TK_ERROR_GPU_ERROR: A generic, unspecified GPU error occurred.";
        case TK_ERROR_GPU_DEVICE_NOT_FOUND:
            return "TK_ERROR_GPU_DEVICE_NOT_FOUND: No compatible GPU device was found.";
        case TK_ERROR_GPU_DRIVER_VERSION:
            return "TK_ERROR_GPU_DRIVER_VERSION: The installed GPU driver version is incompatible.";
        case TK_ERROR_GPU_CUDA_ERROR:
            return "TK_ERROR_GPU_CUDA_ERROR: A specific error from the CUDA runtime API occurred.";
        case TK_ERROR_GPU_METAL_ERROR:
            return "TK_ERROR_GPU_METAL_ERROR: A specific error from the Metal framework occurred.";
        case TK_ERROR_GPU_ROCM_ERROR:
            return "TK_ERROR_GPU_ROCM_ERROR: A specific error from the ROCm/HIP runtime occurred.";
        case TK_ERROR_GPU_KERNEL_LAUNCH:
            return "TK_ERROR_GPU_KERNEL_LAUNCH: Failed to launch a compute kernel on the GPU.";
        case TK_ERROR_GPU_MEMORY:
            return "TK_ERROR_GPU_MEMORY: An error occurred with GPU memory allocation or transfer.";

        // Networking Errors
        case TK_ERROR_NETWORK_ERROR:
            return "TK_ERROR_NETWORK_ERROR: A generic networking error occurred.";
        case TK_ERROR_CONNECTION_FAILED:
            return "TK_ERROR_CONNECTION_FAILED: Failed to establish a network connection.";
        case TK_ERROR_CONNECTION_CLOSED:
            return "TK_ERROR_CONNECTION_CLOSED: The connection was closed unexpectedly.";
        case TK_ERROR_DNS_RESOLUTION_FAILED:
            return "TK_ERROR_DNS_RESOLUTION_FAILED: Failed to resolve a hostname.";
        case TK_ERROR_SOCKET_ERROR:
            return "TK_ERROR_SOCKET_ERROR: A low-level socket operation failed.";

        // Concurrency and Tasking Errors
        case TK_ERROR_THREAD_CREATE_FAILED:
            return "TK_ERROR_THREAD_CREATE_FAILED: Failed to create a new thread.";
        case TK_ERROR_MUTEX_ERROR:
            return "TK_ERROR_MUTEX_ERROR: An error occurred with a mutex operation (lock, unlock).";
        case TK_ERROR_SEMAPHORE_ERROR:
            return "TK_ERROR_SEMAPHORE_ERROR: An error occurred with a semaphore operation.";
        case TK_ERROR_TASK_QUEUE_FULL:
            return "TK_ERROR_TASK_QUEUE_FULL: The task scheduler's queue is full.";
        case TK_ERROR_FUTURE_CANCELLED:
            return "TK_ERROR_FUTURE_CANCELLED: An asynchronous task was cancelled before completion.";

        // FFI Errors
        case TK_ERROR_FFI_PANIC:
            return "TK_ERROR_FFI_PANIC: A panic occurred within the Rust side of an FFI call.";
        case TK_ERROR_FFI_INVALID_STRING:
            return "TK_ERROR_FFI_INVALID_STRING: A string passed across the FFI boundary was not valid UTF-8.";

        // The sentinel value should not be matched.
        case TK_ERROR_CODE_COUNT:
            break;
    }

    // Default case for any unhandled or invalid error codes.
    return "UNRECOGNIZED_ERROR_CODE: The provided error code is not valid or recognized.";
}


#include <stdarg.h>
#include <stdio.h> // For vsnprintf

// Define a thread-local buffer for detailed error messages.
// Each thread will have its own instance of this buffer.
// Using __thread, a common extension in GCC/Clang.
#define TK_ERROR_DETAIL_BUFFER_SIZE 1024
static __thread char g_error_detail_buffer[TK_ERROR_DETAIL_BUFFER_SIZE] = {0};

/**
 * @brief Implementation of tk_error_set_detail.
 */
void tk_error_set_detail(const char* fmt, ...) {
    if (!fmt) {
        g_error_detail_buffer[0] = '\0';
        return;
    }

    va_list args;
    va_start(args, fmt);
    // vsnprintf is safe and prevents buffer overflows.
    vsnprintf(g_error_detail_buffer, TK_ERROR_DETAIL_BUFFER_SIZE, fmt, args);
    va_end(args);
}

/**
 * @brief Implementation of tk_error_get_detail.
 */
const char* tk_error_get_detail(void) {
    return g_error_detail_buffer;
}