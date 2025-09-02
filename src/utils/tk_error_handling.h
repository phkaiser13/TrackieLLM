/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_error_handling.h
*
* This header file defines the core error handling infrastructure for the
* TrackieLLM project. It establishes a comprehensive set of error codes,
* provides utilities for converting these codes into human-readable strings,
* and defines macros to enforce best practices, such as checking function
* return values. A centralized error system is critical for building a robust,
* maintainable, and debuggable application.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TRACKIELLM_UTILS_TK_ERROR_HANDLING_H
#define TRACKIELLM_UTILS_TK_ERROR_HANDLING_H

#include <stddef.h> // For NULL
#include <stdint.h> // For standard integer types

// Define a cross-compiler attribute for functions whose return value must be used.
// This helps prevent bugs where an error code is returned but ignored by the caller.
#if defined(__GNUC__) || defined(__clang__)
    #define TK_NODISCARD __attribute__((warn_unused_result))
#elif defined(_MSC_VER)
    #define TK_NODISCARD _Check_return_
#else
    #define TK_NODISCARD
#endif

/**
 * @enum tk_error_code_t
 * @brief Defines all possible error codes within the TrackieLLM system.
 *
 * The error codes are organized by module to provide context. Each error code
 * has a unique, stable integer value. This enum is the single source of truth
* for error reporting in the C/C++ parts of the codebase.
 */
typedef enum tk_error_code_t {
    //--------------------------------------------------------------------------
    // Success Codes
    //--------------------------------------------------------------------------
    TK_SUCCESS = 0,                  /**< Operation completed successfully. */

    //--------------------------------------------------------------------------
    // General Errors (1000-1999)
    //--------------------------------------------------------------------------
    TK_ERROR_UNKNOWN = 1000,         /**< An unknown or unspecified error occurred. */
    TK_ERROR_INVALID_ARGUMENT,       /**< A function argument was invalid (e.g., NULL pointer). */
    TK_ERROR_INVALID_STATE,          /**< The system or module was in an invalid state for the operation. */
    TK_ERROR_NOT_IMPLEMENTED,        /**< The requested feature or function is not yet implemented. */
    TK_ERROR_BUFFER_TOO_SMALL,       /**< A provided buffer was too small to hold the result. */
    TK_ERROR_TIMEOUT,                /**< The operation timed out. */
    TK_ERROR_PERMISSION_DENIED,      /**< Insufficient permissions to perform the operation. */
    TK_ERROR_NOT_INITIALIZED,        /**< A required subsystem or module has not been initialized. */

    //--------------------------------------------------------------------------
    // Memory Errors (2000-2999)
    //--------------------------------------------------------------------------
    TK_ERROR_OUT_OF_MEMORY = 2000,   /**< Failed to allocate memory (e.g., malloc returned NULL). */
    TK_ERROR_MEMORY_ALIGNMENT,       /**< A memory alignment requirement was not met. */
    TK_ERROR_MEMORY_POOL_EXHAUSTED,  /**< A fixed-size memory pool has no available blocks. */
    TK_ERROR_MEMORY_DOUBLE_FREE,     /**< Attempted to free an already freed memory block. */
    TK_ERROR_MEMORY_INVALID_POINTER, /**< A pointer was invalid for the requested memory operation. */

    //--------------------------------------------------------------------------
    // I/O and Filesystem Errors (3000-3999)
    //--------------------------------------------------------------------------
    TK_ERROR_IO = 3000,              /**< A generic input/output error occurred. */
    TK_ERROR_FILE_NOT_FOUND,         /**< The specified file or path does not exist. */
    TK_ERROR_FILE_READ,              /**< An error occurred while reading from a file. */
    TK_ERROR_FILE_WRITE,             /**< An error occurred while writing to a file. */
    TK_ERROR_FILE_CORRUPT,           /**< The file is corrupted or in an unexpected format. */
    TK_ERROR_CONFIG_PARSE_FAILED,    /**< Failed to parse a configuration file. */

    //--------------------------------------------------------------------------
    // AI Model and Inference Errors (4000-4999)
    //--------------------------------------------------------------------------
    TK_ERROR_MODEL_LOAD_FAILED = 4000, /**< Failed to load an AI model from a file. */
    TK_ERROR_MODEL_VERIFICATION_FAILED,/**< The model file failed a verification or integrity check. */
    TK_ERROR_INFERENCE_FAILED,       /**< The inference engine failed to process the input. */
    TK_ERROR_INVALID_INPUT_TENSOR,   /**< The input tensor has incorrect dimensions, type, or data. */
    TK_ERROR_INVALID_OUTPUT_TENSOR,  /**< The output tensor has incorrect dimensions, type, or data. */
    TK_ERROR_BACKEND_NOT_SUPPORTED,  /**< The selected inference backend is not supported on this hardware. */

    //--------------------------------------------------------------------------
    // GPU and Hardware Acceleration Errors (5000-5999)
    //--------------------------------------------------------------------------
    TK_ERROR_GPU_ERROR = 5000,       /**< A generic, unspecified GPU error occurred. */
    TK_ERROR_GPU_DEVICE_NOT_FOUND,   /**< No compatible GPU device was found. */
    TK_ERROR_GPU_DRIVER_VERSION,     /**< The installed GPU driver version is incompatible. */
    TK_ERROR_GPU_CUDA_ERROR,         /**< A specific error from the CUDA runtime API occurred. */
    TK_ERROR_GPU_METAL_ERROR,        /**< A specific error from the Metal framework occurred. */
    TK_ERROR_GPU_ROCM_ERROR,         /**< A specific error from the ROCm/HIP runtime occurred. */
    TK_ERROR_GPU_KERNEL_LAUNCH,      /**< Failed to launch a compute kernel on the GPU. */
    TK_ERROR_GPU_MEMORY,             /**< An error occurred with GPU memory allocation or transfer. */

    //--------------------------------------------------------------------------
    // Networking Errors (6000-6999)
    //--------------------------------------------------------------------------
    TK_ERROR_NETWORK_ERROR = 6000,   /**< A generic networking error occurred. */
    TK_ERROR_CONNECTION_FAILED,      /**< Failed to establish a network connection. */
    TK_ERROR_CONNECTION_CLOSED,      /**< The connection was closed unexpectedly. */
    TK_ERROR_DNS_RESOLUTION_FAILED,  /**< Failed to resolve a hostname. */
    TK_ERROR_SOCKET_ERROR,           /**< A low-level socket operation failed. */

    //--------------------------------------------------------------------------
    // Concurrency and Tasking Errors (7000-7999)
    //--------------------------------------------------------------------------
    TK_ERROR_THREAD_CREATE_FAILED = 7000, /**< Failed to create a new thread. */
    TK_ERROR_MUTEX_ERROR,            /**< An error occurred with a mutex operation (lock, unlock). */
    TK_ERROR_SEMAPHORE_ERROR,        /**< An error occurred with a semaphore operation. */
    TK_ERROR_TASK_QUEUE_FULL,        /**< The task scheduler's queue is full. */
    TK_ERROR_FUTURE_CANCELLED,       /**< An asynchronous task was cancelled before completion. */

    //--------------------------------------------------------------------------
    // FFI Errors (8000-8999)
    //--------------------------------------------------------------------------
    TK_ERROR_FFI_PANIC = 8000,       /**< A panic occurred within the Rust side of an FFI call. */
    TK_ERROR_FFI_INVALID_STRING,     /**< A string passed across the FFI boundary was not valid UTF-8. */

    TK_ERROR_CODE_COUNT              /**< Sentinel value; represents the total number of error codes. */
} tk_error_code_t;


#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Converts a TrackieLLM error code into a static, human-readable string.
 *
 * This function provides a convenient way to get a descriptive string for a given
 * error code, which is invaluable for logging, debugging, and user-facing error
 * messages. The returned string is a constant literal and should not be modified
 * or freed.
 *
 * @param[in] code The error code to convert.
 *
 * @return A null-terminated C-string describing the error. If the code is
 *         unrecognized, a string indicating an "Unknown Error Code" is returned.
 *         This function never returns NULL.
 *
 * @par Thread-Safety
 * This function is thread-safe as it only reads from static data.
 *
 * @par Complexity
 * Time: O(1) - The implementation should use a direct lookup (e.g., switch statement or array).
 * Space: O(1) - No memory is allocated.
 *
 * @par Example
 * @code
 * tk_error_code_t result = some_function();
 * if (result != TK_SUCCESS) {
 *     printf("An error occurred: %s\n", tk_error_to_string(result));
 * }
 * @endcode
 */
const char* tk_error_to_string(tk_error_code_t code);

#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_UTILS_TK_ERROR_HANDLING_H