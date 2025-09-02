/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_file_manager.h
*
* This header defines the interface for the TrackieLLM Filesystem Abstraction
* Layer (FAL). The primary goal of this module is to provide a secure,
* platform-independent, and robust API for all filesystem operations.
*
* It replaces the direct use of C-style string paths with an opaque `tk_path_t`
* object. This design choice is deliberate and critical for several reasons:
*   1. Security: It enables centralized validation and sanitization of paths,
*      mitigating risks such as directory traversal attacks (`../../`).
*   2. Portability: It abstracts away platform-specific path separators (`/` vs `\`)
*      and path length limits (e.g., MAX_PATH on Windows).
*   3. Robustness: It enforces a clear ownership model for path resources,
*      reducing memory management errors.
*   4. Abstraction: It provides a clean way to reference well-known system
*      directories (e.g., application data, model cache) without scattering
*      platform-specific logic throughout the codebase.
*
* This module is a foundational component for resource loading, configuration
* management, and data persistence.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TRACKIELLM_INTERNAL_TOOLS_TK_FILE_MANAGER_H
#define TRACKIELLM_INTERNAL_TOOLS_TK_FILE_MANAGER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"

// Forward-declare the main path structure as an opaque type.
// All interactions are performed through functions, enforcing encapsulation
// and hiding the platform-specific internal representation.
typedef struct tk_path_s tk_path_t;

/**
 * @enum tk_base_path_e
 * @brief Enumerates well-known base directories managed by the file manager.
 *
 * This abstraction allows the system to request paths for specific purposes
 * without needing to know the underlying OS conventions (e.g., ~/.config vs
 * %APPDATA%).
 */
typedef enum {
    /**
     * @brief The primary, persistent data directory for the application.
     * On Linux: ~/.config/trackiellm
     * On Windows: %APPDATA%/TrackieLLM
     * On macOS: ~/Library/Application Support/TrackieLLM
     */
    TK_BASE_PATH_APP_CONFIG,

    /**
     * @brief A directory for storing cached data that can be regenerated.
     * This includes downloaded model weights or intermediate data.
     * On Linux: ~/.cache/trackiellm
     * On Windows: %LOCALAPPDATA%/TrackieLLM/Cache
     * On macOS: ~/Library/Caches/TrackieLLM
     */
    TK_BASE_PATH_CACHE,

    /**
     * @brief The directory where the main executable is located.
     * Useful for finding resources bundled with the application.
     */
    TK_BASE_PATH_EXECUTABLE_DIR,

    /**
     * @brief The current working directory from which the application was launched.
     * Use with caution, as it can be unpredictable.
     */
    TK_BASE_PATH_WORKING_DIR

} tk_base_path_e;


#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Path Object Lifecycle Management
//------------------------------------------------------------------------------

/**
 * @brief Creates a path object from a standard null-terminated string.
 *
 * The input string is copied and normalized to the internal platform-specific
 * format. The returned path object must be freed by the caller using
* tk_path_destroy().
 *
 * @param[out] out_path A pointer to a tk_path_t* that will receive the address
 *                      of the newly created path object.
 * @param[in] path_str The UTF-8 encoded string representing the path.
 *
 * @return TK_SUCCESS on successful creation.
 * @return TK_ERROR_INVALID_ARGUMENT if out_path or path_str is NULL.
 * @return TK_ERROR_OUT_OF_MEMORY if memory allocation fails.
 *
 * @par Thread-Safety
 * This function is thread-safe.
 *
 * @par Example
 * @code
 * tk_path_t* my_path = NULL;
 * tk_error_code_t err = tk_path_create_from_string(&my_path, "/home/user/models");
 * if (err == TK_SUCCESS) {
 *     // ... use my_path ...
 *     tk_path_destroy(&my_path);
 * }
 * @endcode
 */
TK_NODISCARD tk_error_code_t tk_path_create_from_string(tk_path_t** out_path, const char* path_str);

/**
 * @brief Creates a path object by resolving a well-known base directory.
 *
 * This is the preferred method for creating paths to application resources,
 * as it is platform-independent.
 *
 * @param[out] out_path A pointer to a tk_path_t* that will receive the address
 *                      of the newly created path object.
 * @param[in] base The well-known base directory to use as the root.
 *
 * @return TK_SUCCESS on successful creation.
 * @return TK_ERROR_INVALID_ARGUMENT if out_path is NULL or base is invalid.
 * @return TK_ERROR_OUT_OF_MEMORY if memory allocation fails.
 * @return TK_ERROR_NOT_FOUND if the platform-specific path for the base
 *         directory cannot be determined.
 *
 * @par Thread-Safety
 * This function is thread-safe.
 */
TK_NODISCARD tk_error_code_t tk_path_create_from_base(tk_path_t** out_path, tk_base_path_e base);

/**
 * @brief Creates a deep copy of an existing path object.
 *
 * The new object is independent of the original and must also be freed using
 * tk_path_destroy().
 *
 * @param[out] out_new_path Pointer to receive the newly created path object.
 * @param[in]  source_path The path object to clone.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL.
 * @return TK_ERROR_OUT_OF_MEMORY if memory allocation fails.
 *
 * @par Thread-Safety
 * This function is thread-safe.
 */
TK_NODISCARD tk_error_code_t tk_path_clone(tk_path_t** out_new_path, const tk_path_t* source_path);

/**
 * @brief Destroys a path object and frees all associated memory.
 *
 * @param[in,out] path A pointer to the tk_path_t* object to be destroyed.
 *                     The pointer is set to NULL after destruction to prevent
 *                     use-after-free errors. If *path is NULL, the function
 *                     does nothing.
 *
 * @par Thread-Safety
 * This function is thread-safe, but the caller must ensure no other thread
 * is using the path object at the same time.
 */
void tk_path_destroy(tk_path_t** path);


//------------------------------------------------------------------------------
// Path Manipulation and Conversion
//------------------------------------------------------------------------------

/**
 * @brief Appends a path segment to an existing path object.
 *
 * This function safely handles path separators. The modification is performed
 * in-place.
 *
 * @param[in,out] path The path object to modify.
 * @param[in] segment The UTF-8 encoded path segment to append (e.g., "models", "yolo.onnx").
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_INVALID_ARGUMENT if path or segment is NULL.
 * @return TK_ERROR_OUT_OF_MEMORY if resizing the internal path buffer fails.
 *
 * @par Thread-Safety
 * This function is NOT thread-safe as it modifies the path object.
 */
TK_NODISCARD tk_error_code_t tk_path_join(tk_path_t* path, const char* segment);

/**
 * @brief Retrieves the null-terminated string representation of the path.
 *
 * The returned string is owned by the path object and must NOT be freed by the
 * caller. It is valid only for the lifetime of the path object or until the
 * path is next modified.
 *
 * @param[in] path The path object.
 *
 * @return A constant, null-terminated, UTF-8 encoded string. Returns an empty
 *         string "" if the path object is NULL.
 *
 * @par Thread-Safety
 * This function is thread-safe.
 */
const char* tk_path_get_string(const tk_path_t* path);

/**
 * @brief Canonicalizes the path, resolving `.` and `..` components.
 *
 * This is a critical security function to prevent directory traversal attacks.
 * It creates a new, cleaned path object. The original path is unmodified.
 *
 * @param[out] out_canonical_path Pointer to receive the new, canonicalized path.
 * @param[in]  source_path The path to canonicalize.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL.
 * @return TK_ERROR_OUT_OF_MEMORY on allocation failure.
 * @return TK_ERROR_INVALID_STATE if the path resolves to a location outside
 *         of an expected root (if sandboxing is implemented).
 *
 * @par Thread-Safety
 * This function is thread-safe.
 */
TK_NODISCARD tk_error_code_t tk_path_canonicalize(tk_path_t** out_canonical_path, const tk_path_t* source_path);


//------------------------------------------------------------------------------
// Filesystem Query Operations
//------------------------------------------------------------------------------

/**
 * @brief Checks if a file or directory exists at the given path.
 *
 * @param[in] path The path to check.
 * @param[out] exists Pointer to a boolean that will be set to true if the
 *                    path exists, and false otherwise.
 *
 * @return TK_SUCCESS on a successful check, regardless of whether it exists.
 * @return TK_ERROR_INVALID_ARGUMENT if path or exists is NULL.
 * @return TK_ERROR_PERMISSION_DENIED if permissions prevent checking.
 *
 * @par Thread-Safety
 * This function is thread-safe.
 */
TK_NODISCARD tk_error_code_t tk_fs_exists(const tk_path_t* path, bool* exists);

/**
 * @brief Checks if the path points to a regular file.
 *
 * @param[in] path The path to check.
 * @param[out] is_file Pointer to a boolean that will be set to true if the
 *                     path points to a file, and false otherwise.
 *
 * @return TK_SUCCESS on a successful check.
 * @return TK_ERROR_INVALID_ARGUMENT if path or is_file is NULL.
 *
 * @par Thread-Safety
 * This function is thread-safe.
 */
TK_NODISCARD tk_error_code_t tk_fs_is_file(const tk_path_t* path, bool* is_file);

/**
 * @brief Checks if the path points to a directory.
 *
 * @param[in] path The path to check.
 * @param[out] is_directory Pointer to a boolean that will be set to true if the
 *                          path points to a directory, and false otherwise.
 *
 * @return TK_SUCCESS on a successful check.
 * @return TK_ERROR_INVALID_ARGUMENT if path or is_directory is NULL.
 *
 * @par Thread-Safety
 * This function is thread-safe.
 */
TK_NODISCARD tk_error_code_t tk_fs_is_directory(const tk_path_t* path, bool* is_directory);


//------------------------------------------------------------------------------
// Filesystem Mutation and I/O Operations
//------------------------------------------------------------------------------

/**
 * @brief Creates a directory and all its parent directories if they do not exist.
 *
 * This function is equivalent to `mkdir -p`.
 *
 * @param[in] path The directory path to create.
 *
 * @return TK_SUCCESS if the directory was created or already existed.
 * @return TK_ERROR_INVALID_ARGUMENT if path is NULL.
 * @return TK_ERROR_PERMISSION_DENIED if creation is not permitted.
 * @return TK_ERROR_IO for other filesystem errors.
 *
 * @par Thread-Safety
 * This function is thread-safe, but race conditions are possible if multiple
 * processes/threads manipulate the same path concurrently.
 */
TK_NODISCARD tk_error_code_t tk_dir_create_recursive(const tk_path_t* path);

/**
 * @brief Reads the entire content of a file into a dynamically allocated buffer.
 *
 * The caller is responsible for freeing the allocated buffer using `free()`.
 *
 * @param[in] path The path to the file to read.
 * @param[out] out_buffer Pointer to a `uint8_t*` that will receive the address
 *                        of the allocated buffer containing the file data.
 * @param[out] out_size Pointer to a `size_t` that will receive the size of the
 *                      buffer in bytes.
 * @param[in] max_size The maximum number of bytes to read. This is a security
 *                     measure to prevent allocating excessive memory. Use 0 for
 *                     no limit.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_INVALID_ARGUMENT if any pointer arguments are NULL.
 * @return TK_ERROR_FILE_NOT_FOUND if the file does not exist.
 * @return TK_ERROR_OUT_OF_MEMORY if buffer allocation fails.
 * @return TK_ERROR_BUFFER_TOO_SMALL if the file size exceeds max_size.
 * @return TK_ERROR_FILE_READ for I/O errors during reading.
 *
 * @par Thread-Safety
 * This function is thread-safe.
 */
TK_NODISCARD tk_error_code_t tk_file_read_all_bytes(const tk_path_t* path, uint8_t** out_buffer, size_t* out_size, size_t max_size);

/**
 * @brief Writes the content of a buffer to a file, overwriting it if it exists.
 *
 * @param[in] path The path to the file to write.
 * @param[in] buffer Pointer to the data buffer to write.
 * @param[in] size The number of bytes to write from the buffer.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_INVALID_ARGUMENT if path or buffer is NULL.
 * @return TK_ERROR_PERMISSION_DENIED if writing is not permitted.
 * @return TK_ERROR_FILE_WRITE for I/O errors during writing.
 *
 * @par Thread-Safety
 * This function is thread-safe, but subject to filesystem race conditions.
 */
TK_NODISCARD tk_error_code_t tk_file_write_buffer(const tk_path_t* path, const uint8_t* buffer, size_t size);


#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_INTERNAL_TOOLS_TK_FILE_MANAGER_H