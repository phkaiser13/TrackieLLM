/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_file_manager.c
*
* This source file provides the concrete implementation for the TrackieLLM
* Filesystem Abstraction Layer (FAL). It is designed with portability and
* security as primary concerns.
*
* The implementation uses preprocessor directives to select the correct underlying
* OS APIs for POSIX-compliant systems (Linux, macOS) and Windows. This includes
* handling different path separators, system calls for filesystem queries (stat),
* and APIs for resolving well-known directory paths.
*
* Memory management is handled meticulously. The core `tk_path_s` structure uses
* a capacity field to allow for efficient path joining operations, minimizing
* the number of reallocations. All external inputs are validated, and all system
* call return values are checked to ensure robust error reporting.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#include "tk_file_manager.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

// Platform-specific includes
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shlobj.h>
#include <direct.h> // For _mkdir
#define TK_STAT_STRUCT __stat64
#define TK_STAT_FUNC _stat64
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <limits.h> // For PATH_MAX
#define TK_STAT_STRUCT stat
#define TK_STAT_FUNC stat
#endif

// Define platform-specific constants
#ifdef _WIN32
static const char PLATFORM_SEPARATOR = '\\';
#else
static const char PLATFORM_SEPARATOR = '/';
#endif

//------------------------------------------------------------------------------
// Internal Data Structures
//------------------------------------------------------------------------------

struct tk_path_s {
    char*  buffer;
    size_t length;
    size_t capacity;
};

//------------------------------------------------------------------------------
// Internal Helper Functions
//------------------------------------------------------------------------------

static void normalize_separators(char* path_str) {
    if (!path_str) return;
    char wrong_separator = (PLATFORM_SEPARATOR == '/') ? '\\' : '/';
    for (char* p = path_str; *p; ++p) {
        if (*p == wrong_separator) {
            *p = PLATFORM_SEPARATOR;
        }
    }
}

static TK_NODISCARD tk_error_code_t ensure_capacity(tk_path_t* path, size_t required_capacity) {
    if (path->capacity >= required_capacity) {
        return TK_SUCCESS;
    }
    size_t new_capacity = path->capacity > 0 ? path->capacity : 256;
    while (new_capacity < required_capacity) {
        new_capacity *= 2;
    }
    char* new_buffer = realloc(path->buffer, new_capacity);
    if (!new_buffer) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    path->buffer = new_buffer;
    path->capacity = new_capacity;
    return TK_SUCCESS;
}

static TK_NODISCARD tk_error_code_t resolve_base_path(tk_base_path_e base, char** out_path_str) {
    char path_buffer[4096] = {0};
#ifdef _WIN32
    int csidl = -1;
    switch (base) {
        case TK_BASE_PATH_APP_CONFIG: csidl = CSIDL_APPDATA; break;
        case TK_BASE_PATH_CACHE: csidl = CSIDL_LOCAL_APPDATA; break;
        case TK_BASE_PATH_EXECUTABLE_DIR:
            if (GetModuleFileNameA(NULL, path_buffer, sizeof(path_buffer)) == 0) return TK_ERROR_UNKNOWN;
            char* last_sep = strrchr(path_buffer, '\\');
            if (last_sep) *last_sep = '\0';
            break;
        case TK_BASE_PATH_WORKING_DIR:
            if (_getcwd(path_buffer, sizeof(path_buffer)) == NULL) return TK_ERROR_UNKNOWN;
            break;
        default: return TK_ERROR_INVALID_ARGUMENT;
    }
    if (csidl != -1 && SHGetFolderPathA(NULL, csidl, NULL, 0, path_buffer) != S_OK) return TK_ERROR_NOT_FOUND;
#else
    const char* home_dir = getenv("HOME");
    if (!home_dir) return TK_ERROR_NOT_FOUND;
    switch (base) {
        case TK_BASE_PATH_APP_CONFIG:
            snprintf(path_buffer, sizeof(path_buffer), "%s/.config", home_dir);
            break;
        case TK_BASE_PATH_CACHE:
            snprintf(path_buffer, sizeof(path_buffer), "%s/.cache", home_dir);
            break;
        case TK_BASE_PATH_EXECUTABLE_DIR:
            ssize_t len = readlink("/proc/self/exe", path_buffer, sizeof(path_buffer) - 1);
            if (len != -1) {
                path_buffer[len] = '\0';
                char* last_sep = strrchr(path_buffer, '/');
                if (last_sep) *last_sep = '\0';
            } else return TK_ERROR_UNKNOWN;
            break;
        case TK_BASE_PATH_WORKING_DIR:
            if (getcwd(path_buffer, sizeof(path_buffer)) == NULL) return TK_ERROR_UNKNOWN;
            break;
        default: return TK_ERROR_INVALID_ARGUMENT;
    }
#endif
    *out_path_str = strdup(path_buffer);
    return (*out_path_str) ? TK_SUCCESS : TK_ERROR_OUT_OF_MEMORY;
}

//------------------------------------------------------------------------------
// Public API Implementation: Lifecycle
//------------------------------------------------------------------------------

TK_NODISCARD tk_error_code_t tk_path_create_from_string(tk_path_t** out_path, const char* path_str) {
    if (!out_path || !path_str) return TK_ERROR_INVALID_ARGUMENT;
    tk_path_t* path = calloc(1, sizeof(tk_path_t));
    if (!path) return TK_ERROR_OUT_OF_MEMORY;
    path->length = strlen(path_str);
    path->capacity = path->length + 1;
    path->buffer = malloc(path->capacity);
    if (!path->buffer) {
        free(path);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    memcpy(path->buffer, path_str, path->capacity);
    normalize_separators(path->buffer);
    *out_path = path;
    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t tk_path_create_from_base(tk_path_t** out_path, tk_base_path_e base) {
    if (!out_path) return TK_ERROR_INVALID_ARGUMENT;
    char* base_str = NULL;
    tk_error_code_t err = resolve_base_path(base, &base_str);
    if (err != TK_SUCCESS) return err;
    
    const char* app_dir_name = "trackiellm";
    size_t final_len = strlen(base_str) + 1 + strlen(app_dir_name);
    char* final_path_str = malloc(final_len + 1);
    if (!final_path_str) {
        free(base_str);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    if (base == TK_BASE_PATH_APP_CONFIG || base == TK_BASE_PATH_CACHE) {
        snprintf(final_path_str, final_len + 1, "%s%c%s", base_str, PLATFORM_SEPARATOR, app_dir_name);
    } else {
        strncpy(final_path_str, base_str, final_len + 1);
    }
    free(base_str);
    err = tk_path_create_from_string(out_path, final_path_str);
    free(final_path_str);
    return err;
}

TK_NODISCARD tk_error_code_t tk_path_clone(tk_path_t** out_new_path, const tk_path_t* source_path) {
    if (!out_new_path || !source_path) return TK_ERROR_INVALID_ARGUMENT;
    return tk_path_create_from_string(out_new_path, source_path->buffer);
}

void tk_path_destroy(tk_path_t** path) {
    if (!path || !*path) return;
    free((*path)->buffer);
    free(*path);
    *path = NULL;
}

//------------------------------------------------------------------------------
// Public API Implementation: Manipulation & Conversion
//------------------------------------------------------------------------------

TK_NODISCARD tk_error_code_t tk_path_join(tk_path_t* path, const char* segment) {
    if (!path || !segment) return TK_ERROR_INVALID_ARGUMENT;
    size_t segment_len = strlen(segment);
    if (segment_len == 0) return TK_SUCCESS;
    size_t required_len = path->length + 1 + segment_len + 1;
    tk_error_code_t err = ensure_capacity(path, required_len);
    if (err != TK_SUCCESS) return err;
    if (path->length > 0 && path->buffer[path->length - 1] != PLATFORM_SEPARATOR) {
        path->buffer[path->length++] = PLATFORM_SEPARATOR;
    }
    memcpy(path->buffer + path->length, segment, segment_len + 1);
    path->length += segment_len;
    return TK_SUCCESS;
}

const char* tk_path_get_string(const tk_path_t* path) {
    return path ? path->buffer : "";
}

TK_NODISCARD tk_error_code_t tk_path_canonicalize(tk_path_t** out_canonical_path, const tk_path_t* source_path) {
    if (!out_canonical_path || !source_path) return TK_ERROR_INVALID_ARGUMENT;
    
    char* resolved_path = malloc(source_path->capacity);
    if (!resolved_path) return TK_ERROR_OUT_OF_MEMORY;

    char* output_ptr = resolved_path;
    const char* input_ptr = source_path->buffer;
    const char* input_end = source_path->buffer + source_path->length;

    // Handle root directory for absolute paths
    if (*input_ptr == PLATFORM_SEPARATOR) {
        *output_ptr++ = *input_ptr++;
    }

    while (input_ptr < input_end) {
        const char* segment_start = input_ptr;
        const char* segment_end = strchr(segment_start, PLATFORM_SEPARATOR);
        if (!segment_end) segment_end = input_end;

        size_t segment_len = segment_end - segment_start;

        if (segment_len == 0 || (segment_len == 1 && *segment_start == '.')) {
            // Ignore empty segments ('//') or '.' segments
        } else if (segment_len == 2 && strncmp(segment_start, "..", 2) == 0) {
            // Handle '..' segment
            if (output_ptr > resolved_path + 1) {
                output_ptr--; // Move back past the last separator
                while (output_ptr > resolved_path && *(output_ptr - 1) != PLATFORM_SEPARATOR) {
                    output_ptr--;
                }
            }
        } else {
            // Copy segment
            if (output_ptr > resolved_path && *(output_ptr - 1) != PLATFORM_SEPARATOR) {
                *output_ptr++ = PLATFORM_SEPARATOR;
            }
            memcpy(output_ptr, segment_start, segment_len);
            output_ptr += segment_len;
        }
        input_ptr = segment_end;
        if (*input_ptr == PLATFORM_SEPARATOR) input_ptr++;
    }
    *output_ptr = '\0';
    if (resolved_path == output_ptr && source_path->length > 0 && source_path->buffer[0] == PLATFORM_SEPARATOR) {
        // Handle root case "/"
        strcpy(resolved_path, (char[2]){PLATFORM_SEPARATOR, '\0'});
    }


    tk_error_code_t err = tk_path_create_from_string(out_canonical_path, resolved_path);
    free(resolved_path);
    return err;
}

//------------------------------------------------------------------------------
// Public API Implementation: Filesystem Query Operations
//------------------------------------------------------------------------------

TK_NODISCARD tk_error_code_t tk_fs_exists(const tk_path_t* path, bool* exists) {
    if (!path || !exists) return TK_ERROR_INVALID_ARGUMENT;
    struct TK_STAT_STRUCT stat_buf;
    if (TK_STAT_FUNC(path->buffer, &stat_buf) == 0) {
        *exists = true;
    } else {
        if (errno == ENOENT) {
            *exists = false;
        } else {
            return TK_ERROR_PERMISSION_DENIED; // Or other I/O error
        }
    }
    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t tk_fs_is_file(const tk_path_t* path, bool* is_file) {
    if (!path || !is_file) return TK_ERROR_INVALID_ARGUMENT;
    struct TK_STAT_STRUCT stat_buf;
    if (TK_STAT_FUNC(path->buffer, &stat_buf) != 0) {
        *is_file = false;
        return (errno == ENOENT) ? TK_SUCCESS : TK_ERROR_IO;
    }
    #ifdef _WIN32
    *is_file = (stat_buf.st_mode & S_IFREG) != 0;
    #else
    *is_file = S_ISREG(stat_buf.st_mode);
    #endif
    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t tk_fs_is_directory(const tk_path_t* path, bool* is_directory) {
    if (!path || !is_directory) return TK_ERROR_INVALID_ARGUMENT;
    struct TK_STAT_STRUCT stat_buf;
    if (TK_STAT_FUNC(path->buffer, &stat_buf) != 0) {
        *is_directory = false;
        return (errno == ENOENT) ? TK_SUCCESS : TK_ERROR_IO;
    }
    #ifdef _WIN32
    *is_directory = (stat_buf.st_mode & S_IFDIR) != 0;
    #else
    *is_directory = S_ISDIR(stat_buf.st_mode);
    #endif
    return TK_SUCCESS;
}

//------------------------------------------------------------------------------
// Public API Implementation: Filesystem Mutation and I/O
//------------------------------------------------------------------------------

TK_NODISCARD tk_error_code_t tk_dir_create_recursive(const tk_path_t* path) {
    if (!path) return TK_ERROR_INVALID_ARGUMENT;
    char* path_copy = strdup(path->buffer);
    if (!path_copy) return TK_ERROR_OUT_OF_MEMORY;

    tk_error_code_t result = TK_SUCCESS;
    for (char* p = path_copy + 1; *p; p++) {
        if (*p == PLATFORM_SEPARATOR) {
            *p = '\0';
#ifdef _WIN32
            if (_mkdir(path_copy) != 0 && errno != EEXIST) {
                result = TK_ERROR_IO;
                break;
            }
#else
            if (mkdir(path_copy, 0755) != 0 && errno != EEXIST) {
                result = TK_ERROR_IO;
                break;
            }
#endif
            *p = PLATFORM_SEPARATOR;
        }
    }
    if (result == TK_SUCCESS) {
#ifdef _WIN32
        if (_mkdir(path_copy) != 0 && errno != EEXIST) result = TK_ERROR_IO;
#else
        if (mkdir(path_copy, 0755) != 0 && errno != EEXIST) result = TK_ERROR_IO;
#endif
    }
    free(path_copy);
    return result;
}

TK_NODISCARD tk_error_code_t tk_file_read_all_bytes(const tk_path_t* path, uint8_t** out_buffer, size_t* out_size, size_t max_size) {
    if (!path || !out_buffer || !out_size) return TK_ERROR_INVALID_ARGUMENT;
    
    FILE* file = fopen(path->buffer, "rb");
    if (!file) return TK_ERROR_FILE_NOT_FOUND;

    fseek(file, 0, SEEK_END);
    long file_size_long = ftell(file);
    if (file_size_long < 0) {
        fclose(file);
        return TK_ERROR_FILE_READ;
    }
    size_t file_size = (size_t)file_size_long;
    rewind(file);

    if (max_size > 0 && file_size > max_size) {
        fclose(file);
        return TK_ERROR_BUFFER_TOO_SMALL;
    }

    uint8_t* buffer = malloc(file_size);
    if (!buffer) {
        fclose(file);
        return TK_ERROR_OUT_OF_MEMORY;
    }

    if (fread(buffer, 1, file_size, file) != file_size) {
        free(buffer);
        fclose(file);
        return TK_ERROR_FILE_READ;
    }

    fclose(file);
    *out_buffer = buffer;
    *out_size = file_size;
    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t tk_file_write_buffer(const tk_path_t* path, const uint8_t* buffer, size_t size) {
    if (!path || !buffer) return TK_ERROR_INVALID_ARGUMENT;

    FILE* file = fopen(path->buffer, "wb");
    if (!file) return TK_ERROR_PERMISSION_DENIED;

    if (fwrite(buffer, 1, size, file) != size) {
        fclose(file);
        return TK_ERROR_FILE_WRITE;
    }

    fclose(file);
    return TK_SUCCESS;
}