/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_config_parser.h
*
* This header file defines the interface for a simple key-value configuration
* file parser. This utility is essential for loading runtime settings, such as
* model paths, device IDs, and performance thresholds, without hardcoding them
* into the application. The design uses an opaque handle (tk_config_t) to
* encapsulate the internal data structures.
*
* SPDX-License-Identifier:
*/

#ifndef TRACKIELLM_INTERNAL_TOOLS_TK_CONFIG_PARSER_H
#define TRACKIELLM_INTERNAL_TOOLS_TK_CONFIG_PARSER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"

// Forward-declare the main configuration structure as an opaque type.
// The user of this API only interacts with a pointer to it, promoting encapsulation.
typedef struct tk_config_s tk_config_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates and initializes a new configuration object.
 *
 * This function allocates memory for a new, empty configuration context.
 * This context must be freed later by calling tk_config_destroy.
 *
 * @param[out] out_config A pointer to a tk_config_t* that will receive the
 *                        address of the newly created configuration object.
 *
 * @return TK_SUCCESS on successful creation.
 * @return TK_ERROR_INVALID_ARGUMENT if out_config is NULL.
 * @return TK_ERROR_OUT_OF_MEMORY if memory allocation fails.
 *
 * @par Thread-Safety
 * This function is not thread-safe. It should be called during initialization.
 */
TK_NODISCARD tk_error_code_t tk_config_create(tk_config_t** out_config);

/**
 * @brief Destroys a configuration object and frees all associated memory.
 *
 * @param[in] config A pointer to the tk_config_t object to be destroyed.
 *                   If *config is NULL, the function does nothing.
 *                   The pointer is set to NULL after destruction to prevent
 *                   use-after-free errors.
 *
 * @par Thread-Safety
 * This function is not thread-safe.
 */
void tk_config_destroy(tk_config_t** config);

/**
 * @brief Loads and parses configuration settings from a specified file.
 *
 * Reads a file line by line, parsing `key = value` pairs. Lines starting with
 * '#' or ';' are treated as comments and ignored. Whitespace around keys and
 * values is trimmed.
 *
 * @param[in] config The configuration object to populate.
 * @param[in] filepath The path to the configuration file.
 *
 * @return TK_SUCCESS on successful loading and parsing.
 * @return TK_ERROR_INVALID_ARGUMENT if config or filepath is NULL.
 * @return TK_ERROR_FILE_NOT_FOUND if the file does not exist.
 * @return TK_ERROR_FILE_READ if there was an error reading the file.
 * @return TK_ERROR_OUT_OF_MEMORY if memory allocation for key-value pairs fails.
 *
 * @par Thread-Safety
 * This function is not thread-safe.
 */
TK_NODISCARD tk_error_code_t tk_config_load_from_file(tk_config_t* config, const char* filepath);

/**
 * @brief Retrieves a string value associated with a given key.
 *
 * @param[in] config The configuration object.
 * @param[in] key The key for the desired value.
 * @param[in] default_value A default value to return if the key is not found.
 *
 * @return The string value from the configuration file, or default_value if
 *         the key is not found. The returned string is owned by the config
 *         object and must not be freed by the caller. It is valid until
 *         tk_config_destroy is called.
 *
 * @par Thread-Safety
 * This function is thread-safe, assuming no other thread is modifying the
 * config object via tk_config_load_from_file.
 */
const char* tk_config_get_string(const tk_config_t* config, const char* key, const char* default_value);

/**
 * @brief Retrieves an integer value associated with a given key.
 *
 * The function attempts to parse the string value as a 64-bit signed integer.
 *
 * @param[in] config The configuration object.
 * @param[in] key The key for the desired value.
 * @param[in] default_value A default value to return if the key is not found
 *                          or if the value cannot be parsed as an integer.
 *
 * @return The parsed integer value, or default_value on failure.
 *
 * @par Thread-Safety
 * This function is thread-safe (read-only).
 */
int64_t tk_config_get_int(const tk_config_t* config, const char* key, int64_t default_value);

/**
 * @brief Retrieves a double-precision floating-point value for a given key.
 *
 * @param[in] config The configuration object.
 * @param[in] key The key for the desired value.
 * @param[in] default_value A default value to return if the key is not found
 *                          or if the value cannot be parsed as a double.
 *
 * @return The parsed double value, or default_value on failure.
 *
 * @par Thread-Safety
 * This function is thread-safe (read-only).
 */
double tk_config_get_double(const tk_config_t* config, const char* key, double default_value);

/**
 * @brief Retrieves a boolean value for a given key.
 *
 * The function interprets "true", "yes", "on", and "1" (case-insensitive) as
 * true. All other values are interpreted as false.
 *
 * @param[in] config The configuration object.
 * @param[in] key The key for the desired value.
 * @param[in] default_value A default value to return if the key is not found.
 *
 * @return The parsed boolean value, or default_value if the key is not found.
 *
 * @par Thread-Safety
 * This function is thread-safe (read-only).
 */
bool tk_config_get_bool(const tk_config_t* config, const char* key, bool default_value);


#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_INTERNAL_TOOLS_TK_CONFIG_PARSER_H