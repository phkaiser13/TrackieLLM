/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_config_parser.c
*
* This source file implements the key-value configuration file parser for the
* TrackieLLM project. It handles the lifecycle of a configuration object,
* including creation, parsing from a file, value retrieval, and destruction.
* The internal storage is a dynamically growing array of key-value pairs.
*
* SPDX-License-Identifier:
*/

#include "tk_config_parser.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INITIAL_CONFIG_CAPACITY 16
#define MAX_LINE_LENGTH 1024

//------------------------------------------------------------------------------
// Internal Data Structures
//------------------------------------------------------------------------------

/**
 * @struct tk_key_value_pair_t
 * @brief Represents a single key-value entry in the configuration.
 */
typedef struct {
    char* key;
    char* value;
} tk_key_value_pair_t;

/**
 * @struct tk_config_s
 * @brief The concrete implementation of the opaque tk_config_t type.
 */
struct tk_config_s {
    tk_key_value_pair_t** pairs;
    size_t count;
    size_t capacity;
};

//------------------------------------------------------------------------------
// Internal Helper Functions
//------------------------------------------------------------------------------

/**
 * @brief Trims leading and trailing whitespace from a string in-place.
 * @param str The string to trim.
 * @return The modified string.
 */
static char* trim_whitespace(char* str) {
    if (!str) return NULL;

    char* end;

    // Trim leading space
    while (isspace((unsigned char)*str)) {
        str++;
    }

    if (*str == 0) { // All spaces?
        return str;
    }

    // Trim trailing space
    end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) {
        end--;
    }

    // Write new null terminator
    *(end + 1) = 0;

    return str;
}

/**
 * @brief Adds a new key-value pair to the configuration object.
 *        Handles resizing the internal array if necessary.
 * @param config The configuration object.
 * @param key The key string.
 * @param value The value string.
 * @return TK_SUCCESS on success, TK_ERROR_OUT_OF_MEMORY on failure.
 */
static TK_NODISCARD tk_error_code_t add_pair(tk_config_t* config, const char* key, const char* value) {
    // Resize if necessary
    if (config->count >= config->capacity) {
        size_t new_capacity = (config->capacity == 0) ? INITIAL_CONFIG_CAPACITY : config->capacity * 2;
        tk_key_value_pair_t** new_pairs = realloc(config->pairs, new_capacity * sizeof(tk_key_value_pair_t*));
        if (!new_pairs) {
            return TK_ERROR_OUT_OF_MEMORY;
        }
        config->pairs = new_pairs;
        config->capacity = new_capacity;
    }

    // Allocate memory for the new pair and its contents
    tk_key_value_pair_t* new_pair = malloc(sizeof(tk_key_value_pair_t));
    if (!new_pair) return TK_ERROR_OUT_OF_MEMORY;

    new_pair->key = strdup(key);
    if (!new_pair->key) {
        free(new_pair);
        return TK_ERROR_OUT_OF_MEMORY;
    }

    new_pair->value = strdup(value);
    if (!new_pair->value) {
        free(new_pair->key);
        free(new_pair);
        return TK_ERROR_OUT_OF_MEMORY;
    }

    config->pairs[config->count++] = new_pair;
    return TK_SUCCESS;
}

//------------------------------------------------------------------------------
// Public API Implementation
//------------------------------------------------------------------------------

TK_NODISCARD tk_error_code_t tk_config_create(tk_config_t** out_config) {
    if (!out_config) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    *out_config = calloc(1, sizeof(tk_config_t));
    if (!*out_config) {
        return TK_ERROR_OUT_OF_MEMORY;
    }

    return TK_SUCCESS;
}

void tk_config_destroy(tk_config_t** config) {
    if (!config || !*config) {
        return;
    }

    tk_config_t* cfg = *config;
    for (size_t i = 0; i < cfg->count; ++i) {
        free(cfg->pairs[i]->key);
        free(cfg->pairs[i]->value);
        free(cfg->pairs[i]);
    }
    free(cfg->pairs);
    free(cfg);

    *config = NULL;
}

TK_NODISCARD tk_error_code_t tk_config_load_from_file(tk_config_t* config, const char* filepath) {
    if (!config || !filepath) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    FILE* file = fopen(filepath, "r");
    if (!file) {
        return TK_ERROR_FILE_NOT_FOUND;
    }

    char line[MAX_LINE_LENGTH];
    tk_error_code_t err = TK_SUCCESS;

    while (fgets(line, sizeof(line), file)) {
        char* trimmed_line = trim_whitespace(line);

        // Skip empty or comment lines
        if (trimmed_line[0] == '\0' || trimmed_line[0] == '#' || trimmed_line[0] == ';') {
            continue;
        }

        char* separator = strchr(trimmed_line, '=');
        if (!separator) {
            // Line is malformed, skip it. Could also be a warning.
            continue;
        }

        // Split key and value
        *separator = '\0';
        char* key = trim_whitespace(trimmed_line);
        char* value = trim_whitespace(separator + 1);

        if (strlen(key) == 0) {
            // Key cannot be empty.
            continue;
        }

        err = add_pair(config, key, value);
        if (err != TK_SUCCESS) {
            break;
        }
    }

    fclose(file);
    return err;
}

const char* tk_config_get_string(const tk_config_t* config, const char* key, const char* default_value) {
    if (!config || !key) {
        return default_value;
    }

    for (size_t i = 0; i < config->count; ++i) {
        if (strcmp(config->pairs[i]->key, key) == 0) {
            return config->pairs[i]->value;
        }
    }

    return default_value;
}

int64_t tk_config_get_int(const tk_config_t* config, const char* key, int64_t default_value) {
    const char* value_str = tk_config_get_string(config, key, NULL);
    if (!value_str) {
        return default_value;
    }

    char* endptr;
    long long result = strtoll(value_str, &endptr, 10);

    // Check for parsing errors: no digits found, or extra characters after number
    if (endptr == value_str || *trim_whitespace(endptr) != '\0') {
        return default_value;
    }

    return (int64_t)result;
}

double tk_config_get_double(const tk_config_t* config, const char* key, double default_value) {
    const char* value_str = tk_config_get_string(config, key, NULL);
    if (!value_str) {
        return default_value;
    }

    char* endptr;
    double result = strtod(value_str, &endptr);

    if (endptr == value_str || *trim_whitespace(endptr) != '\0') {
        return default_value;
    }

    return result;
}

bool tk_config_get_bool(const tk_config_t* config, const char* key, bool default_value) {
    const char* value_str = tk_config_get_string(config, key, NULL);
    if (!value_str) {
        return default_value;
    }

    if (strcasecmp(value_str, "true") == 0 ||
        strcasecmp(value_str, "yes") == 0 ||
        strcasecmp(value_str, "on") == 0 ||
        strcmp(value_str, "1") == 0) {
        return true;
    }

    return false;
}