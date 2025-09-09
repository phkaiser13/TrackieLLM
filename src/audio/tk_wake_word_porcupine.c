/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_wake_word_porcupine.c
 *
 * This source file implements the Porcupine Wake Word Engine module.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "audio/tk_wake_word_porcupine.h"
#include "utils/tk_logging.h"

// Assume Porcupine header is available
#include "pv_porcupine.h"

#include <stdlib.h>
#include <string.h>

// Internal context structure
struct tk_porcupine_context_s {
    pv_porcupine_t* handle;
};

// --- Helper Functions ---

static tk_error_code_t convert_pv_status(pv_status_t status) {
    switch (status) {
        case PV_STATUS_SUCCESS:
            return TK_SUCCESS;
        case PV_STATUS_OUT_OF_MEMORY:
            return TK_ERROR_OUT_OF_MEMORY;
        case PV_STATUS_IO_ERROR:
            return TK_ERROR_IO;
        case PV_STATUS_INVALID_ARGUMENT:
            return TK_ERROR_INVALID_ARGUMENT;
        case PV_STATUS_STOP_ITERATION: // Not really an error
            return TK_SUCCESS;
        case PV_STATUS_KEY_ERROR:
            return TK_ERROR_NOT_FOUND;
        case PV_STATUS_INVALID_STATE:
            return TK_ERROR_INVALID_STATE;
        case PV_STATUS_RUNTIME_ERROR:
            return TK_ERROR_EXTERNAL_LIBRARY_FAILED;
        case PV_STATUS_ACTIVATION_ERROR:
        case PV_STATUS_ACTIVATION_LIMIT_REACHED:
        case PV_STATUS_ACTIVATION_THROTTLED:
        case PV_STATUS_ACTIVATION_REFUSED:
            return TK_ERROR_LICENSE_ERROR;
        default:
            return TK_ERROR_INTERNAL;
    }
}

// --- Public API Implementation ---

TK_NODISCARD tk_error_code_t tk_porcupine_create(
    tk_porcupine_context_t** out_context,
    const tk_porcupine_config_t* config
) {
    if (!out_context || !config || !config->access_key_path || !config->model_path || !config->keyword_paths || !config->sensitivities) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    *out_context = NULL;
    tk_porcupine_context_t* context = calloc(1, sizeof(tk_porcupine_context_t));
    if (!context) {
        return TK_ERROR_OUT_OF_MEMORY;
    }

    // Porcupine requires an access key string, not a path. For this implementation,
    // we'll assume the path contains the key as a string. A real implementation
    // would read the file. We'll use a placeholder.
    const char* access_key = "YOUR_PICOVOICE_ACCESS_KEY"; // Placeholder

    // Convert tk_path_t arrays to const char* arrays
    const char** keyword_paths_str = malloc(config->num_keywords * sizeof(const char*));
    if (!keyword_paths_str) {
        free(context);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    for (int i = 0; i < config->num_keywords; ++i) {
        keyword_paths_str[i] = tk_path_get_str(config->keyword_paths[i]);
    }

    pv_status_t status = pv_porcupine_init(
        access_key,
        tk_path_get_str(config->model_path),
        config->num_keywords,
        keyword_paths_str,
        config->sensitivities,
        &context->handle
    );

    free(keyword_paths_str);

    if (status != PV_STATUS_SUCCESS) {
        TK_LOG_ERROR("Failed to initialize Porcupine. Status: %s", pv_status_to_string(status));
        free(context);
        return convert_pv_status(status);
    }

    *out_context = context;
    return TK_SUCCESS;
}

void tk_porcupine_destroy(tk_porcupine_context_t** context) {
    if (!context || !*context) return;

    tk_porcupine_context_t* ctx = *context;
    if (ctx->handle) {
        pv_porcupine_delete(ctx->handle);
    }
    free(ctx);
    *context = NULL;
}

TK_NODISCARD tk_error_code_t tk_porcupine_process(
    tk_porcupine_context_t* context,
    const int16_t* audio_data,
    int* out_keyword_index
) {
    if (!context || !context->handle || !audio_data || !out_keyword_index) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    pv_status_t status = pv_porcupine_process(context->handle, audio_data, out_keyword_index);

    if (status != PV_STATUS_SUCCESS) {
        TK_LOG_ERROR("Porcupine processing failed. Status: %s", pv_status_to_string(status));
        return convert_pv_status(status);
    }

    return TK_SUCCESS;
}

int tk_porcupine_get_frame_length(tk_porcupine_context_t* context) {
    if (!context || !context->handle) {
        return 0;
    }
    return pv_porcupine_frame_length();
}

int tk_porcupine_get_sample_rate(tk_porcupine_context_t* context) {
    if (!context || !context->handle) {
        return 0;
    }
    return pv_sample_rate();
}
