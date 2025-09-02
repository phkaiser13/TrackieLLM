/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_tts_piper.c
 *
 * This source file implements the Piper TTS (Text-to-Speech) module.
 * It provides a high-level interface to the Piper library, abstracting away
 * the complexities of model loading, inference, and audio generation.
 *
 * The implementation is designed for real-time operation in embedded environments,
 * with support for streaming audio output and configurable voice parameters.
 *
 * Dependencies:
 *   - piper (https://github.com/rhasspy/piper)
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "audio/tk_tts_piper.h"
#include "utils/tk_logging.h"
#include "utils/tk_error_handling.h"

// Include Piper headers
#include "piper.h"
#include "piper_config.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

// Maximum text length for a single synthesis request
#define MAX_TEXT_LENGTH 1024

// Internal structure for Piper context
struct tk_tts_piper_context_s {
    struct piper_context* piper_ctx;        // Piper context
    tk_tts_piper_config_t config;           // Configuration copy
    struct piper_model_info model_info;     // Model information
    struct piper_voice_params voice_params; // Voice parameters
};

//------------------------------------------------------------------------------
// Private helper functions
//------------------------------------------------------------------------------

/**
 * @brief Converts Piper error codes to Trackie error codes
 */
static tk_error_code_t convert_piper_error(enum piper_error_code piper_error) {
    switch (piper_error) {
        case PIPER_ERROR_SUCCESS:
            return TK_SUCCESS;
        case PIPER_ERROR_INVALID_ARGUMENT:
            return TK_ERROR_INVALID_ARGUMENT;
        case PIPER_ERROR_MODEL_LOAD_FAILED:
            return TK_ERROR_MODEL_LOAD_FAILED;
        case PIPER_ERROR_MODEL_INFERENCE_FAILED:
            return TK_ERROR_MODEL_INFERENCE_FAILED;
        case PIPER_ERROR_OUT_OF_MEMORY:
            return TK_ERROR_OUT_OF_MEMORY;
        default:
            return TK_ERROR_INTERNAL;
    }
}

/**
 * @brief Allocates and copies a string
 */
static char* duplicate_string(const char* src) {
    if (!src) return NULL;
    
    size_t len = strlen(src);
    char* dup = malloc(len + 1);
    if (!dup) return NULL;
    
    memcpy(dup, src, len + 1);
    return dup;
}

/**
 * @brief Frees memory allocated for a string
 */
static void free_string(char* str) {
    if (str) {
        free(str);
    }
}

/**
 * @brief Initializes Piper parameters with default values
 */
static void init_piper_params(struct piper_params* params, const tk_tts_piper_config_t* config) {
    // Initialize with defaults
    piper_params_init(params);
    
    // Set configuration parameters
    params->n_threads = config->n_threads > 0 ? config->n_threads : 4;
    params->length_scale = config->voice_params.length_scale > 0.0f ? 
                          config->voice_params.length_scale : 1.0f;
    params->noise_scale = config->voice_params.noise_scale > 0.0f ? 
                         config->voice_params.noise_scale : 0.667f;
    params->noise_w = config->voice_params.noise_w > 0.0f ? 
                     config->voice_params.noise_w : 0.8f;
    params->speaker_id = config->voice_params.speaker_id;
}

/**
 * @brief Internal audio callback that accumulates audio data
 */
struct audio_accumulator {
    int16_t* buffer;
    size_t   capacity;
    size_t   size;
    tk_error_code_t error;
};

static void accumulate_audio_callback(
    const int16_t* audio_data,
    size_t frame_count,
    void* user_data
) {
    struct audio_accumulator* accumulator = (struct audio_accumulator*)user_data;
    
    // Check if we have enough space
    if (accumulator->size + frame_count > accumulator->capacity) {
        // Reallocate buffer with more space
        size_t new_capacity = accumulator->capacity * 2;
        if (new_capacity < accumulator->size + frame_count) {
            new_capacity = accumulator->size + frame_count;
        }
        
        int16_t* new_buffer = realloc(accumulator->buffer, new_capacity * sizeof(int16_t));
        if (!new_buffer) {
            accumulator->error = TK_ERROR_OUT_OF_MEMORY;
            return;
        }
        
        accumulator->buffer = new_buffer;
        accumulator->capacity = new_capacity;
    }
    
    // Copy audio data
    memcpy(
        accumulator->buffer + accumulator->size,
        audio_data,
        frame_count * sizeof(int16_t)
    );
    
    accumulator->size += frame_count;
}

/**
 * @brief Validates text for synthesis
 */
static bool validate_text(const char* text) {
    if (!text || strlen(text) == 0) {
        return false;
    }
    
    // Check for excessive length
    if (strlen(text) > MAX_TEXT_LENGTH) {
        return false;
    }
    
    // Check for non-printable characters
    for (size_t i = 0; i < strlen(text); i++) {
        if (text[i] < 32 && text[i] != '\n' && text[i] != '\t') {
            return false;
        }
    }
    
    return true;
}

/**
 * @brief Sanitizes text for synthesis
 */
static char* sanitize_text(const char* text) {
    if (!text) return NULL;
    
    size_t len = strlen(text);
    char* sanitized = malloc(len + 1);
    if (!sanitized) return NULL;
    
    size_t j = 0;
    for (size_t i = 0; i < len; i++) {
        // Replace non-printable characters with space
        if (text[i] >= 32 || text[i] == '\n' || text[i] == '\t') {
            sanitized[j++] = text[i];
        } else {
            sanitized[j++] = ' ';
        }
    }
    
    sanitized[j] = '\0';
    return sanitized;
}

//------------------------------------------------------------------------------
// Public API Implementation
//------------------------------------------------------------------------------

tk_error_code_t tk_tts_piper_create(
    tk_tts_piper_context_t** out_context,
    const tk_tts_piper_config_t* config
) {
    if (!out_context || !config || !config->model_path || !config->config_path) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_context = NULL;
    
    // Allocate context structure
    tk_tts_piper_context_t* context = calloc(1, sizeof(tk_tts_piper_context_t));
    if (!context) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Copy configuration paths
    context->config.model_path = config->model_path;
    context->config.config_path = config->config_path;
    context->config.language = duplicate_string(config->language);
    if (config->language && !context->config.language) {
        free(context);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    context->config.sample_rate = config->sample_rate;
    context->config.user_data = config->user_data;
    context->config.voice_params = config->voice_params;
    context->config.n_threads = config->n_threads;
    context->config.audio_buffer_size = config->audio_buffer_size;
    
    // Initialize Piper context
    context->piper_ctx = piper_context_create();
    if (!context->piper_ctx) {
        free_string(context->config.language);
        free(context);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Load Piper model
    enum piper_error_code result = piper_load_model(
        context->piper_ctx,
        config->model_path->path_str,
        config->config_path->path_str
    );
    
    if (result != PIPER_ERROR_SUCCESS) {
        TK_LOG_ERROR("Failed to load Piper model from: %s", config->model_path->path_str);
        piper_context_destroy(context->piper_ctx);
        free_string(context->config.language);
        free(context);
        return convert_piper_error(result);
    }
    
    // Get model information
    result = piper_get_model_info(context->piper_ctx, &context->model_info);
    if (result != PIPER_ERROR_SUCCESS) {
        TK_LOG_ERROR("Failed to get Piper model info");
        piper_context_destroy(context->piper_ctx);
        free_string(context->config.language);
        free(context);
        return convert_piper_error(result);
    }
    
    // Set voice parameters
    context->voice_params.speaker_id = config->voice_params.speaker_id;
    context->voice_params.length_scale = config->voice_params.length_scale;
    context->voice_params.noise_scale = config->voice_params.noise_scale;
    context->voice_params.noise_w = config->voice_params.noise_w;
    
    *out_context = context;
    return TK_SUCCESS;
}

void tk_tts_piper_destroy(tk_tts_piper_context_t** context) {
    if (!context || !*context) return;
    
    tk_tts_piper_context_t* ctx = *context;
    
    // Free Piper context
    if (ctx->piper_ctx) {
        piper_context_destroy(ctx->piper_ctx);
    }
    
    // Free configuration strings
    free_string(ctx->config.language);
    
    // Free context itself
    free(ctx);
    *context = NULL;
}

tk_error_code_t tk_tts_piper_synthesize(
    tk_tts_piper_context_t* context,
    const char* text,
    tk_tts_piper_audio_callback_t callback,
    void* user_data
) {
    if (!context || !text || !callback) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Validate and sanitize text
    if (!validate_text(text)) {
        TK_LOG_WARN("Invalid text for synthesis: %s", text);
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    char* sanitized_text = sanitize_text(text);
    if (!sanitized_text) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Initialize Piper parameters
    struct piper_params params;
    init_piper_params(&params, &context->config);
    
    // Set up audio callback adapter
    struct piper_audio_callback audio_cb = {
        .callback = (void (*)(const int16_t*, size_t, void*))callback,
        .user_data = user_data
    };
    
    // Run Piper synthesis
    enum piper_error_code result = piper_synthesize(
        context->piper_ctx,
        sanitized_text,
        &params,
        &audio_cb
    );
    
    free(sanitized_text);
    
    if (result != PIPER_ERROR_SUCCESS) {
        TK_LOG_ERROR("Piper synthesis failed with code: %d", result);
        return convert_piper_error(result);
    }
    
    return TK_SUCCESS;
}

tk_error_code_t tk_tts_piper_synthesize_to_buffer(
    tk_tts_piper_context_t* context,
    const char* text,
    int16_t** out_audio_data,
    size_t* out_frame_count
) {
    if (!context || !text || !out_audio_data || !out_frame_count) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_audio_data = NULL;
    *out_frame_count = 0;
    
    // Validate and sanitize text
    if (!validate_text(text)) {
        TK_LOG_WARN("Invalid text for synthesis: %s", text);
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    char* sanitized_text = sanitize_text(text);
    if (!sanitized_text) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Initialize audio accumulator
    struct audio_accumulator accumulator = {
        .buffer = malloc(44100 * 10 * sizeof(int16_t)), // 10 seconds at 44.1kHz
        .capacity = 44100 * 10,
        .size = 0,
        .error = TK_SUCCESS
    };
    
    if (!accumulator.buffer) {
        free(sanitized_text);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Initialize Piper parameters
    struct piper_params params;
    init_piper_params(&params, &context->config);
    
    // Set up audio callback adapter
    struct piper_audio_callback audio_cb = {
        .callback = (void (*)(const int16_t*, size_t, void*))accumulate_audio_callback,
        .user_data = &accumulator
    };
    
    // Run Piper synthesis
    enum piper_error_code result = piper_synthesize(
        context->piper_ctx,
        sanitized_text,
        &params,
        &audio_cb
    );
    
    free(sanitized_text);
    
    if (result != PIPER_ERROR_SUCCESS) {
        TK_LOG_ERROR("Piper synthesis failed with code: %d", result);
        free(accumulator.buffer);
        return convert_piper_error(result);
    }
    
    if (accumulator.error != TK_SUCCESS) {
        free(accumulator.buffer);
        return accumulator.error;
    }
    
    // Return the accumulated audio data
    *out_audio_data = accumulator.buffer;
    *out_frame_count = accumulator.size;
    
    return TK_SUCCESS;
}

tk_error_code_t tk_tts_piper_set_voice_params(
    tk_tts_piper_context_t* context,
    const tk_tts_piper_voice_params_t* params
) {
    if (!context || !params) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Update internal voice parameters
    context->voice_params.speaker_id = params->speaker_id;
    context->voice_params.length_scale = params->length_scale;
    context->voice_params.noise_scale = params->noise_scale;
    context->voice_params.noise_w = params->noise_w;
    
    return TK_SUCCESS;
}

tk_error_code_t tk_tts_piper_get_voice_params(
    tk_tts_piper_context_t* context,
    tk_tts_piper_voice_params_t* out_params
) {
    if (!context || !out_params) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    out_params->speaker_id = context->voice_params.speaker_id;
    out_params->length_scale = context->voice_params.length_scale;
    out_params->noise_scale = context->voice_params.noise_scale;
    out_params->noise_w = context->voice_params.noise_w;
    
    return TK_SUCCESS;
}

tk_error_code_t tk_tts_piper_get_model_info(
    tk_tts_piper_context_t* context,
    uint32_t* out_sample_rate,
    int* out_num_speakers
) {
    if (!context) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    if (out_sample_rate) {
        *out_sample_rate = context->model_info.sample_rate;
    }
    
    if (out_num_speakers) {
        *out_num_speakers = context->model_info.num_speakers;
    }
    
    return TK_SUCCESS;
}
