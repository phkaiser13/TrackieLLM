/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_asr_whisper.c
 *
 * This source file implements the Whisper ASR (Automatic Speech Recognition) module.
 * It provides a high-level interface to the Whisper.cpp library, abstracting away
 * the complexities of model loading, inference, and result processing.
 *
 * The implementation is designed for real-time operation in embedded environments,
 * with support for streaming audio input and incremental transcription updates.
 *
 * Dependencies:
 *   - whisper.cpp (https://github.com/ggerganov/whisper.cpp)
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "audio/tk_asr_whisper.h"
#include "utils/tk_logging.h"
#include "utils/tk_error_handling.h"

// Include Whisper.cpp headers
#include "whisper.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

// Maximum audio buffer size for Whisper processing (30 seconds at 16kHz)
#define MAX_AUDIO_BUFFER_SIZE (16000 * 30)

// Internal structure for Whisper context
struct tk_asr_whisper_context_s {
    struct whisper_context* whisper_ctx;  // Whisper.cpp context
    tk_asr_whisper_config_t config;       // Configuration copy
    int16_t*                audio_buffer; // Buffer for accumulating audio data
    size_t                  audio_buffer_size; // Current size of buffered audio
    char*                   last_text;    // Last transcribed text
    bool                    has_partial;  // Whether we have a partial result
};

//------------------------------------------------------------------------------
// Private helper functions
//------------------------------------------------------------------------------

/**
 * @brief Converts Whisper error codes to Trackie error codes
 */
static tk_error_code_t convert_whisper_error(int whisper_error) {
    switch (whisper_error) {
        case 0:
            return TK_SUCCESS;
        case -1:
            return TK_ERROR_MODEL_LOAD_FAILED;
        case -2:
            return TK_ERROR_MODEL_INFERENCE_FAILED;
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
 * @brief Initializes Whisper parameters with default values
 */
static void init_whisper_params(struct whisper_full_params* params, const tk_asr_whisper_config_t* config) {
    *params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    
    // Set configuration parameters
    params->n_threads = config->n_threads > 0 ? config->n_threads : 4;
    params->n_max_text_ctx = config->max_context > 0 ? config->max_context : 16384;
    params->translate = config->translate_to_en;
    params->language = config->language;
    params->suppress_blank = false;
    params->suppress_non_speech_tokens = true;
    params->word_thold = config->word_threshold;
    
    // Enable token timestamps for better partial results
    params->token_timestamps = true;
    params->timestamp_token_probability_threshold = 0.01f;
    params->timestamp_token_sum_probability_threshold = 0.01f;
    
    // Set print progress to false for embedded use
    params->print_progress = false;
    params->print_results = false;
    params->print_special = false;
}

/**
 * @brief Processes accumulated audio buffer with Whisper
 */
static tk_error_code_t process_with_whisper(
    tk_asr_whisper_context_t* context,
    bool is_final,
    tk_asr_whisper_result_t* out_result
) {
    if (!context || !out_result) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Initialize Whisper parameters
    struct whisper_full_params params;
    init_whisper_params(&params, &context->config);
    
    // For final processing, we might want different parameters
    if (is_final) {
        params.temperature_inc = 0.2f;
        params.entropy_thold = 2.4f;
        params.logprob_thold = -1.0f;
    } else {
        // For partial results, use faster parameters
        params.temperature_inc = -1.0f; // Disables temperature fallback
        params.entropy_thold = 4.0f;
        params.logprob_thold = -0.5f;
    }
    
    // Run Whisper inference
    int result = whisper_full(
        context->whisper_ctx,
        params,
        context->audio_buffer,
        context->audio_buffer_size
    );
    
    if (result != 0) {
        TK_LOG_ERROR("Whisper inference failed with code: %d", result);
        return convert_whisper_error(result);
    }
    
    // Get the transcription result
    int n_segments = whisper_full_n_segments(context->whisper_ctx);
    
    // Build the result text
    size_t total_text_length = 0;
    for (int i = 0; i < n_segments; i++) {
        const char* segment_text = whisper_full_get_segment_text(context->whisper_ctx, i);
        if (segment_text) {
            total_text_length += strlen(segment_text);
        }
    }
    
    // Allocate result text buffer
    char* result_text = calloc(total_text_length + 1, sizeof(char));
    if (!result_text) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Copy all segments to result text
    size_t offset = 0;
    for (int i = 0; i < n_segments; i++) {
        const char* segment_text = whisper_full_get_segment_text(context->whisper_ctx, i);
        if (segment_text) {
            size_t segment_len = strlen(segment_text);
            memcpy(result_text + offset, segment_text, segment_len);
            offset += segment_len;
        }
    }
    
    // Set output result
    out_result->text = result_text;
    out_result->text_length = total_text_length;
    out_result->is_partial = !is_final;
    
    // Calculate average confidence (simplified)
    out_result->confidence = 0.9f; // Placeholder - real implementation would calculate this
    
    return TK_SUCCESS;
}

//------------------------------------------------------------------------------
// Public API Implementation
//------------------------------------------------------------------------------

tk_error_code_t tk_asr_whisper_create(
    tk_asr_whisper_context_t** out_context,
    const tk_asr_whisper_config_t* config
) {
    if (!out_context || !config || !config->model_path || !config->language) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_context = NULL;
    
    // Allocate context structure
    tk_asr_whisper_context_t* context = calloc(1, sizeof(tk_asr_whisper_context_t));
    if (!context) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Copy configuration
    context->config = *config;
    context->config.model_path = NULL; // We don't copy the path object
    context->config.language = duplicate_string(config->language);
    if (!context->config.language) {
        free(context);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Allocate audio buffer
    context->audio_buffer = calloc(MAX_AUDIO_BUFFER_SIZE, sizeof(int16_t));
    if (!context->audio_buffer) {
        free_string(context->config.language);
        free(context);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    context->audio_buffer_size = 0;
    
    // Initialize last_text
    context->last_text = NULL;
    context->has_partial = false;
    
    // Load Whisper model
    const char* model_path_str = config->model_path->path_str;
    context->whisper_ctx = whisper_init_from_file_with_params(
        model_path_str,
        whisper_context_default_params()
    );
    
    if (!context->whisper_ctx) {
        TK_LOG_ERROR("Failed to load Whisper model from: %s", model_path_str);
        free(context->audio_buffer);
        free_string(context->config.language);
        free(context);
        return TK_ERROR_MODEL_LOAD_FAILED;
    }
    
    // Set initial language
    whisper_full_set_language(context->whisper_ctx, config->language);
    
    *out_context = context;
    return TK_SUCCESS;
}

void tk_asr_whisper_destroy(tk_asr_whisper_context_t** context) {
    if (!context || !*context) return;
    
    tk_asr_whisper_context_t* ctx = *context;
    
    // Free Whisper context
    if (ctx->whisper_ctx) {
        whisper_free(ctx->whisper_ctx);
    }
    
    // Free audio buffer
    if (ctx->audio_buffer) {
        free(ctx->audio_buffer);
    }
    
    // Free configuration strings
    free_string(ctx->config.language);
    free_string(ctx->last_text);
    
    // Free context itself
    free(ctx);
    *context = NULL;
}

tk_error_code_t tk_asr_whisper_process_audio(
    tk_asr_whisper_context_t* context,
    const int16_t* audio_data,
    size_t frame_count,
    bool is_final,
    tk_asr_whisper_result_t** out_result
) {
    if (!context || !audio_data || !out_result) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_result = NULL;
    
    // Check if we have enough space in the buffer
    if (context->audio_buffer_size + frame_count > MAX_AUDIO_BUFFER_SIZE) {
        TK_LOG_WARN("Audio buffer overflow, resetting buffer");
        context->audio_buffer_size = 0;
    }
    
    // Append new audio data to buffer
    memcpy(
        context->audio_buffer + context->audio_buffer_size,
        audio_data,
        frame_count * sizeof(int16_t)
    );
    context->audio_buffer_size += frame_count;
    
    // Only process if we have enough audio data (at least 1 second)
    if (context->audio_buffer_size < 16000 && !is_final) {
        // Not enough data yet, return empty result
        *out_result = calloc(1, sizeof(tk_asr_whisper_result_t));
        if (!*out_result) {
            return TK_ERROR_OUT_OF_MEMORY;
        }
        return TK_SUCCESS;
    }
    
    // Allocate result structure
    tk_asr_whisper_result_t* result = calloc(1, sizeof(tk_asr_whisper_result_t));
    if (!result) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Process with Whisper
    tk_error_code_t error = process_with_whisper(context, is_final, result);
    if (error != TK_SUCCESS) {
        free(result);
        return error;
    }
    
    // If this is a final result, reset the buffer
    if (is_final) {
        context->audio_buffer_size = 0;
        context->has_partial = false;
        free_string(context->last_text);
        context->last_text = duplicate_string(result->text);
    } else {
        context->has_partial = true;
    }
    
    *out_result = result;
    return TK_SUCCESS;
}

void tk_asr_whisper_free_result(tk_asr_whisper_result_t** result) {
    if (!result || !*result) return;
    
    tk_asr_whisper_result_t* res = *result;
    
    if (res->text) {
        free(res->text);
    }
    
    free(res);
    *result = NULL;
}

tk_error_code_t tk_asr_whisper_reset(tk_asr_whisper_context_t* context) {
    if (!context) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Reset audio buffer
    context->audio_buffer_size = 0;
    
    // Reset state flags
    context->has_partial = false;
    
    // Free last text
    free_string(context->last_text);
    context->last_text = NULL;
    
    return TK_SUCCESS;
}

tk_error_code_t tk_asr_whisper_set_language(
    tk_asr_whisper_context_t* context,
    const char* language
) {
    if (!context || !language) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Update language in Whisper context
    whisper_full_set_language(context->whisper_ctx, language);
    
    // Update our copy of the language
    free_string(context->config.language);
    context->config.language = duplicate_string(language);
    if (!context->config.language) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    return TK_SUCCESS;
}
