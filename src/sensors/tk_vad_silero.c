/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_vad_silero.c
 *
 * This source file implements the Silero VAD (Voice Activity Detection) module.
 * It provides a high-level interface to the Silero VAD ONNX model, abstracting away
 * the complexities of model loading, inference, and result processing.
 *
 * The implementation is designed for real-time operation in embedded environments,
 * with support for streaming audio input and configurable sensitivity parameters.
 *
 * Dependencies:
 *   - ONNX Runtime (https://github.com/microsoft/onnxruntime)
 *   - Silero VAD model (https://github.com/snakers4/silero-vad)
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "audio/tk_vad_silero.h"
#include "utils/tk_logging.h"
#include "utils/tk_error_handling.h"

// Include ONNX Runtime headers
#include <onnxruntime_c_api.h>
#include <cpu_provider_factory.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

// Maximum audio buffer size for Silero VAD processing (30 seconds at 16kHz)
#define MAX_AUDIO_BUFFER_SIZE (16000 * 30)

// Internal structure for Silero VAD context
struct tk_vad_silero_context_s {
    // ONNX Runtime components
    OrtEnv*                 env;
    OrtSession*             session;
    OrtAllocator*           allocator;
    OrtMemoryInfo*          memory_info;
    
    // Model information
    tk_vad_silero_config_t  config;
    int                     input_sample_rate;
    
    // Internal state
    tk_vad_silero_state_t   state;
    float                   last_probability;
    
    // Audio buffering
    float*                  audio_buffer;       // Float buffer for model input
    size_t                  audio_buffer_size;  // Current size of buffered audio
    size_t                  window_size;        // Size of processing window in samples
    size_t                  step_size;          // Step size between windows
    
    // Timing tracking
    float                   time_since_last_event_ms;
    bool                    triggered_speech_start;
};

//------------------------------------------------------------------------------
// Private helper functions
//------------------------------------------------------------------------------

/**
 * @brief Checks if the sample rate is supported by Silero VAD
 */
static bool is_supported_sample_rate(uint32_t sample_rate) {
    return (sample_rate == 8000) || (sample_rate == 16000) || (sample_rate == 48000);
}

/**
 * @brief Converts integer audio samples to float
 */
static void convert_int16_to_float(const int16_t* input, float* output, size_t count) {
    for (size_t i = 0; i < count; i++) {
        output[i] = (float)input[i] / 32768.0f;
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
 * @brief Initializes ONNX Runtime environment
 */
static tk_error_code_t init_onnx_runtime(tk_vad_silero_context_t* context) {
    // Create environment
    OrtStatus* status = OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "SileroVAD", &context->env);
    if (status != NULL) {
        TK_LOG_ERROR("Failed to create ONNX Runtime environment");
        OrtReleaseStatus(status);
        return TK_ERROR_MODEL_LOAD_FAILED;
    }
    
    // Create session options
    OrtSessionOptions* session_options;
    status = OrtCreateSessionOptions(&session_options);
    if (status != NULL) {
        TK_LOG_ERROR("Failed to create ONNX Runtime session options");
        OrtReleaseEnv(context->env);
        OrtReleaseStatus(status);
        return TK_ERROR_MODEL_LOAD_FAILED;
    }
    
    // Set number of threads
    OrtSetIntraOpNumThreads(session_options, 1);
    OrtSetInterOpNumThreads(session_options, 1);
    
    // Enable CPU memory arena
    OrtEnableCpuMemArena(session_options);
    
    // Disable profiling
    OrtDisableProfiling(session_options);
    
    // Create session
    const char* model_path_str = context->config.model_path->path_str;
    status = OrtCreateSession(
        context->env,
        model_path_str,
        session_options,
        &context->session
    );
    
    OrtReleaseSessionOptions(session_options);
    
    if (status != NULL) {
        TK_LOG_ERROR("Failed to create ONNX Runtime session from: %s", model_path_str);
        OrtReleaseEnv(context->env);
        OrtReleaseStatus(status);
        return TK_ERROR_MODEL_LOAD_FAILED;
    }
    
    // Get allocator
    status = OrtGetAllocatorWithDefaultOptions(&context->allocator);
    if (status != NULL) {
        TK_LOG_ERROR("Failed to get ONNX Runtime allocator");
        OrtReleaseSession(context->session);
        OrtReleaseEnv(context->env);
        OrtReleaseStatus(status);
        return TK_ERROR_MODEL_LOAD_FAILED;
    }
    
    // Get memory info
    context->memory_info = OrtAllocatorGetInfo(context->allocator);
    
    return TK_SUCCESS;
}

/**
 * @brief Releases ONNX Runtime resources
 */
static void release_onnx_runtime(tk_vad_silero_context_t* context) {
    if (context->memory_info) {
        OrtReleaseMemoryInfo(context->memory_info);
    }
    
    if (context->session) {
        OrtReleaseSession(context->session);
    }
    
    if (context->env) {
        OrtReleaseEnv(context->env);
    }
}

/**
 * @brief Runs inference on the Silero VAD model
 */
static tk_error_code_t run_vad_inference(
    tk_vad_silero_context_t* context,
    const float* audio_data,
    size_t frame_count,
    float* out_probability
) {
    if (!context || !audio_data || !out_probability) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Create input tensor
    int64_t input_shape[] = {1, (int64_t)frame_count};
    OrtValue* input_tensor = NULL;
    
    OrtStatus* status = OrtCreateTensorWithDataAsOrtValue(
        context->memory_info,
        (void*)audio_data,
        frame_count * sizeof(float),
        input_shape,
        2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor
    );
    
    if (status != NULL) {
        TK_LOG_ERROR("Failed to create input tensor");
        OrtReleaseStatus(status);
        return TK_ERROR_MODEL_INFERENCE_FAILED;
    }
    
    // Get input name
    char* input_name;
    status = OrtSessionGetInputName(context->session, 0, context->allocator, &input_name);
    if (status != NULL) {
        TK_LOG_ERROR("Failed to get input name");
        OrtReleaseValue(input_tensor);
        OrtReleaseStatus(status);
        return TK_ERROR_MODEL_INFERENCE_FAILED;
    }
    
    // Prepare output
    OrtValue* output_tensor = NULL;
    const char* input_names[] = {input_name};
    const OrtValue* input_values[] = {input_tensor};
    
    // Run inference
    status = OrtRun(
        context->session,
        NULL, // RunOptions
        input_names,
        input_values,
        1, // Input count
        NULL, // Output names (NULL means use default)
        0, // Output count (0 means use default)
        &output_tensor
    );
    
    OrtFree(input_name);
    
    if (status != NULL) {
        TK_LOG_ERROR("Failed to run inference");
        OrtReleaseValue(input_tensor);
        OrtReleaseStatus(status);
        return TK_ERROR_MODEL_INFERENCE_FAILED;
    }
    
    // Get output data
    float* output_data;
    status = OrtGetTensorMutableData(output_tensor, (void**)&output_data);
    if (status != NULL) {
        TK_LOG_ERROR("Failed to get output tensor data");
        OrtReleaseValue(output_tensor);
        OrtReleaseValue(input_tensor);
        OrtReleaseStatus(status);
        return TK_ERROR_MODEL_INFERENCE_FAILED;
    }
    
    // Copy probability result
    *out_probability = output_data[0];
    
    // Clean up
    OrtReleaseValue(output_tensor);
    OrtReleaseValue(input_tensor);
    
    return TK_SUCCESS;
}

/**
 * @brief Updates the VAD state machine based on probability
 */
static void update_vad_state(
    tk_vad_silero_context_t* context,
    float probability,
    float time_delta_ms
) {
    if (!context) return;
    
    context->last_probability = probability;
    context->state.speech_probability = probability;
    context->time_since_last_event_ms += time_delta_ms;
    
    bool speech_detected = (probability >= context->config.threshold);
    
    if (speech_detected) {
        // Update speech duration
        context->state.speech_duration_ms += time_delta_ms;
        context->state.silence_duration_ms = 0.0f;
        
        // Check if we should trigger speech start
        if (!context->state.is_speech_active && 
            context->state.speech_duration_ms >= context->config.min_speech_duration_ms &&
            !context->triggered_speech_start) {
            context->state.is_speech_active = true;
            context->triggered_speech_start = true;
            context->time_since_last_event_ms = 0.0f;
        }
    } else {
        // Update silence duration
        context->state.silence_duration_ms += time_delta_ms;
        context->state.speech_duration_ms = 0.0f;
        
        // Check if we should trigger speech end
        if (context->state.is_speech_active && 
            context->state.silence_duration_ms >= context->config.min_silence_duration_ms) {
            context->state.is_speech_active = false;
            context->triggered_speech_start = false;
            context->time_since_last_event_ms = 0.0f;
        }
    }
}

/**
 * @brief Processes buffered audio in windows
 */
static tk_error_code_t process_buffered_audio(
    tk_vad_silero_context_t* context,
    tk_vad_silero_event_callback_t callback,
    void* user_data
) {
    if (!context) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Process audio in windows
    size_t processed_samples = 0;
    bool last_speech_state = context->state.is_speech_active;
    
    while (processed_samples + context->window_size <= context->audio_buffer_size) {
        // Run inference on current window
        float probability = 0.0f;
        tk_error_code_t result = run_vad_inference(
            context,
            context->audio_buffer + processed_samples,
            context->window_size,
            &probability
        );
        
        if (result != TK_SUCCESS) {
            return result;
        }
        
        // Calculate time delta for this window
        float time_delta_ms = (float)(context->window_size * 1000) / context->input_sample_rate;
        
        // Update state machine
        update_vad_state(context, probability, time_delta_ms);
        
        // Check for state transitions
        if (callback) {
            if (!last_speech_state && context->state.is_speech_active) {
                callback(TK_VAD_EVENT_SPEECH_STARTED, user_data);
            } else if (last_speech_state && !context->state.is_speech_active) {
                callback(TK_VAD_EVENT_SPEECH_ENDED, user_data);
            }
        }
        
        last_speech_state = context->state.is_speech_active;
        processed_samples += context->step_size;
    }
    
    // Shift remaining audio to beginning of buffer
    if (processed_samples > 0 && context->audio_buffer_size > processed_samples) {
        size_t remaining_samples = context->audio_buffer_size - processed_samples;
        memmove(
            context->audio_buffer,
            context->audio_buffer + processed_samples,
            remaining_samples * sizeof(float)
        );
        context->audio_buffer_size = remaining_samples;
    } else if (processed_samples > 0) {
        context->audio_buffer_size = 0;
    }
    
    return TK_SUCCESS;
}

//------------------------------------------------------------------------------
// Public API Implementation
//------------------------------------------------------------------------------

tk_error_code_t tk_vad_silero_create(
    tk_vad_silero_context_t** out_context,
    const tk_vad_silero_config_t* config
) {
    if (!out_context || !config || !config->model_path) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    if (!is_supported_sample_rate(config->sample_rate)) {
        TK_LOG_ERROR("Unsupported sample rate: %u", config->sample_rate);
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_context = NULL;
    
    // Allocate context structure
    tk_vad_silero_context_t* context = calloc(1, sizeof(tk_vad_silero_context_t));
    if (!context) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Copy configuration
    context->config = *config;
    context->config.model_path = NULL; // We don't copy the path object
    context->input_sample_rate = config->sample_rate;
    
    // Set default parameters if not provided
    if (context->config.threshold <= 0.0f) {
        context->config.threshold = 0.5f;
    }
    
    if (context->config.min_silence_duration_ms <= 0.0f) {
        context->config.min_silence_duration_ms = 300.0f;
    }
    
    if (context->config.min_speech_duration_ms <= 0.0f) {
        context->config.min_speech_duration_ms = 250.0f;
    }
    
    if (context->config.speech_pad_ms < 0.0f) {
        context->config.speech_pad_ms = 30.0f;
    }
    
    // Initialize state
    context->state.is_speech_active = false;
    context->state.speech_probability = 0.0f;
    context->state.silence_duration_ms = 0.0f;
    context->state.speech_duration_ms = 0.0f;
    context->last_probability = 0.0f;
    context->time_since_last_event_ms = 0.0f;
    context->triggered_speech_start = false;
    
    // Allocate audio buffer (30 seconds of audio)
    context->audio_buffer = calloc(MAX_AUDIO_BUFFER_SIZE, sizeof(float));
    if (!context->audio_buffer) {
        free(context);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    context->audio_buffer_size = 0;
    
    // Calculate window and step sizes
    // For Silero VAD, we typically use 30ms windows with 10ms steps
    context->window_size = (context->input_sample_rate * 30) / 1000; // 30ms
    context->step_size = (context->input_sample_rate * 10) / 1000;    // 10ms
    
    // Initialize ONNX Runtime
    tk_error_code_t result = init_onnx_runtime(context);
    if (result != TK_SUCCESS) {
        free(context->audio_buffer);
        free(context);
        return result;
    }
    
    *out_context = context;
    return TK_SUCCESS;
}

void tk_vad_silero_destroy(tk_vad_silero_context_t** context) {
    if (!context || !*context) return;
    
    tk_vad_silero_context_t* ctx = *context;
    
    // Release ONNX Runtime resources
    release_onnx_runtime(ctx);
    
    // Free audio buffer
    if (ctx->audio_buffer) {
        free(ctx->audio_buffer);
    }
    
    // Free context itself
    free(ctx);
    *context = NULL;
}

tk_error_code_t tk_vad_silero_process_audio(
    tk_vad_silero_context_t* context,
    const int16_t* audio_data,
    size_t frame_count,
    float* out_probability
) {
    if (!context || !audio_data || !out_probability) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_probability = 0.0f;
    
    // Convert audio to float
    float* float_buffer = malloc(frame_count * sizeof(float));
    if (!float_buffer) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    convert_int16_to_float(audio_data, float_buffer, frame_count);
    
    // Run inference
    tk_error_code_t result = run_vad_inference(
        context,
        float_buffer,
        frame_count,
        out_probability
    );
    
    free(float_buffer);
    return result;
}

tk_error_code_t tk_vad_silero_process_audio_with_events(
    tk_vad_silero_context_t* context,
    const int16_t* audio_data,
    size_t frame_count,
    tk_vad_silero_event_callback_t callback,
    void* user_data
) {
    if (!context || !audio_data) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Check if we have enough space in the buffer
    if (context->audio_buffer_size + frame_count > MAX_AUDIO_BUFFER_SIZE) {
        TK_LOG_WARN("Audio buffer overflow, resetting buffer");
        context->audio_buffer_size = 0;
    }
    
    // Convert audio to float and append to buffer
    convert_int16_to_float(
        audio_data,
        context->audio_buffer + context->audio_buffer_size,
        frame_count
    );
    context->audio_buffer_size += frame_count;
    
    // Process buffered audio if we have enough for at least one window
    if (context->audio_buffer_size >= context->window_size) {
        return process_buffered_audio(context, callback, user_data);
    }
    
    return TK_SUCCESS;
}

tk_error_code_t tk_vad_silero_get_state(
    tk_vad_silero_context_t* context,
    tk_vad_silero_state_t* out_state
) {
    if (!context || !out_state) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_state = context->state;
    return TK_SUCCESS;
}

tk_error_code_t tk_vad_silero_reset(tk_vad_silero_context_t* context) {
    if (!context) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Reset state
    context->state.is_speech_active = false;
    context->state.speech_probability = 0.0f;
    context->state.silence_duration_ms = 0.0f;
    context->state.speech_duration_ms = 0.0f;
    context->last_probability = 0.0f;
    context->time_since_last_event_ms = 0.0f;
    context->triggered_speech_start = false;
    
    // Reset audio buffer
    context->audio_buffer_size = 0;
    
    return TK_SUCCESS;
}

tk_error_code_t tk_vad_silero_set_threshold(
    tk_vad_silero_context_t* context,
    float threshold
) {
    if (!context || threshold < 0.0f || threshold > 1.0f) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    context->config.threshold = threshold;
    return TK_SUCCESS;
}
