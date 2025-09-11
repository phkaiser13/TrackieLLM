/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_sound_classifier.c
*
* This file implements the Ambient Sound Classifier module.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#include "tk_sound_classifier.h"
#include "utils/tk_logging.h"

#include <onnxruntime_c_api.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define CLASSIFIER_WINDOW_SIZE_SAMPLES 15600 // YAMNet expects ~0.975s at 16kHz
#define CLASSIFIER_STEP_SIZE_SAMPLES 7800   // 50% overlap

// Internal context structure
struct tk_sound_classifier_context_s {
    OrtEnv* ort_env;
    OrtSession* ort_session;
    OrtAllocator* ort_allocator;
    OrtMemoryInfo* memory_info;

    tk_sound_classifier_config_t config;

    // Model input/output details
    char* input_name;
    char* output_name;
    int64_t input_shape[2];

    // Audio buffering
    float* audio_buffer;
    size_t audio_buffer_size;
};

// --- Private Helper Functions ---

static void convert_s16_to_float(const int16_t* input, float* output, size_t count) {
    for (size_t i = 0; i < count; i++) {
        output[i] = (float)input[i] / 32768.0f;
    }
}

static tk_error_code_t init_onnx_runtime(tk_sound_classifier_context_t* context) {
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtStatus* status;

    status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "sound-classifier", &context->ort_env);
    if (status) {
        TK_LOG_ERROR("ONNX: Failed to create environment: %s", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        return TK_ERROR_EXTERNAL_LIBRARY_FAILED;
    }

    OrtSessionOptions* session_options;
    status = g_ort->CreateSessionOptions(&session_options);
    if (status) { /* ... error handling ... */ return TK_ERROR_EXTERNAL_LIBRARY_FAILED; }
    g_ort->SetIntraOpNumThreads(session_options, context->config.n_threads);

    status = g_ort->CreateSession(context->ort_env, context->config.model_path->path_str, session_options, &context->ort_session);
    if (status) {
        TK_LOG_ERROR("ONNX: Failed to create session: %s", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(session_options);
        return TK_ERROR_MODEL_LOAD_FAILED;
    }
    g_ort->ReleaseSessionOptions(session_options);

    status = g_ort->GetAllocatorWithDefaultOptions(&context->ort_allocator);
    if (status) { /* ... error handling ... */ return TK_ERROR_EXTERNAL_LIBRARY_FAILED; }

    status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &context->memory_info);
    if (status) { /* ... error handling ... */ return TK_ERROR_EXTERNAL_LIBRARY_FAILED; }

    // Get input/output names
    status = g_ort->SessionGetInputName(context->ort_session, 0, context->ort_allocator, &context->input_name);
    if(status) { /* ... error handling ... */ return TK_ERROR_MODEL_LOAD_FAILED; }
    status = g_ort->SessionGetOutputName(context->ort_session, 0, context->ort_allocator, &context->output_name);
    if(status) { /* ... error handling ... */ return TK_ERROR_MODEL_LOAD_FAILED; }

    return TK_SUCCESS;
}

static void release_onnx_runtime(tk_sound_classifier_context_t* context) {
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) return;

    if (context->input_name) g_ort->AllocatorFree(context->ort_allocator, context->input_name);
    if (context->output_name) g_ort->AllocatorFree(context->ort_allocator, context->output_name);
    if (context->memory_info) g_ort->ReleaseMemoryInfo(context->memory_info);
    if (context->ort_session) g_ort->ReleaseSession(context->ort_session);
    if (context->ort_env) g_ort->ReleaseEnv(context->ort_env);
}

// --- Public API Implementation ---

TK_NODISCARD tk_error_code_t tk_sound_classifier_create(
    tk_sound_classifier_context_t** out_context,
    const tk_sound_classifier_config_t* config)
{
    if (!out_context || !config || !config->model_path) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    tk_sound_classifier_context_t* context = calloc(1, sizeof(tk_sound_classifier_context_t));
    if (!context) return TK_ERROR_OUT_OF_MEMORY;

    context->config = *config;
    if (context->config.n_threads <= 0) context->config.n_threads = 1;
    if (context->config.detection_threshold <= 0.0f) context->config.detection_threshold = 0.5f;

    tk_error_code_t err = init_onnx_runtime(context);
    if (err != TK_SUCCESS) {
        free(context);
        return err;
    }

    context->audio_buffer = malloc(CLASSIFIER_WINDOW_SIZE_SAMPLES * sizeof(float));
    if (!context->audio_buffer) {
        release_onnx_runtime(context);
        free(context);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    context->audio_buffer_size = 0;

    context->input_shape[0] = 1;
    context->input_shape[1] = CLASSIFIER_WINDOW_SIZE_SAMPLES;

    *out_context = context;
    return TK_SUCCESS;
}

void tk_sound_classifier_destroy(tk_sound_classifier_context_t** context)
{
    if (!context || !*context) return;

    tk_sound_classifier_context_t* ctx = *context;
    release_onnx_runtime(ctx);
    if (ctx->audio_buffer) free(ctx->audio_buffer);
    free(ctx);
    *context = NULL;
}

TK_NODISCARD tk_error_code_t tk_sound_classifier_process(
    tk_sound_classifier_context_t* context,
    const int16_t* audio_chunk,
    size_t frame_count,
    tk_sound_detection_result_t* out_result)
{
    if (!context || !audio_chunk || !out_result) return TK_ERROR_INVALID_ARGUMENT;

    // Set default result
    out_result->sound_class = TK_SOUND_UNKNOWN;
    out_result->confidence = 0.0f;

    // Append new audio data to internal buffer
    // This is a simplified buffering logic. A real implementation would use a ring buffer.
    size_t remaining_space = CLASSIFIER_WINDOW_SIZE_SAMPLES - context->audio_buffer_size;
    size_t frames_to_copy = frame_count < remaining_space ? frame_count : remaining_space;

    float* float_buffer_start = context->audio_buffer + context->audio_buffer_size;
    convert_s16_to_float(audio_chunk, float_buffer_start, frames_to_copy);
    context->audio_buffer_size += frames_to_copy;

    // If buffer is not full, wait for more data
    if (context->audio_buffer_size < CLASSIFIER_WINDOW_SIZE_SAMPLES) {
        return TK_SUCCESS;
    }

    // Buffer is full, run inference
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtValue* input_tensor = NULL;
    OrtStatus* status = g_ort->CreateTensorWithDataAsOrtValue(context->memory_info, context->audio_buffer,
        CLASSIFIER_WINDOW_SIZE_SAMPLES * sizeof(float), context->input_shape, 2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
    if(status) { return TK_ERROR_MODEL_INFERENCE_FAILED; }

    const char* input_names[] = {context->input_name};
    const char* output_names[] = {context->output_name};
    OrtValue* output_tensor = NULL;

    status = g_ort->Run(context->ort_session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor);
    if (status) {
        g_ort->ReleaseValue(input_tensor);
        return TK_ERROR_MODEL_INFERENCE_FAILED;
    }

    float* output_data;
    g_ort->GetTensorMutableData(output_tensor, (void**)&output_data);

    // Post-process: Find max score and its index
    // Assuming YAMNet-style output shape [1, num_classes]
    OrtTensorTypeAndShapeInfo* shape_info;
    g_ort->GetTensorTypeAndShape(output_tensor, &shape_info);
    size_t num_dims;
    g_ort->GetDimensionsCount(shape_info, &num_dims);
    int64_t dims[num_dims];
    g_ort->GetDimensions(shape_info, dims, num_dims);
    size_t num_classes = (num_dims > 1) ? dims[1] : 0;

    float max_score = -1.0f;
    int max_index = -1;
    for (size_t i = 0; i < num_classes; ++i) {
        if (output_data[i] > max_score) {
            max_score = output_data[i];
            max_index = i;
        }
    }

    g_ort->ReleaseTensorTypeAndShapeInfo(shape_info);
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(input_tensor);

    // If score is above threshold, map to our enum and report it
    // THIS IS A SIMPLIFIED MAPPING. A real one would use a label map file.
    if (max_score > context->config.detection_threshold) {
        out_result->confidence = max_score;
        switch (max_index) {
            case 0: // Speech (example index)
            case 12: // Chirp (example index)
                // Ignore these sounds
                break;
            case 500: // Siren (example index)
                out_result->sound_class = TK_SOUND_SIREN;
                break;
            case 389: // Alarm clock (example index)
                out_result->sound_class = TK_SOUND_ALARM;
                break;
            case 137: // Dog (example index)
                out_result->sound_class = TK_SOUND_DOG_BARK;
                break;
            // ... other mappings
            default:
                out_result->sound_class = TK_SOUND_UNKNOWN;
                break;
        }
    }

    // Shift buffer to make space for new data (50% overlap)
    memmove(context->audio_buffer, context->audio_buffer + CLASSIFIER_STEP_SIZE_SAMPLES,
            (CLASSIFIER_WINDOW_SIZE_SAMPLES - CLASSIFIER_STEP_SIZE_SAMPLES) * sizeof(float));
    context->audio_buffer_size -= CLASSIFIER_STEP_SIZE_SAMPLES;

    return TK_SUCCESS;
}
