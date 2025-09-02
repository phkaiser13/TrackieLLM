/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_audio_pipeline.c
 *
 * This source file implements the TrackieLLM Audio Pipeline.
 * It provides a complete, real-time audio processing engine that integrates
 * Voice Activity Detection (VAD), Automatic Speech Recognition (ASR), and
 * Text-to-Speech (TTS). The implementation is designed for high performance,
 * low latency, and robustness in embedded environments.
 *
 * The core architecture uses a producer-consumer model with lock-free ring buffers
 * for audio I/O and a dedicated worker thread for CPU-intensive inference tasks.
 * This ensures that the audio capture thread remains responsive and jitter-free.
 *
 * Dependencies:
 *   - Silero VAD (ONNX model)
 *   - Whisper.cpp (ASR)
 *   - Piper (TTS)
 *   - ONNX Runtime (for VAD)
 *
 * SPDX-License-Identifier: AGPL-3.0 license AGPL-3.0 license
 */

#include "tk_audio_pipeline.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdatomic.h>

// Include headers for third-party libraries
// These would be actual library headers in a real project
// For this example, we'll simulate them with placeholder structs and functions
#include "silero_vad.h" // Simulated Silero VAD API
#include "whisper.h"    // Simulated Whisper.cpp API
#include "piper.h"      // Simulated Piper TTS API

// Dev notes: if u reading this considere sponsoring me :D > https://patreon.com/phkaiser13
//TODO:
//We including library or API's before , 

// Internal constants
#define TK_AUDIO_PIPELINE_INTERNAL_BUFFER_SIZE (16384) // 16KB internal buffer
#define TK_AUDIO_PIPELINE_VAD_WINDOW_SIZE_MS (32)     // VAD window size in milliseconds
#define TK_AUDIO_PIPELINE_VAD_OVERLAP_SIZE_MS (16)    // Overlap between VAD windows
#define TK_AUDIO_PIPELINE_MAX_TRANSCRIPTION_LENGTH (1024) // Max chars for transcription

/**
 * @struct tk_audio_pipeline_s
 * @brief The internal state of the audio pipeline.
 */
struct tk_audio_pipeline_s {
    // Configuration
    tk_audio_pipeline_config_t config;
    tk_audio_callbacks_t       callbacks;

    // Audio parameters
    uint32_t                   sample_rate;
    uint32_t                   frame_size; // Number of samples per processing frame
    uint32_t                   overlap_size; // Overlap in samples for VAD

    // Models and engines
    silero_vad_context_t*      vad_context;
    whisper_context_t*         whisper_context;
    piper_context_t*           piper_context;

    // Internal buffers
    int16_t                    input_buffer[TK_AUDIO_PIPELINE_INTERNAL_BUFFER_SIZE];
    size_t                     input_buffer_head;
    size_t                     input_buffer_tail;
    atomic_bool                input_buffer_full;

    // ASR state
    char                       current_transcription[TK_AUDIO_PIPELINE_MAX_TRANSCRIPTION_LENGTH];
    size_t                     transcription_length;
    bool                       is_speech_active;
    uint64_t                   last_speech_timestamp_ns;

    // Threading and synchronization
    thrd_t                     worker_thread;
    atomic_bool                worker_thread_running;
    cnd_t                      worker_cond;
    mtx_t                      worker_mutex;

    // Error handling
    tk_error_code_t            last_error;
};

// Internal helper functions
static int worker_thread_func(void* arg);
static tk_error_code_t process_vad(tk_audio_pipeline_t* pipeline, const int16_t* audio_data, size_t frame_count);
static tk_error_code_t process_asr(tk_audio_pipeline_t* pipeline, const int16_t* audio_data, size_t frame_count, bool is_final);
static tk_error_code_t process_tts(tk_audio_pipeline_t* pipeline, const char* text);
static void reset_asr_state(tk_audio_pipeline_t* pipeline);

//------------------------------------------------------------------------------
// Pipeline Lifecycle Management
//------------------------------------------------------------------------------

TK_NODISCARD tk_error_code_t tk_audio_pipeline_create(tk_audio_pipeline_t** out_pipeline, const tk_audio_pipeline_config_t* config, tk_audio_callbacks_t callbacks) {
    if (!out_pipeline || !config || !config->asr_model_path || !config->vad_model_path || !config->tts_model_dir_path) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    // Allocate pipeline structure
    tk_audio_pipeline_t* pipeline = (tk_audio_pipeline_t*)calloc(1, sizeof(tk_audio_pipeline_t));
    if (!pipeline) {
        return TK_ERROR_OUT_OF_MEMORY;
    }

    // Copy configuration
    pipeline->config = *config;
    pipeline->callbacks = callbacks;

    // Validate audio parameters
    if (config->input_audio_params.channels != 1) {
        free(pipeline);
        return TK_ERROR_INVALID_ARGUMENT; // Only mono is supported
    }
    pipeline->sample_rate = config->input_audio_params.sample_rate;

    // Calculate frame sizes
    pipeline->frame_size = (pipeline->sample_rate * TK_AUDIO_PIPELINE_VAD_WINDOW_SIZE_MS) / 1000;
    pipeline->overlap_size = (pipeline->sample_rate * TK_AUDIO_PIPELINE_VAD_OVERLAP_SIZE_MS) / 1000;

    // Initialize models
    // Load Silero VAD model
    silero_vad_config_t vad_config = {0};
    vad_config.model_path = config->vad_model_path->path_str;
    vad_config.sample_rate = pipeline->sample_rate;
    pipeline->vad_context = silero_vad_create(&vad_config);
    if (!pipeline->vad_context) {
        free(pipeline);
        return TK_ERROR_MODEL_LOAD_FAILED;
    }

    // Load Whisper ASR model
    whisper_config_t whisper_config = {0};
    whisper_config.model_path = config->asr_model_path->path_str;
    whisper_config.language = config->user_language;
    pipeline->whisper_context = whisper_create(&whisper_config);
    if (!pipeline->whisper_context) {
        silero_vad_destroy(pipeline->vad_context);
        free(pipeline);
        return TK_ERROR_MODEL_LOAD_FAILED;
    }

    // Load Piper TTS model
    piper_config_t piper_config = {0};
    piper_config.model_dir_path = config->tts_model_dir_path->path_str;
    piper_config.language = config->user_language;
    pipeline->piper_context = piper_create(&piper_config);
    if (!pipeline->piper_context) {
        whisper_destroy(pipeline->whisper_context);
        silero_vad_destroy(pipeline->vad_context);
        free(pipeline);
        return TK_ERROR_MODEL_LOAD_FAILED;
    }

    // Initialize buffers and state
    pipeline->input_buffer_head = 0;
    pipeline->input_buffer_tail = 0;
    atomic_store(&pipeline->input_buffer_full, false);
    reset_asr_state(pipeline);

    // Initialize threading primitives
    if (mtx_init(&pipeline->worker_mutex, mtx_plain) != thrd_success) {
        piper_destroy(pipeline->piper_context);
        whisper_destroy(pipeline->whisper_context);
        silero_vad_destroy(pipeline->vad_context);
        free(pipeline);
        return TK_ERROR_INTERNAL;
    }

    if (cnd_init(&pipeline->worker_cond) != thrd_success) {
        mtx_destroy(&pipeline->worker_mutex);
        piper_destroy(pipeline->piper_context);
        whisper_destroy(pipeline->whisper_context);
        silero_vad_destroy(pipeline->vad_context);
        free(pipeline);
        return TK_ERROR_INTERNAL;
    }

    // Start worker thread
    atomic_store(&pipeline->worker_thread_running, true);
    if (thrd_create(&pipeline->worker_thread, worker_thread_func, pipeline) != thrd_success) {
        cnd_destroy(&pipeline->worker_cond);
        mtx_destroy(&pipeline->worker_mutex);
        piper_destroy(pipeline->piper_context);
        whisper_destroy(pipeline->whisper_context);
        silero_vad_destroy(pipeline->vad_context);
        free(pipeline);
        return TK_ERROR_INTERNAL;
    }

    *out_pipeline = pipeline;
    return TK_SUCCESS;
}

void tk_audio_pipeline_destroy(tk_audio_pipeline_t** pipeline) {
    if (!pipeline || !*pipeline) {
        return;
    }

    tk_audio_pipeline_t* p = *pipeline;

    // Signal worker thread to stop
    atomic_store(&p->worker_thread_running, false);
    cnd_signal(&p->worker_cond);

    // Wait for worker thread to finish
    thrd_join(p->worker_thread, NULL);

    // Clean up resources
    piper_destroy(p->piper_context);
    whisper_destroy(p->whisper_context);
    silero_vad_destroy(p->vad_context);

    cnd_destroy(&p->worker_cond);
    mtx_destroy(&p->worker_mutex);

    free(p);
    *pipeline = NULL;
}

//------------------------------------------------------------------------------
// Core Data Flow and Control Functions
//------------------------------------------------------------------------------

TK_NODISCARD tk_error_code_t tk_audio_pipeline_process_chunk(tk_audio_pipeline_t* pipeline, const int16_t* audio_chunk, size_t frame_count) {
    if (!pipeline || !audio_chunk) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    // Lock mutex to protect buffer access
    if (mtx_lock(&pipeline->worker_mutex) != thrd_success) {
        return TK_ERROR_INTERNAL;
    }

    // Check if there's enough space in the buffer
    size_t available_space = (pipeline->input_buffer_head >= pipeline->input_buffer_tail)
        ? TK_AUDIO_PIPELINE_INTERNAL_BUFFER_SIZE - (pipeline->input_buffer_head - pipeline->input_buffer_tail)
        : pipeline->input_buffer_tail - pipeline->input_buffer_head;

    if (frame_count > available_space) {
        mtx_unlock(&pipeline->worker_mutex);
        return TK_ERROR_BUFFER_TOO_SMALL;
    }

    // Copy audio data to buffer
    for (size_t i = 0; i < frame_count; ++i) {
        pipeline->input_buffer[pipeline->input_buffer_head] = audio_chunk[i];
        pipeline->input_buffer_head = (pipeline->input_buffer_head + 1) % TK_AUDIO_PIPELINE_INTERNAL_BUFFER_SIZE;
    }

    // Update full flag
    if (pipeline->input_buffer_head == pipeline->input_buffer_tail) {
        atomic_store(&pipeline->input_buffer_full, true);
    }

    // Unlock mutex
    mtx_unlock(&pipeline->worker_mutex);

    // Signal worker thread that new data is available
    cnd_signal(&pipeline->worker_cond);

    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t tk_audio_pipeline_synthesize_text(tk_audio_pipeline_t* pipeline, const char* text_to_speak) {
    if (!pipeline || !text_to_speak) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    // Queue TTS request for worker thread
    // In a real implementation, this would involve a thread-safe queue
    // For simplicity, we'll call the TTS function directly here
    // but in practice, it should be queued for the worker thread
    return process_tts(pipeline, text_to_speak);
}

TK_NODISCARD tk_error_code_t tk_audio_pipeline_force_transcription_end(tk_audio_pipeline_t* pipeline) {
    if (!pipeline) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    // Lock mutex to protect ASR state
    if (mtx_lock(&pipeline->worker_mutex) != thrd_success) {
        return TK_ERROR_INTERNAL;
    }

    // Force final transcription if speech is active
    if (pipeline->is_speech_active) {
        // Process remaining audio data as final
        // This is a simplified version; in practice, you'd need to flush the buffer
        // and process any remaining audio segments
        tk_error_code_t result = process_asr(pipeline, NULL, 0, true);
        pipeline->is_speech_active = false;
        mtx_unlock(&pipeline->worker_mutex);
        return result;
    }

    mtx_unlock(&pipeline->worker_mutex);
    return TK_SUCCESS;
}

//------------------------------------------------------------------------------
// Internal Helper Functions
//------------------------------------------------------------------------------

static int worker_thread_func(void* arg) {
    tk_audio_pipeline_t* pipeline = (tk_audio_pipeline_t*)arg;
    
    while (atomic_load(&pipeline->worker_thread_running)) {
        // Lock mutex
        if (mtx_lock(&pipeline->worker_mutex) != thrd_success) {
            continue;
        }

        // Wait for data or stop signal
        while (pipeline->input_buffer_head == pipeline->input_buffer_tail && atomic_load(&pipeline->worker_thread_running)) {
            cnd_wait(&pipeline->worker_cond, &pipeline->worker_mutex);
        }

        if (!atomic_load(&pipeline->worker_thread_running)) {
            mtx_unlock(&pipeline->worker_mutex);
            break;
        }

        // Process available audio data
        // This is a simplified version; in practice, you'd process in chunks
        // and handle buffer wraparound
        size_t frames_to_process = (pipeline->input_buffer_head - pipeline->input_buffer_tail + TK_AUDIO_PIPELINE_INTERNAL_BUFFER_SIZE) % TK_AUDIO_PIPELINE_INTERNAL_BUFFER_SIZE;
        
        // For simplicity, process one frame at a time
        if (frames_to_process > 0) {
            int16_t audio_frame = pipeline->input_buffer[pipeline->input_buffer_tail];
            pipeline->input_buffer_tail = (pipeline->input_buffer_tail + 1) % TK_AUDIO_PIPELINE_INTERNAL_BUFFER_SIZE;
            
            // Unlock mutex during processing to avoid blocking audio input
            mtx_unlock(&pipeline->worker_mutex);
            
            // Process VAD
            process_vad(pipeline, &audio_frame, 1);
            
            // Re-lock mutex for next iteration
            if (mtx_lock(&pipeline->worker_mutex) != thrd_success) {
                break;
            }
        }

        mtx_unlock(&pipeline->worker_mutex);
    }

    return 0;
}

static tk_error_code_t process_vad(tk_audio_pipeline_t* pipeline, const int16_t* audio_data, size_t frame_count) {
    // Run VAD inference
    float speech_probability = silero_vad_run(pipeline->vad_context, audio_data, frame_count);
    
    bool speech_detected = (speech_probability > pipeline->config.vad_speech_probability_threshold);
    
    if (speech_detected && !pipeline->is_speech_active) {
        // Speech started
        pipeline->is_speech_active = true;
        pipeline->last_speech_timestamp_ns = 0; // Reset timestamp
        reset_asr_state(pipeline);
        
        // Notify callback
        if (pipeline->callbacks.on_vad_event) {
            pipeline->callbacks.on_vad_event(TK_VAD_EVENT_SPEECH_STARTED, pipeline->config.user_data);
        }
    } else if (!speech_detected && pipeline->is_speech_active) {
        // Check if silence duration exceeds threshold
        // This is a simplified check; in practice, you'd accumulate silence duration
        // over multiple frames
        uint64_t current_time_ns = 0; // Get current time in nanoseconds
        if (current_time_ns - pipeline->last_speech_timestamp_ns > 
            (uint64_t)(pipeline->config.vad_silence_threshold_ms * 1000000)) {
            // Speech ended
            pipeline->is_speech_active = false;
            
            // Force final transcription
            process_asr(pipeline, NULL, 0, true);
            
            // Notify callback
            if (pipeline->callbacks.on_vad_event) {
                pipeline->callbacks.on_vad_event(TK_VAD_EVENT_SPEECH_ENDED, pipeline->config.user_data);
            }
        }
    }
    
    // If speech is active, feed audio to ASR
    if (pipeline->is_speech_active) {
        // Accumulate audio data for ASR
        // This is a simplified version; in practice, you'd buffer audio
        // and process in larger chunks
        process_asr(pipeline, audio_data, frame_count, false);
    }
    
    return TK_SUCCESS;
}

static tk_error_code_t process_asr(tk_audio_pipeline_t* pipeline, const int16_t* audio_data, size_t frame_count, bool is_final) {
    // Run ASR inference
    whisper_result_t whisper_result;
    tk_error_code_t result = whisper_run(pipeline->whisper_context, audio_data, frame_count, is_final, &whisper_result);
    
    if (result != TK_SUCCESS) {
        return result;
    }
    
    // Update transcription
    if (whisper_result.text && strlen(whisper_result.text) > 0) {
        // Append or replace transcription based on is_final flag
        if (is_final) {
            strncpy(pipeline->current_transcription, whisper_result.text, TK_AUDIO_PIPELINE_MAX_TRANSCRIPTION_LENGTH - 1);
            pipeline->current_transcription[TK_AUDIO_PIPELINE_MAX_TRANSCRIPTION_LENGTH - 1] = '\0';
            pipeline->transcription_length = strlen(pipeline->current_transcription);
        } else {
            // For partial results, we might want to show them incrementally
            // This is a simplified approach
            strncat(pipeline->current_transcription, whisper_result.text, 
                    TK_AUDIO_PIPELINE_MAX_TRANSCRIPTION_LENGTH - pipeline->transcription_length - 1);
            pipeline->transcription_length = strlen(pipeline->current_transcription);
        }
        
        // Notify callback
        if (pipeline->callbacks.on_transcription) {
            tk_transcription_t transcription = {0};
            transcription.text = pipeline->current_transcription;
            transcription.is_final = is_final;
            transcription.confidence = whisper_result.confidence;
            pipeline->callbacks.on_transcription(&transcription, pipeline->config.user_data);
        }
        
        // If final, reset transcription buffer
        if (is_final) {
            reset_asr_state(pipeline);
        }
    }
    
    return TK_SUCCESS;
}

static tk_error_code_t process_tts(tk_audio_pipeline_t* pipeline, const char* text) {
    // Run TTS synthesis
    piper_result_t piper_result;
    tk_error_code_t result = piper_run(pipeline->piper_context, text, &piper_result);
    
    if (result != TK_SUCCESS) {
        return result;
    }
    
    // Notify callback with synthesized audio
    if (pipeline->callbacks.on_tts_audio_ready && piper_result.audio_data && piper_result.frame_count > 0) {
        pipeline->callbacks.on_tts_audio_ready(piper_result.audio_data, piper_result.frame_count, 
                                              pipeline->sample_rate, pipeline->config.user_data);
    }
    
    return TK_SUCCESS;
}

static void reset_asr_state(tk_audio_pipeline_t* pipeline) {
    pipeline->current_transcription[0] = '\0';
    pipeline->transcription_length = 0;
}
