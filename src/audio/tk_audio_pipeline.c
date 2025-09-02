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
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "tk_audio_pipeline.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdatomic.h>
#include <time.h>

// Include headers for third-party libraries
#include "tk_vad_silero.h"
#include "tk_asr_whisper.h"
#include "tk_tts_piper.h"
#include "utils/tk_logging.h"
#include "utils/tk_error_handling.h"

// Internal constants
#define TK_AUDIO_PIPELINE_INTERNAL_BUFFER_SIZE (16384) // 16KB internal buffer
#define TK_AUDIO_PIPELINE_VAD_WINDOW_SIZE_MS (32)     // VAD window size in milliseconds
#define TK_AUDIO_PIPELINE_VAD_OVERLAP_SIZE_MS (16)    // Overlap between VAD windows
#define TK_AUDIO_PIPELINE_MAX_TRANSCRIPTION_LENGTH (1024) // Max chars for transcription
#define TK_AUDIO_PIPELINE_MAX_TTS_QUEUE_SIZE (16)     // Max TTS requests in queue

/**
 * @struct tk_tts_request_queue_item_t
 * @brief Represents a queued TTS request
 */
typedef struct {
    char* text;
    bool  is_processing;
} tk_tts_request_queue_item_t;

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
    tk_vad_silero_context_t*   vad_context;
    tk_asr_whisper_context_t*  whisper_context;
    tk_tts_piper_context_t*    piper_context;

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
    int16_t*                   asr_audio_buffer;
    size_t                     asr_audio_buffer_size;
    size_t                     asr_audio_buffer_capacity;

    // TTS queue
    tk_tts_request_queue_item_t tts_queue[TK_AUDIO_PIPELINE_MAX_TTS_QUEUE_SIZE];
    size_t                     tts_queue_head;
    size_t                     tts_queue_tail;
    size_t                     tts_queue_size;

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
static void vad_event_callback(tk_vad_silero_event_e event, void* user_data);
static void tts_audio_callback(const tk_tts_piper_audio_chunk_t* chunk, void* user_data);
static tk_error_code_t enqueue_tts_request(tk_audio_pipeline_t* pipeline, const char* text);
static tk_error_code_t process_next_tts_request(tk_audio_pipeline_t* pipeline);

//------------------------------------------------------------------------------
// Pipeline Lifecycle Management
//------------------------------------------------------------------------------

TK_NODISCARD tk_error_code_t tk_audio_pipeline_create(
    tk_audio_pipeline_t** out_pipeline, 
    const tk_audio_pipeline_config_t* config, 
    tk_audio_callbacks_t callbacks
) {
    if (!out_pipeline || !config) {
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

    // Initialize ASR audio buffer (30 seconds at sample rate)
    pipeline->asr_audio_buffer_capacity = pipeline->sample_rate * 30;
    pipeline->asr_audio_buffer = calloc(pipeline->asr_audio_buffer_capacity, sizeof(int16_t));
    if (!pipeline->asr_audio_buffer) {
        free(pipeline);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    pipeline->asr_audio_buffer_size = 0;

    // Initialize models
    // Load Silero VAD model
    tk_vad_silero_config_t vad_config = {0};
    vad_config.model_path = config->vad_model_path;
    vad_config.sample_rate = pipeline->sample_rate;
    vad_config.threshold = config->vad_speech_probability_threshold;
    vad_config.min_silence_duration_ms = config->vad_silence_threshold_ms;
    vad_config.min_speech_duration_ms = 250.0f; // Default value
    vad_config.speech_pad_ms = 30.0f; // Default value
    
    tk_error_code_t result = tk_vad_silero_create(&pipeline->vad_context, &vad_config);
    if (result != TK_SUCCESS) {
        TK_LOG_ERROR("Failed to initialize Silero VAD: %d", result);
        free(pipeline->asr_audio_buffer);
        free(pipeline);
        return result;
    }

    // Load Whisper ASR model
    tk_asr_whisper_config_t whisper_config = {0};
    whisper_config.model_path = config->asr_model_path;
    whisper_config.language = config->user_language;
    whisper_config.translate_to_en = false; // Default value
    whisper_config.sample_rate = pipeline->sample_rate;
    whisper_config.user_data = config->user_data;
    whisper_config.n_threads = 4; // Default value
    whisper_config.max_context = 16384; // Default value
    whisper_config.word_threshold = 0.1f; // Default value
    
    result = tk_asr_whisper_create(&pipeline->whisper_context, &whisper_config);
    if (result != TK_SUCCESS) {
        TK_LOG_ERROR("Failed to initialize Whisper ASR: %d", result);
        tk_vad_silero_destroy(&pipeline->vad_context);
        free(pipeline->asr_audio_buffer);
        free(pipeline);
        return result;
    }

    // Load Piper TTS model
    tk_tts_piper_config_t piper_config = {0};
    piper_config.model_path = config->tts_model_path; // Assuming this is added to config
    piper_config.config_path = config->tts_config_path; // Assuming this is added to config
    piper_config.language = config->user_language;
    piper_config.sample_rate = 22050; // Default Piper sample rate
    piper_config.user_data = config->user_data;
    piper_config.voice_params.speaker_id = 0; // Default speaker
    piper_config.voice_params.length_scale = 1.0f; // Normal speed
    piper_config.voice_params.noise_scale = 0.667f; // Default value
    piper_config.voice_params.noise_w = 0.8f; // Default value
    piper_config.n_threads = 4; // Default value
    piper_config.audio_buffer_size = 44100 * 10; // 10 seconds at 44.1kHz
    
    result = tk_tts_piper_create(&pipeline->piper_context, &piper_config);
    if (result != TK_SUCCESS) {
        TK_LOG_ERROR("Failed to initialize Piper TTS: %d", result);
        tk_asr_whisper_destroy(&pipeline->whisper_context);
        tk_vad_silero_destroy(&pipeline->vad_context);
        free(pipeline->asr_audio_buffer);
        free(pipeline);
        return result;
    }

    // Initialize buffers and state
    pipeline->input_buffer_head = 0;
    pipeline->input_buffer_tail = 0;
    atomic_store(&pipeline->input_buffer_full, false);
    reset_asr_state(pipeline);

    // Initialize TTS queue
    pipeline->tts_queue_head = 0;
    pipeline->tts_queue_tail = 0;
    pipeline->tts_queue_size = 0;

    // Initialize threading primitives
    if (mtx_init(&pipeline->worker_mutex, mtx_plain) != thrd_success) {
        tk_tts_piper_destroy(&pipeline->piper_context);
        tk_asr_whisper_destroy(&pipeline->whisper_context);
        tk_vad_silero_destroy(&pipeline->vad_context);
        free(pipeline->asr_audio_buffer);
        free(pipeline);
        return TK_ERROR_INTERNAL;
    }

    if (cnd_init(&pipeline->worker_cond) != thrd_success) {
        mtx_destroy(&pipeline->worker_mutex);
        tk_tts_piper_destroy(&pipeline->piper_context);
        tk_asr_whisper_destroy(&pipeline->whisper_context);
        tk_vad_silero_destroy(&pipeline->vad_context);
        free(pipeline->asr_audio_buffer);
        free(pipeline);
        return TK_ERROR_INTERNAL;
    }

    // Start worker thread
    atomic_store(&pipeline->worker_thread_running, true);
    if (thrd_create(&pipeline->worker_thread, worker_thread_func, pipeline) != thrd_success) {
        cnd_destroy(&pipeline->worker_cond);
        mtx_destroy(&pipeline->worker_mutex);
        tk_tts_piper_destroy(&pipeline->piper_context);
        tk_asr_whisper_destroy(&pipeline->whisper_context);
        tk_vad_silero_destroy(&pipeline->vad_context);
        free(pipeline->asr_audio_buffer);
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
    tk_tts_piper_destroy(&p->piper_context);
    tk_asr_whisper_destroy(&p->whisper_context);
    tk_vad_silero_destroy(&p->vad_context);

    // Clean up ASR audio buffer
    if (p->asr_audio_buffer) {
        free(p->asr_audio_buffer);
    }

    // Clean up TTS queue items
    size_t index = p->tts_queue_head;
    for (size_t i = 0; i < p->tts_queue_size; i++) {
        if (p->tts_queue[index].text) {
            free(p->tts_queue[index].text);
        }
        index = (index + 1) % TK_AUDIO_PIPELINE_MAX_TTS_QUEUE_SIZE;
    }

    cnd_destroy(&p->worker_cond);
    mtx_destroy(&p->worker_mutex);

    free(p);
    *pipeline = NULL;
}

//------------------------------------------------------------------------------
// Core Data Flow and Control Functions
//------------------------------------------------------------------------------

TK_NODISCARD tk_error_code_t tk_audio_pipeline_process_chunk(
    tk_audio_pipeline_t* pipeline, 
    const int16_t* audio_chunk, 
    size_t frame_count
) {
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

TK_NODISCARD tk_error_code_t tk_audio_pipeline_synthesize_text(
    tk_audio_pipeline_t* pipeline, 
    const char* text_to_speak
) {
    if (!pipeline || !text_to_speak) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    // Queue TTS request for worker thread
    return enqueue_tts_request(pipeline, text_to_speak);
}

TK_NODISCARD tk_error_code_t tk_audio_pipeline_force_transcription_end(
    tk_audio_pipeline_t* pipeline
) {
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
        while (pipeline->input_buffer_head == pipeline->input_buffer_tail && 
               pipeline->tts_queue_size == 0 && 
               atomic_load(&pipeline->worker_thread_running)) {
            cnd_wait(&pipeline->worker_cond, &pipeline->worker_mutex);
        }

        if (!atomic_load(&pipeline->worker_thread_running)) {
            mtx_unlock(&pipeline->worker_mutex);
            break;
        }

        // Process available audio data
        size_t frames_to_process = (pipeline->input_buffer_head - pipeline->input_buffer_tail + TK_AUDIO_PIPELINE_INTERNAL_BUFFER_SIZE) % TK_AUDIO_PIPELINE_INTERNAL_BUFFER_SIZE;
        
        if (frames_to_process > 0) {
            // Process audio in chunks of frame_size
            size_t chunk_size = (frames_to_process > pipeline->frame_size) ? pipeline->frame_size : frames_to_process;
            int16_t audio_chunk[chunk_size];
            
            // Copy chunk from circular buffer
            for (size_t i = 0; i < chunk_size; i++) {
                audio_chunk[i] = pipeline->input_buffer[pipeline->input_buffer_tail];
                pipeline->input_buffer_tail = (pipeline->input_buffer_tail + 1) % TK_AUDIO_PIPELINE_INTERNAL_BUFFER_SIZE;
            }
            
            // Unlock mutex during processing to avoid blocking audio input
            mtx_unlock(&pipeline->worker_mutex);
            
            // Process VAD
            process_vad(pipeline, audio_chunk, chunk_size);
            
            // Re-lock mutex for next iteration
            if (mtx_lock(&pipeline->worker_mutex) != thrd_success) {
                break;
            }
        }

        // Process TTS queue if there are items
        if (pipeline->tts_queue_size > 0) {
            // Process next TTS request
            process_next_tts_request(pipeline);
        }

        mtx_unlock(&pipeline->worker_mutex);
    }

    return 0;
}

static tk_error_code_t process_vad(
    tk_audio_pipeline_t* pipeline, 
    const int16_t* audio_data, 
    size_t frame_count
) {
    // Run VAD inference with event callback
    tk_error_code_t result = tk_vad_silero_process_audio_with_events(
        pipeline->vad_context,
        audio_data,
        frame_count,
        vad_event_callback,
        pipeline
    );
    
    if (result != TK_SUCCESS) {
        TK_LOG_ERROR("VAD processing failed: %d", result);
        return result;
    }
    
    // Get current VAD state
    tk_vad_silero_state_t vad_state;
    result = tk_vad_silero_get_state(pipeline->vad_context, &vad_state);
    if (result != TK_SUCCESS) {
        return result;
    }
    
    // Update pipeline state
    pipeline->is_speech_active = vad_state.is_speech_active;
    
    // If speech is active, accumulate audio for ASR
    if (pipeline->is_speech_active) {
        // Check if we have enough space in ASR buffer
        if (pipeline->asr_audio_buffer_size + frame_count > pipeline->asr_audio_buffer_capacity) {
            // Process current buffer as partial result
            process_asr(pipeline, NULL, 0, false);
        }
        
        // Append audio data to ASR buffer
        memcpy(
            pipeline->asr_audio_buffer + pipeline->asr_audio_buffer_size,
            audio_data,
            frame_count * sizeof(int16_t)
        );
        pipeline->asr_audio_buffer_size += frame_count;
    }
    
    return TK_SUCCESS;
}

static tk_error_code_t process_asr(
    tk_audio_pipeline_t* pipeline, 
    const int16_t* audio_data, 
    size_t frame_count, 
    bool is_final
) {
    // If this is a final processing, use accumulated audio
    if (is_final && pipeline->asr_audio_buffer_size > 0) {
        audio_data = pipeline->asr_audio_buffer;
        frame_count = pipeline->asr_audio_buffer_size;
    }
    
    // If we have no audio data, nothing to process
    if (!audio_data || frame_count == 0) {
        return TK_SUCCESS;
    }
    
    // Run ASR inference
    tk_asr_whisper_result_t* whisper_result = NULL;
    tk_error_code_t result = tk_asr_whisper_process_audio(
        pipeline->whisper_context,
        audio_data,
        frame_count,
        is_final,
        &whisper_result
    );
    
    if (result != TK_SUCCESS) {
        TK_LOG_ERROR("ASR processing failed: %d", result);
        return result;
    }
    
    // Update transcription
    if (whisper_result && whisper_result->text && strlen(whisper_result->text) > 0) {
        // Append or replace transcription based on is_final flag
        if (is_final) {
            strncpy(pipeline->current_transcription, whisper_result->text, TK_AUDIO_PIPELINE_MAX_TRANSCRIPTION_LENGTH - 1);
            pipeline->current_transcription[TK_AUDIO_PIPELINE_MAX_TRANSCRIPTION_LENGTH - 1] = '\0';
            pipeline->transcription_length = strlen(pipeline->current_transcription);
        } else {
            // For partial results, we might want to show them incrementally
            // This is a simplified approach
            strncat(pipeline->current_transcription, whisper_result->text, 
                    TK_AUDIO_PIPELINE_MAX_TRANSCRIPTION_LENGTH - pipeline->transcription_length - 1);
            pipeline->transcription_length = strlen(pipeline->current_transcription);
        }
        
        // Notify callback
        if (pipeline->callbacks.on_transcription) {
            tk_transcription_t transcription = {0};
            transcription.text = pipeline->current_transcription;
            transcription.is_final = is_final;
            transcription.confidence = whisper_result->confidence;
            pipeline->callbacks.on_transcription(&transcription, pipeline->config.user_data);
        }
        
        // If final, reset transcription buffer
        if (is_final) {
            reset_asr_state(pipeline);
        }
    }
    
    // Free ASR result
    if (whisper_result) {
        tk_asr_whisper_free_result(&whisper_result);
    }
    
    // If this was final processing, reset ASR buffer
    if (is_final) {
        pipeline->asr_audio_buffer_size = 0;
    }
    
    return TK_SUCCESS;
}

static tk_error_code_t process_tts(tk_audio_pipeline_t* pipeline, const char* text) {
    if (!pipeline || !text) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Run TTS synthesis with audio callback
    tk_error_code_t result = tk_tts_piper_synthesize(
        pipeline->piper_context,
        text,
        tts_audio_callback,
        pipeline
    );
    
    if (result != TK_SUCCESS) {
        TK_LOG_ERROR("TTS synthesis failed: %d", result);
        return result;
    }
    
    return TK_SUCCESS;
}

static void reset_asr_state(tk_audio_pipeline_t* pipeline) {
    pipeline->current_transcription[0] = '\0';
    pipeline->transcription_length = 0;
}

static void vad_event_callback(tk_vad_silero_event_e event, void* user_data) {
    tk_audio_pipeline_t* pipeline = (tk_audio_pipeline_t*)user_data;
    
    if (!pipeline) return;
    
    // Notify callback
    if (pipeline->callbacks.on_vad_event) {
        pipeline->callbacks.on_vad_event(event, pipeline->config.user_data);
    }
    
    // Handle speech start/end events
    switch (event) {
        case TK_VAD_EVENT_SPEECH_STARTED:
            // Reset ASR state when speech starts
            reset_asr_state(pipeline);
            break;
            
        case TK_VAD_EVENT_SPEECH_ENDED:
            // Process accumulated audio as final transcription
            if (pipeline->asr_audio_buffer_size > 0) {
                process_asr(pipeline, NULL, 0, true);
            }
            break;
    }
}

static void tts_audio_callback(const tk_tts_piper_audio_chunk_t* chunk, void* user_data) {
    tk_audio_pipeline_t* pipeline = (tk_audio_pipeline_t*)user_data;
    
    if (!pipeline || !chunk) return;
    
    // Notify callback with synthesized audio
    if (pipeline->callbacks.on_tts_audio_ready) {
        pipeline->callbacks.on_tts_audio_ready(
            chunk->audio_data,
            chunk->frame_count,
            chunk->sample_rate,
            pipeline->config.user_data
        );
    }
}

static tk_error_code_t enqueue_tts_request(tk_audio_pipeline_t* pipeline, const char* text) {
    if (!pipeline || !text) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Lock mutex to protect queue access
    if (mtx_lock(&pipeline->worker_mutex) != thrd_success) {
        return TK_ERROR_INTERNAL;
    }
    
    // Check if queue is full
    if (pipeline->tts_queue_size >= TK_AUDIO_PIPELINE_MAX_TTS_QUEUE_SIZE) {
        mtx_unlock(&pipeline->worker_mutex);
        return TK_ERROR_BUFFER_TOO_SMALL;
    }
    
    // Allocate and copy text
    char* text_copy = malloc(strlen(text) + 1);
    if (!text_copy) {
        mtx_unlock(&pipeline->worker_mutex);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    strcpy(text_copy, text);
    
    // Add to queue
    pipeline->tts_queue[pipeline->tts_queue_tail].text = text_copy;
    pipeline->tts_queue[pipeline->tts_queue_tail].is_processing = false;
    pipeline->tts_queue_tail = (pipeline->tts_queue_tail + 1) % TK_AUDIO_PIPELINE_MAX_TTS_QUEUE_SIZE;
    pipeline->tts_queue_size++;
    
    // Unlock mutex
    mtx_unlock(&pipeline->worker_mutex);
    
    // Signal worker thread that new TTS request is available
    cnd_signal(&pipeline->worker_cond);
    
    return TK_SUCCESS;
}

static tk_error_code_t process_next_tts_request(tk_audio_pipeline_t* pipeline) {
    if (!pipeline || pipeline->tts_queue_size == 0) {
        return TK_SUCCESS;
    }
    
    // Get the next TTS request
    tk_tts_request_queue_item_t* item = &pipeline->tts_queue[pipeline->tts_queue_head];
    
    // If already processing, skip
    if (item->is_processing) {
        return TK_SUCCESS;
    }
    
    // Mark as processing
    item->is_processing = true;
    
    // Process TTS
    tk_error_code_t result = process_tts(pipeline, item->text);
    
    // Remove from queue
    free(item->text);
    item->text = NULL;
    pipeline->tts_queue_head = (pipeline->tts_queue_head + 1) % TK_AUDIO_PIPELINE_MAX_TTS_QUEUE_SIZE;
    pipeline->tts_queue_size--;
    
    return result;
}
