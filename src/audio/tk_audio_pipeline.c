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
#include "tk_decision_engine.h" // Include for priority enum

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
#include "audio/tk_wake_word_porcupine.h" // Added for Wake Word
#include "audio/tk_sound_classifier.h" // Added for Sound Classifier
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
 * @brief Represents a queued TTS request with priority
 */
typedef struct {
    char* text;
    tk_response_priority_e priority; // Added priority field
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

    // Pipeline State
    atomic_int                 current_state; // Using atomic for thread-safe state checks

    // Audio parameters
    uint32_t                   sample_rate;
    uint32_t                   frame_size; // Number of samples per processing frame for VAD
    uint32_t                   porcupine_frame_length; // Number of samples for Porcupine

    // Models and engines
    tk_porcupine_context_t*    porcupine_context; // Wake Word engine
    tk_sound_classifier_context_t* sound_classifier_context; // Sound Classifier engine
    tk_vad_silero_context_t*   vad_context;
    tk_asr_whisper_context_t*  whisper_context;
    tk_tts_piper_context_t*    piper_context;

    // Internal buffers
    int16_t                    input_buffer[TK_AUDIO_PIPELINE_INTERNAL_BUFFER_SIZE];
    size_t                     input_buffer_head;
    size_t                     input_buffer_tail;
    atomic_bool                input_buffer_full;

    // Wake Word state
    int16_t*                   porcupine_buffer; // Buffer sized for one Porcupine frame
    size_t                     porcupine_buffer_size;

    // ASR state
    char                       current_transcription[TK_AUDIO_PIPELINE_MAX_TRANSCRIPTION_LENGTH];
    size_t                     transcription_length;
    bool                       is_speech_active;
    uint64_t                   last_speech_timestamp_ns;
    int16_t*                   asr_audio_buffer;
    size_t                     asr_audio_buffer_size;
    size_t                     asr_audio_buffer_capacity;

    // TTS queue with priority
    tk_tts_request_queue_item_t tts_queue[TK_AUDIO_PIPELINE_MAX_TTS_QUEUE_SIZE];
    size_t                     tts_queue_head;
    size_t                     tts_queue_tail;
    size_t                     tts_queue_size;

    // TTS interruption control
    atomic_bool                tts_interrupt_requested;
    tk_tts_request_queue_item_t* current_tts_item; // Currently processing item

    // Threading and synchronization
    thrd_t                     worker_thread;
    atomic_bool                worker_thread_running;
    cnd_t                      worker_cond;
    mtx_t                      worker_mutex;

    // State transition timing
    uint64_t                   state_transition_time_ns;
    bool                       vad_detected_speech_since_transition;

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
static tk_error_code_t enqueue_tts_request_with_priority(tk_audio_pipeline_t* pipeline, const char* text, tk_response_priority_e priority);
static tk_error_code_t process_next_tts_request(tk_audio_pipeline_t* pipeline);
static void interrupt_current_tts(tk_audio_pipeline_t* pipeline);
static int get_priority_value(tk_response_priority_e priority);

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

    // Initialize interruption control
    atomic_store(&pipeline->tts_interrupt_requested, false);
    pipeline->current_tts_item = NULL;

    // Initialize models
    // Load Porcupine Wake Word model
    tk_porcupine_config_t ww_config = {0};
    ww_config.access_key_path = NULL; // Assuming access key is hardcoded in wrapper
    ww_config.model_path = config->ww_model_path;
    ww_config.num_keywords = 1;
    ww_config.keyword_paths = &config->ww_keyword_path;
    ww_config.sensitivities = &config->ww_sensitivity;

    tk_error_code_t result = tk_porcupine_create(&pipeline->porcupine_context, &ww_config);
    if (result != TK_SUCCESS) {
        TK_LOG_ERROR("Failed to initialize Porcupine Wake Word engine: %d", result);
        free(pipeline->asr_audio_buffer);
        free(pipeline);
        return result;
    }

    // Load Sound Classifier model
    if (config->sc_model_path) {
        tk_sound_classifier_config_t sc_config = {0};
        sc_config.model_path = config->sc_model_path;
        sc_config.sample_rate = pipeline->sample_rate;
        sc_config.n_threads = 2; // Default value
        sc_config.detection_threshold = 0.7f; // Default value, can be tuned
        result = tk_sound_classifier_create(&pipeline->sound_classifier_context, &sc_config);
        if (result != TK_SUCCESS) {
            TK_LOG_ERROR("Failed to initialize Sound Classifier engine: %d", result);
            tk_porcupine_destroy(&pipeline->porcupine_context);
            free(pipeline->asr_audio_buffer);
            free(pipeline);
            return result;
        }
    }
    pipeline->porcupine_frame_length = tk_porcupine_get_frame_length(pipeline->porcupine_context);
    pipeline->porcupine_buffer = malloc(pipeline->porcupine_frame_length * sizeof(int16_t));


    // Load Silero VAD model
    tk_vad_silero_config_t vad_config = {0};
    vad_config.model_path = config->vad_model_path;
    vad_config.sample_rate = pipeline->sample_rate;
    vad_config.threshold = config->vad_speech_probability_threshold;
    vad_config.min_silence_duration_ms = config->vad_silence_threshold_ms;
    
    result = tk_vad_silero_create(&pipeline->vad_context, &vad_config);
    if (result != TK_SUCCESS) {
        TK_LOG_ERROR("Failed to initialize Silero VAD: %d", result);
        tk_porcupine_destroy(&pipeline->porcupine_context);
        free(pipeline->porcupine_buffer);
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
    atomic_store(&pipeline->current_state, TK_PIPELINE_STATE_AWAITING_WAKE_WORD);
    pipeline->porcupine_buffer_size = 0;
    pipeline->input_buffer_head = 0;
    pipeline->input_buffer_tail = 0;
    atomic_store(&pipeline->input_buffer_full, false);
    reset_asr_state(pipeline);

    // Initialize TTS queue
    pipeline->tts_queue_head = 0;
    pipeline->tts_queue_tail = 0;
    pipeline->tts_queue_size = 0;
    pipeline->state_transition_time_ns = 0;
    pipeline->vad_detected_speech_since_transition = false;

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
    tk_porcupine_destroy(&p->porcupine_context);
    tk_sound_classifier_destroy(&p->sound_classifier_context);

    // Clean up buffers
    if (p->porcupine_buffer) {
        free(p->porcupine_buffer);
    }
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
    const char* text_to_speak,
    tk_response_priority_e priority // Added priority parameter
) {
    if (!pipeline || !text_to_speak) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    // Queue TTS request with priority for worker thread
    return enqueue_tts_request_with_priority(pipeline, text_to_speak, priority);
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

tk_pipeline_state_e tk_audio_pipeline_get_state(tk_audio_pipeline_t* pipeline) {
    if (!pipeline) {
        return TK_PIPELINE_STATE_IDLE; // Or some other error indicator
    }
    return (tk_pipeline_state_e)atomic_load(&pipeline->current_state);
}

//------------------------------------------------------------------------------
// Internal Helper Functions
//------------------------------------------------------------------------------

static void process_audio_for_passive_listening(tk_audio_pipeline_t* pipeline) {
    size_t available_frames = (pipeline->input_buffer_head - pipeline->input_buffer_tail + TK_AUDIO_PIPELINE_INTERNAL_BUFFER_SIZE) % TK_AUDIO_PIPELINE_INTERNAL_BUFFER_SIZE;

    // We process audio in chunks matching Porcupine's frame length
    while (available_frames >= pipeline->porcupine_frame_length) {
        // Copy a chunk for processing
        int16_t process_buffer[pipeline->porcupine_frame_length];
        for (size_t i = 0; i < pipeline->porcupine_frame_length; i++) {
            process_buffer[i] = pipeline->input_buffer[pipeline->input_buffer_tail];
            pipeline->input_buffer_tail = (pipeline->input_buffer_tail + 1) % TK_AUDIO_PIPELINE_INTERNAL_BUFFER_SIZE;
        }
        available_frames -= pipeline->porcupine_frame_length;

        // 1. Process for Wake Word
        int keyword_index = -1;
        tk_error_code_t ww_result = tk_porcupine_process(pipeline->porcupine_context, process_buffer, &keyword_index);
        if (ww_result == TK_SUCCESS && keyword_index != -1) {
            TK_LOG_INFO("Wake word detected! Keyword index: %d", keyword_index);
            atomic_store(&pipeline->current_state, TK_PIPELINE_STATE_LISTENING_FOR_COMMAND);

            struct timespec ts;
            timespec_get(&ts, TIME_UTC);
            pipeline->state_transition_time_ns = (uint64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
            pipeline->vad_detected_speech_since_transition = false;

            // Wake word detected, no need to process for ambient sound in this chunk
            return;
        }

        // 2. Process for Ambient Sound (if classifier exists)
        if (pipeline->sound_classifier_context) {
            tk_sound_detection_result_t sc_result;
            tk_error_code_t sc_err = tk_sound_classifier_process(
                pipeline->sound_classifier_context,
                process_buffer,
                pipeline->porcupine_frame_length,
                &sc_result
            );

            if (sc_err == TK_SUCCESS && sc_result.sound_class != TK_SOUND_UNKNOWN) {
                if (pipeline->callbacks.on_ambient_sound_detected) {
                    TK_LOG_INFO("Ambient sound detected: %d (Confidence: %.2f)", sc_result.sound_class, sc_result.confidence);
                    pipeline->callbacks.on_ambient_sound_detected(&sc_result, pipeline->config.user_data);
                }
            }
        }
    }
}

static void process_audio_for_vad(tk_audio_pipeline_t* pipeline) {
    size_t frames_to_process = (pipeline->input_buffer_head - pipeline->input_buffer_tail + TK_AUDIO_PIPELINE_INTERNAL_BUFFER_SIZE) % TK_AUDIO_PIPELINE_INTERNAL_BUFFER_SIZE;

    if (frames_to_process > 0) {
        size_t chunk_size = (frames_to_process > pipeline->frame_size) ? pipeline->frame_size : frames_to_process;
        int16_t audio_chunk[chunk_size];

        for (size_t i = 0; i < chunk_size; i++) {
            audio_chunk[i] = pipeline->input_buffer[pipeline->input_buffer_tail];
            pipeline->input_buffer_tail = (pipeline->input_buffer_tail + 1) % TK_AUDIO_PIPELINE_INTERNAL_BUFFER_SIZE;
        }

        mtx_unlock(&pipeline->worker_mutex);
        process_vad(pipeline, audio_chunk, chunk_size);
        if (mtx_lock(&pipeline->worker_mutex) != thrd_success) {
            atomic_store(&pipeline->worker_thread_running, false); // Stop thread on mutex error
        }
    }
}

static int worker_thread_func(void* arg) {
    tk_audio_pipeline_t* pipeline = (tk_audio_pipeline_t*)arg;
    const uint64_t listen_timeout_ns = 5 * 1000000000ULL; // 5 seconds

    while (atomic_load(&pipeline->worker_thread_running)) {
        if (mtx_lock(&pipeline->worker_mutex) != thrd_success) { continue; }

        while (pipeline->input_buffer_head == pipeline->input_buffer_tail &&
               pipeline->tts_queue_size == 0 &&
               atomic_load(&pipeline->worker_thread_running)) {
            cnd_wait(&pipeline->worker_cond, &pipeline->worker_mutex);
        }

        if (!atomic_load(&pipeline->worker_thread_running)) {
            mtx_unlock(&pipeline->worker_mutex);
            break;
        }

        // Main State Machine Logic
        tk_pipeline_state_e state = atomic_load(&pipeline->current_state);
        switch (state) {
            case TK_PIPELINE_STATE_AWAITING_WAKE_WORD:
                process_audio_for_passive_listening(pipeline);
                break;

            case TK_PIPELINE_STATE_LISTENING_FOR_COMMAND:
                {
                    // Check for listening timeout
                    struct timespec ts;
                    timespec_get(&ts, TIME_UTC);
                    uint64_t current_time_ns = (uint64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;

                    if (!pipeline->vad_detected_speech_since_transition &&
                        (current_time_ns - pipeline->state_transition_time_ns) > listen_timeout_ns) {

                        TK_LOG_INFO("Listening timeout. No speech detected. Returning to AWAITING_WAKE_WORD state.");
                        atomic_store(&pipeline->current_state, TK_PIPELINE_STATE_AWAITING_WAKE_WORD);
                    } else {
                        process_audio_for_vad(pipeline);
                    }
                }
                break;

            default:
                // Other states (TRANSCRIBING, etc.) are handled implicitly
                // by the VAD->ASR->TTS flow. We just need to make sure
                // we don't consume audio when not listening.
                break;
        }

        // Process TTS queue if there are items
        if (pipeline->tts_queue_size > 0) {
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
        
        // Log the final transcription for debugging/testing purposes
        if (is_final) {
            TK_LOG_INFO("Final transcription: \"%s\"", pipeline->current_transcription);
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
            TK_LOG_DEBUG("VAD: Speech started.");
            pipeline->vad_detected_speech_since_transition = true;
            reset_asr_state(pipeline);
            break;
            
        case TK_VAD_EVENT_SPEECH_ENDED:
            TK_LOG_DEBUG("VAD: Speech ended.");
            if (pipeline->asr_audio_buffer_size > 0) {
                TK_LOG_INFO("Transitioning to state: TRANSCRIBING");
                atomic_store(&pipeline->current_state, TK_PIPELINE_STATE_TRANSCRIBING);
                process_asr(pipeline, NULL, 0, true);

                // After transcription, go back to waiting for wake word
                TK_LOG_INFO("Transitioning to state: AWAITING_WAKE_WORD");
                atomic_store(&pipeline->current_state, TK_PIPELINE_STATE_AWAITING_WAKE_WORD);
            }
            break;
    }
}

static void tts_audio_callback(const tk_tts_piper_audio_chunk_t* chunk, void* user_data) {
    tk_audio_pipeline_t* pipeline = (tk_audio_pipeline_t*)user_data;
    
    if (!pipeline || !chunk) return;
    
    // Check if interruption was requested
    if (atomic_load(&pipeline->tts_interrupt_requested)) {
        // Skip this chunk if interruption is requested
        return;
    }
    
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

/**
 * @brief Get numeric value for priority (higher number = higher priority)
 */
static int get_priority_value(tk_response_priority_e priority) {
    switch (priority) {
        case TK_RESPONSE_PRIORITY_CRITICAL: return 4;
        case TK_RESPONSE_PRIORITY_HIGH:    return 3;
        case TK_RESPONSE_PRIORITY_NORMAL:  return 2;
        case TK_RESPONSE_PRIORITY_LOW:     return 1;
        default:                           return 0;
    }
}

/**
 * @brief Enqueue TTS request with priority-based insertion
 */
static tk_error_code_t enqueue_tts_request_with_priority(
    tk_audio_pipeline_t* pipeline, 
    const char* text, 
    tk_response_priority_e priority
) {
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
    
    // Find the correct position based on priority
    size_t insert_position = pipeline->tts_queue_tail;
    int new_priority_value = get_priority_value(priority);
    
    // If queue is not empty, find the correct insertion point
    if (pipeline->tts_queue_size > 0) {
        // Start from head and find where to insert
        size_t current_pos = pipeline->tts_queue_head;
        bool inserted = false;
        
        for (size_t i = 0; i < pipeline->tts_queue_size; i++) {
            int current_priority_value = get_priority_value(pipeline->tts_queue[current_pos].priority);
            
            // If new item has higher priority, insert here
            if (new_priority_value > current_priority_value) {
                insert_position = current_pos;
                inserted = true;
                break;
            }
            
            current_pos = (current_pos + 1) % TK_AUDIO_PIPELINE_MAX_TTS_QUEUE_SIZE;
        }
        
        // If not inserted, it goes at the end (lower priority than all existing)
        if (!inserted) {
            insert_position = (pipeline->tts_queue_head + pipeline->tts_queue_size) % TK_AUDIO_PIPELINE_MAX_TTS_QUEUE_SIZE;
        }
    } else {
        // Queue is empty, insert at head
        insert_position = pipeline->tts_queue_head;
    }
    
    // Shift items to make space if needed
    if (insert_position != pipeline->tts_queue_tail) {
        // Need to shift items
        size_t shift_pos = (pipeline->tts_queue_head + pipeline->tts_queue_size) % TK_AUDIO_PIPELINE_MAX_TTS_QUEUE_SIZE;
        while (shift_pos != insert_position) {
            size_t prev_pos = (shift_pos == 0) ? 
                (TK_AUDIO_PIPELINE_MAX_TTS_QUEUE_SIZE - 1) : (shift_pos - 1);
            pipeline->tts_queue[shift_pos] = pipeline->tts_queue[prev_pos];
            shift_pos = prev_pos;
        }
    }
    
    // Insert the new item
    pipeline->tts_queue[insert_position].text = text_copy;
    pipeline->tts_queue[insert_position].priority = priority;
    pipeline->tts_queue[insert_position].is_processing = false;
    
    // Update queue pointers and size
    if (pipeline->tts_queue_size == 0) {
        // First item
        pipeline->tts_queue_head = insert_position;
        pipeline->tts_queue_tail = (insert_position + 1) % TK_AUDIO_PIPELINE_MAX_TTS_QUEUE_SIZE;
    } else if (insert_position == pipeline->tts_queue_head) {
        // Inserted at head
        pipeline->tts_queue_head = (pipeline->tts_queue_head == 0) ? 
            (TK_AUDIO_PIPELINE_MAX_TTS_QUEUE_SIZE - 1) : (pipeline->tts_queue_head - 1);
    } else if (insert_position == pipeline->tts_queue_tail) {
        // Inserted at tail
        pipeline->tts_queue_tail = (pipeline->tts_queue_tail + 1) % TK_AUDIO_PIPELINE_MAX_TTS_QUEUE_SIZE;
    }
    
    pipeline->tts_queue_size++;
    
    // Check if we need to interrupt current TTS processing
    if (pipeline->current_tts_item && 
        get_priority_value(priority) > get_priority_value(pipeline->current_tts_item->priority)) {
        // New request has higher priority than current processing
        // Check if current is low/normal priority
        if (get_priority_value(pipeline->current_tts_item->priority) <= 2) { // LOW or NORMAL
            interrupt_current_tts(pipeline);
        }
    }
    
    // Unlock mutex
    mtx_unlock(&pipeline->worker_mutex);
    
    // Signal worker thread that new TTS request is available
    cnd_signal(&pipeline->worker_cond);
    
    return TK_SUCCESS;
}

/**
 * @brief Interrupt current TTS processing
 */
static void interrupt_current_tts(tk_audio_pipeline_t* pipeline) {
    TK_LOG_INFO("Interrupting current TTS playback due to higher priority request.");
    // Set interruption flag to stop feeding new audio chunks from the callback
    atomic_store(&pipeline->tts_interrupt_requested, true);

    // Signal the host application to stop its audio player immediately
    if (pipeline->callbacks.on_tts_interrupt) {
        pipeline->callbacks.on_tts_interrupt(pipeline->config.user_data);
    }

    // Note: We don't call tk_tts_piper_interrupt because piper is blocking
    // and doesn't have an interruption mechanism. The playback must be stopped
    // by the consumer of the audio chunks.
}

static tk_error_code_t process_next_tts_request(tk_audio_pipeline_t* pipeline) {
    if (!pipeline || pipeline->tts_queue_size == 0) {
        return TK_SUCCESS;
    }
    
    // Get the next TTS request (highest priority is at head)
    tk_tts_request_queue_item_t* item = &pipeline->tts_queue[pipeline->tts_queue_head];
    
    // If already processing, skip
    if (item->is_processing) {
        return TK_SUCCESS;
    }
    
    // Mark as processing and store reference
    item->is_processing = true;
    pipeline->current_tts_item = item;
    
    // Clear interruption flag before starting
    atomic_store(&pipeline->tts_interrupt_requested, false);
    
    // Process TTS
    tk_error_code_t result = process_tts(pipeline, item->text);
    
    // Remove from queue regardless of success/failure
    free(item->text);
    item->text = NULL;
    pipeline->tts_queue_head = (pipeline->tts_queue_head + 1) % TK_AUDIO_PIPELINE_MAX_TTS_QUEUE_SIZE;
    pipeline->tts_queue_size--;
    
    // Clear current item reference
    pipeline->current_tts_item = NULL;
    
    return result;
}

// --- MINIAUDIO IMPLEMENTATION ---
// Define the implementation of the miniaudio library in this source file.
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
// --------------------------------

// --- Standalone Utility Function Implementation ---

static void audio_playback_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount)
{
    int16_t* buffer = (int16_t*)pDevice->pUserData;
    ma_uint32 frames_to_write = frameCount;
    ma_uint32* p_frames_played = (ma_uint32*)((char*)buffer - sizeof(ma_uint32)); // User data trick

    if (*p_frames_played + frames_to_write > pDevice->playback.internalPeriodSizeInFrames) {
        frames_to_write = pDevice->playback.internalPeriodSizeInFrames - *p_frames_played;
    }

    memcpy(pOutput, buffer + *p_frames_played, frames_to_write * sizeof(int16_t));
    *p_frames_played += frames_to_write;

    (void)pInput;
}


TK_NODISCARD tk_error_code_t tk_audio_pipeline_say(
    const char* text,
    const char* model_path_str,
    const char* config_path_str
) {
    if (!text || !model_path_str || !config_path_str) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    tk_error_code_t final_result = TK_SUCCESS;

    // 1. Create Piper TTS context
    tk_tts_piper_context_t* tts_context = NULL;
    tk_path_t* model_path = tk_path_create(model_path_str);
    tk_path_t* config_path = tk_path_create(config_path_str);

    if (!model_path || !config_path) {
        if (model_path) tk_path_destroy(model_path);
        if (config_path) tk_path_destroy(config_path);
        return TK_ERROR_OUT_OF_MEMORY;
    }

    tk_tts_piper_config_t tts_config = {
        .model_path = model_path,
        .config_path = config_path,
        .n_threads = 2,
    };

    tk_error_code_t result = tk_tts_piper_create(&tts_context, &tts_config);
    tk_path_destroy(model_path);
    tk_path_destroy(config_path);

    if (result != TK_SUCCESS) {
        TK_LOG_ERROR("Failed to create temporary TTS context: %d", result);
        return result;
    }

    // 2. Synthesize audio to buffer
    int16_t* audio_buffer = NULL;
    size_t frame_count = 0;
    result = tk_tts_piper_synthesize_to_buffer(tts_context, text, &audio_buffer, &frame_count);

    if (result != TK_SUCCESS) {
        TK_LOG_ERROR("Failed to synthesize text for say(): %d", result);
        final_result = result;
        goto cleanup_tts;
    }

    if (audio_buffer == NULL || frame_count == 0) {
        TK_LOG_WARN("TTS synthesis produced no audio data.");
        final_result = TK_SUCCESS; // Not an error, just nothing to play
        goto cleanup_tts;
    }

    // 3. Play audio using miniaudio
    ma_device_config device_config;
    ma_device device;
    ma_uint32 frames_played = 0;

    // We use a small trick to pass both the buffer and a counter to the callback
    void* user_data_with_counter = malloc(frame_count * sizeof(int16_t) + sizeof(ma_uint32));
    if (!user_data_with_counter) {
        final_result = TK_ERROR_OUT_OF_MEMORY;
        goto cleanup_buffer;
    }
    *(ma_uint32*)user_data_with_counter = 0;
    memcpy((char*)user_data_with_counter + sizeof(ma_uint32), audio_buffer, frame_count * sizeof(int16_t));

    device_config = ma_device_config_init(ma_device_type_playback);
    device_config.playback.format   = ma_format_s16;
    device_config.playback.channels = 1;
    device_config.sampleRate        = 22050; // Piper default
    device_config.dataCallback      = audio_playback_callback;
    device_config.pUserData         = (char*)user_data_with_counter + sizeof(ma_uint32);

    if (ma_device_init(NULL, &device_config, &device) != MA_SUCCESS) {
        TK_LOG_ERROR("Failed to initialize miniaudio playback device.");
        final_result = TK_ERROR_EXTERNAL_LIBRARY_FAILED;
        free(user_data_with_counter);
        goto cleanup_buffer;
    }

    if (ma_device_start(&device) != MA_SUCCESS) {
        TK_LOG_ERROR("Failed to start miniaudio playback device.");
        ma_device_uninit(&device);
        free(user_data_with_counter);
        final_result = TK_ERROR_EXTERNAL_LIBRARY_FAILED;
        goto cleanup_buffer;
    }

    TK_LOG_INFO("Playing audio... Press Enter to stop.");
    // In a real app, we would wait for the callback to finish.
    // For this test, we can just sleep or wait for user input.
    // Here we'll just let it play out. In a test executable, you'd wait.
    // This is a blocking call in a simple implementation.
    // A better way is to check `ma_device_get_state`.
    // For now, we'll just sleep for the duration of the audio.
    float duration_sec = (float)frame_count / 22050.0f;
    struct timespec sleep_time = { (time_t)duration_sec, (long)((duration_sec - (time_t)duration_sec) * 1e9) };
    nanosleep(&sleep_time, NULL);


    ma_device_uninit(&device);
    free(user_data_with_counter);

cleanup_buffer:
    // 4. Free the audio buffer
    free(audio_buffer);

cleanup_tts:
    // 5. Destroy the TTS context
    tk_tts_piper_destroy(&tts_context);

    return final_result;
}
