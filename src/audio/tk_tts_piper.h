/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_tts_piper.h
 *
 * This header file defines the public API for the Piper TTS (Text-to-Speech) module.
 * This module provides a high-level interface to the Piper library, abstracting away
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

#ifndef TRACKIELLM_AUDIO_TK_TTS_PIPER_H
#define TRACKIELLM_AUDIO_TK_TTS_PIPER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "internal_tools/tk_file_manager.h"

// Forward-declare the primary TTS context as an opaque type.
typedef struct tk_tts_piper_context_s tk_tts_piper_context_t;

/**
 * @struct tk_tts_piper_voice_params_t
 * @brief Voice parameters for TTS synthesis.
 */
typedef struct {
    int speaker_id;        /**< Speaker ID for multi-speaker models (0 for single speaker). */
    float length_scale;    /**< Controls speech speed (1.0 is normal, <1.0 is faster, >1.0 is slower). */
    float noise_scale;     /**< Controls pronunciation variability (0.667 is default). */
    float noise_w;         /**< Controls emotion variability (0.8 is default). */
} tk_tts_piper_voice_params_t;

/**
 * @struct tk_tts_piper_config_t
 * @brief Configuration for initializing the Piper TTS module.
 */
typedef struct {
    tk_path_t* model_path;         /**< Path to the Piper ONNX model file. */
    tk_path_t* config_path;        /**< Path to the Piper JSON config file. */
    const char* language;          /**< Language code (e.g., "en", "pt") for TTS. */
    uint32_t    sample_rate;       /**< Audio sample rate (typically 22050 for Piper). */
    void*       user_data;         /**< Opaque pointer passed to all callbacks. */
    
    // Voice parameters
    tk_tts_piper_voice_params_t voice_params; /**< Voice parameters for synthesis. */
    
    // Performance parameters
    int         n_threads;         /**< Number of CPU threads to use for inference. */
    size_t      audio_buffer_size; /**< Size of internal audio buffer in samples. */
} tk_tts_piper_config_t;

/**
 * @struct tk_tts_piper_audio_chunk_t
 * @brief Represents a chunk of synthesized audio data.
 */
typedef struct {
    int16_t*    audio_data;        /**< Pointer to the raw PCM audio data (16-bit signed). */
    size_t      frame_count;       /**< Number of audio frames in the buffer. */
    uint32_t    sample_rate;       /**< Sample rate of the audio data. */
    bool        is_last_chunk;     /**< True if this is the final chunk of audio. */
} tk_tts_piper_audio_chunk_t;

/**
 * @typedef tk_tts_piper_audio_callback_t
 * @brief Callback function for delivering synthesized audio chunks.
 * @param chunk Pointer to the audio chunk structure.
 * @param user_data The opaque user data pointer from the configuration.
 */
typedef void (*tk_tts_piper_audio_callback_t)(
    const tk_tts_piper_audio_chunk_t* chunk,
    void* user_data
);

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Context Lifecycle Management
//------------------------------------------------------------------------------

/**
 * @brief Creates and initializes a new Piper TTS context.
 *
 * This function loads the Piper model and initializes the inference engine.
 * This can be a time-consuming operation.
 *
 * @param[out] out_context A pointer to receive the address of the new context.
 * @param[in] config A pointer to the configuration structure.
 *
 * @return TK_SUCCESS on successful creation.
 * @return TK_ERROR_INVALID_ARGUMENT if any required pointers are NULL or config is invalid.
 * @return TK_ERROR_OUT_OF_MEMORY if memory allocation fails.
 * @return TK_ERROR_MODEL_LOAD_FAILED if the Piper model cannot be loaded.
 */
TK_NODISCARD tk_error_code_t tk_tts_piper_create(
    tk_tts_piper_context_t** out_context,
    const tk_tts_piper_config_t* config
);

/**
 * @brief Destroys a Piper TTS context and frees all associated resources.
 *
 * @param[in,out] context A pointer to the context to be destroyed.
 */
void tk_tts_piper_destroy(tk_tts_piper_context_t** context);

//------------------------------------------------------------------------------
// TTS Processing Functions
//------------------------------------------------------------------------------

/**
 * @brief Synthesizes text into speech.
 *
 * This function performs Piper inference on the provided text and generates audio.
 * The audio is delivered through a callback mechanism to support streaming output.
 *
 * @param[in] context The Piper TTS context.
 * @param[in] text The UTF-8 encoded text to be synthesized.
 * @param[in] callback Function to call with each audio chunk.
 * @param[in] user_data User data to pass to the callback.
 *
 * @return TK_SUCCESS on successful synthesis.
 * @return TK_ERROR_INVALID_ARGUMENT if any pointers are NULL.
 * @return TK_ERROR_MODEL_INFERENCE_FAILED if the Piper inference fails.
 */
TK_NODISCARD tk_error_code_t tk_tts_piper_synthesize(
    tk_tts_piper_context_t* context,
    const char* text,
    tk_tts_piper_audio_callback_t callback,
    void* user_data
);

/**
 * @brief Synthesizes text into speech and returns all audio in one buffer.
 *
 * This function performs Piper inference on the provided text and generates audio.
 * All audio is accumulated in a single buffer and returned.
 *
 * @param[in] context The Piper TTS context.
 * @param[in] text The UTF-8 encoded text to be synthesized.
 * @param[out] out_audio_data Pointer to receive the audio data buffer (must be freed by caller).
 * @param[out] out_frame_count Pointer to receive the number of audio frames.
 *
 * @return TK_SUCCESS on successful synthesis.
 * @return TK_ERROR_INVALID_ARGUMENT if any pointers are NULL.
 * @return TK_ERROR_OUT_OF_MEMORY if memory allocation fails.
 * @return TK_ERROR_MODEL_INFERENCE_FAILED if the Piper inference fails.
 */
TK_NODISCARD tk_error_code_t tk_tts_piper_synthesize_to_buffer(
    tk_tts_piper_context_t* context,
    const char* text,
    int16_t** out_audio_data,
    size_t* out_frame_count
);

//------------------------------------------------------------------------------
// Model and State Management
//------------------------------------------------------------------------------

/**
 * @brief Updates the voice parameters for the Piper context.
 *
 * This function allows changing voice characteristics dynamically.
 *
 * @param[in] context The Piper TTS context.
 * @param[in] params The new voice parameters.
 *
 * @return TK_SUCCESS on successful update.
 * @return TK_ERROR_INVALID_ARGUMENT if context or params is NULL.
 */
TK_NODISCARD tk_error_code_t tk_tts_piper_set_voice_params(
    tk_tts_piper_context_t* context,
    const tk_tts_piper_voice_params_t* params
);

/**
 * @brief Gets the current voice parameters from the Piper context.
 *
 * @param[in] context The Piper TTS context.
 * @param[out] out_params Pointer to receive the current voice parameters.
 *
 * @return TK_SUCCESS on successful retrieval.
 * @return TK_ERROR_INVALID_ARGUMENT if context or out_params is NULL.
 */
TK_NODISCARD tk_error_code_t tk_tts_piper_get_voice_params(
    tk_tts_piper_context_t* context,
    tk_tts_piper_voice_params_t* out_params
);

/**
 * @brief Gets information about the loaded Piper model.
 *
 * @param[in] context The Piper TTS context.
 * @param[out] out_sample_rate Pointer to receive the model's sample rate.
 * @param[out] out_num_speakers Pointer to receive the number of speakers in the model.
 *
 * @return TK_SUCCESS on successful retrieval.
 * @return TK_ERROR_INVALID_ARGUMENT if context is NULL.
 */
TK_NODISCARD tk_error_code_t tk_tts_piper_get_model_info(
    tk_tts_piper_context_t* context,
    uint32_t* out_sample_rate,
    int* out_num_speakers
);

#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_AUDIO_TK_TTS_PIPER_H
