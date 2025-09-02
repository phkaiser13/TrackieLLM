/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_asr_whisper.h
 *
 * This header file defines the public API for the Whisper ASR (Automatic Speech Recognition)
 * module. This module provides a high-level interface to the Whisper.cpp library,
 * abstracting away the complexities of model loading, inference, and result processing.
 *
 * The implementation is designed for real-time operation in embedded environments,
 * with support for streaming audio input and incremental transcription updates.
 *
 * Dependencies:
 *   - whisper.cpp (https://github.com/ggerganov/whisper.cpp)
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#ifndef TRACKIELLM_AUDIO_TK_ASR_WHISPER_H
#define TRACKIELLM_AUDIO_TK_ASR_WHISPER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "internal_tools/tk_file_manager.h"

// Forward-declare the primary ASR context as an opaque type.
typedef struct tk_asr_whisper_context_s tk_asr_whisper_context_t;

/**
 * @struct tk_asr_whisper_config_t
 * @brief Configuration for initializing the Whisper ASR module.
 */
typedef struct {
    tk_path_t* model_path;        /**< Path to the Whisper GGML model file. */
    const char* language;         /**< Language code (e.g., "en", "pt") for ASR. */
    bool        translate_to_en;  /**< If true, translate non-English speech to English. */
    uint32_t    sample_rate;      /**< Audio sample rate (must be 16000 for Whisper). */
    void*       user_data;        /**< Opaque pointer passed to all callbacks. */
    
    // Performance parameters
    int         n_threads;        /**< Number of CPU threads to use for inference. */
    int         max_context;      /**< Maximum number of text context tokens to use. */
    float       word_threshold;   /**< Minimum probability for a word to be considered. */
} tk_asr_whisper_config_t;

/**
 * @struct tk_asr_whisper_result_t
 * @brief Holds the result of a speech recognition operation.
 */
typedef struct {
    char*  text;           /**< The transcribed text (UTF-8). Must be freed by caller. */
    size_t text_length;    /**< Length of the transcribed text. */
    float  confidence;     /**< Overall confidence score for the transcription (0.0 to 1.0). */
    bool   is_partial;     /**< True if this is a partial result, false if final. */
} tk_asr_whisper_result_t;

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Context Lifecycle Management
//------------------------------------------------------------------------------

/**
 * @brief Creates and initializes a new Whisper ASR context.
 *
 * This function loads the Whisper model and initializes the inference engine.
 * This can be a time-consuming operation.
 *
 * @param[out] out_context A pointer to receive the address of the new context.
 * @param[in] config A pointer to the configuration structure.
 *
 * @return TK_SUCCESS on successful creation.
 * @return TK_ERROR_INVALID_ARGUMENT if any required pointers are NULL or config is invalid.
 * @return TK_ERROR_OUT_OF_MEMORY if memory allocation fails.
 * @return TK_ERROR_MODEL_LOAD_FAILED if the Whisper model cannot be loaded.
 */
TK_NODISCARD tk_error_code_t tk_asr_whisper_create(
    tk_asr_whisper_context_t** out_context,
    const tk_asr_whisper_config_t* config
);

/**
 * @brief Destroys a Whisper ASR context and frees all associated resources.
 *
 * @param[in,out] context A pointer to the context to be destroyed.
 */
void tk_asr_whisper_destroy(tk_asr_whisper_context_t** context);

//------------------------------------------------------------------------------
// ASR Processing Functions
//------------------------------------------------------------------------------

/**
 * @brief Processes a chunk of raw audio data for speech recognition.
 *
 * This function performs Whisper inference on the provided audio data.
 * It can produce either partial or final results depending on the `is_final` parameter.
 *
 * @param[in] context The Whisper ASR context.
 * @param[in] audio_data Pointer to the raw audio data (16-bit signed mono PCM).
 * @param[in] frame_count The number of audio frames in the data.
 * @param[in] is_final If true, forces a final transcription; if false, allows partial results.
 * @param[out] out_result A pointer to receive the transcription result.
 *                        The caller assumes ownership of the result and must free it.
 *
 * @return TK_SUCCESS on successful processing.
 * @return TK_ERROR_INVALID_ARGUMENT if any pointers are NULL.
 * @return TK_ERROR_OUT_OF_MEMORY if memory allocation for the result fails.
 * @return TK_ERROR_MODEL_INFERENCE_FAILED if the Whisper inference fails.
 */
TK_NODISCARD tk_error_code_t tk_asr_whisper_process_audio(
    tk_asr_whisper_context_t* context,
    const int16_t* audio_data,
    size_t frame_count,
    bool is_final,
    tk_asr_whisper_result_t** out_result
);

/**
 * @brief Frees the memory allocated for an ASR result.
 *
 * @param[in,out] result A pointer to the result to be freed.
 */
void tk_asr_whisper_free_result(tk_asr_whisper_result_t** result);

//------------------------------------------------------------------------------
// Model and State Management
//------------------------------------------------------------------------------

/**
 * @brief Resets the internal state of the Whisper context.
 *
 * This function clears any buffered audio data and resets the transcription state.
 * It should be called when starting a new transcription session.
 *
 * @param[in] context The Whisper ASR context.
 *
 * @return TK_SUCCESS on successful reset.
 * @return TK_ERROR_INVALID_ARGUMENT if context is NULL.
 */
TK_NODISCARD tk_error_code_t tk_asr_whisper_reset(tk_asr_whisper_context_t* context);

/**
 * @brief Updates the language setting for the Whisper context.
 *
 * This function allows changing the language dynamically without reloading the model.
 *
 * @param[in] context The Whisper ASR context.
 * @param[in] language The new language code (e.g., "en", "pt").
 *
 * @return TK_SUCCESS on successful update.
 * @return TK_ERROR_INVALID_ARGUMENT if context or language is NULL.
 */
TK_NODISCARD tk_error_code_t tk_asr_whisper_set_language(
    tk_asr_whisper_context_t* context,
    const char* language
);

#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_AUDIO_TK_ASR_WHISPER_H
