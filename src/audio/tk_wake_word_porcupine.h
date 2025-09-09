/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_wake_word_porcupine.h
 *
 * This header file defines the public API for the Porcupine Wake Word Engine module.
 * This module provides a high-level interface to the Porcupine library, abstracting
 * away the complexities of model loading and processing.
 *
 * Dependencies:
 *   - Porcupine (https://github.com/Picovoice/Porcupine)
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#ifndef TRACKIELLM_AUDIO_TK_WAKE_WORD_PORCUPINE_H
#define TRACKIELLM_AUDIO_TK_WAKE_WORD_PORCUPINE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "internal_tools/tk_file_manager.h"

// Forward-declare the primary context as an opaque type.
typedef struct tk_porcupine_context_s tk_porcupine_context_t;

/**
 * @struct tk_porcupine_config_t
 * @brief Configuration for initializing the Porcupine wake word engine.
 */
typedef struct {
    tk_path_t* access_key_path;   /**< Path to the Porcupine access key file. */
    tk_path_t* model_path;        /**< Path to the Porcupine model file (.pv). */
    int        num_keywords;      /**< Number of keyword models to use. */
    tk_path_t** keyword_paths;    /**< Array of paths to keyword model files (.ppn). */
    float*     sensitivities;     /**< Array of sensitivities for each keyword (0.0 to 1.0). */
} tk_porcupine_config_t;


#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Context Lifecycle Management
//------------------------------------------------------------------------------

/**
 * @brief Creates and initializes a new Porcupine context.
 *
 * @param[out] out_context A pointer to receive the address of the new context.
 * @param[in] config A pointer to the configuration structure.
 *
 * @return TK_SUCCESS on successful creation.
 * @return TK_ERROR_INVALID_ARGUMENT if any required pointers are NULL.
 * @return TK_ERROR_OUT_OF_MEMORY if memory allocation fails.
 * @return TK_ERROR_MODEL_LOAD_FAILED if the Porcupine model cannot be loaded.
 */
TK_NODISCARD tk_error_code_t tk_porcupine_create(
    tk_porcupine_context_t** out_context,
    const tk_porcupine_config_t* config
);

/**
 * @brief Destroys a Porcupine context and frees all associated resources.
 *
 * @param[in,out] context A pointer to the context to be destroyed.
 */
void tk_porcupine_destroy(tk_porcupine_context_t** context);

//------------------------------------------------------------------------------
// Wake Word Processing
//------------------------------------------------------------------------------

/**
 * @brief Processes a chunk of raw audio data for wake word detection.
 *
 * @param[in] context The Porcupine context.
 * @param[in] audio_data Pointer to the raw audio data (16-bit signed mono PCM).
 *                       The number of frames must be equal to `pv_porcupine_frame_length()`.
 * @param[out] out_keyword_index The index of the detected keyword, or -1 if none detected.
 *
 * @return TK_SUCCESS on successful processing.
 * @return TK_ERROR_INVALID_ARGUMENT if context or audio_data is NULL.
 * @return TK_ERROR_EXTERNAL_LIBRARY_FAILED if the Porcupine processing fails.
 */
TK_NODISCARD tk_error_code_t tk_porcupine_process(
    tk_porcupine_context_t* context,
    const int16_t* audio_data,
    int* out_keyword_index
);

/**
 * @brief Gets the required number of audio frames per processing call.
 *
 * @param[in] context The Porcupine context.
 * @return The number of audio frames required by `tk_porcupine_process`.
 */
int tk_porcupine_get_frame_length(tk_porcupine_context_t* context);

/**
 * @brief Gets the audio sample rate required by the Porcupine model.
 *
 * @param[in] context The Porcupine context.
 * @return The required sample rate (e.g., 16000).
 */
int tk_porcupine_get_sample_rate(tk_porcupine_context_t* context);


#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_AUDIO_TK_WAKE_WORD_PORCUPINE_H
