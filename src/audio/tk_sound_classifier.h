/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_sound_classifier.h
*
* This header file defines the public API for the Ambient Sound Classifier module.
* This module uses a pre-trained audio classification model (e.g., YAMNet) in
* the ONNX format to detect and classify various background sounds from an
* audio stream. This enables the system to be aware of its environment,
* reacting to important sounds like alarms, sirens, or running water.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TRACKIELLM_AUDIO_TK_SOUND_CLASSIFIER_H
#define TRACKIELLM_AUDIO_TK_SOUND_CLASSIFIER_H

#include <stdbool.h>
#include <stdint.h>
#include "utils/tk_error_handling.h"
#include "internal_tools/tk_file_manager.h"

// Forward-declare the context as an opaque type
typedef struct tk_sound_classifier_context_s tk_sound_classifier_context_t;

/**
 * @enum tk_sound_class_e
 * @brief Defines the classes of ambient sounds that can be detected.
 * These should correspond to the output classes of the underlying ONNX model.
 */
typedef enum {
    TK_SOUND_UNKNOWN = 0,
    TK_SOUND_ALARM,
    TK_SOUND_SIREN,
    TK_SOUND_WATER_RUNNING,
    TK_SOUND_DOG_BARK,
    // Add other relevant sound classes here
    TK_SOUND_CLASS_COUNT // Keep this last for array sizing
} tk_sound_class_e;

/**
 * @struct tk_sound_classifier_config_t
 * @brief Configuration for initializing the sound classifier.
 */
typedef struct {
    tk_path_t* model_path;              /**< Path to the sound classification ONNX model. */
    uint32_t   sample_rate;             /**< Sample rate of the input audio (must match model requirements). */
    int        n_threads;               /**< Number of CPU threads for ONNX Runtime. */
    float      detection_threshold;     /**< Confidence threshold (0.0 to 1.0) to report a sound. */
} tk_sound_classifier_config_t;

/**
 * @struct tk_sound_detection_result_t
 * @brief Holds the result of a sound classification operation.
 */
typedef struct {
    tk_sound_class_e sound_class; /**< The most likely sound class detected. */
    float            confidence;  /**< The confidence score for the detected class. */
} tk_sound_detection_result_t;


#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a new sound classifier instance.
 * @param[out] out_context A pointer to receive the new context.
 * @param[in]  config      Configuration for the classifier.
 * @return TK_SUCCESS on success, or an error code on failure.
 */
TK_NODISCARD tk_error_code_t tk_sound_classifier_create(
    tk_sound_classifier_context_t** out_context,
    const tk_sound_classifier_config_t* config
);

/**
 * @brief Destroys a sound classifier instance and frees all resources.
 * @param[in,out] context The context to destroy.
 */
void tk_sound_classifier_destroy(tk_sound_classifier_context_t** context);

/**
 * @brief Processes a chunk of audio to detect ambient sounds.
 * @param[in]  context     The sound classifier instance.
 * @param[in]  audio_chunk Pointer to the raw audio data (16-bit signed mono PCM).
 * @param[in]  frame_count The number of audio frames in the chunk.
 * @param[out] out_result  A pointer to a result structure to be filled if a sound is detected.
 * @return TK_SUCCESS if processing was successful. A detected sound is indicated by
 *         the content of out_result, not the return code.
 * @return An error code on failure.
 */
TK_NODISCARD tk_error_code_t tk_sound_classifier_process(
    tk_sound_classifier_context_t* context,
    const int16_t* audio_chunk,
    size_t frame_count,
    tk_sound_detection_result_t* out_result
);

#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_AUDIO_TK_SOUND_CLASSIFIER_H
