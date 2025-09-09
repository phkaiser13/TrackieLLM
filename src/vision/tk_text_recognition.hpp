/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_text_recognition.hpp
 *
 * This header file defines the public API for the Text Recognition (OCR) module.
 * This component integrates Tesseract OCR to extract text from images captured
 * by the vision pipeline.
 *
 * The implementation is optimized for real-time performance in embedded environments,
 * with support for multiple languages, preprocessing, and result formatting.
 *
 * Dependencies:
 *   - Tesseract OCR (https://github.com/tesseract-ocr/tesseract)
 *   - Leptonica (https://github.com/DanBloomberg/leptonica)
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#ifndef TRACKIELLM_VISION_TK_TEXT_RECOGNITION_H
#define TRACKIELLM_VISION_TK_TEXT_RECOGNITION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "internal_tools/tk_file_manager.h"

// Forward-declare the primary OCR context as an opaque type.
typedef struct tk_text_recognition_context_s tk_text_recognition_context_t;

/**
 * @enum tk_ocr_language_e
 * @brief Supported OCR languages
 */
typedef enum {
    TK_OCR_LANG_ENGLISH,
    TK_OCR_LANG_PORTUGUESE
} tk_ocr_language_e;

/**
 * @struct tk_ocr_config_t
 * @brief Simplified configuration for initializing the OCR module.
 */
typedef struct {
    tk_path_t* data_path;           /**< Path to Tesseract data files ("tessdata"). */
    tk_ocr_language_e language;     /**< Primary language for OCR. */
    uint32_t dpi;                   /**< Assumed image DPI (dots per inch), e.g., 300. */
    bool enable_upscaling;          /**< Enable image upscaling for small text. */
    bool enable_binarization;       /**< Enable image binarization (thresholding). */
} tk_ocr_config_t;

/**
 * @struct tk_ocr_image_params_t
 * @brief Parameters for an image to be processed by OCR.
 */
typedef struct {
    const uint8_t* image_data;      /**< Pointer to raw pixel data. */
    uint32_t width;                 /**< Image width in pixels. */
    uint32_t height;                /**< Image height in pixels. */
    uint32_t channels;              /**< Number of channels (e.g., 3 for RGB). */
    uint32_t stride;                /**< Number of bytes per row in memory. */
    uint32_t dpi;                   /**< Optional: overrides default config DPI. */
} tk_ocr_image_params_t;

/**
 * @struct tk_ocr_result_t
 * @brief Simplified OCR result for an image.
 */
typedef struct {
    char* full_text;                /**< Full recognized text, owned by this struct. */
    size_t full_text_length;        /**< Length of the recognized text string. */
    float average_confidence;       /**< Average confidence score from Tesseract (0-100). */
    float processing_time_ms;       /**< Time taken for OCR processing in milliseconds. */
} tk_ocr_result_t;

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Context Lifecycle Management
//------------------------------------------------------------------------------

/**
 * @brief Creates and initializes a new Text Recognition context.
 *
 * @param[out] out_context Pointer to receive the address of the new context.
 * @param[in] config The configuration for the OCR module.
 *
 * @return TK_SUCCESS on successful creation.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL or config is invalid.
 * @return TK_ERROR_OUT_OF_MEMORY if memory allocation fails.
 * @return TK_ERROR_MODEL_LOAD_FAILED if OCR models cannot be loaded.
 */
TK_NODISCARD tk_error_code_t tk_text_recognition_create(
    tk_text_recognition_context_t** out_context,
    const tk_ocr_config_t* config
);

/**
 * @brief Destroys a Text Recognition context and frees all associated resources.
 *
 * @param[in,out] context Pointer to the context to be destroyed.
 */
void tk_text_recognition_destroy(tk_text_recognition_context_t** context);

//------------------------------------------------------------------------------
// OCR Processing Functions
//------------------------------------------------------------------------------

/**
 * @brief Recognizes text in an image.
 *
 * @param[in] context The OCR context.
 * @param[in] image_params Parameters describing the image to process.
 * @param[out] out_result Pointer to receive the OCR result.
 *
 * @return TK_SUCCESS on successful recognition.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL.
 * @return TK_ERROR_OUT_OF_MEMORY if memory allocation fails.
 * @return TK_ERROR_MODEL_INFERENCE_FAILED if OCR processing fails.
 */
TK_NODISCARD tk_error_code_t tk_text_recognition_process_image(
    tk_text_recognition_context_t* context,
    const tk_ocr_image_params_t* image_params,
    tk_ocr_result_t** out_result
);

/**
 * @brief Recognizes text in a region of an image.
 *
 * @param[in] context The OCR context.
 * @param[in] image_params Parameters describing the image to process.
 * @param[in] region_x X coordinate of region.
 * @param[in] region_y Y coordinate of region.
 * @param[in] region_width Width of region.
 * @param[in] region_height Height of region.
 * @param[out] out_result Pointer to receive the OCR result.
 *
 * @return TK_SUCCESS on successful recognition.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL.
 * @return TK_ERROR_OUT_OF_MEMORY if memory allocation fails.
 * @return TK_ERROR_MODEL_INFERENCE_FAILED if OCR processing fails.
 */
TK_NODISCARD tk_error_code_t tk_text_recognition_process_region(
    tk_text_recognition_context_t* context,
    const tk_ocr_image_params_t* image_params,
    uint32_t region_x,
    uint32_t region_y,
    uint32_t region_width,
    uint32_t region_height,
    tk_ocr_result_t** out_result
);

/**
 * @brief Frees an OCR result.
 *
 * @param[in,out] result Pointer to the result to free.
 */
void tk_text_recognition_free_result(tk_ocr_result_t** result);

//------------------------------------------------------------------------------
// Model and Configuration Management
//------------------------------------------------------------------------------

/**
 * @brief Sets the language for OCR processing.
 *
 * @param[in] context The OCR context.
 * @param[in] language The language to set.
 *
 * @return TK_SUCCESS on successful update.
 * @return TK_ERROR_INVALID_ARGUMENT if context is NULL.
 * @return TK_ERROR_MODEL_LOAD_FAILED if language model cannot be loaded.
 */
TK_NODISCARD tk_error_code_t tk_text_recognition_set_language(
    tk_text_recognition_context_t* context,
    tk_ocr_language_e language
);

/**
 * @brief Gets information about the loaded OCR models.
 *
 * @param[in] context The OCR context.
 * @param[out] out_language Pointer to receive current language.
 * @param[out] out_engine_version Pointer to receive engine version string.
 * @param[out] out_model_path Pointer to receive model path string.
 *
 * @return TK_SUCCESS on successful retrieval.
 * @return TK_ERROR_INVALID_ARGUMENT if context is NULL.
 */
TK_NODISCARD tk_error_code_t tk_text_recognition_get_model_info(
    tk_text_recognition_context_t* context,
    tk_ocr_language_e* out_language,
    char** out_engine_version,
    char** out_model_path
);

/**
 * @brief Updates the OCR configuration.
 *
 * @param[in] context The OCR context.
 * @param[in] config The new configuration.
 *
 * @return TK_SUCCESS on successful update.
 * @return TK_ERROR_INVALID_ARGUMENT if context or config is NULL.
 */
TK_NODISCARD tk_error_code_t tk_text_recognition_update_config(
    tk_text_recognition_context_t* context,
    const tk_ocr_config_t* config
);

#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_VISION_TK_TEXT_RECOGNITION_H
