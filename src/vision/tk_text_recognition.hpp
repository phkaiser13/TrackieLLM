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
    TK_OCR_LANG_UNKNOWN = 0,
    TK_OCR_LANG_ENGLISH,       /**< English */
    TK_OCR_LANG_PORTUGUESE,    /**< Portuguese */
    TK_OCR_LANG_SPANISH,       /**< Spanish */
    TK_OCR_LANG_FRENCH,        /**< French */
    TK_OCR_LANG_GERMAN,        /**< German */
    TK_OCR_LANG_ITALIAN,       /**< Italian */
    TK_OCR_LANG_DUTCH,         /**< Dutch */
    TK_OCR_LANG_RUSSIAN,       /**< Russian */
    TK_OCR_LANG_CHINESE,       /**< Chinese */
    TK_OCR_LANG_JAPANESE,      /**< Japanese */
    TK_OCR_LANG_KOREAN,        /**< Korean */
    TK_OCR_LANG_ARABIC,        /**< Arabic */
    TK_OCR_LANG_HINDI,         /**< Hindi */
    TK_OCR_LANG_BRAZILIAN_PORTUGUESE = TK_OCR_LANG_PORTUGUESE /**< Brazilian Portuguese */
} tk_ocr_language_e;

/**
 * @enum tk_ocr_engine_mode_e
 * @brief OCR engine modes
 */
typedef enum {
    TK_OCR_ENGINE_DEFAULT = 0,      /**< Default engine mode */
    TK_OCR_ENGINE_LSTM_ONLY,        /**< LSTM neural nets only */
    TK_OCR_ENGINE_TESSERACT_ONLY,   /**< Tesseract legacy only */
    TK_OCR_ENGINE_TESSERACT_LSTM    /**< Both Tesseract and LSTM */
} tk_ocr_engine_mode_e;

/**
 * @enum tk_ocr_page_seg_mode_e
 * @brief Page segmentation modes
 */
typedef enum {
    TK_OCR_PSM_OSD_ONLY = 0,        /**< Orientation and script detection only */
    TK_OCR_PSM_AUTO_OSD,            /**< Automatic page segmentation with OSD */
    TK_OCR_PSM_AUTO,                /**< Fully automatic page segmentation */
    TK_OCR_PSM_SINGLE_COLUMN,       /**< Assume a single column of text */
    TK_OCR_PSM_SINGLE_BLOCK_VERT,   /**< Assume a single uniform block of vertically aligned text */
    TK_OCR_PSM_SINGLE_BLOCK,        /**< Assume a single uniform block of text */
    TK_OCR_PSM_SINGLE_LINE,         /**< Treat the image as a single text line */
    TK_OCR_PSM_SINGLE_WORD,         /**< Treat the image as a single word */
    TK_OCR_PSM_CIRCLE_WORD,         /**< Treat the image as a single word in a circle */
    TK_OCR_PSM_SINGLE_CHAR,         /**< Treat the image as a single character */
    TK_OCR_PSM_SPARSE_TEXT,         /**< Find as much text as possible in no particular order */
    TK_OCR_PSM_SPARSE_TEXT_OSD      /**< Sparse text with orientation and script detection */
} tk_ocr_page_seg_mode_e;

/**
 * @struct tk_ocr_config_t
 * @brief Configuration for initializing the OCR module
 */
typedef struct {
    tk_path_t* data_path;           /**< Path to Tesseract data files */
    tk_ocr_language_e language;     /**< Primary language for OCR */
    char* additional_languages;     /**< Additional languages (comma-separated) */
    tk_ocr_engine_mode_e engine_mode; /**< OCR engine mode */
    tk_ocr_page_seg_mode_e psm;     /**< Page segmentation mode */
    uint32_t dpi;                   /**< Image DPI (dots per inch) */
    bool enable_spellchecker;       /**< Enable spell checker */
    bool enable_dictionary;         /**< Enable dictionary correction */
    uint32_t timeout_ms;            /**< Timeout for OCR operations in milliseconds */
    uint32_t max_image_width;       /**< Maximum image width for processing */
    uint32_t max_image_height;      /**< Maximum image height for processing */
    bool enable_preprocessing;      /**< Enable automatic image preprocessing */
    bool enable_auto_rotate;        /**< Enable automatic image rotation */
    bool enable_deskew;             /**< Enable image deskewing */
    bool enable_contrast_enhancement; /**< Enable contrast enhancement */
    bool enable_noise_reduction;    /**< Enable noise reduction */
    bool enable_sharpening;         /**< Enable image sharpening */
    bool enable_binarization;       /**< Enable image binarization */
    bool enable_inversion;          /**< Enable image inversion for white text on dark background */
    bool enable_upscaling;          /**< Enable image upscaling for small text */
    float upscale_factor;           /**< Upscaling factor (1.0 = no scaling) */
    bool enable_confidence_scoring; /**< Enable confidence scoring for results */
    float min_confidence;           /**< Minimum confidence threshold (0.0-1.0) */
    bool enable_word_detection;     /**< Enable word-level detection */
    bool enable_line_detection;     /**< Enable line-level detection */
    bool enable_paragraph_detection; /**< Enable paragraph-level detection */
    bool enable_reading_order;      /**< Enable reading order detection */
    bool enable_multi_column;       /**< Enable multi-column text detection */
    bool enable_table_detection;    /**< Enable table structure detection */
    bool enable_form_field_detection; /**< Enable form field detection */
    bool enable_qr_code_detection;  /**< Enable QR code detection */
    bool enable_barcode_detection;  /**< Enable barcode detection */
    bool enable_math_detection;     /**< Enable mathematical expression detection */
    bool enable_handwriting_recognition; /**< Enable handwriting recognition */
    uint32_t num_threads;           /**< Number of threads for OCR processing */
    bool enable_gpu_acceleration;   /**< Enable GPU acceleration if available */
    uint32_t gpu_device_id;         /**< GPU device ID for acceleration */
    bool enable_batch_processing;   /**< Enable batch processing of images */
    uint32_t batch_size;            /**< Batch size for processing */
    bool enable_progress_reporting; /**< Enable progress reporting */
    bool enable_debug_output;       /**< Enable debug output */
    uint32_t debug_verbosity;       /**< Debug verbosity level */
    char* debug_output_path;        /**< Path for debug output files */
    bool enable_result_caching;     /**< Enable caching of OCR results */
    uint32_t cache_size_mb;         /**< Size of result cache in MB */
    bool enable_result_compression; /**< Enable compression of cached results */
    bool enable_result_encryption;  /**< Enable encryption of cached results */
    char* encryption_key;           /**< Encryption key for cached results */
    bool enable_result_validation;  /**< Enable validation of OCR results */
    float validation_threshold;     /**< Validation threshold for result accuracy */
    bool enable_result_filtering;   /**< Enable filtering of OCR results */
    char* filter_regex;             /**< Regular expression for result filtering */
    bool enable_result_formatting;  /**< Enable formatting of OCR results */
    char* output_format;            /**< Output format (text, hocr, pdf, etc.) */
    bool enable_result_export;      /**< Enable export of results */
    char* export_path;              /**< Path for exported results */
    bool enable_result_notification; /**< Enable notification of results */
    char* notification_callback;    /**< Callback function for notifications */
    bool enable_result_streaming;   /**< Enable streaming of results */
    uint32_t streaming_buffer_size; /**< Streaming buffer size */
    bool enable_result_synchronization; /**< Enable synchronization of results */
    char* sync_endpoint;            /**< Synchronization endpoint */
    bool enable_result_analytics;   /**< Enable analytics collection */
    char* analytics_endpoint;       /**< Analytics collection endpoint */
    bool enable_result_security;    /**< Enable security features */
    char* security_policy;          /**< Security policy file */
    bool enable_result_compliance;  /**< Enable compliance checking */
    char* compliance_standard;      /**< Compliance standard */
    bool enable_result_archiving;   /**< Enable archiving of results */
    char* archive_path;             /**< Archive path */
    bool enable_result_backup;      /**< Enable backup of results */
    char* backup_path;              /**< Backup path */
    bool enable_result_restore;     /**< Enable restore of results */
    char* restore_path;             /**< Restore path */
    bool enable_result_migration;   /**< Enable migration of results */
    char* migration_target;         /**< Migration target */
    bool enable_result_replication; /**< Enable replication of results */
    uint32_t replication_factor;    /**< Replication factor */
    bool enable_result_sharding;    /**< Enable sharding of results */
    uint32_t shard_count;           /**< Number of shards */
    bool enable_result_federation;  /**< Enable federation of results */
    char* federation_config;        /**< Federation configuration */
    bool enable_result_synchronization_protocol; /**< Enable synchronization protocol */
    char* sync_protocol;            /**< Synchronization protocol */
    bool enable_result_consistency; /**< Enable consistency checking */
    char* consistency_protocol;     /**< Consistency protocol */
    bool enable_result_transaction; /**< Enable transaction support */
    char* transaction_manager;      /**< Transaction manager */
    bool enable_result_caching_strategy; /**< Enable caching strategy */
    char* caching_strategy;         /**< Caching strategy */
    bool enable_result_compression_strategy; /**< Enable compression strategy */
    char* compression_strategy;     /**< Compression strategy */
    bool enable_result_security_strategy; /**< Enable security strategy */
    char* security_strategy;        /**< Security strategy */
    bool enable_result_analytics_strategy; /**< Enable analytics strategy */
    char* analytics_strategy;       /**< Analytics strategy */
    bool enable_result_debugging_strategy; /**< Enable debugging strategy */
    char* debugging_strategy;       /**< Debugging strategy */
    bool enable_result_custom_strategy; /**< Enable custom strategy */
    char* custom_strategy;          /**< Custom strategy */
} tk_ocr_config_t;

/**
 * @struct tk_ocr_image_params_t
 * @brief Parameters for OCR image processing
 */
typedef struct {
    const uint8_t* image_data;      /**< Pointer to image data */
    uint32_t width;                 /**< Image width in pixels */
    uint32_t height;                /**< Image height in pixels */
    uint32_t channels;              /**< Number of channels (1=grayscale, 3=RGB, 4=RGBA) */
    uint32_t stride;                /**< Number of bytes per row */
    bool is_packed;                 /**< Whether image data is packed (no padding) */
    uint32_t dpi;                   /**< Image DPI (overrides config) */
    tk_ocr_page_seg_mode_e psm;     /**< Page segmentation mode (overrides config) */
    bool enable_preprocessing;      /**< Enable preprocessing (overrides config) */
    bool enable_auto_rotate;        /**< Enable auto rotation (overrides config) */
    bool enable_deskew;             /**< Enable deskewing (overrides config) */
    bool enable_contrast_enhancement; /**< Enable contrast enhancement (overrides config) */
    bool enable_noise_reduction;    /**< Enable noise reduction (overrides config) */
    bool enable_sharpening;         /**< Enable sharpening (overrides config) */
    bool enable_binarization;       /**< Enable binarization (overrides config) */
    bool enable_inversion;          /**< Enable inversion (overrides config) */
    bool enable_upscaling;          /**< Enable upscaling (overrides config) */
    float upscale_factor;           /**< Upscaling factor (overrides config) */
} tk_ocr_image_params_t;

/**
 * @struct tk_ocr_text_region_t
 * @brief Represents a text region in an image
 */
typedef struct {
    uint32_t x;                     /**< X coordinate of region */
    uint32_t y;                     /**< Y coordinate of region */
    uint32_t width;                 /**< Width of region */
    uint32_t height;                /**< Height of region */
    float confidence;               /**< Confidence score (0.0-1.0) */
    char* text;                     /**< Recognized text */
    size_t text_length;             /**< Length of recognized text */
    uint32_t word_count;            /**< Number of words in text */
    uint32_t line_count;            /**< Number of lines in text */
    bool is_handwritten;            /**< Whether text is handwritten */
    bool is_mathematical;           /**< Whether text is mathematical */
    bool is_qr_code;                /**< Whether region contains QR code */
    bool is_barcode;                /**< Whether region contains barcode */
    char* font_name;                /**< Font name (if detected) */
    uint32_t font_size;             /**< Font size (if detected) */
    bool is_bold;                   /**< Whether text is bold */
    bool is_italic;                 /**< Whether text is italic */
    bool is_underlined;             /**< Whether text is underlined */
    uint32_t color_fg;              /**< Foreground color (RGB) */
    uint32_t color_bg;              /**< Background color (RGB) */
    uint32_t orientation;           /**< Text orientation in degrees */
    bool is_vertical;               /**< Whether text is vertical */
    bool is_upside_down;            /**< Whether text is upside down */
    char* language;                 /**< Detected language */
    uint32_t page_number;           /**< Page number (for multi-page documents) */
    uint32_t block_number;          /**< Block number */
    uint32_t paragraph_number;      /**< Paragraph number */
    uint32_t line_number;           /**< Line number */
    uint32_t word_number;           /**< Word number */
} tk_ocr_text_region_t;

/**
 * @struct tk_ocr_result_t
 * @brief Complete OCR result for an image
 */
typedef struct {
    char* full_text;                /**< Full recognized text */
    size_t full_text_length;        /**< Length of full text */
    tk_ocr_text_region_t* regions;  /**< Array of text regions */
    size_t region_count;            /**< Number of text regions */
    uint32_t word_count;            /**< Total number of words */
    uint32_t line_count;            /**< Total number of lines */
    uint32_t page_count;            /**< Number of pages processed */
    float average_confidence;       /**< Average confidence score */
    float processing_time_ms;       /**< Processing time in milliseconds */
    uint32_t image_width;           /**< Width of processed image */
    uint32_t image_height;          /**< Height of processed image */
    char* detected_language;        /**< Detected language */
    bool is_multilingual;           /**< Whether multiple languages were detected */
    char** languages;               /**< Array of detected languages */
    uint32_t language_count;        /**< Number of detected languages */
    bool has_qr_codes;              /**< Whether QR codes were detected */
    uint32_t qr_code_count;         /**< Number of QR codes detected */
    bool has_barcodes;              /**< Whether barcodes were detected */
    uint32_t barcode_count;         /**< Number of barcodes detected */
    bool has_mathematical_expressions; /**< Whether mathematical expressions were detected */
    uint32_t math_expression_count; /**< Number of mathematical expressions detected */
    bool has_handwritten_text;      /**< Whether handwritten text was detected */
    uint32_t handwritten_word_count; /**< Number of handwritten words detected */
    char* processing_timestamp;     /**< Timestamp of processing */
    char* model_version;            /**< OCR model version used */
    char* engine_version;           /**< OCR engine version used */
    bool is_valid;                  /**< Whether result is valid */
    char* validation_message;       /**< Validation message if not valid */
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
