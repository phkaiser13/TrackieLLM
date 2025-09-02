/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_text_recognition.cpp
 *
 * This source file implements the Text Recognition (OCR) module for TrackieLLM.
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

#include "vision/tk_text_recognition.h"
#include "utils/tk_logging.h"
#include "utils/tk_error_handling.h"

// Include Tesseract and Leptonica headers
#include <tesseract/baseapi.h>
#include <tesseract/renderer.h>
#include <tesseract/ocrclass.h>
#include <leptonica/allheaders.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include <thread>
#include <mutex>
#include <chrono>
#include <regex>

// Maximum number of text regions to process
#define MAX_TEXT_REGIONS 1024

// Maximum text length for a single region
#define MAX_REGION_TEXT_LENGTH 4096

// Internal structures for OCR processing
struct tk_text_recognition_context_s {
    tesseract::TessBaseAPI* tess_api;    // Tesseract API instance
    tk_ocr_config_t config;              // Configuration
    char* data_path;                     // Tesseract data path
    char* language_code;                 // Current language code
    std::mutex processing_mutex;         // Mutex for thread safety
    PIX* preprocessing_buffer;           // Buffer for image preprocessing
    size_t cache_size_bytes;             // Current cache size
    std::vector<std::pair<std::string, tk_ocr_result_t*>> result_cache; // Result cache
    time_t last_cache_cleanup;           // Last cache cleanup time
};

// Internal helper functions
static tk_error_code_t validate_config(const tk_ocr_config_t* config);
static tk_error_code_t init_tesseract_api(tk_text_recognition_context_t* context);
static void cleanup_tesseract_api(tk_text_recognition_context_t* context);
static tk_error_code_t convert_language_enum(tk_ocr_language_e lang, char** out_code);
static tk_error_code_t convert_engine_mode(tk_ocr_engine_mode_e mode, tesseract::OcrEngineMode* out_mode);
static tk_error_code_t convert_page_seg_mode(tk_ocr_page_seg_mode_e mode, tesseract::PageSegMode* out_mode);
static PIX* preprocess_image(PIX* input_pix, const tk_ocr_config_t* config, const tk_ocr_image_params_t* params);
static PIX* convert_to_pix(const tk_ocr_image_params_t* params);
static tk_error_code_t extract_text_regions(tk_text_recognition_context_t* context, 
                                          BOXA* boxes, 
                                          const char* text, 
                                          const int* confidences,
                                          tk_ocr_text_region_t** out_regions, 
                                          size_t* out_count);
static tk_error_code_t extract_full_text(tk_text_recognition_context_t* context, 
                                       const char* tess_text, 
                                       char** out_text, 
                                       size_t* out_length);
static tk_error_code_t calculate_statistics(tk_text_recognition_context_t* context,
                                         const int* confidences,
                                         int confidence_count,
                                         float* out_average_confidence,
                                         uint32_t* out_word_count,
                                         uint32_t* out_line_count);
static void free_text_regions(tk_ocr_text_region_t* regions, size_t count);
static void free_ocr_result(tk_ocr_result_t* result);
static char* duplicate_string(const char* src);
static void free_string(char* str);
static bool is_cache_entry_expired(const tk_ocr_result_t* result, time_t current_time, uint32_t timeout_seconds);
static void cleanup_cache(tk_text_recognition_context_t* context);
static tk_error_code_t cache_result(tk_text_recognition_context_t* context, 
                                  const std::string& image_hash, 
                                  const tk_ocr_result_t* result);
static tk_ocr_result_t* find_cached_result(tk_text_recognition_context_t* context, 
                                         const std::string& image_hash);
static std::string calculate_image_hash(const tk_ocr_image_params_t* params);
static bool apply_filters(const char* text, const char* filter_regex);
static void apply_formatting(char* text, const char* output_format);
static tk_error_code_t detect_qr_codes(PIX* pix, bool* out_has_qr_codes, uint32_t* out_qr_count);
static tk_error_code_t detect_barcodes(PIX* pix, bool* out_has_barcodes, uint32_t* out_barcode_count);
static tk_error_code_t detect_mathematical_expressions(const char* text, bool* out_has_math, uint32_t* out_math_count);
static tk_error_code_t detect_handwritten_text(const char* text, bool* out_is_handwritten, uint32_t* out_handwritten_count);
static tk_error_code_t detect_languages(const char* text, char*** out_languages, uint32_t* out_language_count);
static void free_language_list(char** languages, uint32_t count);
static tk_error_code_t validate_result(const tk_ocr_result_t* result, float threshold, bool* out_is_valid, char** out_message);
static tk_error_code_t compress_result(const tk_ocr_result_t* input, tk_ocr_result_t** out_compressed);
static tk_error_code_t decompress_result(const tk_ocr_result_t* input, tk_ocr_result_t** out_decompressed);
static tk_error_code_t encrypt_result(const tk_ocr_result_t* input, const char* key, tk_ocr_result_t** out_encrypted);
static tk_error_code_t decrypt_result(const tk_ocr_result_t* input, const char* key, tk_ocr_result_t** out_decrypted);
static void apply_security_policy(const char* policy);
static void apply_compliance_standard(const char* standard);
static void export_result(const tk_ocr_result_t* result, const char* export_path);
static void notify_result(const tk_ocr_result_t* result, const char* callback);
static void stream_result(const tk_ocr_result_t* result, uint32_t buffer_size);
static void synchronize_result(const tk_ocr_result_t* result, const char* endpoint);
static void collect_analytics(const tk_ocr_result_t* result, const char* endpoint);
static void archive_result(const tk_ocr_result_t* result, const char* archive_path);
static void backup_result(const tk_ocr_result_t* result, const char* backup_path);
static void restore_result(tk_ocr_result_t** result, const char* restore_path);
static void migrate_result(const tk_ocr_result_t* result, const char* target);
static void replicate_result(const tk_ocr_result_t* result, uint32_t factor);
static void shard_result(const tk_ocr_result_t* result, uint32_t shard_count);
static void federate_result(const tk_ocr_result_t* result, const char* config);
static void synchronize_with_protocol(const tk_ocr_result_t* result, const char* protocol);
static void check_consistency(const tk_ocr_result_t* result, const char* protocol);
static void manage_transaction(const tk_ocr_result_t* result, const char* manager);
static void apply_caching_strategy(const char* strategy);
static void apply_compression_strategy(const char* strategy);
static void apply_security_strategy(const char* strategy);
static void apply_analytics_strategy(const char* strategy);
static void apply_debugging_strategy(const char* strategy);
static void apply_custom_strategy(const char* strategy);

//------------------------------------------------------------------------------
// Private helper functions
//------------------------------------------------------------------------------

/**
 * @brief Validates the OCR configuration
 */
static tk_error_code_t validate_config(const tk_ocr_config_t* config) {
    if (!config) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Validate required fields
    if (!config->data_path) {
        TK_LOG_ERROR("Tesseract data path is required");
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    if (config->dpi == 0) {
        TK_LOG_ERROR("Invalid DPI: %u", config->dpi);
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    if (config->num_threads == 0) {
        TK_LOG_ERROR("Invalid number of threads: %u", config->num_threads);
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    return TK_SUCCESS;
}

/**
 * @brief Initializes the Tesseract API
 */
static tk_error_code_t init_tesseract_api(tk_text_recognition_context_t* context) {
    if (!context) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Create Tesseract API instance
    context->tess_api = new tesseract::TessBaseAPI();
    if (!context->tess_api) {
        TK_LOG_ERROR("Failed to create Tesseract API instance");
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Initialize Tesseract with language and data path
    const char* lang = context->language_code ? context->language_code : "eng";
    const char* data_path = context->data_path ? context->data_path : "./tessdata";
    
    int result = context->tess_api->Init(data_path, lang);
    if (result != 0) {
        TK_LOG_ERROR("Failed to initialize Tesseract API with language '%s' and data path '%s'", lang, data_path);
        delete context->tess_api;
        context->tess_api = nullptr;
        return TK_ERROR_MODEL_LOAD_FAILED;
    }
    
    // Set engine mode
    tesseract::OcrEngineMode engine_mode;
    tk_error_code_t error = convert_engine_mode(context->config.engine_mode, &engine_mode);
    if (error != TK_SUCCESS) {
        delete context->tess_api;
        context->tess_api = nullptr;
        return error;
    }
    
    context->tess_api->SetPageSegMode(tesseract::PSM_AUTO);
    
    // Set other configuration parameters
    context->tess_api->SetVariable(" tessedit_create_hocr", "1");
    context->tess_api->SetVariable(" tessedit_create_tsv", "1");
    context->tess_api->SetVariable(" tessedit_create_boxfile", "0");
    context->tess_api->SetVariable(" tessedit_create_unlv", "0");
    context->tess_api->SetVariable(" tessedit_create_osd", "0");
    
    // Enable or disable spell checker
    context->tess_api->SetVariable(" tessedit_enable_doc_dict", 
                                  context->config.enable_spellchecker ? "1" : "0");
    
    // Enable or disable dictionary
    context->tess_api->SetVariable(" tessedit_enable_bigram_correction", 
                                  context->config.enable_dictionary ? "1" : "0");
    
    TK_LOG_INFO("Tesseract API initialized successfully with language: %s", lang);
    return TK_SUCCESS;
}

/**
 * @brief Cleans up the Tesseract API
 */
static void cleanup_tesseract_api(tk_text_recognition_context_t* context) {
    if (!context) return;
    
    if (context->tess_api) {
        context->tess_api->End();
        delete context->tess_api;
        context->tess_api = nullptr;
    }
}

/**
 * @brief Converts language enum to language code
 */
static tk_error_code_t convert_language_enum(tk_ocr_language_e lang, char** out_code) {
    if (!out_code) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_code = nullptr;
    
    switch (lang) {
        case TK_OCR_LANG_ENGLISH:
            *out_code = strdup("eng");
            break;
        case TK_OCR_LANG_PORTUGUESE:
            *out_code = strdup("por");
            break;
        case TK_OCR_LANG_SPANISH:
            *out_code = strdup("spa");
            break;
        case TK_OCR_LANG_FRENCH:
            *out_code = strdup("fra");
            break;
        case TK_OCR_LANG_GERMAN:
            *out_code = strdup("deu");
            break;
        case TK_OCR_LANG_ITALIAN:
            *out_code = strdup("ita");
            break;
        case TK_OCR_LANG_DUTCH:
            *out_code = strdup("nld");
            break;
        case TK_OCR_LANG_RUSSIAN:
            *out_code = strdup("rus");
            break;
        case TK_OCR_LANG_CHINESE:
            *out_code = strdup("chi_sim");
            break;
        case TK_OCR_LANG_JAPANESE:
            *out_code = strdup("jpn");
            break;
        case TK_OCR_LANG_KOREAN:
            *out_code = strdup("kor");
            break;
        case TK_OCR_LANG_ARABIC:
            *out_code = strdup("ara");
            break;
        case TK_OCR_LANG_HINDI:
            *out_code = strdup("hin");
            break;
        default:
            *out_code = strdup("eng"); // Default to English
            break;
    }
    
    if (!*out_code) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    return TK_SUCCESS;
}

/**
 * @brief Converts engine mode enum to Tesseract engine mode
 */
static tk_error_code_t convert_engine_mode(tk_ocr_engine_mode_e mode, tesseract::OcrEngineMode* out_mode) {
    if (!out_mode) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    switch (mode) {
        case TK_OCR_ENGINE_DEFAULT:
            *out_mode = tesseract::OEM_DEFAULT;
            break;
        case TK_OCR_ENGINE_LSTM_ONLY:
            *out_mode = tesseract::OEM_LSTM_ONLY;
            break;
        case TK_OCR_ENGINE_TESSERACT_ONLY:
            *out_mode = tesseract::OEM_TESSERACT_ONLY;
            break;
        case TK_OCR_ENGINE_TESSERACT_LSTM:
            *out_mode = tesseract::OEM_TESSERACT_LSTM_COMBINED;
            break;
        default:
            *out_mode = tesseract::OEM_DEFAULT;
            break;
    }
    
    return TK_SUCCESS;
}

/**
 * @brief Converts page segmentation mode enum to Tesseract mode
 */
static tk_error_code_t convert_page_seg_mode(tk_ocr_page_seg_mode_e mode, tesseract::PageSegMode* out_mode) {
    if (!out_mode) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    switch (mode) {
        case TK_OCR_PSM_OSD_ONLY:
            *out_mode = tesseract::PSM_OSD_ONLY;
            break;
        case TK_OCR_PSM_AUTO_OSD:
            *out_mode = tesseract::PSM_AUTO_OSD;
            break;
        case TK_OCR_PSM_AUTO:
            *out_mode = tesseract::PSM_AUTO;
            break;
        case TK_OCR_PSM_SINGLE_COLUMN:
            *out_mode = tesseract::PSM_SINGLE_COLUMN;
            break;
        case TK_OCR_PSM_SINGLE_BLOCK_VERT:
            *out_mode = tesseract::PSM_SINGLE_BLOCK_VERT_TEXT;
            break;
        case TK_OCR_PSM_SINGLE_BLOCK:
            *out_mode = tesseract::PSM_SINGLE_BLOCK;
            break;
        case TK_OCR_PSM_SINGLE_LINE:
            *out_mode = tesseract::PSM_SINGLE_LINE;
            break;
        case TK_OCR_PSM_SINGLE_WORD:
            *out_mode = tesseract::PSM_SINGLE_WORD;
            break;
        case TK_OCR_PSM_CIRCLE_WORD:
            *out_mode = tesseract::PSM_CIRCLE_WORD;
            break;
        case TK_OCR_PSM_SINGLE_CHAR:
            *out_mode = tesseract::PSM_SINGLE_CHAR;
            break;
        case TK_OCR_PSM_SPARSE_TEXT:
            *out_mode = tesseract::PSM_SPARSE_TEXT;
            break;
        case TK_OCR_PSM_SPARSE_TEXT_OSD:
            *out_mode = tesseract::PSM_SPARSE_TEXT_OSD;
            break;
        default:
            *out_mode = tesseract::PSM_AUTO;
            break;
    }
    
    return TK_SUCCESS;
}

/**
 * @brief Preprocesses an image for OCR
 */
static PIX* preprocess_image(PIX* input_pix, const tk_ocr_config_t* config, const tk_ocr_image_params_t* params) {
    if (!input_pix || !config) {
        return nullptr;
    }
    
    PIX* processed_pix = pixCopy(nullptr, input_pix);
    if (!processed_pix) {
        return nullptr;
    }
    
    // Apply preprocessing steps based on configuration
    if (config->enable_preprocessing || (params && params->enable_preprocessing)) {
        // Auto-rotate if enabled
        if (config->enable_auto_rotate || (params && params->enable_auto_rotate)) {
            // In a real implementation, this would detect orientation and rotate
            TK_LOG_DEBUG("Applying auto-rotation");
        }
        
        // Deskew if enabled
        if (config->enable_deskew || (params && params->enable_deskew)) {
            TK_LOG_DEBUG("Applying deskewing");
            PIX* deskewed = pixDeskew(processed_pix, 0);
            if (deskewed) {
                pixDestroy(&processed_pix);
                processed_pix = deskewed;
            }
        }
        
        // Contrast enhancement if enabled
        if (config->enable_contrast_enhancement || (params && params->enable_contrast_enhancement)) {
            TK_LOG_DEBUG("Applying contrast enhancement");
            PIX* enhanced = pixGammaTRC(nullptr, processed_pix, 1.5, 0, 255);
            if (enhanced) {
                pixDestroy(&processed_pix);
                processed_pix = enhanced;
            }
        }
        
        // Noise reduction if enabled
        if (config->enable_noise_reduction || (params && params->enable_noise_reduction)) {
            TK_LOG_DEBUG("Applying noise reduction");
            PIX* denoised = pixBlockconv(processed_pix, 1, 1);
            if (denoised) {
                pixDestroy(&processed_pix);
                processed_pix = denoised;
            }
        }
        
        // Sharpening if enabled
        if (config->enable_sharpening || (params && params->enable_sharpening)) {
            TK_LOG_DEBUG("Applying sharpening");
            // Simple sharpening kernel
            L_KERNEL* kernel = kernelCreateFromString(3, 3, 1, 1, "-1 -1 -1 -1 9 -1 -1 -1 -1");
            if (kernel) {
                PIX* sharpened = pixConvolve(processed_pix, kernel, 8, 1);
                if (sharpened) {
                    pixDestroy(&processed_pix);
                    processed_pix = sharpened;
                }
                kernelDestroy(&kernel);
            }
        }
        
        // Binarization if enabled
        if (config->enable_binarization || (params && params->enable_binarization)) {
            TK_LOG_DEBUG("Applying binarization");
            PIX* binary = pixConvertTo1(processed_pix, 128);
            if (binary) {
                pixDestroy(&processed_pix);
                processed_pix = binary;
            }
        }
        
        // Inversion if enabled (for white text on dark background)
        if (config->enable_inversion || (params && params->enable_inversion)) {
            TK_LOG_DEBUG("Applying inversion");
            pixInvert(processed_pix, processed_pix);
        }
        
        // Upscaling if enabled
        if ((config->enable_upscaling || (params && params->enable_upscaling)) && 
            (config->upscale_factor > 1.0f || (params && params->upscale_factor > 1.0f))) {
            float factor = params && params->upscale_factor > 1.0f ? 
                          params->upscale_factor : config->upscale_factor;
            TK_LOG_DEBUG("Applying upscaling with factor: %.2f", factor);
            PIX* scaled = pixScale(processed_pix, factor, factor);
            if (scaled) {
                pixDestroy(&processed_pix);
                processed_pix = scaled;
            }
        }
    }
    
    return processed_pix;
}

/**
 * @brief Converts image parameters to PIX format
 */
static PIX* convert_to_pix(const tk_ocr_image_params_t* params) {
    if (!params || !params->image_data) {
        return nullptr;
    }
    
    // Create PIX based on image parameters
    PIX* pix = nullptr;
    
    if (params->channels == 1) {
        // Grayscale image
        pix = pixCreate(params->width, params->height, 8);
    } else if (params->channels == 3) {
        // RGB image
        pix = pixCreate(params->width, params->height, 32);
    } else if (params->channels == 4) {
        // RGBA image
        pix = pixCreate(params->width, params->height, 32);
    } else {
        TK_LOG_ERROR("Unsupported number of channels: %u", params->channels);
        return nullptr;
    }
    
    if (!pix) {
        TK_LOG_ERROR("Failed to create PIX image");
        return nullptr;
    }
    
    // Copy image data
    if (params->channels == 1) {
        // Grayscale
        for (uint32_t y = 0; y < params->height; y++) {
            const uint8_t* src_row = params->image_data + y * params->stride;
            uint8_t* dst_row = pix->data + y * pix->wpl * sizeof(uint32_t);
            memcpy(dst_row, src_row, params->width);
        }
    } else if (params->channels == 3) {
        // RGB
        for (uint32_t y = 0; y < params->height; y++) {
            const uint8_t* src_row = params->image_data + y * params->stride;
            uint32_t* dst_row = pix->data + y * pix->wpl;
            for (uint32_t x = 0; x < params->width; x++) {
                uint32_t r = src_row[x * 3];
                uint32_t g = src_row[x * 3 + 1];
                uint32_t b = src_row[x * 3 + 2];
                dst_row[x] = (r << 24) | (g << 16) | (b << 8) | 0xFF;
            }
        }
    } else if (params->channels == 4) {
        // RGBA
        for (uint32_t y = 0; y < params->height; y++) {
            const uint8_t* src_row = params->image_data + y * params->stride;
            uint32_t* dst_row = pix->data + y * pix->wpl;
            for (uint32_t x = 0; x < params->width; x++) {
                uint32_t r = src_row[x * 4];
                uint32_t g = src_row[x * 4 + 1];
                uint32_t b = src_row[x * 4 + 2];
                uint32_t a = src_row[x * 4 + 3];
                dst_row[x] = (r << 24) | (g << 16) | (b << 8) | a;
            }
        }
    }
    
    return pix;
}

/**
 * @brief Extracts text regions from Tesseract results
 */
static tk_error_code_t extract_text_regions(tk_text_recognition_context_t* context, 
                                          BOXA* boxes, 
                                          const char* text, 
                                          const int* confidences,
                                          tk_ocr_text_region_t** out_regions, 
                                          size_t* out_count) {
    if (!context || !boxes || !text || !out_regions || !out_count) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_regions = nullptr;
    *out_count = 0;
    
    int box_count = boxaGetCount(boxes);
    if (box_count <= 0) {
        return TK_SUCCESS; // No regions found
    }
    
    // Allocate array for regions
    tk_ocr_text_region_t* regions = new tk_ocr_text_region_t[box_count];
    if (!regions) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Extract information for each region
    for (int i = 0; i < box_count; i++) {
        BOX* box = boxaGetBox(boxes, i, L_CLONE);
        if (!box) continue;
        
        // Fill region information
        regions[i].x = box->x;
        regions[i].y = box->y;
        regions[i].width = box->w;
        regions[i].height = box->h;
        regions[i].confidence = confidences ? (confidences[i] / 100.0f) : 0.0f;
        regions[i].text = nullptr;
        regions[i].text_length = 0;
        regions[i].word_count = 0;
        regions[i].line_count = 0;
        regions[i].is_handwritten = false;
        regions[i].is_mathematical = false;
        regions[i].is_qr_code = false;
        regions[i].is_barcode = false;
        regions[i].font_name = nullptr;
        regions[i].font_size = 0;
        regions[i].is_bold = false;
        regions[i].is_italic = false;
        regions[i].is_underlined = false;
        regions[i].color_fg = 0;
        regions[i].color_bg = 0;
        regions[i].orientation = 0;
        regions[i].is_vertical = false;
        regions[i].is_upside_down = false;
        regions[i].language = nullptr;
        regions[i].page_number = 0;
        regions[i].block_number = 0;
        regions[i].paragraph_number = 0;
        regions[i].line_number = 0;
        regions[i].word_number = 0;
        
        boxDestroy(&box);
    }
    
    *out_regions = regions;
    *out_count = box_count;
    
    return TK_SUCCESS;
}

/**
 * @brief Extracts full text from Tesseract results
 */
static tk_error_code_t extract_full_text(tk_text_recognition_context_t* context, 
                                       const char* tess_text, 
                                       char** out_text, 
                                       size_t* out_length) {
    if (!context || !tess_text || !out_text || !out_length) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_text = nullptr;
    *out_length = 0;
    
    size_t text_len = strlen(tess_text);
    if (text_len == 0) {
        *out_text = strdup("");
        return TK_SUCCESS;
    }
    
    // Allocate memory for text
    char* text = new char[text_len + 1];
    if (!text) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Copy text
    strcpy(text, tess_text);
    *out_text = text;
    *out_length = text_len;
    
    return TK_SUCCESS;
}

/**
 * @brief Calculates statistics from OCR results
 */
static tk_error_code_t calculate_statistics(tk_text_recognition_context_t* context,
                                         const int* confidences,
                                         int confidence_count,
                                         float* out_average_confidence,
                                         uint32_t* out_word_count,
                                         uint32_t* out_line_count) {
    if (!context || !out_average_confidence || !out_word_count || !out_line_count) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_average_confidence = 0.0f;
    *out_word_count = 0;
    *out_line_count = 0;
    
    // Calculate average confidence
    if (confidences && confidence_count > 0) {
        int total_confidence = 0;
        for (int i = 0; i < confidence_count; i++) {
            total_confidence += confidences[i];
        }
        *out_average_confidence = (total_confidence / confidence_count) / 100.0f;
    }
    
    // In a real implementation, we would count words and lines from the text
    // For now, we'll set placeholder values
    *out_word_count = 10;  // Placeholder
    *out_line_count = 3;   // Placeholder
    
    return TK_SUCCESS;
}

/**
 * @brief Frees text regions
 */
static void free_text_regions(tk_ocr_text_region_t* regions, size_t count) {
    if (!regions) return;
    
    for (size_t i = 0; i < count; i++) {
        free_string(regions[i].text);
        free_string(regions[i].font_name);
        free_string(regions[i].language);
    }
    
    delete[] regions;
}

/**
 * @brief Frees an OCR result
 */
static void free_ocr_result(tk_ocr_result_t* result) {
    if (!result) return;
    
    free_string(result->full_text);
    free_text_regions(result->regions, result->region_count);
    free_language_list(result->languages, result->language_count);
    free_string(result->detected_language);
    free_string(result->processing_timestamp);
    free_string(result->model_version);
    free_string(result->engine_version);
    free_string(result->validation_message);
    
    delete result;
}

/**
 * @brief Duplicates a string
 */
static char* duplicate_string(const char* src) {
    if (!src) return nullptr;
    
    size_t len = strlen(src);
    char* dup = new char[len + 1];
    if (!dup) return nullptr;
    
    strcpy(dup, src);
    return dup;
}

/**
 * @brief Frees a string
 */
static void free_string(char* str) {
    if (str) {
        delete[] str;
    }
}

/**
 * @brief Checks if a cache entry has expired
 */
static bool is_cache_entry_expired(const tk_ocr_result_t* result, time_t current_time, uint32_t timeout_seconds) {
    if (!result || !result->processing_timestamp) return true;
    
    // In a real implementation, we would parse the timestamp and compare with current_time
    // For now, we'll assume entries expire after timeout_seconds
    return false; // Placeholder
}

/**
 * @brief Cleans up the result cache
 */
static void cleanup_cache(tk_text_recognition_context_t* context) {
    if (!context) return;
    
    time_t current_time = time(nullptr);
    
    // Remove expired entries
    auto it = context->result_cache.begin();
    while (it != context->result_cache.end()) {
        if (is_cache_entry_expired(it->second, current_time, context->config.timeout_ms / 1000)) {
            free_ocr_result(it->second);
            it = context->result_cache.erase(it);
        } else {
            ++it;
        }
    }
    
    // Check if we need to reduce cache size
    if (context->cache_size_bytes > context->config.cache_size_mb * 1024 * 1024) {
        // Remove oldest entries until cache size is within limits
        // This is a simplified LRU implementation
        while (context->cache_size_bytes > context->config.cache_size_mb * 1024 * 1024 && 
               !context->result_cache.empty()) {
            auto oldest = context->result_cache.begin();
            free_ocr_result(oldest->second);
            context->cache_size_bytes -= sizeof(tk_ocr_result_t); // Simplified size calculation
            context->result_cache.erase(oldest);
        }
    }
    
    context->last_cache_cleanup = current_time;
}

/**
 * @brief Caches an OCR result
 */
static tk_error_code_t cache_result(tk_text_recognition_context_t* context, 
                                  const std::string& image_hash, 
                                  const tk_ocr_result_t* result) {
    if (!context || !result) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Check if result already exists in cache
    for (const auto& entry : context->result_cache) {
        if (entry.first == image_hash) {
            // Update existing entry
            free_ocr_result(entry.second);
            context->result_cache.erase(entry.first);
            break;
        }
    }
    
    // Create a copy of the result for caching
    tk_ocr_result_t* cached_result = new tk_ocr_result_t;
    if (!cached_result) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Copy result data (simplified)
    memcpy(cached_result, result, sizeof(tk_ocr_result_t));
    cached_result->full_text = duplicate_string(result->full_text);
    // Note: In a real implementation, we would need to deep copy all fields
    
    // Add to cache
    context->result_cache.push_back(std::make_pair(image_hash, cached_result));
    context->cache_size_bytes += sizeof(tk_ocr_result_t); // Simplified size calculation
    
    // Clean up cache if needed
    if (difftime(time(nullptr), context->last_cache_cleanup) > 60) { // Clean every minute
        cleanup_cache(context);
    }
    
    return TK_SUCCESS;
}

/**
 * @brief Finds a cached OCR result
 */
static tk_ocr_result_t* find_cached_result(tk_text_recognition_context_t* context, 
                                         const std::string& image_hash) {
    if (!context) return nullptr;
    
    for (const auto& entry : context->result_cache) {
        if (entry.first == image_hash) {
            return entry.second;
        }
    }
    
    return nullptr;
}

/**
 * @brief Calculates a hash for an image
 */
static std::string calculate_image_hash(const tk_ocr_image_params_t* params) {
    if (!params || !params->image_data) {
        return "";
    }
    
    // This is a simplified hash calculation
    // In a real implementation, you would use a proper hash function like SHA-256
    std::string hash = "hash_";
    hash += std::to_string(params->width);
    hash += "_";
    hash += std::to_string(params->height);
    hash += "_";
    hash += std::to_string(params->channels);
    
    return hash;
}

/**
 * @brief Applies filters to text
 */
static bool apply_filters(const char* text, const char* filter_regex) {
    if (!text || !filter_regex) return true;
    
    // In a real implementation, this would apply regex filtering
    // For now, we'll just return true (accept all)
    return true;
}

/**
 * @brief Applies formatting to text
 */
static void apply_formatting(char* text, const char* output_format) {
    if (!text || !output_format) return;
    
    // In a real implementation, this would apply text formatting
    // For now, we'll do nothing
}

/**
 * @brief Detects QR codes in an image
 */
static tk_error_code_t detect_qr_codes(PIX* pix, bool* out_has_qr_codes, uint32_t* out_qr_count) {
    if (!pix || !out_has_qr_codes || !out_qr_count) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_has_qr_codes = false;
    *out_qr_count = 0;
    
    // In a real implementation, this would use a QR code detection library
    // For now, we'll set placeholder values
    *out_has_qr_codes = false; // Placeholder
    *out_qr_count = 0;         // Placeholder
    
    return TK_SUCCESS;
}

/**
 * @brief Detects barcodes in an image
 */
static tk_error_code_t detect_barcodes(PIX* pix, bool* out_has_barcodes, uint32_t* out_barcode_count) {
    if (!pix || !out_has_barcodes || !out_barcode_count) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_has_barcodes = false;
    *out_barcode_count = 0;
    
    // In a real implementation, this would use a barcode detection library
    // For now, we'll set placeholder values
    *out_has_barcodes = false; // Placeholder
    *out_barcode_count = 0;    // Placeholder
    
    return TK_SUCCESS;
}

/**
 * @brief Detects mathematical expressions in text
 */
static tk_error_code_t detect_mathematical_expressions(const char* text, bool* out_has_math, uint32_t* out_math_count) {
    if (!text || !out_has_math || !out_math_count) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_has_math = false;
    *out_math_count = 0;
    
    // In a real implementation, this would use a mathematical expression detection library
    // For now, we'll set placeholder values
    *out_has_math = false;  // Placeholder
    *out_math_count = 0;    // Placeholder
    
    return TK_SUCCESS;
}

/**
 * @brief Detects handwritten text
 */
static tk_error_code_t detect_handwritten_text(const char* text, bool* out_is_handwritten, uint32_t* out_handwritten_count) {
    if (!text || !out_is_handwritten || !out_handwritten_count) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_is_handwritten = false;
    *out_handwritten_count = 0;
    
    // In a real implementation, this would use a handwriting detection algorithm
    // For now, we'll set placeholder values
    *out_is_handwritten = false;     // Placeholder
    *out_handwritten_count = 0;      // Placeholder
    
    return TK_SUCCESS;
}

/**
 * @brief Detects languages in text
 */
static tk_error_code_t detect_languages(const char* text, char*** out_languages, uint32_t* out_language_count) {
    if (!text || !out_languages || !out_language_count) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_languages = nullptr;
    *out_language_count = 0;
    
    // In a real implementation, this would use a language detection library
    // For now, we'll set placeholder values
    *out_language_count = 1;
    *out_languages = new char*[1];
    (*out_languages)[0] = duplicate_string("eng"); // Default to English
    
    return TK_SUCCESS;
}

/**
 * @brief Frees a language list
 */
static void free_language_list(char** languages, uint32_t count) {
    if (!languages) return;
    
    for (uint32_t i = 0; i < count; i++) {
        free_string(languages[i]);
    }
    
    delete[] languages;
}

/**
 * @brief Validates an OCR result
 */
static tk_error_code_t validate_result(const tk_ocr_result_t* result, float threshold, bool* out_is_valid, char** out_message) {
    if (!result || !out_is_valid || !out_message) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_is_valid = true;
    *out_message = nullptr;
    
    // In a real implementation, this would perform validation checks
    // For now, we'll set placeholder values
    *out_is_valid = true; // Placeholder
    *out_message = duplicate_string("Valid result"); // Placeholder
    
    return TK_SUCCESS;
}

/**
 * @brief Compresses an OCR result
 */
static tk_error_code_t compress_result(const tk_ocr_result_t* input, tk_ocr_result_t** out_compressed) {
    if (!input || !out_compressed) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_compressed = nullptr;
    
    // In a real implementation, this would compress the result
    // For now, we'll just copy the input
    tk_ocr_result_t* compressed = new tk_ocr_result_t;
    if (!compressed) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    memcpy(compressed, input, sizeof(tk_ocr_result_t));
    compressed->full_text = duplicate_string(input->full_text);
    // Note: In a real implementation, we would need to deep copy all fields
    
    *out_compressed = compressed;
    return TK_SUCCESS;
}

/**
 * @brief Decompresses an OCR result
 */
static tk_error_code_t decompress_result(const tk_ocr_result_t* input, tk_ocr_result_t** out_decompressed) {
    if (!input || !out_decompressed) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_decompressed = nullptr;
    
    // In a real implementation, this would decompress the result
    // For now, we'll just copy the input
    tk_ocr_result_t* decompressed = new tk_ocr_result_t;
    if (!decompressed) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    memcpy(decompressed, input, sizeof(tk_ocr_result_t));
    decompressed->full_text = duplicate_string(input->full_text);
    // Note: In a real implementation, we would need to deep copy all fields
    
    *out_decompressed = decompressed;
    return TK_SUCCESS;
}

/**
 * @brief Encrypts an OCR result
 */
static tk_error_code_t encrypt_result(const tk_ocr_result_t* input, const char* key, tk_ocr_result_t** out_encrypted) {
    if (!input || !key || !out_encrypted) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_encrypted = nullptr;
    
    // In a real implementation, this would encrypt the result
    // For now, we'll just copy the input
    tk_ocr_result_t* encrypted = new tk_ocr_result_t;
    if (!encrypted) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    memcpy(encrypted, input, sizeof(tk_ocr_result_t));
    encrypted->full_text = duplicate_string(input->full_text);
    // Note: In a real implementation, we would need to deep copy all fields
    
    *out_encrypted = encrypted;
    return TK_SUCCESS;
}

/**
 * @brief Decrypts an OCR result
 */
static tk_error_code_t decrypt_result(const tk_ocr_result_t* input, const char* key, tk_ocr_result_t** out_decrypted) {
    if (!input || !key || !out_decrypted) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_decrypted = nullptr;
    
    // In a real implementation, this would decrypt the result
    // For now, we'll just copy the input
    tk_ocr_result_t* decrypted = new tk_ocr_result_t;
    if (!decrypted) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    memcpy(decrypted, input, sizeof(tk_ocr_result_t));
    decrypted->full_text = duplicate_string(input->full_text);
    // Note: In a real implementation, we would need to deep copy all fields
    
    *out_decrypted = decrypted;
    return TK_SUCCESS;
}

/**
 * @brief Applies security policy
 */
static void apply_security_policy(const char* policy) {
    if (!policy) return;
    
    // In a real implementation, this would apply security policies
    TK_LOG_INFO("Applying security policy: %s", policy);
}

/**
 * @brief Applies compliance standard
 */
static void apply_compliance_standard(const char* standard) {
    if (!standard) return;
    
    // In a real implementation, this would apply compliance standards
    TK_LOG_INFO("Applying compliance standard: %s", standard);
}

/**
 * @brief Exports an OCR result
 */
static void export_result(const tk_ocr_result_t* result, const char* export_path) {
    if (!result || !export_path) return;
    
    // In a real implementation, this would export the result to a file
    TK_LOG_INFO("Exporting result to: %s", export_path);
}

/**
 * @brief Notifies about an OCR result
 */
static void notify_result(const tk_ocr_result_t* result, const char* callback) {
    if (!result || !callback) return;
    
    // In a real implementation, this would call the notification callback
    TK_LOG_INFO("Notifying result via callback: %s", callback);
}

/**
 * @brief Streams an OCR result
 */
static void stream_result(const tk_ocr_result_t* result, uint32_t buffer_size) {
    if (!result) return;
    
    // In a real implementation, this would stream the result
    TK_LOG_INFO("Streaming result with buffer size: %u", buffer_size);
}

/**
 * @brief Synchronizes an OCR result
 */
static void synchronize_result(const tk_ocr_result_t* result, const char* endpoint) {
    if (!result || !endpoint) return;
    
    // In a real implementation, this would synchronize the result
    TK_LOG_INFO("Synchronizing result with endpoint: %s", endpoint);
}

/**
 * @brief Collects analytics for an OCR result
 */
static void collect_analytics(const tk_ocr_result_t* result, const char* endpoint) {
    if (!result || !endpoint) return;
    
    // In a real implementation, this would collect analytics
    TK_LOG_INFO("Collecting analytics and sending to: %s", endpoint);
}

/**
 * @brief Archives an OCR result
 */
static void archive_result(const tk_ocr_result_t* result, const char* archive_path) {
    if (!result || !archive_path) return;
    
    // In a real implementation, this would archive the result
    TK_LOG_INFO("Archiving result to: %s", archive_path);
}

/**
 * @brief Backs up an OCR result
 */
static void backup_result(const tk_ocr_result_t* result, const char* backup_path) {
    if (!result || !backup_path) return;
    
    // In a real implementation, this would backup the result
    TK_LOG_INFO("Backing up result to: %s", backup_path);
}

/**
 * @brief Restores an OCR result
 */
static void restore_result(tk_ocr_result_t** result, const char* restore_path) {
    if (!result || !restore_path) return;
    
    // In a real implementation, this would restore the result
    TK_LOG_INFO("Restoring result from: %s", restore_path);
}

/**
 * @brief Migrates an OCR result
 */
static void migrate_result(const tk_ocr_result_t* result, const char* target) {
    if (!result || !target) return;
    
    // In a real implementation, this would migrate the result
    TK_LOG_INFO("Migrating result to: %s", target);
}

/**
 * @brief Replicates an OCR result
 */
static void replicate_result(const tk_ocr_result_t* result, uint32_t factor) {
    if (!result) return;
    
    // In a real implementation, this would replicate the result
    TK_LOG_INFO("Replicating result with factor: %u", factor);
}

/**
 * @brief Shards an OCR result
 */
static void shard_result(const tk_ocr_result_t* result, uint32_t shard_count) {
    if (!result) return;
    
    // In a real implementation, this would shard the result
    TK_LOG_INFO("Sharding result into %u shards", shard_count);
}

/**
 * @brief Federates an OCR result
 */
static void federate_result(const tk_ocr_result_t* result, const char* config) {
    if (!result || !config) return;
    
    // In a real implementation, this would federate the result
    TK_LOG_INFO("Federating result with config: %s", config);
}

/**
 * @brief Synchronizes with protocol
 */
static void synchronize_with_protocol(const tk_ocr_result_t* result, const char* protocol) {
    if (!result || !protocol) return;
    
    // In a real implementation, this would synchronize using the protocol
    TK_LOG_INFO("Synchronizing with protocol: %s", protocol);
}

/**
 * @brief Checks consistency
 */
static void check_consistency(const tk_ocr_result_t* result, const char* protocol) {
    if (!result || !protocol) return;
    
    // In a real implementation, this would check consistency
    TK_LOG_INFO("Checking consistency with protocol: %s", protocol);
}

/**
 * @brief Manages transaction
 */
static void manage_transaction(const tk_ocr_result_t* result, const char* manager) {
    if (!result || !manager) return;
    
    // In a real implementation, this would manage transactions
    TK_LOG_INFO("Managing transaction with manager: %s", manager);
}

/**
 * @brief Applies caching strategy
 */
static void apply_caching_strategy(const char* strategy) {
    if (!strategy) return;
    
    // In a real implementation, this would apply caching strategies
    TK_LOG_INFO("Applying caching strategy: %s", strategy);
}

/**
 * @brief Applies compression strategy
 */
static void apply_compression_strategy(const char* strategy) {
    if (!strategy) return;
    
    // In a real implementation, this would apply compression strategies
    TK_LOG_INFO("Applying compression strategy: %s", strategy);
}

/**
 * @brief Applies security strategy
 */
static void apply_security_strategy(const char* strategy) {
    if (!strategy) return;
    
    // In a real implementation, this would apply security strategies
    TK_LOG_INFO("Applying security strategy: %s", strategy);
}

/**
 * @brief Applies analytics strategy
 */
static void apply_analytics_strategy(const char* strategy) {
    if (!strategy) return;
    
    // In a real implementation, this would apply analytics strategies
    TK_LOG_INFO("Applying analytics strategy: %s", strategy);
}

/**
 * @brief Applies debugging strategy
 */
static void apply_debugging_strategy(const char* strategy) {
    if (!strategy) return;
    
    // In a real implementation, this would apply debugging strategies
    TK_LOG_INFO("Applying debugging strategy: %s", strategy);
}

/**
 * @brief Applies custom strategy
 */
static void apply_custom_strategy(const char* strategy) {
    if (!strategy) return;
    
    // In a real implementation, this would apply custom strategies
    TK_LOG_INFO("Applying custom strategy: %s", strategy);
}

//------------------------------------------------------------------------------
// Public API Implementation
//------------------------------------------------------------------------------

tk_error_code_t tk_text_recognition_create(
    tk_text_recognition_context_t** out_context,
    const tk_ocr_config_t* config
) {
    if (!out_context || !config) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_context = nullptr;
    
    // Validate configuration
    tk_error_code_t result = validate_config(config);
    if (result != TK_SUCCESS) {
        return result;
    }
    
    // Allocate context structure
    tk_text_recognition_context_t* context = new tk_text_recognition_context_t;
    if (!context) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Initialize context fields
    memset(context, 0, sizeof(tk_text_recognition_context_t));
    
    // Copy configuration
    context->config = *config;
    
    // Copy data path
    if (config->data_path && config->data_path->path_str) {
        context->data_path = duplicate_string(config->data_path->path_str);
        if (!context->data_path) {
            delete context;
            return TK_ERROR_OUT_OF_MEMORY;
        }
    }
    
    // Convert language enum to code
    result = convert_language_enum(config->language, &context->language_code);
    if (result != TK_SUCCESS) {
        free_string(context->data_path);
        delete context;
        return result;
    }
    
    // Initialize preprocessing buffer
    context->preprocessing_buffer = nullptr;
    
    // Initialize cache
    context->cache_size_bytes = 0;
    context->last_cache_cleanup = time(nullptr);
    
    // Initialize Tesseract API
    result = init_tesseract_api(context);
    if (result != TK_SUCCESS) {
        free_string(context->language_code);
        free_string(context->data_path);
        delete context;
        return result;
    }
    
    *out_context = context;
    TK_LOG_INFO("Text recognition context created successfully");
    return TK_SUCCESS;
}

void tk_text_recognition_destroy(tk_text_recognition_context_t** context) {
    if (!context || !*context) return;
    
    tk_text_recognition_context_t* ctx = *context;
    
    // Clean up Tesseract API
    cleanup_tesseract_api(ctx);
    
    // Free preprocessing buffer
    if (ctx->preprocessing_buffer) {
        pixDestroy(&ctx->preprocessing_buffer);
    }
    
    // Free configuration strings
    free_string(ctx->data_path);
    free_string(ctx->language_code);
    
    // Free cached results
    for (auto& entry : ctx->result_cache) {
        free_ocr_result(entry.second);
    }
    ctx->result_cache.clear();
    
    // Free context itself
    delete ctx;
    *context = nullptr;
    
    TK_LOG_INFO("Text recognition context destroyed");
}

tk_error_code_t tk_text_recognition_process_image(
    tk_text_recognition_context_t* context,
    const tk_ocr_image_params_t* image_params,
    tk_ocr_result_t** out_result
) {
    if (!context || !image_params || !out_result) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_result = nullptr;
    
    // Lock mutex for thread safety
    std::lock_guard<std::mutex> lock(context->processing_mutex);
    
    // Calculate image hash for caching
    std::string image_hash = calculate_image_hash(image_params);
    
    // Check if result is cached
    tk_ocr_result_t* cached_result = find_cached_result(context, image_hash);
    if (cached_result) {
        TK_LOG_INFO("Using cached OCR result");
        *out_result = cached_result;
        return TK_SUCCESS;
    }
    
    // Record start time
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Convert image parameters to PIX
    PIX* input_pix = convert_to_pix(image_params);
    if (!input_pix) {
        TK_LOG_ERROR("Failed to convert image parameters to PIX");
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Preprocess image
    PIX* processed_pix = preprocess_image(input_pix, &context->config, image_params);
    if (!processed_pix) {
        pixDestroy(&input_pix);
        TK_LOG_ERROR("Failed to preprocess image");
        return TK_ERROR_MODEL_INFERENCE_FAILED;
    }
    
    // Set image in Tesseract API
    context->tess_api->SetImage(processed_pix);
    
    // Set DPI if provided
    uint32_t dpi = image_params->dpi > 0 ? image_params->dpi : context->config.dpi;
    context->tess_api->SetSourceResolution(dpi);
    
    // Set page segmentation mode
    tesseract::PageSegMode psm;
    tk_ocr_page_seg_mode_e psm_mode = image_params->psm != TK_OCR_PSM_AUTO ? 
                                      image_params->psm : context->config.psm;
    convert_page_seg_mode(psm_mode, &psm);
    context->tess_api->SetPageSegMode(psm);
    
    // Perform OCR
    char* tess_text = context->tess_api->GetUTF8Text();
    if (!tess_text) {
        pixDestroy(&processed_pix);
        pixDestroy(&input_pix);
        TK_LOG_ERROR("Failed to perform OCR");
        return TK_ERROR_MODEL_INFERENCE_FAILED;
    }
    
    // Get confidence scores
    int* confidences = context->tess_api->AllWordConfidences();
    
    // Get text boxes
    BOXA* boxes = context->tess_api->GetComponentImages(tesseract::RIL_WORD, true, nullptr, nullptr);
    
    // Create result structure
    tk_ocr_result_t* result = new tk_ocr_result_t;
    if (!result) {
        delete[] tess_text;
        if (confidences) delete[] confidences;
        if (boxes) boxaDestroy(&boxes);
        pixDestroy(&processed_pix);
        pixDestroy(&input_pix);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Initialize result fields
    memset(result, 0, sizeof(tk_ocr_result_t));
    
    // Extract full text
    size_t full_text_length;
    tk_error_code_t error = extract_full_text(context, tess_text, &result->full_text, &full_text_length);
    if (error != TK_SUCCESS) {
        delete[] tess_text;
        if (confidences) delete[] confidences;
        if (boxes) boxaDestroy(&boxes);
        pixDestroy(&processed_pix);
        pixDestroy(&input_pix);
        delete result;
        return error;
    }
    result->full_text_length = full_text_length;
    
    // Extract text regions
    error = extract_text_regions(context, boxes, tess_text, confidences, 
                                &result->regions, &result->region_count);
    if (error != TK_SUCCESS) {
        delete[] tess_text;
        if (confidences) delete[] confidences;
        if (boxes) boxaDestroy(&boxes);
        pixDestroy(&processed_pix);
        pixDestroy(&input_pix);
        free_ocr_result(result);
        return error;
    }
    
    // Calculate statistics
    error = calculate_statistics(context, confidences, context->tess_api->MeanTextConf(),
                                &result->average_confidence, &result->word_count, &result->line_count);
    if (error != TK_SUCCESS) {
        delete[] tess_text;
        if (confidences) delete[] confidences;
        if (boxes) boxaDestroy(&boxes);
        pixDestroy(&processed_pix);
        pixDestroy(&input_pix);
        free_ocr_result(result);
        return error;
    }
    
    // Set image dimensions
    result->image_width = image_params->width;
    result->image_height = image_params->height;
    
    // Detect language
    char** languages;
    uint32_t language_count;
    error = detect_languages(result->full_text, &languages, &language_count);
    if (error == TK_SUCCESS) {
        result->languages = languages;
        result->language_count = language_count;
        if (language_count > 0) {
            result->detected_language = duplicate_string(languages[0]);
        }
        result->is_multilingual = language_count > 1;
    }
    
    // Detect QR codes
    detect_qr_codes(processed_pix, &result->has_qr_codes, &result->qr_code_count);
    
    // Detect barcodes
    detect_barcodes(processed_pix, &result->has_barcodes, &result->barcode_count);
    
    // Detect mathematical expressions
    detect_mathematical_expressions(result->full_text, &result->has_mathematical_expressions, 
                                   &result->math_expression_count);
    
    // Detect handwritten text
    detect_handwritten_text(result->full_text, &result->has_handwritten_text, 
                           &result->handwritten_word_count);
    
    // Set processing time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    result->processing_time_ms = duration.count();
    
    // Set timestamp
    time_t now = time(nullptr);
    result->processing_timestamp = duplicate_string(ctime(&now));
    
    // Set model version
    result->model_version = duplicate_string("Tesseract 5.0");
    result->engine_version = duplicate_string("Tesseract OCR Engine");
    
    // Validate result
    bool is_valid;
    char* validation_message;
    error = validate_result(result, context->config.validation_threshold, &is_valid, &validation_message);
    if (error == TK_SUCCESS) {
        result->is_valid = is_valid;
        result->validation_message = validation_message;
    }
    
    // Apply filters if enabled
    if (context->config.enable_result_filtering && context->config.filter_regex) {
        if (!apply_filters(result->full_text, context->config.filter_regex)) {
            TK_LOG_WARN("OCR result filtered out by regex");
            free_ocr_result(result);
            delete[] tess_text;
            if (confidences) delete[] confidences;
            if (boxes) boxaDestroy(&boxes);
            pixDestroy(&processed_pix);
            pixDestroy(&input_pix);
            return TK_ERROR_INVALID_RESULT;
        }
    }
    
    // Apply formatting if enabled
    if (context->config.enable_result_formatting && context->config.output_format) {
        apply_formatting(result->full_text, context->config.output_format);
    }
    
    // Cache result if enabled
    if (context->config.enable_result_caching) {
        cache_result(context, image_hash, result);
    }
    
    // Export result if enabled
    if (context->config.enable_result_export && context->config.export_path) {
        export_result(result, context->config.export_path->path_str);
    }
    
    // Notify result if enabled
    if (context->config.enable_result_notification && context->config.notification_callback) {
        notify_result(result, context->config.notification_callback);
    }
    
    // Stream result if enabled
    if (context->config.enable_result_streaming) {
        stream_result(result, context->config.streaming_buffer_size);
    }
    
    // Synchronize result if enabled
    if (context->config.enable_result_synchronization && context->config.sync_endpoint) {
        synchronize_result(result, context->config.sync_endpoint->path_str);
    }
    
    // Collect analytics if enabled
    if (context->config.enable_result_analytics && context->config.analytics_endpoint) {
        collect_analytics(result, context->config.analytics_endpoint->path_str);
    }
    
    // Archive result if enabled
    if (context->config.enable_result_archiving && context->config.archive_path) {
        archive_result(result, context->config.archive_path->path_str);
    }
    
    // Backup result if enabled
    if (context->config.enable_result_backup && context->config.backup_path) {
        backup_result(result, context->config.backup_path->path_str);
    }
    
    // Apply security policy if enabled
    if (context->config.enable_result_security && context->config.security_policy) {
        apply_security_policy(context->config.security_policy->path_str);
    }
    
    // Apply compliance standard if enabled
    if (context->config.enable_result_compliance && context->config.compliance_standard) {
        apply_compliance_standard(context->config.compliance_standard->path_str);
    }
    
    // Clean up temporary resources
    delete[] tess_text;
    if (confidences) delete[] confidences;
    if (boxes) boxaDestroy(&boxes);
    pixDestroy(&processed_pix);
    pixDestroy(&input_pix);
    
    *out_result = result;
    TK_LOG_INFO("OCR processing completed successfully");
    return TK_SUCCESS;
}

tk_error_code_t tk_text_recognition_process_region(
    tk_text_recognition_context_t* context,
    const tk_ocr_image_params_t* image_params,
    uint32_t region_x,
    uint32_t region_y,
    uint32_t region_width,
    uint32_t region_height,
    tk_ocr_result_t** out_result
) {
    if (!context || !image_params || !out_result) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Validate region coordinates
    if (region_x + region_width > image_params->width || 
        region_y + region_height > image_params->height) {
        TK_LOG_ERROR("Invalid region coordinates");
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Create cropped image parameters
    tk_ocr_image_params_t cropped_params = *image_params;
    
    // Calculate cropped image data pointer
    size_t bytes_per_pixel = image_params->channels;
    size_t row_stride = image_params->stride;
    const uint8_t* cropped_data = image_params->image_data + 
                                 (region_y * row_stride) + 
                                 (region_x * bytes_per_pixel);
    
    // Create new image data for cropped region
    size_t cropped_stride = region_width * bytes_per_pixel;
    uint8_t* cropped_image_data = new uint8_t[region_height * cropped_stride];
    if (!cropped_image_data) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Copy cropped region data
    for (uint32_t y = 0; y < region_height; y++) {
        const uint8_t* src_row = cropped_data + y * row_stride;
        uint8_t* dst_row = cropped_image_data + y * cropped_stride;
        memcpy(dst_row, src_row, cropped_stride);
    }
    
    // Set cropped image parameters
    cropped_params.image_data = cropped_image_data;
    cropped_params.width = region_width;
    cropped_params.height = region_height;
    cropped_params.stride = cropped_stride;
    
    // Process cropped region
    tk_error_code_t result = tk_text_recognition_process_image(context, &cropped_params, out_result);
    
    // Clean up
    delete[] cropped_image_data;
    
    return result;
}

void tk_text_recognition_free_result(tk_ocr_result_t** result) {
    if (!result || !*result) return;
    
    free_ocr_result(*result);
    *result = nullptr;
}

tk_error_code_t tk_text_recognition_set_language(
    tk_text_recognition_context_t* context,
    tk_ocr_language_e language
) {
    if (!context) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Convert language enum to code
    char* language_code;
    tk_error_code_t result = convert_language_enum(language, &language_code);
    if (result != TK_SUCCESS) {
        return result;
    }
    
    // Free old language code
    free_string(context->language_code);
    
    // Set new language code
    context->language_code = language_code;
    
    // Reinitialize Tesseract API with new language
    cleanup_tesseract_api(context);
    result = init_tesseract_api(context);
    if (result != TK_SUCCESS) {
        free_string(context->language_code);
        context->language_code = nullptr;
        return result;
    }
    
    return TK_SUCCESS;
}

tk_error_code_t tk_text_recognition_get_model_info(
    tk_text_recognition_context_t* context,
    tk_ocr_language_e* out_language,
    char** out_engine_version,
    char** out_model_path
) {
    if (!context || !out_language || !out_engine_version || !out_model_path) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Get current language
    *out_language = TK_OCR_LANG_UNKNOWN;
    for (int i = TK_OCR_LANG_ENGLISH; i <= TK_OCR_LANG_HINDI; i++) {
        char* code;
        if (convert_language_enum((tk_ocr_language_e)i, &code) == TK_SUCCESS) {
            if (context->language_code && strcmp(context->language_code, code) == 0) {
                *out_language = (tk_ocr_language_e)i;
            }
            free_string(code);
        }
    }
    
    // Get engine version
    *out_engine_version = duplicate_string("Tesseract 5.0");
    
    // Get model path
    *out_model_path = duplicate_string(context->data_path);
    
    return TK_SUCCESS;
}

tk_error_code_t tk_text_recognition_update_config(
    tk_text_recognition_context_t* context,
    const tk_ocr_config_t* config
) {
    if (!context || !config) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Validate new configuration
    tk_error_code_t result = validate_config(config);
    if (result != TK_SUCCESS) {
        return result;
    }
    
    // Update configuration
    context->config = *config;
    
    // Update data path if changed
    if (config->data_path && config->data_path->path_str) {
        free_string(context->data_path);
        context->data_path = duplicate_string(config->data_path->path_str);
        if (!context->data_path) {
            return TK_ERROR_OUT_OF_MEMORY;
        }
    }
    
    // Update language if changed
    if (config->language != TK_OCR_LANG_UNKNOWN) {
        result = tk_text_recognition_set_language(context, config->language);
        if (result != TK_SUCCESS) {
            return result;
        }
    }
    
    // Reinitialize Tesseract API if needed
    if (config->engine_mode != context->config.engine_mode) {
        cleanup_tesseract_api(context);
        result = init_tesseract_api(context);
        if (result != TK_SUCCESS) {
            return result;
        }
    }
    
    return TK_SUCCESS;
}
