/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_vision_pipeline.c
 *
 * This source file implements the TrackieLLM Vision Pipeline.
 * It provides the core logic for loading models, processing video frames,
 * and fusing sensor data to create a semantic understanding of the environment.
 *
 * The implementation is designed to be modular, efficient, and robust,
 * handling model loading, inference execution, and memory management with care.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "tk_vision_pipeline.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <float.h>

// Assume these headers exist and provide the necessary functionality
#include "vision/tk_object_detector.h"
#include "vision/tk_depth_midas.h"
#include "vision/tk_text_recognition.hpp"
#include "utils/tk_logging.h"
#include "vision/tk_vision_ffi_defs.h"
#include "event_bus/event_bus_ffi.h"

// Opaque structure for the vision pipeline
struct tk_vision_pipeline_s {
    tk_event_bus_t* event_bus; // Handle to the Rust event bus
    tk_vision_backend_e backend;
    int gpu_device_id;
    
    // Model instances
    tk_object_detector_t* object_detector;
    tk_depth_midas_t* depth_estimator;
    tk_text_recognition_context_t* text_recognizer;
    
    // Configuration
    float object_confidence_threshold;
    uint32_t max_detected_objects;
    float focal_length_x;
    float focal_length_y;
};

// FFI declarations for Rust functions are now in the included headers

// Internal helper functions
static tk_error_code_t load_models(tk_vision_pipeline_t* pipeline, const tk_vision_pipeline_config_t* config);
static void unload_models(tk_vision_pipeline_t* pipeline);
static bool is_ocr_relevant_object(const char* label);
static tk_error_code_t perform_conditional_ocr(tk_vision_pipeline_t* pipeline, const tk_video_frame_t* frame, tk_vision_result_t* result);
static void publish_vision_event(tk_vision_pipeline_t* pipeline, const tk_vision_result_t* result);
static tk_error_code_t fuse_object_depth(tk_vision_pipeline_t* pipeline, tk_vision_result_t* result, const tk_video_frame_t* frame);

//------------------------------------------------------------------------------
// Pipeline Lifecycle Management
//------------------------------------------------------------------------------

TK_NODISCARD tk_error_code_t tk_vision_pipeline_create(tk_vision_pipeline_t** out_pipeline, const tk_vision_pipeline_config_t* config) {
    if (!out_pipeline || !config) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    tk_vision_pipeline_t* pipeline = (tk_vision_pipeline_t*)calloc(1, sizeof(tk_vision_pipeline_t));
    if (!pipeline) {
        return TK_ERROR_OUT_OF_MEMORY;
    }

    pipeline->event_bus = config->event_bus;
    pipeline->backend = config->backend;
    pipeline->gpu_device_id = config->gpu_device_id;
    pipeline->object_confidence_threshold = config->object_confidence_threshold;
    pipeline->max_detected_objects = config->max_detected_objects;
    // Initialize camera parameters from config
    pipeline->focal_length_x = config->focal_length_x;
    pipeline->focal_length_y = config->focal_length_y;

    tk_error_code_t err = load_models(pipeline, config);
    if (err != TK_SUCCESS) {
        free(pipeline);
        return err;
    }

    *out_pipeline = pipeline;
    tk_log_info("Vision pipeline created successfully with focal lengths: fx=%.2f, fy=%.2f", 
                pipeline->focal_length_x, pipeline->focal_length_y);
    return TK_SUCCESS;
}

void tk_vision_pipeline_destroy(tk_vision_pipeline_t** pipeline) {
    if (pipeline && *pipeline) {
        tk_log_info("Destroying vision pipeline");
        unload_models(*pipeline);
        free(*pipeline);
        *pipeline = NULL;
    }
}

//------------------------------------------------------------------------------
// Core Processing Function
//------------------------------------------------------------------------------

TK_NODISCARD tk_error_code_t tk_vision_pipeline_process_frame(
    tk_vision_pipeline_t* pipeline,
    const tk_video_frame_t* video_frame,
    tk_vision_analysis_flags_t analysis_flags,
    uint64_t timestamp_ns,
    tk_vision_result_t** out_result
) {
    if (!pipeline || !video_frame || !out_result) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    tk_detection_result_t* raw_detections = NULL;
    size_t raw_detection_count = 0;
    tk_vision_depth_map_t* depth_map = NULL;
    tk_error_code_t err = TK_SUCCESS;

    // --- Perform Raw Model Inference ---
    if (analysis_flags & TK_VISION_ANALYZE_OBJECT_DETECTION) {
        err = tk_object_detector_detect(pipeline->object_detector, video_frame, &raw_detections, &raw_detection_count);
        if (err != TK_SUCCESS) {
            tk_log_error("Object detection failed: %d", err);
            return err;
        }
    }

    if (analysis_flags & TK_VISION_ANALYZE_DEPTH_ESTIMATION) {
        err = tk_depth_estimator_estimate(pipeline->depth_estimator, video_frame, &depth_map);
        if (err != TK_SUCCESS) {
            tk_log_error("Depth estimation failed: %d", err);
            tk_object_detector_free_results(&raw_detections);
            return err;
        }
    }

    // --- FFI Call to Rust for Fusion and Analysis ---
    tk_yolo_output_t yolo_out = {
        .detections = raw_detections,
        .count = raw_detection_count,
        .frame_width = video_frame->width,
        .frame_height = video_frame->height
    };
    tk_midas_output_t midas_out = { .depth_map = depth_map };
    
    tk_fused_result_t* fused_result = rust_process_vision_data(&yolo_out, &midas_out);

    // We are done with the raw detections, free them now.
    tk_object_detector_free_results(&raw_detections);

    // --- Create Final C-Managed Result ---
    tk_vision_result_t* final_result = (tk_vision_result_t*)calloc(1, sizeof(tk_vision_result_t));
    if (!final_result) {
        rust_free_fused_result(fused_result);
        tk_depth_estimator_free_map(&depth_map);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    final_result->source_frame_timestamp_ns = timestamp_ns;
    final_result->depth_map = depth_map; // Transfer ownership of depth map

    if (fused_result && fused_result->count > 0) {
        // Copy the enriched objects from the Rust-managed memory to C-managed memory
        final_result->object_count = fused_result->count;
        final_result->objects = (tk_vision_object_t*)calloc(fused_result->count, sizeof(tk_vision_object_t));
        if (!final_result->objects) {
            rust_free_fused_result(fused_result);
            tk_vision_result_destroy(&final_result);
            return TK_ERROR_OUT_OF_MEMORY;
        }

        for (size_t i = 0; i < fused_result->count; ++i) {
            final_result->objects[i].class_id = fused_result->objects[i].class_id;
            final_result->objects[i].label = fused_result->objects[i].label;
            final_result->objects[i].confidence = fused_result->objects[i].confidence;
            final_result->objects[i].bbox = fused_result->objects[i].bbox;
            final_result->objects[i].distance_meters = fused_result->objects[i].distance_meters;
            final_result->objects[i].width_meters = -1.0f;
            final_result->objects[i].height_meters = -1.0f;
        }
    }

    // Free the memory allocated by Rust now that we have our own copy.
    rust_free_fused_result(fused_result);

    // --- Perform Conditional OCR ---
    if (analysis_flags & TK_VISION_ANALYZE_OCR) {
        err = perform_conditional_ocr(pipeline, video_frame, final_result);
        if (err != TK_SUCCESS) {
            // Log a warning but don't fail the entire pipeline for OCR failure
            tk_log_warn("Conditional OCR failed with error code: %d", err);
        }
    }

    // --- Publish Final Result to Event Bus ---
    publish_vision_event(pipeline, final_result);

    *out_result = final_result;
    return TK_SUCCESS;
}

//------------------------------------------------------------------------------
// Result Data Management
//------------------------------------------------------------------------------

void tk_vision_result_destroy(tk_vision_result_t** result) {
    if (result && *result) {
        // Free objects
        if ((*result)->objects) {
            // Note: The `label` field within tk_vision_object_t points to C-managed
            // config data, so we don't free it individually. We just free the array.
            free((*result)->objects);
        }

        // Free text blocks
        if ((*result)->text_blocks) {
            for (size_t i = 0; i < (*result)->text_block_count; ++i) {
                if ((*result)->text_blocks[i].text) {
                    free((void*)((*result)->text_blocks[i].text));
                }
            }
            free((*result)->text_blocks);
        }

        // Free depth map
        if ((*result)->depth_map) {
            if ((*result)->depth_map->data) {
                free((*result)->depth_map->data);
            }
            free((*result)->depth_map);
        }

        free(*result);
        *result = NULL;
    }
}

//------------------------------------------------------------------------------
// Internal Helper Functions
//------------------------------------------------------------------------------

static tk_error_code_t load_models(tk_vision_pipeline_t* pipeline, const tk_vision_pipeline_config_t* config) {
    tk_error_code_t err;

    // Validate config paths
    if (!config->object_detection_model_path || 
        !config->depth_estimation_model_path || 
        !config->tesseract_data_path) {
        tk_log_error("Missing required model paths in configuration");
        return TK_ERROR_INVALID_ARGUMENT;
    }

    tk_log_info("Loading vision models...");

    // Load object detector
    err = tk_object_detector_create(&pipeline->object_detector, config->object_detection_model_path);
    if (err != TK_SUCCESS) {
        tk_log_error("Failed to load object detection model from %s", config->object_detection_model_path);
        return TK_ERROR_MODEL_LOAD_FAILED;
    }
    tk_log_debug("Object detection model loaded successfully");

    // Load depth estimator
    err = tk_depth_midas_create(&pipeline->depth_estimator, config->depth_estimation_model_path);
    if (err != TK_SUCCESS) {
        tk_log_error("Failed to load depth estimation model from %s", config->depth_estimation_model_path);
        tk_object_detector_destroy(&pipeline->object_detector);
        return TK_ERROR_MODEL_LOAD_FAILED;
    }
    tk_log_debug("Depth estimation model loaded successfully");

    // Load text recognizer
    err = tk_text_recognition_create(&pipeline->text_recognizer, config->tesseract_data_path);
    if (err != TK_SUCCESS) {
        tk_log_error("Failed to initialize text recognizer with data path %s", config->tesseract_data_path);
        tk_object_detector_destroy(&pipeline->object_detector);
        tk_depth_midas_destroy(&pipeline->depth_estimator);
        return TK_ERROR_MODEL_LOAD_FAILED;
    }
    tk_log_debug("Text recognizer initialized successfully");

    return TK_SUCCESS;
}

static void unload_models(tk_vision_pipeline_t* pipeline) {
    if (pipeline->object_detector) {
        tk_object_detector_destroy(&pipeline->object_detector);
    }
    if (pipeline->depth_estimator) {
        tk_depth_midas_destroy(&pipeline->depth_estimator);
    }
    if (pipeline->text_recognizer) {
        tk_text_recognition_destroy(&pipeline->text_recognizer);
    }
}

static bool is_ocr_relevant_object(const char* label) {
    if (!label) return false;
    // Add more relevant labels as needed by the model
    const char* relevant_labels[] = {"sign", "book", "label", "text", "screen", "monitor", "plate", NULL};
    for (int i = 0; relevant_labels[i] != NULL; ++i) {
        if (strstr(label, relevant_labels[i]) != NULL) {
            return true;
        }
    }
    return false;
}

static tk_error_code_t perform_conditional_ocr(tk_vision_pipeline_t* pipeline, const tk_video_frame_t* frame, tk_vision_result_t* result) {
    if (!pipeline || !frame || !result || result->object_count == 0) {
        return TK_SUCCESS; // Nothing to do
    }

    tk_ocr_image_params_t image_params = {};
    image_params.image_data = frame->data;
    image_params.width = frame->width;
    image_params.height = frame->height;
    image_params.channels = 3; // Assuming RGB
    image_params.stride = frame->width * 3;

    // Use a dynamic array for text blocks
    size_t text_block_capacity = 4;
    result->text_blocks = (tk_vision_text_block_t*)malloc(text_block_capacity * sizeof(tk_vision_text_block_t));
    if (!result->text_blocks) return TK_ERROR_OUT_OF_MEMORY;
    result->text_block_count = 0;

    for (size_t i = 0; i < result->object_count; ++i) {
        if (is_ocr_relevant_object(result->objects[i].label)) {
            tk_rect_t bbox = result->objects[i].bbox;

            // Ensure bbox is within frame bounds
            if (bbox.x < 0) { bbox.w += bbox.x; bbox.x = 0; }
            if (bbox.y < 0) { bbox.h += bbox.y; bbox.y = 0; }
            if (bbox.x + bbox.w > frame->width) { bbox.w = frame->width - bbox.x; }
            if (bbox.y + bbox.h > frame->height) { bbox.h = frame->height - bbox.y; }
            if (bbox.w <= 0 || bbox.h <= 0) continue;

            tk_ocr_result_t* ocr_result = NULL;
            tk_error_code_t err = tk_text_recognition_process_region(
                pipeline->text_recognizer, &image_params, bbox.x, bbox.y, bbox.w, bbox.h, &ocr_result
            );

            if (err == TK_SUCCESS && ocr_result && ocr_result->full_text && ocr_result->full_text[0] != '\0') {
                if (result->text_block_count >= text_block_capacity) {
                    text_block_capacity *= 2;
                    tk_vision_text_block_t* new_blocks = (tk_vision_text_block_t*)realloc(result->text_blocks, text_block_capacity * sizeof(tk_vision_text_block_t));
                    if (!new_blocks) {
                        tk_text_recognition_free_result(&ocr_result);
                        continue; // Continue without this result
                    }
                    result->text_blocks = new_blocks;
                }

                tk_vision_text_block_t* new_block = &result->text_blocks[result->text_block_count++];
                new_block->text = strdup(ocr_result->full_text);
                new_block->confidence = ocr_result->average_confidence;
                new_block->bbox = bbox; // The bbox of the object that triggered the OCR

                tk_log_debug("OCR found text '%s' in object '%s'", new_block->text, result->objects[i].label);
            }

            tk_text_recognition_free_result(&ocr_result);
        }
    }

    return TK_SUCCESS;
}

static void publish_vision_event(tk_vision_pipeline_t* pipeline, const tk_vision_result_t* result) {
    if (!pipeline->event_bus || !result) {
        return;
    }

    // Create a temporary array of FFI-safe objects
    FfiVisionObject* ffi_objects = NULL;
    if (result->object_count > 0) {
        ffi_objects = (FfiVisionObject*)malloc(result->object_count * sizeof(FfiVisionObject));
        if (!ffi_objects) return; // Cannot publish if allocation fails

        for (size_t i = 0; i < result->object_count; ++i) {
            ffi_objects[i].label = result->objects[i].label;
            ffi_objects[i].confidence = result->objects[i].confidence;
            ffi_objects[i].distance_meters = result->objects[i].distance_meters;
        }
    }

    FfiVisionResult ffi_result = {
        .objects = ffi_objects,
        .object_count = result->object_count,
        .timestamp_ns = result->source_frame_timestamp_ns
    };

    vision_publish_result(pipeline->event_bus, &ffi_result);

    // Clean up the temporary array
    if (ffi_objects) {
        free(ffi_objects);
    }
}
