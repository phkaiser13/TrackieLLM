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
#include "memory/tk_memory_pool.h"

// Opaque structure for the vision pipeline
struct tk_vision_pipeline_s {
    tk_vision_backend_e backend;
    int gpu_device_id;
    
    // Model instances
    tk_object_detector_t* object_detector;
    tk_depth_midas_t* depth_estimator;
    tk_text_recognizer_t* text_recognizer;
    
    // Configuration
    float object_confidence_threshold;
    uint32_t max_detected_objects;
    float focal_length_x;  // Added: Camera focal length in x direction
    float focal_length_y;  // Added: Camera focal length in y direction
};

// Internal helper functions
static tk_error_code_t load_models(tk_vision_pipeline_t* pipeline, const tk_vision_pipeline_config_t* config);
static void unload_models(tk_vision_pipeline_t* pipeline);
static tk_error_code_t perform_object_detection(tk_vision_pipeline_t* pipeline, const tk_video_frame_t* frame, tk_vision_result_t* result);
static tk_error_code_t perform_depth_estimation(tk_vision_pipeline_t* pipeline, const tk_video_frame_t* frame, tk_vision_result_t* result);
static tk_error_code_t perform_ocr(tk_vision_pipeline_t* pipeline, const tk_video_frame_t* frame, tk_vision_result_t* result);
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

    tk_vision_result_t* result = (tk_vision_result_t*)calloc(1, sizeof(tk_vision_result_t));
    if (!result) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    result->source_frame_timestamp_ns = timestamp_ns;

    tk_error_code_t err = TK_SUCCESS;

    // Perform requested analyses
    if (analysis_flags & TK_VISION_ANALYZE_OBJECT_DETECTION) {
        err = perform_object_detection(pipeline, video_frame, result);
        if (err != TK_SUCCESS) {
            tk_log_error("Object detection failed: %d", err);
            tk_vision_result_destroy(&result);
            return err;
        }
        tk_log_debug("Object detection completed, found %zu objects", result->object_count);
    }

    if (analysis_flags & TK_VISION_ANALYZE_DEPTH_ESTIMATION) {
        err = perform_depth_estimation(pipeline, video_frame, result);
        if (err != TK_SUCCESS) {
            tk_log_error("Depth estimation failed: %d", err);
            tk_vision_result_destroy(&result);
            return err;
        }
        tk_log_debug("Depth estimation completed");
    }

    // OCR is performed only when explicitly requested or when relevant objects are detected
    bool should_perform_ocr = (analysis_flags & TK_VISION_ANALYZE_OCR);
    
    // Check if any detected objects are text-related (e.g., signs, labels)
    if (!should_perform_ocr && (analysis_flags & TK_VISION_ANALYZE_OBJECT_DETECTION)) {
        for (size_t i = 0; i < result->object_count; ++i) {
            // Common text-related object classes (these would be defined in your object detection model)
            if (result->objects[i].class_id == 10 ||  // "sign"
                result->objects[i].class_id == 15 ||  // "label"
                result->objects[i].class_id == 20 ||  // "text"
                strstr(result->objects[i].label, "sign") ||
                strstr(result->objects[i].label, "text")) {
                should_perform_ocr = true;
                break;
            }
        }
    }

    if (should_perform_ocr) {
        err = perform_ocr(pipeline, video_frame, result);
        if (err != TK_SUCCESS) {
            tk_log_warning("OCR failed: %d, continuing without text recognition", err);
            // Don't fail the entire pipeline for OCR failure
        } else {
            tk_log_debug("OCR completed, found %zu text blocks", result->text_block_count);
        }
    }

    // Fusion is performed only when both object detection and depth estimation were successful
    if ((analysis_flags & TK_VISION_ANALYZE_FUSION_DISTANCE) &&
        (result->object_count > 0) && 
        (result->depth_map != NULL)) {
        err = fuse_object_depth(pipeline, result, video_frame);
        if (err != TK_SUCCESS) {
            tk_log_warning("Object-depth fusion failed: %d", err);
            // Don't fail the entire pipeline for fusion failure
        } else {
            tk_log_debug("Object-depth fusion completed");
        }
    }

    *out_result = result;
    return TK_SUCCESS;
}

//------------------------------------------------------------------------------
// Result Data Management
//------------------------------------------------------------------------------

void tk_vision_result_destroy(tk_vision_result_t** result) {
    if (result && *result) {
        // Free objects
        if ((*result)->objects) {
            for (size_t i = 0; i < (*result)->object_count; ++i) {
                if ((*result)->objects[i].label) {
                    free((void*)((*result)->objects[i].label));
                }
            }
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
    err = tk_text_recognizer_create(&pipeline->text_recognizer, config->tesseract_data_path);
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
        tk_text_recognizer_destroy(&pipeline->text_recognizer);
    }
}

static tk_error_code_t perform_object_detection(tk_vision_pipeline_t* pipeline, const tk_video_frame_t* frame, tk_vision_result_t* result) {
    tk_object_detection_result_t* detection_result = NULL;
    tk_error_code_t err = tk_object_detector_process(pipeline->object_detector, frame, &detection_result);
    if (err != TK_SUCCESS) {
        return err;
    }

    // Filter and copy results
    size_t valid_detections = 0;
    for (size_t i = 0; i < detection_result->detection_count; ++i) {
        if (detection_result->detections[i].confidence >= pipeline->object_confidence_threshold) {
            valid_detections++;
        }
    }

    valid_detections = (valid_detections > pipeline->max_detected_objects) ? pipeline->max_detected_objects : valid_detections;

    if (valid_detections > 0) {
        result->objects = (tk_vision_object_t*)calloc(valid_detections, sizeof(tk_vision_object_t));
        if (!result->objects) {
            tk_object_detection_result_destroy(&detection_result);
            return TK_ERROR_OUT_OF_MEMORY;
        }

        size_t idx = 0;
        for (size_t i = 0; i < detection_result->detection_count && idx < valid_detections; ++i) {
            if (detection_result->detections[i].confidence >= pipeline->object_confidence_threshold) {
                result->objects[idx].class_id = detection_result->detections[i].class_id;
                
                // Copy label string
                if (detection_result->detections[i].label) {
                    result->objects[idx].label = strdup(detection_result->detections[i].label);
                } else {
                    result->objects[idx].label = strdup("unknown");
                }
                
                result->objects[idx].confidence = detection_result->detections[i].confidence;
                result->objects[idx].bbox = detection_result->detections[i].bbox;
                result->objects[idx].distance_meters = 0.0f; // Will be populated by fusion
                result->objects[idx].width_meters = 0.0f;    // Will be populated by fusion
                result->objects[idx].height_meters = 0.0f;   // Will be populated by fusion
                idx++;
            }
        }
        result->object_count = valid_detections;
    }

    tk_object_detection_result_destroy(&detection_result);
    return TK_SUCCESS;
}

static tk_error_code_t perform_depth_estimation(tk_vision_pipeline_t* pipeline, const tk_video_frame_t* frame, tk_vision_result_t* result) {
    tk_depth_map_t* depth_map = NULL;
    tk_error_code_t err = tk_depth_midas_process(pipeline->depth_estimator, frame, &depth_map);
    if (err != TK_SUCCESS) {
        return err;
    }

    result->depth_map = (tk_vision_depth_map_t*)malloc(sizeof(tk_vision_depth_map_t));
    if (!result->depth_map) {
        tk_depth_map_destroy(&depth_map);
        return TK_ERROR_OUT_OF_MEMORY;
    }

    result->depth_map->width = depth_map->width;
    result->depth_map->height = depth_map->height;
    size_t data_size = depth_map->width * depth_map->height * sizeof(float);
    result->depth_map->data = (float*)malloc(data_size);
    if (!result->depth_map->data) {
        free(result->depth_map);
        result->depth_map = NULL;
        tk_depth_map_destroy(&depth_map);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    memcpy(result->depth_map->data, depth_map->data, data_size);

    tk_depth_map_destroy(&depth_map);
    return TK_SUCCESS;
}

static tk_error_code_t perform_ocr(tk_vision_pipeline_t* pipeline, const tk_video_frame_t* frame, tk_vision_result_t* result) {
    tk_ocr_result_t* ocr_result = NULL;
    tk_error_code_t err = tk_text_recognizer_process(pipeline->text_recognizer, frame, &ocr_result);
    if (err != TK_SUCCESS) {
        return err;
    }

    if (ocr_result->block_count > 0) {
        result->text_blocks = (tk_vision_text_block_t*)calloc(ocr_result->block_count, sizeof(tk_vision_text_block_t));
        if (!result->text_blocks) {
            tk_ocr_result_destroy(&ocr_result);
            return TK_ERROR_OUT_OF_MEMORY;
        }

        for (size_t i = 0; i < ocr_result->block_count; ++i) {
            // Copy text string
            if (ocr_result->blocks[i].text) {
                result->text_blocks[i].text = strdup(ocr_result->blocks[i].text);
            } else {
                result->text_blocks[i].text = strdup("");
            }
            result->text_blocks[i].confidence = ocr_result->blocks[i].confidence;
            result->text_blocks[i].bbox = ocr_result->blocks[i].bbox;
        }
        result->text_block_count = ocr_result->block_count;
    }

    tk_ocr_result_destroy(&ocr_result);
    return TK_SUCCESS;
}

/**
 * @brief Enhanced object-depth fusion with robust distance calculation and size estimation
 * 
 * This function implements a more sophisticated approach to fusing object detection
 * results with depth information:
 * 1. Calculates robust distance using median/minimum of central pixels instead of single point
 * 2. Estimates real-world dimensions using camera parameters and triangulation
 */
static tk_error_code_t fuse_object_depth(tk_vision_pipeline_t* pipeline, tk_vision_result_t* result, const tk_video_frame_t* frame) {
    if (!result->depth_map || result->object_count == 0) {
        return TK_SUCCESS; // Nothing to fuse
    }

    uint32_t depth_width = result->depth_map->width;
    uint32_t depth_height = result->depth_map->height;
    float* depth_data = result->depth_map->data;
    
    uint32_t frame_width = frame->width;
    uint32_t frame_height = frame->height;

    // Camera parameters for size estimation
    float focal_length_x = pipeline->focal_length_x;
    float focal_length_y = pipeline->focal_length_y;

    for (size_t i = 0; i < result->object_count; ++i) {
        tk_rect_t bbox = result->objects[i].bbox;
        
        // Normalize bounding box coordinates to depth map space
        float norm_x_min = (float)bbox.x / (float)frame_width;
        float norm_y_min = (float)bbox.y / (float)frame_height;
        float norm_x_max = (float)(bbox.x + bbox.w) / (float)frame_width;
        float norm_y_max = (float)(bbox.y + bbox.h) / (float)frame_height;
        
        // Map to depth coordinates
        uint32_t depth_x_min = (uint32_t)(norm_x_min * (depth_width - 1));
        uint32_t depth_y_min = (uint32_t)(norm_y_min * (depth_height - 1));
        uint32_t depth_x_max = (uint32_t)(norm_x_max * (depth_width - 1));
        uint32_t depth_y_max = (uint32_t)(norm_y_max * (depth_height - 1));
        
        // Boundary checks and clamping
        depth_x_min = (depth_x_min < depth_width) ? depth_x_min : depth_width - 1;
        depth_y_min = (depth_y_min < depth_height) ? depth_y_min : depth_height - 1;
        depth_x_max = (depth_x_max < depth_width) ? depth_x_max : depth_width - 1;
        depth_y_max = (depth_y_max < depth_height) ? depth_y_max : depth_height - 1;
        
        if (depth_x_min >= depth_x_max || depth_y_min >= depth_y_max) {
            result->objects[i].distance_meters = -1.0f;
            result->objects[i].width_meters = -1.0f;
            result->objects[i].height_meters = -1.0f;
            continue;
        }

        // Focus on the central 25% of the bounding box for more robust distance estimation
        uint32_t center_width = depth_x_max - depth_x_min;
        uint32_t center_height = depth_y_max - depth_y_min;
        uint32_t crop_margin_x = center_width / 4;
        uint32_t crop_margin_y = center_height / 4;
        
        uint32_t center_x_min = depth_x_min + crop_margin_x;
        uint32_t center_y_min = depth_y_min + crop_margin_y;
        uint32_t center_x_max = depth_x_max - crop_margin_x;
        uint32_t center_y_max = depth_y_max - crop_margin_y;
        
        // Collect valid depth values from the central region
        float* valid_depths = NULL;
        size_t valid_count = 0;
        size_t max_samples = (center_x_max - center_x_min + 1) * (center_y_max - center_y_min + 1);
        
        if (max_samples > 0) {
            valid_depths = (float*)malloc(max_samples * sizeof(float));
            if (!valid_depths) {
                result->objects[i].distance_meters = -1.0f;
                result->objects[i].width_meters = -1.0f;
                result->objects[i].height_meters = -1.0f;
                continue;
            }
            
            for (uint32_t y = center_y_min; y <= center_y_max; ++y) {
                for (uint32_t x = center_x_min; x <= center_x_max; ++x) {
                    size_t index = y * depth_width + x;
                    float depth_value = depth_data[index];
                    
                    // Only consider valid positive depth values
                    if (depth_value > 0.0f && depth_value < 100.0f) { // Reasonable depth range
                        valid_depths[valid_count++] = depth_value;
                    }
                }
            }
        }
        
        // Calculate robust distance (minimum or median of valid depths)
        float robust_distance = -1.0f;
        if (valid_count > 0) {
            // Sort valid depths to find median
            for (size_t j = 0; j < valid_count - 1; ++j) {
                for (size_t k = j + 1; k < valid_count; ++k) {
                    if (valid_depths[j] > valid_depths[k]) {
                        float temp = valid_depths[j];
                        valid_depths[j] = valid_depths[k];
                        valid_depths[k] = temp;
                    }
                }
            }
            
            // Use minimum for safety-critical applications, or median for general use
            // For assistive technology, minimum is often better (closest point)
            robust_distance = valid_depths[0]; // Minimum distance
            
            // Alternative: use median
            // robust_distance = (valid_count % 2 == 1) ? 
            //                   valid_depths[valid_count/2] : 
            //                   (valid_depths[valid_count/2-1] + valid_depths[valid_count/2]) / 2.0f;
        }
        
        result->objects[i].distance_meters = robust_distance;
        
        // Estimate real-world dimensions using triangulation
        if (robust_distance > 0.0f && focal_length_x > 0.0f && focal_length_y > 0.0f) {
            // Convert pixel dimensions to meters using the triangulation formula:
            // real_size = (pixel_size * distance) / focal_length
            result->objects[i].width_meters = (bbox.w * robust_distance) / focal_length_x;
            result->objects[i].height_meters = (bbox.h * robust_distance) / focal_length_y;
        } else {
            result->objects[i].width_meters = -1.0f;
            result->objects[i].height_meters = -1.0f;
        }
        
        // Clean up temporary array
        if (valid_depths) {
            free(valid_depths);
        }
        
        tk_log_debug("Object %zu (%s) at (%d,%d) distance: %.2fm, size: %.2fx%.2fm", 
                    i, result->objects[i].label, bbox.x, bbox.y, 
                    result->objects[i].distance_meters,
                    result->objects[i].width_meters, result->objects[i].height_meters);
    }

    return TK_SUCCESS;
}
