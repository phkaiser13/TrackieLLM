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
 * SPDX-License-Identifier: AGPL-3.0 license AGPL-3.0 license
 */

#include "tk_vision_pipeline.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

// Assume these headers exist and provide the necessary functionality
#include "vision/tk_object_detector.h"
#include "vision/tk_depth_midas.h"
#include "vision/tk_text_recognition.h"
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
};

// Internal helper functions
static tk_error_code_t load_models(tk_vision_pipeline_t* pipeline, const tk_vision_pipeline_config_t* config);
static void unload_models(tk_vision_pipeline_t* pipeline);
static tk_error_code_t perform_object_detection(tk_vision_pipeline_t* pipeline, const tk_video_frame_t* frame, tk_vision_result_t* result);
static tk_error_code_t perform_depth_estimation(tk_vision_pipeline_t* pipeline, const tk_video_frame_t* frame, tk_vision_result_t* result);
static tk_error_code_t perform_ocr(tk_vision_pipeline_t* pipeline, const tk_video_frame_t* frame, tk_vision_result_t* result);
static tk_error_code_t fuse_object_depth(tk_vision_pipeline_t* pipeline, tk_vision_result_t* result);

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

    tk_error_code_t err = load_models(pipeline, config);
    if (err != TK_SUCCESS) {
        free(pipeline);
        return err;
    }

    *out_pipeline = pipeline;
    return TK_SUCCESS;
}

void tk_vision_pipeline_destroy(tk_vision_pipeline_t** pipeline) {
    if (pipeline && *pipeline) {
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
            tk_vision_result_destroy(&result);
            return err;
        }
    }

    if (analysis_flags & TK_VISION_ANALYZE_DEPTH_ESTIMATION) {
        err = perform_depth_estimation(pipeline, video_frame, result);
        if (err != TK_SUCCESS) {
            tk_vision_result_destroy(&result);
            return err;
        }
    }

    if (analysis_flags & TK_VISION_ANALYZE_OCR) {
        err = perform_ocr(pipeline, video_frame, result);
        if (err != TK_SUCCESS) {
            tk_vision_result_destroy(&result);
            return err;
        }
    }

    if ((analysis_flags & TK_VISION_ANALYZE_FUSION_DISTANCE) &&
        (analysis_flags & TK_VISION_ANALYZE_OBJECT_DETECTION) &&
        (analysis_flags & TK_VISION_ANALYZE_DEPTH_ESTIMATION)) {
        err = fuse_object_depth(pipeline, result);
        if (err != TK_SUCCESS) {
            tk_vision_result_destroy(&result);
            return err;
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
                free((void*)((*result)->objects[i].label)); // Assuming label is dynamically allocated
            }
            free((*result)->objects);
        }

        // Free text blocks
        if ((*result)->text_blocks) {
            for (size_t i = 0; i < (*result)->text_block_count; ++i) {
                free((*result)->text_blocks[i].text);
            }
            free((*result)->text_blocks);
        }

        // Free depth map
        if ((*result)->depth_map) {
            free((*result)->depth_map->data);
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

    // Load object detector
    err = tk_object_detector_create(&pipeline->object_detector, config->object_detection_model_path);
    if (err != TK_SUCCESS) {
        tk_log_error("Failed to load object detection model");
        return TK_ERROR_MODEL_LOAD_FAILED;
    }

    // Load depth estimator
    err = tk_depth_midas_create(&pipeline->depth_estimator, config->depth_estimation_model_path);
    if (err != TK_SUCCESS) {
        tk_log_error("Failed to load depth estimation model");
        tk_object_detector_destroy(&pipeline->object_detector);
        return TK_ERROR_MODEL_LOAD_FAILED;
    }

    // Load text recognizer
    err = tk_text_recognizer_create(&pipeline->text_recognizer, config->tesseract_data_path);
    if (err != TK_SUCCESS) {
        tk_log_error("Failed to initialize text recognizer");
        tk_object_detector_destroy(&pipeline->object_detector);
        tk_depth_midas_destroy(&pipeline->depth_estimator);
        return TK_ERROR_MODEL_LOAD_FAILED;
    }

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

    result->objects = (tk_vision_object_t*)calloc(valid_detections, sizeof(tk_vision_object_t));
    if (!result->objects) {
        tk_object_detection_result_destroy(&detection_result);
        return TK_ERROR_OUT_OF_MEMORY;
    }

    size_t idx = 0;
    for (size_t i = 0; i < detection_result->detection_count && idx < valid_detections; ++i) {
        if (detection_result->detections[i].confidence >= pipeline->object_confidence_threshold) {
            result->objects[idx].class_id = detection_result->detections[i].class_id;
            result->objects[idx].label = strdup(detection_result->detections[i].label); // Assume strdup is available or implemented
            result->objects[idx].confidence = detection_result->detections[i].confidence;
            result->objects[idx].bbox = detection_result->detections[i].bbox;
            result->objects[idx].distance_meters = 0.0f; // Will be populated by fusion
            idx++;
        }
    }
    result->object_count = valid_detections;

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

    result->text_blocks = (tk_vision_text_block_t*)calloc(ocr_result->block_count, sizeof(tk_vision_text_block_t));
    if (!result->text_blocks) {
        tk_ocr_result_destroy(&ocr_result);
        return TK_ERROR_OUT_OF_MEMORY;
    }

    for (size_t i = 0; i < ocr_result->block_count; ++i) {
        result->text_blocks[i].text = strdup(ocr_result->blocks[i].text);
        result->text_blocks[i].confidence = ocr_result->blocks[i].confidence;
        result->text_blocks[i].bbox = ocr_result->blocks[i].bbox;
    }
    result->text_block_count = ocr_result->block_count;

    tk_ocr_result_destroy(&ocr_result);
    return TK_SUCCESS;
}

static tk_error_code_t fuse_object_depth(tk_vision_pipeline_t* pipeline, tk_vision_result_t* result) {
    if (!result->depth_map || result->object_count == 0) {
        return TK_SUCCESS; // Nothing to fuse
    }

    uint32_t depth_width = result->depth_map->width;
    uint32_t depth_height = result->depth_map->height;
    float* depth_data = result->depth_map->data;

    for (size_t i = 0; i < result->object_count; ++i) {
        tk_rect_t bbox = result->objects[i].bbox;
        // Calculate center of bounding box
        int center_x = bbox.x + bbox.w / 2;
        int center_y = bbox.y + bbox.h / 2;

        // Normalize to depth map coordinates if sizes differ
        // This is a simplification; in practice, you might need to resize or interpolate
        uint32_t depth_x = (uint32_t)((center_x / (float)frame->width) * depth_width);
        uint32_t depth_y = (uint32_t)((center_y / (float)frame->height) * depth_height);

        if (depth_x < depth_width && depth_y < depth_height) {
            size_t index = depth_y * depth_width + depth_x;
            result->objects[i].distance_meters = depth_data[index];
        } else {
            result->objects[i].distance_meters = -1.0f; // Invalid distance
        }
    }

    return TK_SUCCESS;
}
