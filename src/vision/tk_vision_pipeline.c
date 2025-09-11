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

    // Navigation analysis is performed if requested and depth is available
    if ((analysis_flags & TK_VISION_ANALYZE_NAVIGATION_CUES) && (result->depth_map != NULL)) {
        CNavigationCues* nav_cues = tk_vision_rust_analyze_navigation(result->depth_map);
        if (nav_cues) {
            tk_log_debug("Navigation analysis completed. Found %zu vertical changes.", nav_cues->vertical_changes_count);
            // In a full implementation, this data would be attached to the result.
            // For now, we just log and free it.
            tk_vision_rust_free_navigation_cues(nav_cues);
        } else {
            tk_log_warning("Navigation analysis failed or returned no data.");
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
                if ((*result)->objects[i].recognized_text) {
                    free((*result)->objects[i].recognized_text);
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

// A placeholder for COCO class labels. A real implementation would load this from a file.
const char* COCO_LABELS[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};
const size_t COCO_CLASS_COUNT = 80;


static tk_error_code_t load_models(tk_vision_pipeline_t* pipeline, const tk_vision_pipeline_config_t* config) {
    tk_error_code_t err;

    if (!config->object_detection_model_path || !config->depth_estimation_model_path || !config->tesseract_data_path) {
        tk_log_error("Missing required model paths in configuration");
        return TK_ERROR_INVALID_ARGUMENT;
    }

    tk_log_info("Loading vision models...");

    // --- Load Object Detector ---
    tk_object_detector_config_t detector_config = {
        .backend = config->backend,
        .gpu_device_id = config->gpu_device_id,
        .model_path = config->object_detection_model_path,
        .input_width = 640, // Common for YOLOv5
        .input_height = 640,
        .class_labels = COCO_LABELS,
        .class_count = COCO_CLASS_COUNT,
        .confidence_threshold = config->object_confidence_threshold,
        .iou_threshold = 0.5f // A common default for NMS
    };
    err = tk_object_detector_create(&pipeline->object_detector, &detector_config);
    if (err != TK_SUCCESS) {
        tk_log_error("Failed to create object detector with model from %s", config->object_detection_model_path->path_str);
        return err;
    }
    tk_log_debug("Object detection model loaded successfully");

    // --- Load Depth Estimator ---
    tk_depth_estimator_config_t depth_config = {
        .backend = config->backend,
        .gpu_device_id = config->gpu_device_id,
        .model_path = config->depth_estimation_model_path,
        .input_width = 256, // Common for MiDaS DPT-SwinV2-Tiny
        .input_height = 256
    };
    err = tk_depth_estimator_create(&pipeline->depth_estimator, &depth_config);
    if (err != TK_SUCCESS) {
        tk_log_error("Failed to create depth estimator with model from %s", config->depth_estimation_model_path->path_str);
        tk_object_detector_destroy(&pipeline->object_detector); // Clean up previous success
        return err;
    }
    tk_log_debug("Depth estimation model loaded successfully");

    // --- Load Text Recognizer ---
    tk_ocr_config_t ocr_config = {
        .data_path = config->tesseract_data_path,
        .language = TK_OCR_LANG_ENGLISH, // Default to English
        .engine_mode = TK_OCR_ENGINE_DEFAULT,
        .psm = TK_OCR_PSM_AUTO,
        .dpi = 300, // A reasonable default DPI
        .num_threads = 2, // Default number of threads
    };
    err = tk_text_recognition_create(&pipeline->text_recognizer, &ocr_config);
    if (err != TK_SUCCESS) {
        tk_log_error("Failed to initialize text recognizer with data path %s", config->tesseract_data_path->path_str);
        tk_object_detector_destroy(&pipeline->object_detector);
        tk_depth_estimator_destroy(&pipeline->depth_estimator);
        return err;
    }
    tk_log_debug("Text recognizer initialized successfully");

    return TK_SUCCESS;
}

static void unload_models(tk_vision_pipeline_t* pipeline) {
    if (pipeline->object_detector) {
        tk_object_detector_destroy(&pipeline->object_detector);
    }
    if (pipeline->depth_estimator) {
        tk_depth_estimator_destroy(&pipeline->depth_estimator);
    }
    if (pipeline->text_recognizer) {
        tk_text_recognition_destroy(&pipeline->text_recognizer);
    }
}

static tk_error_code_t perform_object_detection(tk_vision_pipeline_t* pipeline, const tk_video_frame_t* frame, tk_vision_result_t* result) {
    tk_detection_result_t* detections = NULL;
    size_t detection_count = 0;

    tk_error_code_t err = tk_object_detector_detect(pipeline->object_detector, frame, &detections, &detection_count);
    if (err != TK_SUCCESS) {
        return err;
    }

    if (detection_count > 0) {
        result->objects = (tk_vision_object_t*)calloc(detection_count, sizeof(tk_vision_object_t));
        if (!result->objects) {
            tk_object_detector_free_results(&detections);
            return TK_ERROR_OUT_OF_MEMORY;
        }

        for (size_t i = 0; i < detection_count; ++i) {
            result->objects[i].class_id = detections[i].class_id;
            result->objects[i].label = detections[i].label; // Direct pointer, no copy needed as labels are static const
            result->objects[i].confidence = detections[i].confidence;
            result->objects[i].bbox = detections[i].bbox;
            result->objects[i].distance_meters = 0.0f; // Will be populated by fusion
            result->objects[i].width_meters = 0.0f;    // Will be populated by fusion
            result->objects[i].height_meters = 0.0f;   // Will be populated by fusion
            result->objects[i].is_partially_occluded = false; // Will be populated by fusion
            result->objects[i].recognized_text = NULL; // Will be populated by OCR
        }
        result->object_count = detection_count;
    }

    tk_object_detector_free_results(&detections);
    return TK_SUCCESS;
}

static tk_error_code_t perform_depth_estimation(tk_vision_pipeline_t* pipeline, const tk_video_frame_t* frame, tk_vision_result_t* result) {
    // The estimate function allocates the depth map, which will be owned by the result struct.
    tk_error_code_t err = tk_depth_estimator_estimate(pipeline->depth_estimator, frame, &result->depth_map);
    if (err != TK_SUCCESS) {
        return err;
    }

    // The 'result->depth_map' is now populated and owned by the result.
    // It will be freed in tk_vision_result_destroy.

    return TK_SUCCESS;
}

static tk_error_code_t perform_ocr(tk_vision_pipeline_t* pipeline, const tk_video_frame_t* frame, tk_vision_result_t* result) {
    if (result->object_count == 0) {
        return TK_SUCCESS; // Nothing to perform OCR on
    }

    tk_log_debug("Performing targeted OCR on relevant objects.");

    // Define a list of object labels that are likely to contain text.
    const char* text_bearing_labels[] = {
        "sign", "book", "laptop", "keyboard", "remote", "cell phone", "tv", "label", "plate", NULL
    };

    // Base image parameters for the whole frame
    tk_ocr_image_params_t image_params = {
        .image_data = frame->data,
        .width = frame->width,
        .height = frame->height,
        .channels = 3, // Assuming RGB8
        .stride = frame->width * 3,
        .psm = TK_OCR_PSM_SINGLE_BLOCK, // Assume a single block of text within the bbox
    };

    for (size_t i = 0; i < result->object_count; ++i) {
        tk_vision_object_t* object = &result->objects[i];
        bool should_scan = false;

        // Check if the object's label is in our list
        for (int j = 0; text_bearing_labels[j] != NULL; ++j) {
            if (strstr(object->label, text_bearing_labels[j]) != NULL) {
                should_scan = true;
                break;
            }
        }

        if (should_scan) {
            tk_log_debug("Scanning object '%s' for text.", object->label);
            tk_ocr_result_t* ocr_result = NULL;

            // We use the process_region function to constrain OCR to the bounding box
            tk_error_code_t err = tk_text_recognition_process_region(
                pipeline->text_recognizer,
                &image_params,
                object->bbox.x,
                object->bbox.y,
                object->bbox.w,
                object->bbox.h,
                &ocr_result
            );

            if (err == TK_SUCCESS && ocr_result && ocr_result->full_text && ocr_result->full_text_length > 0) {
                // Associate the recognized text directly with the object
                object->recognized_text = strdup(ocr_result->full_text);
                tk_log_info("Found text '%s' on object '%s'", object->recognized_text, object->label);
            } else if (err != TK_SUCCESS) {
                tk_log_warn("OCR on region for object '%s' failed with error %d", object->label, err);
            }

            if (ocr_result) {
                tk_text_recognition_free_result(&ocr_result);
            }
        }
    }

    // The global `text_blocks` array is no longer used. Text is now part of each object.
    result->text_block_count = 0;
    result->text_blocks = NULL;

    return TK_SUCCESS;
}

// --- FFI Declarations for Rust Fusion Library ---

// Define the C-compatible structs that Rust will return
typedef struct {
    uint32_t class_id;
    float confidence;
    tk_rect_t bbox;
    float distance_meters;
    float width_meters;
    float height_meters;
    bool is_partially_occluded;
} EnrichedObject;

typedef struct {
    const EnrichedObject* objects;
    size_t count;
} CFusedResult;

// Declare the external Rust functions
extern CFusedResult* tk_vision_rust_fuse_data(
    const tk_detection_result_t* detections_ptr,
    size_t detection_count,
    const tk_vision_depth_map_t* depth_map_ptr,
    uint32_t frame_width,
    uint32_t frame_height,
    float focal_length_x,
    float focal_length_y
);

extern void tk_vision_rust_free_fused_result(CFusedResult* result_ptr);

// --- FFI Declarations for Rust Navigation Analysis Library ---

// C-compatible mirror of the GroundPlaneStatus enum in Rust
typedef enum {
    C_GROUND_PLANE_STATUS_UNKNOWN,
    C_GROUND_PLANE_STATUS_FLAT,
    C_GROUND_PLANE_STATUS_OBSTACLE,
    C_GROUND_PLANE_STATUS_HOLE,
    C_GROUND_PLANE_STATUS_RAMP_UP,
    C_GROUND_PLANE_STATUS_RAMP_DOWN,
} CGroundPlaneStatus;

// C-compatible mirror of the VerticalChange struct in Rust
typedef struct {
    float height_m;
    CGroundPlaneStatus status;
    uint32_t grid_x;
    uint32_t grid_y;
} CVerticalChange;

// C-compatible mirror of the NavigationCues struct in Rust
typedef struct {
    const CGroundPlaneStatus* traversability_grid;
    size_t grid_size;
    uint32_t grid_width;
    uint32_t grid_height;
    const CVerticalChange* detected_vertical_changes;
    size_t vertical_changes_count;
} CNavigationCues;

// Declare the external Rust functions for navigation analysis
extern CNavigationCues* tk_vision_rust_analyze_navigation(const tk_vision_depth_map_t* depth_map_ptr);
extern void tk_vision_rust_free_navigation_cues(CNavigationCues* cues_ptr);


/**
 * @brief Calls the Rust library to fuse object detection and depth data.
 *
 * This function acts as a bridge to the high-performance, safe Rust implementation
 * of the fusion logic. It passes pointers to the raw C data, and the Rust side
 * returns a new structure with the calculated distances and sizes.
 */
static tk_error_code_t fuse_object_depth(tk_vision_pipeline_t* pipeline, tk_vision_result_t* result, const tk_video_frame_t* frame) {
    if (!result->depth_map || result->object_count == 0) {
        return TK_SUCCESS; // Nothing to fuse
    }

    // The Rust function needs an array of tk_detection_result_t, but our pipeline
    // result has tk_vision_object_t. For this call, we'll create a temporary
    // array of the required input type.
    tk_detection_result_t* raw_detections = (tk_detection_result_t*)malloc(result->object_count * sizeof(tk_detection_result_t));
    if (!raw_detections) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    for (size_t i = 0; i < result->object_count; ++i) {
        raw_detections[i].class_id = result->objects[i].class_id;
        raw_detections[i].label = result->objects[i].label;
        raw_detections[i].confidence = result->objects[i].confidence;
        raw_detections[i].bbox = result->objects[i].bbox;
    }

    // Call the external Rust function
    CFusedResult* fused_result = tk_vision_rust_fuse_data(
        raw_detections,
        result->object_count,
        result->depth_map,
        frame->width,
        frame->height,
        pipeline->focal_length_x,
        pipeline->focal_length_y
    );

    free(raw_detections); // Free the temporary array

    if (!fused_result) {
        tk_log_error("Rust fusion function returned null. Fusion failed.");
        return TK_ERROR_INFERENCE_FAILED; // Or a more specific error
    }

    // The number of objects should match
    if (fused_result->count != result->object_count) {
        tk_log_warn("Mismatch in object count between C (%zu) and Rust (%zu) layers.", result->object_count, fused_result->count);
    }

    // Copy the fused data (distance, width, height) back into our main result struct
    for (size_t i = 0; i < result->object_count && i < fused_result->count; ++i) {
        result->objects[i].distance_meters = fused_result->objects[i].distance_meters;
        result->objects[i].width_meters = fused_result->objects[i].width_meters;
        result->objects[i].height_meters = fused_result->objects[i].height_meters;
        result->objects[i].is_partially_occluded = fused_result->objects[i].is_partially_occluded;

        tk_log_debug("Object %zu (%s) fused distance: %.2fm, size: %.2fx%.2fm, occluded: %s",
                     i, result->objects[i].label,
                     result->objects[i].distance_meters,
                     result->objects[i].width_meters, result->objects[i].height_meters,
                     result->objects[i].is_partially_occluded ? "yes" : "no");
    }

    // IMPORTANT: Free the memory allocated by the Rust library
    tk_vision_rust_free_fused_result(fused_result);

    return TK_SUCCESS;
}
