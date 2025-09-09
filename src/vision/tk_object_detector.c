/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_object_detector.c
 *
 * This source file implements the TrackieLLM Object Detection Engine.
 * It provides a complete, high-performance object detection system based on
 * ONNX models like YOLOv5nu. The implementation handles model loading,
 * inference execution, pre/post-processing, and memory management with care.
 *
 * The engine is designed to be modular and efficient, supporting multiple
 * inference backends (CPU, CUDA, etc.) and providing granular control over
 * detection parameters like confidence thresholds and NMS.
 *
 * Key features:
 * - ONNX Runtime integration with backend selection
 * - Efficient pre-processing pipeline (resize, normalize, tensor conversion)
 * - Robust post-processing with confidence filtering and NMS
 * - Memory pooling for reduced allocations
 * - Comprehensive error handling and logging
 *
 * SPDX-License-Identifier: AGPL-3.0 license AGPL-3.0 license
 */

#include "tk_object_detector.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

// Include ONNX Runtime headers
#include <onnxruntime_c_api.h>
#include <cpu_provider_factory.h>
#ifdef TRACKIELLM_ENABLE_CUDA
#include <cuda_provider_factory.h>
#endif

// Internal constants
#define TK_OBJECT_DETECTOR_MAX_DETECTIONS 500
#define TK_OBJECT_DETECTOR_INPUT_NAME "images"
#define TK_OBJECT_DETECTOR_OUTPUT_NAME "output0"

// Internal structures
struct tk_object_detector_s {
    // Configuration
    tk_object_detector_config_t config;
    
    // ONNX Runtime components
    Ort::Env env;
    Ort::Session* session;
    Ort::AllocatorWithDefaultOptions allocator;
    
    // Model input/output info
    int64_t input_node_dims[4];  // {batch, channels, height, width}
    size_t input_tensor_size;
    char* input_node_name;
    char* output_node_name;
    
    // Pre-processing buffers
    float* input_tensor_values;
    uint8_t* resized_frame_buffer;
    
    // Post-processing buffers
    float* raw_detections;
    size_t max_raw_detections;
    
    // Memory pool for results
    tk_memory_pool_t* result_pool;
};

// Internal helper functions
static tk_error_code_t initialize_onnx_runtime(tk_object_detector_t* detector);
static tk_error_code_t setup_model_io_info(tk_object_detector_t* detector);
static tk_error_code_t allocate_buffers(tk_object_detector_t* detector);
static void release_buffers(tk_object_detector_t* detector);
static tk_error_code_t preprocess_frame(tk_object_detector_t* detector, const tk_video_frame_t* frame);
static tk_error_code_t run_inference(tk_object_detector_t* detector);
static tk_error_code_t postprocess_detections(tk_object_detector_t* detector, 
                                              tk_detection_result_t** out_results, 
                                              size_t* out_result_count);
static void apply_confidence_filter(tk_object_detector_t* detector, 
                                   size_t* valid_detections_count);
static void apply_nms(tk_object_detector_t* detector, 
                     size_t valid_detections_count,
                     size_t* nms_filtered_count);
static float calculate_iou(const tk_rect_t* bbox1, const tk_rect_t* bbox2);
static void convert_to_original_coordinates(tk_detection_result_t* results, 
                                          size_t count,
                                          uint32_t original_width,
                                          uint32_t original_height,
                                          uint32_t processed_width,
                                          uint32_t processed_height);

//------------------------------------------------------------------------------
// Engine Lifecycle Management
//------------------------------------------------------------------------------

TK_NODISCARD tk_error_code_t tk_object_detector_create(tk_object_detector_t** out_detector, 
                                                       const tk_object_detector_config_t* config) {
    if (!out_detector || !config || !config->model_path) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Validate configuration parameters
    if (config->input_width == 0 || config->input_height == 0 || 
        config->class_count == 0 || !config->class_labels) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Allocate detector structure
    tk_object_detector_t* detector = (tk_object_detector_t*)calloc(1, sizeof(tk_object_detector_t));
    if (!detector) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Copy configuration
    detector->config = *config;
    
    // Initialize ONNX Runtime
    tk_error_code_t err = initialize_onnx_runtime(detector);
    if (err != TK_SUCCESS) {
        free(detector);
        return err;
    }
    
    // Setup model I/O information
    err = setup_model_io_info(detector);
    if (err != TK_SUCCESS) {
        tk_object_detector_destroy(&detector);
        return err;
    }
    
    // Allocate internal buffers
    err = allocate_buffers(detector);
    if (err != TK_SUCCESS) {
        tk_object_detector_destroy(&detector);
        return err;
    }
    
    // Create memory pool for results
    err = tk_memory_pool_create(&detector->result_pool, 
                               sizeof(tk_detection_result_t), 
                               TK_OBJECT_DETECTOR_MAX_DETECTIONS);
    if (err != TK_SUCCESS) {
        tk_object_detector_destroy(&detector);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    *out_detector = detector;
    return TK_SUCCESS;
}

void tk_object_detector_destroy(tk_object_detector_t** detector) {
    if (!detector || !*detector) {
        return;
    }
    
    tk_object_detector_t* d = *detector;
    
    // Release ONNX Runtime resources
    if (d->session) {
        delete d->session;
        d->session = nullptr;
    }
    
    // Release buffers
    release_buffers(d);
    
    // Release memory pool
    if (d->result_pool) {
        tk_memory_pool_destroy(&d->result_pool);
    }
    
    // Free detector structure
    free(d);
    *detector = NULL;
}

//------------------------------------------------------------------------------
// Core Inference Function
//------------------------------------------------------------------------------

TK_NODISCARD tk_error_code_t tk_object_detector_detect(
    tk_object_detector_t* detector,
    const tk_video_frame_t* video_frame,
    tk_detection_result_t** out_results,
    size_t* out_result_count) {
    
    if (!detector || !video_frame || !out_results || !out_result_count) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Reset output parameters
    *out_results = NULL;
    *out_result_count = 0;
    
    // Pre-process input frame
    tk_error_code_t err = preprocess_frame(detector, video_frame);
    if (err != TK_SUCCESS) {
        return err;
    }
    
    // Run model inference
    err = run_inference(detector);
    if (err != TK_SUCCESS) {
        return TK_ERROR_INFERENCE_FAILED;
    }
    
    // Post-process detections
    err = postprocess_detections(detector, out_results, out_result_count, video_frame);
    if (err != TK_SUCCESS) {
        return err;
    }
    
    return TK_SUCCESS;
}

//------------------------------------------------------------------------------
// Result Data Management
//------------------------------------------------------------------------------

void tk_object_detector_free_results(tk_detection_result_t** results) {
    if (results && *results) {
        // In this implementation, results are allocated from a memory pool
        // so we don't need to free individual elements
        free(*results);
        *results = NULL;
    }
}

//------------------------------------------------------------------------------
// Internal Helper Functions
//------------------------------------------------------------------------------

static tk_error_code_t initialize_onnx_runtime(tk_object_detector_t* detector) {
    // Create ONNX environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TrackieObjectDetector");
    detector->env = env;
    
    // Create session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);  // Adjust based on CPU cores
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    // Set execution backend
    switch (detector->config.backend) {
        case TK_VISION_BACKEND_CPU:
            // CPU is the default, no additional setup needed
            break;
            
#ifdef TRACKIELLM_ENABLE_CUDA
        case TK_VISION_BACKEND_CUDA:
            {
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = detector->config.gpu_device_id;
                cuda_options.arena_extend_strategy = 0;
                cuda_options.gpu_mem_limit = SIZE_MAX;
                cuda_options.cuda_mem_limit = SIZE_MAX;
                session_options.AppendExecutionProvider_CUDA(cuda_options);
            }
            break;
#endif
            
        case TK_VISION_BACKEND_METAL:
        case TK_VISION_BACKEND_ROCM:
            // These would require specific provider setup
            // For now, fall back to CPU
            tk_log_warn("Backend not fully supported, falling back to CPU");
            break;
            
        default:
            return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Create session
    try {
        Ort::Session* session = new Ort::Session(detector->env, 
                                                detector->config.model_path->path_str, 
                                                session_options);
        detector->session = session;
    } catch (const Ort::Exception& e) {
        tk_log_error("ONNX Runtime error: %s", e.what());
        return TK_ERROR_MODEL_LOAD_FAILED;
    }
    
    return TK_SUCCESS;
}

static tk_error_code_t setup_model_io_info(tk_object_detector_t* detector) {
    // Get input and output node names
    size_t input_count = detector->session->GetInputCount();
    size_t output_count = detector->session->GetOutputCount();
    
    if (input_count == 0 || output_count == 0) {
        return TK_ERROR_MODEL_LOAD_FAILED;
    }
    
    // Get input node name
    detector->input_node_name = detector->session->GetInputNameAllocated(0, detector->allocator).release();
    
    // Get output node name
    detector->output_node_name = detector->session->GetOutputNameAllocated(0, detector->allocator).release();
    
    // Get input node dimensions
    Ort::TypeInfo input_type_info = detector->session->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    detector->input_node_dims = input_tensor_info.GetShape();
    
    // Validate input dimensions
    if (detector->input_node_dims.size() != 4) {
        return TK_ERROR_MODEL_LOAD_FAILED;
    }
    
    // Calculate input tensor size
    detector->input_tensor_size = 1;
    for (int i = 0; i < 4; i++) {
        detector->input_tensor_size *= detector->input_node_dims[i];
    }
    
    return TK_SUCCESS;
}

static tk_error_code_t allocate_buffers(tk_object_detector_t* detector) {
    // Allocate input tensor buffer
    detector->input_tensor_values = (float*)malloc(detector->input_tensor_size * sizeof(float));
    if (!detector->input_tensor_values) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Allocate resized frame buffer
    uint32_t input_width = detector->config.input_width;
    uint32_t input_height = detector->config.input_height;
    detector->resized_frame_buffer = (uint8_t*)malloc(input_width * input_height * 3 * sizeof(uint8_t));
    if (!detector->resized_frame_buffer) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Allocate raw detections buffer
    // Assuming maximum detections per image
    detector->max_raw_detections = 10000; // Adjust based on model capacity
    detector->raw_detections = (float*)malloc(detector->max_raw_detections * 6 * sizeof(float));
    if (!detector->raw_detections) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    return TK_SUCCESS;
}

static void release_buffers(tk_object_detector_t* detector) {
    if (detector->input_tensor_values) {
        free(detector->input_tensor_values);
        detector->input_tensor_values = NULL;
    }
    
    if (detector->resized_frame_buffer) {
        free(detector->resized_frame_buffer);
        detector->resized_frame_buffer = NULL;
    }
    
    if (detector->raw_detections) {
        free(detector->raw_detections);
        detector->raw_detections = NULL;
    }
    
    if (detector->input_node_name) {
        free(detector->input_node_name);
        detector->input_node_name = NULL;
    }
    
    if (detector->output_node_name) {
        free(detector->output_node_name);
        detector->output_node_name = NULL;
    }
}

static tk_error_code_t preprocess_frame(tk_object_detector_t* detector, 
                                       const tk_video_frame_t* frame) {
    // Resize frame to model input size
    // This is a simplified resize - in practice, you might want to use
    // a more sophisticated resizing algorithm like bilinear interpolation
    uint32_t orig_width = frame->width;
    uint32_t orig_height = frame->height;
    uint32_t target_width = detector->config.input_width;
    uint32_t target_height = detector->config.input_height;
    
    // Simple nearest-neighbor resize (for demonstration)
    for (uint32_t y = 0; y < target_height; y++) {
        for (uint32_t x = 0; x < target_width; x++) {
            uint32_t orig_x = (x * orig_width) / target_width;
            uint32_t orig_y = (y * orig_height) / target_height;
            
            if (orig_x >= orig_width) orig_x = orig_width - 1;
            if (orig_y >= orig_height) orig_y = orig_height - 1;
            
            size_t orig_idx = (orig_y * orig_width + orig_x) * 3;
            size_t target_idx = (y * target_width + x) * 3;
            
            detector->resized_frame_buffer[target_idx] = frame->data[orig_idx];
            detector->resized_frame_buffer[target_idx + 1] = frame->data[orig_idx + 1];
            detector->resized_frame_buffer[target_idx + 2] = frame->data[orig_idx + 2];
        }
    }
    
    // Normalize and convert to float tensor
    // Assuming RGB format and normalization to [0, 1]
    for (size_t i = 0; i < target_width * target_height * 3; i++) {
        detector->input_tensor_values[i] = detector->resized_frame_buffer[i] / 255.0f;
    }
    
    return TK_SUCCESS;
}

// A structure to hold candidate detections before NMS
typedef struct {
    tk_rect_t bbox;
    float confidence;
    uint32_t class_id;
    bool active; // Flag to mark if this box is kept after NMS
} tk_candidate_detection_t;


static tk_error_code_t run_inference(tk_object_detector_t* detector) {
    // Create input tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        detector->input_tensor_values, 
        detector->input_tensor_size,
        detector->input_node_dims.data(), 
        detector->input_node_dims.size()
    );

    // Run inference
    const char* input_names[] = { detector->input_node_name };
    const char* output_names[] = { detector->output_node_name };
    
    try {
        Ort::RunOptions run_options;
        auto output_tensors = detector->session->Run(
            run_options, 
            input_names, 
            &input_tensor, 
            1, 
            output_names, 
            1
        );
        
        // Copy output data
        Ort::Value& output_tensor = output_tensors.front();
        float* output_data = output_tensor.GetTensorMutableData<float>();
        
        size_t output_data_size = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

        // Ensure our raw_detections buffer is large enough
        if (output_data_size > detector->max_raw_detections) {
            // This case should be handled gracefully, e.g., by reallocating or logging an error.
            // For now, we'll log an error and truncate.
            tk_log_error("Model output size (%zu) exceeds allocated buffer size (%zu). Truncating.",
                         output_data_size, detector->max_raw_detections);
            output_data_size = detector->max_raw_detections;
        }
        
        memcpy(detector->raw_detections, output_data, output_data_size * sizeof(float));

    } catch (const Ort::Exception& e) {
        tk_log_error("ONNX Runtime inference error: %s", e.what());
        return TK_ERROR_INFERENCE_FAILED;
    }
    
    return TK_SUCCESS;
}


static tk_error_code_t postprocess_detections(tk_object_detector_t* detector, 
                                              tk_detection_result_t** out_results, 
                                              size_t* out_result_count,
                                              const tk_video_frame_t* original_frame) {
    
    // The output of YOLOv5 is typically [batch_size, num_proposals, 5 + num_classes]
    // e.g., [1, 25200, 85] for COCO dataset (80 classes)
    Ort::TypeInfo output_type_info = detector->session->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_dims = output_tensor_info.GetShape();

    if (output_dims.size() != 3 || output_dims[0] != 1) {
        tk_log_error("Unsupported model output shape.");
        return TK_ERROR_INFERENCE_FAILED;
    }

    const size_t num_proposals = output_dims[1];
    const size_t proposal_length = output_dims[2]; // 5 (cx, cy, w, h, conf) + class_count
    const size_t num_classes = detector->config.class_count;

    if (proposal_length != 5 + num_classes) {
        tk_log_error("Model output proposal length (%zu) does not match expected length (%zu).", proposal_length, 5 + num_classes);
        return TK_ERROR_INFERENCE_FAILED;
    }

    tk_candidate_detection_t* candidates = (tk_candidate_detection_t*)calloc(num_proposals, sizeof(tk_candidate_detection_t));
    if (!candidates) return TK_ERROR_OUT_OF_MEMORY;

    size_t candidate_count = 0;

    // 1. Decode and Filter by Confidence
    for (size_t i = 0; i < num_proposals; ++i) {
        float* proposal = detector->raw_detections + i * proposal_length;
        float object_confidence = proposal[4];

        if (object_confidence < detector->config.confidence_threshold) {
            continue;
        }

        // Find the class with the highest score
        uint32_t best_class_id = 0;
        float max_class_score = 0.0f;
        for (size_t j = 0; j < num_classes; ++j) {
            if (proposal[5 + j] > max_class_score) {
                max_class_score = proposal[5 + j];
                best_class_id = j;
            }
        }

        float final_confidence = object_confidence * max_class_score;

        if (final_confidence < detector->config.confidence_threshold) {
            continue;
        }

        // Convert box from [center_x, center_y, width, height] to [x, y, width, height]
        candidates[candidate_count].bbox.x = (int)(proposal[0] - proposal[2] / 2.0f);
        candidates[candidate_count].bbox.y = (int)(proposal[1] - proposal[3] / 2.0f);
        candidates[candidate_count].bbox.w = (int)proposal[2];
        candidates[candidate_count].bbox.h = (int)proposal[3];
        candidates[candidate_count].confidence = final_confidence;
        candidates[candidate_count].class_id = best_class_id;
        candidates[candidate_count].active = true;
        candidate_count++;
    }

    // 2. Non-Max Suppression (NMS)
    for (size_t i = 0; i < candidate_count; ++i) {
        if (!candidates[i].active) continue;

        for (size_t j = i + 1; j < candidate_count; ++j) {
            if (!candidates[j].active) continue;

            if (candidates[i].class_id != candidates[j].class_id) continue;

            float iou = calculate_iou(&candidates[i].bbox, &candidates[j].bbox);
            if (iou > detector->config.iou_threshold) {
                if (candidates[i].confidence > candidates[j].confidence) {
                    candidates[j].active = false;
                } else {
                    candidates[i].active = false;
                    // Break inner loop and re-evaluate `i` with the better box
                    break;
                }
            }
        }
    }

    // 3. Count final results and allocate memory
    size_t final_count = 0;
    for (size_t i = 0; i < candidate_count; ++i) {
        if (candidates[i].active) {
            final_count++;
        }
    }

    if (final_count == 0) {
        *out_results = NULL;
        *out_result_count = 0;
        free(candidates);
        return TK_SUCCESS;
    }

    tk_detection_result_t* final_results = (tk_detection_result_t*)malloc(final_count * sizeof(tk_detection_result_t));
    if (!final_results) {
        free(candidates);
        return TK_ERROR_OUT_OF_MEMORY;
    }

    // 4. Copy final results
    size_t result_idx = 0;
    for (size_t i = 0; i < candidate_count; ++i) {
        if (candidates[i].active) {
            final_results[result_idx].bbox = candidates[i].bbox;
            final_results[result_idx].class_id = candidates[i].class_id;
            final_results[result_idx].confidence = candidates[i].confidence;
            final_results[result_idx].label = detector->config.class_labels[candidates[i].class_id];
            result_idx++;
        }
    }
    
    free(candidates);

    // 5. Convert coordinates to original image space
    convert_to_original_coordinates(final_results, final_count,
                                    original_frame->width, original_frame->height,
                                    detector->config.input_width, detector->config.input_height);

    *out_results = final_results;
    *out_result_count = final_count;

    return TK_SUCCESS;
}

static float calculate_iou(const tk_rect_t* bbox1, const tk_rect_t* bbox2) {
    // Calculate intersection rectangle
    int x1 = (bbox1->x > bbox2->x) ? bbox1->x : bbox2->x;
    int y1 = (bbox1->y > bbox2->y) ? bbox1->y : bbox2->y;
    int x2 = ((bbox1->x + bbox1->w) < (bbox2->x + bbox2->w)) ? 
             (bbox1->x + bbox1->w) : (bbox2->x + bbox2->w);
    int y2 = ((bbox1->y + bbox1->h) < (bbox2->y + bbox2->h)) ? 
             (bbox1->y + bbox1->h) : (bbox2->y + bbox2->h);
    
    // Calculate intersection area
    int intersection_area = (x2 - x1) * (y2 - y1);
    if (intersection_area < 0) intersection_area = 0;
    
    // Calculate union area
    int area1 = bbox1->w * bbox1->h;
    int area2 = bbox2->w * bbox2->h;
    int union_area = area1 + area2 - intersection_area;
    
    // Calculate IoU
    if (union_area == 0) return 0.0f;
    return (float)intersection_area / union_area;
}

static void convert_to_original_coordinates(tk_detection_result_t* results, 
                                          size_t count,
                                          uint32_t original_width,
                                          uint32_t original_height,
                                          uint32_t processed_width,
                                          uint32_t processed_height) {
    // Convert bounding box coordinates from processed image space back to original image space
    float width_ratio = (float)original_width / processed_width;
    float height_ratio = (float)original_height / processed_height;
    
    for (size_t i = 0; i < count; i++) {
        results[i].bbox.x = (int)(results[i].bbox.x * width_ratio);
        results[i].bbox.y = (int)(results[i].bbox.y * height_ratio);
        results[i].bbox.w = (int)(results[i].bbox.w * width_ratio);
        results[i].bbox.h = (int)(results[i].bbox.h * height_ratio);
    }
}
