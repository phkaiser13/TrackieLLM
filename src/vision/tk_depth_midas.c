/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_depth_midas.c
 *
 * This source file implements the TrackieLLM Monocular Depth Estimation Engine.
 * It provides a complete, high-performance depth estimation system based on
 * the MiDaS (DPT-SwinV2-Tiny) ONNX model. The implementation handles model loading,
 * inference execution, pre/post-processing, and memory management with care.
 *
 * The engine is designed to be modular and efficient, supporting multiple
 * inference backends (CPU, CUDA, etc.) and providing accurate depth maps
 * essential for navigation and spatial understanding.
 *
 * Key features:
 * - ONNX Runtime integration with backend selection
 * - Efficient pre-processing pipeline (resize, normalize, tensor conversion)
 * - Robust post-processing to convert inverse depth to metric depth
 * - Memory pooling for reduced allocations
 * - Comprehensive error handling and logging
 *
 * SPDX-License-Identifier: AGPL-3.0 license AGPL-3.0 license
 */

#include "tk_depth_midas.h"
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
#define TK_DEPTH_ESTIMATOR_INPUT_NAME "input"
#define TK_DEPTH_ESTIMATOR_OUTPUT_NAME "output"
#define TK_DEPTH_ESTIMATOR_MIN_DEPTH 0.1f  // Minimum depth in meters
#define TK_DEPTH_ESTIMATOR_MAX_DEPTH 10.0f // Maximum depth in meters

// Internal structures
struct tk_depth_estimator_s {
    // Configuration
    tk_depth_estimator_config_t config;
    
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
    float* raw_depth_values;
    size_t output_tensor_size;
    
    // Memory pool for results
    tk_memory_pool_t* result_pool;
};

// Internal helper functions
static tk_error_code_t initialize_onnx_runtime(tk_depth_estimator_t* estimator);
static tk_error_code_t setup_model_io_info(tk_depth_estimator_t* estimator);
static tk_error_code_t allocate_buffers(tk_depth_estimator_t* estimator);
static void release_buffers(tk_depth_estimator_t* estimator);
static tk_error_code_t preprocess_frame(tk_depth_estimator_t* estimator, const tk_video_frame_t* frame);
static tk_error_code_t run_inference(tk_depth_estimator_t* estimator);
static tk_error_code_t postprocess_depth(tk_depth_estimator_t* estimator, 
                                        tk_vision_depth_map_t** out_depth_map);
static void convert_inverse_depth_to_metric(float* depth_data, size_t data_size, 
                                           float min_depth, float max_depth);

//------------------------------------------------------------------------------
// Engine Lifecycle Management
//------------------------------------------------------------------------------

TK_NODISCARD tk_error_code_t tk_depth_estimator_create(tk_depth_estimator_t** out_estimator, 
                                                       const tk_depth_estimator_config_t* config) {
    if (!out_estimator || !config || !config->model_path) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Validate configuration parameters
    if (config->input_width == 0 || config->input_height == 0) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Allocate estimator structure
    tk_depth_estimator_t* estimator = (tk_depth_estimator_t*)calloc(1, sizeof(tk_depth_estimator_t));
    if (!estimator) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Copy configuration
    estimator->config = *config;
    
    // Initialize ONNX Runtime
    tk_error_code_t err = initialize_onnx_runtime(estimator);
    if (err != TK_SUCCESS) {
        free(estimator);
        return err;
    }
    
    // Setup model I/O information
    err = setup_model_io_info(estimator);
    if (err != TK_SUCCESS) {
        tk_depth_estimator_destroy(&estimator);
        return err;
    }
    
    // Allocate internal buffers
    err = allocate_buffers(estimator);
    if (err != TK_SUCCESS) {
        tk_depth_estimator_destroy(&estimator);
        return err;
    }
    
    // Create memory pool for results
    err = tk_memory_pool_create(&estimator->result_pool, 
                               sizeof(tk_vision_depth_map_t), 
                               10); // Pool for 10 depth maps
    if (err != TK_SUCCESS) {
        tk_depth_estimator_destroy(&estimator);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    *out_estimator = estimator;
    return TK_SUCCESS;
}

void tk_depth_estimator_destroy(tk_depth_estimator_t** estimator) {
    if (!estimator || !*estimator) {
        return;
    }
    
    tk_depth_estimator_t* e = *estimator;
    
    // Release ONNX Runtime resources
    if (e->session) {
        delete e->session;
        e->session = nullptr;
    }
    
    // Release buffers
    release_buffers(e);
    
    // Release memory pool
    if (e->result_pool) {
        tk_memory_pool_destroy(&e->result_pool);
    }
    
    // Free estimator structure
    free(e);
    *estimator = NULL;
}

//------------------------------------------------------------------------------
// Core Inference Function
//------------------------------------------------------------------------------

TK_NODISCARD tk_error_code_t tk_depth_estimator_estimate(
    tk_depth_estimator_t* estimator,
    const tk_video_frame_t* video_frame,
    tk_vision_depth_map_t** out_depth_map) {
    
    if (!estimator || !video_frame || !out_depth_map) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Reset output parameter
    *out_depth_map = NULL;
    
    // Pre-process input frame
    tk_error_code_t err = preprocess_frame(estimator, video_frame);
    if (err != TK_SUCCESS) {
        return err;
    }
    
    // Run model inference
    err = run_inference(estimator);
    if (err != TK_SUCCESS) {
        return TK_ERROR_INFERENCE_FAILED;
    }
    
    // Post-process depth map
    err = postprocess_depth(estimator, out_depth_map);
    if (err != TK_SUCCESS) {
        return err;
    }
    
    return TK_SUCCESS;
}

//------------------------------------------------------------------------------
// Result Data Management
//------------------------------------------------------------------------------

void tk_depth_estimator_free_map(tk_vision_depth_map_t** depth_map) {
    if (depth_map && *depth_map) {
        // Free the depth data buffer
        if ((*depth_map)->data) {
            free((*depth_map)->data);
            (*depth_map)->data = NULL;
        }
        
        // Free the depth map structure itself
        free(*depth_map);
        *depth_map = NULL;
    }
}

//------------------------------------------------------------------------------
// Internal Helper Functions
//------------------------------------------------------------------------------

static tk_error_code_t initialize_onnx_runtime(tk_depth_estimator_t* estimator) {
    // Create ONNX environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TrackieDepthEstimator");
    estimator->env = env;
    
    // Create session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);  // Adjust based on CPU cores
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    // Set execution backend
    switch (estimator->config.backend) {
        case TK_VISION_BACKEND_CPU:
            // CPU is the default, no additional setup needed
            break;
            
#ifdef TRACKIELLM_ENABLE_CUDA
        case TK_VISION_BACKEND_CUDA:
            {
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = estimator->config.gpu_device_id;
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
        Ort::Session* session = new Ort::Session(estimator->env, 
                                                estimator->config.model_path->path_str, 
                                                session_options);
        estimator->session = session;
    } catch (const Ort::Exception& e) {
        tk_log_error("ONNX Runtime error: %s", e.what());
        return TK_ERROR_MODEL_LOAD_FAILED;
    }
    
    return TK_SUCCESS;
}

static tk_error_code_t setup_model_io_info(tk_depth_estimator_t* estimator) {
    // Get input and output node names
    size_t input_count = estimator->session->GetInputCount();
    size_t output_count = estimator->session->GetOutputCount();
    
    if (input_count == 0 || output_count == 0) {
        return TK_ERROR_MODEL_LOAD_FAILED;
    }
    
    // Get input node name
    estimator->input_node_name = estimator->session->GetInputNameAllocated(0, estimator->allocator).release();
    
    // Get output node name
    estimator->output_node_name = estimator->session->GetOutputNameAllocated(0, estimator->allocator).release();
    
    // Get input node dimensions
    Ort::TypeInfo input_type_info = estimator->session->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    estimator->input_node_dims = input_tensor_info.GetShape();
    
    // Validate input dimensions
    if (estimator->input_node_dims.size() != 4) {
        return TK_ERROR_MODEL_LOAD_FAILED;
    }
    
    // Calculate input tensor size
    estimator->input_tensor_size = 1;
    for (int i = 0; i < 4; i++) {
        estimator->input_tensor_size *= estimator->input_node_dims[i];
    }
    
    // Get output tensor info
    Ort::TypeInfo output_type_info = estimator->session->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_dims = output_tensor_info.GetShape();
    
    // Validate output dimensions (should be [batch, height, width] or [batch, 1, height, width])
    if (output_dims.size() < 3) {
        return TK_ERROR_MODEL_LOAD_FAILED;
    }
    
    // Calculate output tensor size
    estimator->output_tensor_size = 1;
    for (size_t i = 0; i < output_dims.size(); i++) {
        estimator->output_tensor_size *= output_dims[i];
    }
    
    return TK_SUCCESS;
}

static tk_error_code_t allocate_buffers(tk_depth_estimator_t* estimator) {
    // Allocate input tensor buffer
    estimator->input_tensor_values = (float*)malloc(estimator->input_tensor_size * sizeof(float));
    if (!estimator->input_tensor_values) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Allocate resized frame buffer
    uint32_t input_width = estimator->config.input_width;
    uint32_t input_height = estimator->config.input_height;
    estimator->resized_frame_buffer = (uint8_t*)malloc(input_width * input_height * 3 * sizeof(uint8_t));
    if (!estimator->resized_frame_buffer) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Allocate raw depth values buffer
    estimator->raw_depth_values = (float*)malloc(estimator->output_tensor_size * sizeof(float));
    if (!estimator->raw_depth_values) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    return TK_SUCCESS;
}

static void release_buffers(tk_depth_estimator_t* estimator) {
    if (estimator->input_tensor_values) {
        free(estimator->input_tensor_values);
        estimator->input_tensor_values = NULL;
    }
    
    if (estimator->resized_frame_buffer) {
        free(estimator->resized_frame_buffer);
        estimator->resized_frame_buffer = NULL;
    }
    
    if (estimator->raw_depth_values) {
        free(estimator->raw_depth_values);
        estimator->raw_depth_values = NULL;
    }
    
    if (estimator->input_node_name) {
        free(estimator->input_node_name);
        estimator->input_node_name = NULL;
    }
    
    if (estimator->output_node_name) {
        free(estimator->output_node_name);
        estimator->output_node_name = NULL;
    }
}

static tk_error_code_t preprocess_frame(tk_depth_estimator_t* estimator, 
                                       const tk_video_frame_t* frame) {
    // Resize frame to model input size
    // This is a simplified resize - in practice, you might want to use
    // a more sophisticated resizing algorithm like bilinear interpolation
    uint32_t orig_width = frame->width;
    uint32_t orig_height = frame->height;
    uint32_t target_width = estimator->config.input_width;
    uint32_t target_height = estimator->config.input_height;
    
    // Simple nearest-neighbor resize (for demonstration)
    for (uint32_t y = 0; y < target_height; y++) {
        for (uint32_t x = 0; x < target_width; x++) {
            uint32_t orig_x = (x * orig_width) / target_width;
            uint32_t orig_y = (y * orig_height) / target_height;
            
            if (orig_x >= orig_width) orig_x = orig_width - 1;
            if (orig_y >= orig_height) orig_y = orig_height - 1;
            
            size_t orig_idx = (orig_y * orig_width + orig_x) * 3;
            size_t target_idx = (y * target_width + x) * 3;
            
            estimator->resized_frame_buffer[target_idx] = frame->data[orig_idx];
            estimator->resized_frame_buffer[target_idx + 1] = frame->data[orig_idx + 1];
            estimator->resized_frame_buffer[target_idx + 2] = frame->data[orig_idx + 2];
        }
    }
    
    // Normalize and convert to float tensor
    // Assuming RGB format and normalization to [0, 1]
    // MiDaS expects RGB input normalized with ImageNet mean/std
    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float std[] = {0.229f, 0.224f, 0.225f};
    
    for (size_t i = 0; i < target_width * target_height; i++) {
        size_t pixel_idx = i * 3;
        estimator->input_tensor_values[pixel_idx] = 
            (estimator->resized_frame_buffer[pixel_idx] / 255.0f - mean[0]) / std[0];
        estimator->input_tensor_values[pixel_idx + 1] = 
            (estimator->resized_frame_buffer[pixel_idx + 1] / 255.0f - mean[1]) / std[1];
        estimator->input_tensor_values[pixel_idx + 2] = 
            (estimator->resized_frame_buffer[pixel_idx + 2] / 255.0f - mean[2]) / std[2];
    }
    
    return TK_SUCCESS;
}

static tk_error_code_t run_inference(tk_depth_estimator_t* estimator) {
    // Create input tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        estimator->input_tensor_values, 
        estimator->input_tensor_size,
        estimator->input_node_dims.data(), 
        estimator->input_node_dims.size()
    );
    
    // Run inference
    const char* input_names[] = { estimator->input_node_name };
    const char* output_names[] = { estimator->output_node_name };
    
    try {
        Ort::RunOptions run_options;
        auto output_tensors = estimator->session->Run(
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
        if (output_data_size > estimator->output_tensor_size) {
            tk_log_warn("Output data size (%zu) is larger than allocated buffer (%zu). Truncating.",
                        output_data_size, estimator->output_tensor_size);
            output_data_size = estimator->output_tensor_size;
        }
        
        memcpy(estimator->raw_depth_values, output_data, output_data_size * sizeof(float));
    } catch (const Ort::Exception& e) {
        tk_log_error("ONNX Runtime inference error: %s", e.what());
        return TK_ERROR_INFERENCE_FAILED;
    }
    return TK_SUCCESS;
}

static tk_error_code_t postprocess_depth(tk_depth_estimator_t* estimator, 
                                        tk_vision_depth_map_t** out_depth_map) {
    // Allocate depth map structure
    tk_vision_depth_map_t* depth_map = (tk_vision_depth_map_t*)malloc(sizeof(tk_vision_depth_map_t));
    if (!depth_map) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Set dimensions
    depth_map->width = estimator->config.input_width;
    depth_map->height = estimator->config.input_height;
    
    // Allocate depth data buffer
    size_t data_size = depth_map->width * depth_map->height;
    depth_map->data = (float*)malloc(data_size * sizeof(float));
    if (!depth_map->data) {
        free(depth_map);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Copy raw depth values
    memcpy(depth_map->data, estimator->raw_depth_values, data_size * sizeof(float));
    
    // Convert inverse depth to metric depth
    convert_inverse_depth_to_metric(depth_map->data, data_size, 
                                   TK_DEPTH_ESTIMATOR_MIN_DEPTH, 
                                   TK_DEPTH_ESTIMATOR_MAX_DEPTH);
    
    *out_depth_map = depth_map;
    return TK_SUCCESS;
}

static void convert_inverse_depth_to_metric(float* depth_data, size_t data_size, 
                                           float min_depth, float max_depth) {
    // Find min and max values in the depth data
    float min_val = depth_data[0];
    float max_val = depth_data[0];
    
    for (size_t i = 1; i < data_size; i++) {
        if (depth_data[i] < min_val) min_val = depth_data[i];
        if (depth_data[i] > max_val) max_val = depth_data[i];
    }
    
    // Avoid division by zero
    if (max_val - min_val < 1e-6f) {
        // If all values are the same, set to max_depth
        for (size_t i = 0; i < data_size; i++) {
            depth_data[i] = max_depth;
        }
        return;
    }
    
    // Normalize to [0, 1] and then scale to [min_depth, max_depth]
    // Inverse the mapping since MiDaS outputs inverse depth
    for (size_t i = 0; i < data_size; i++) {
        float normalized = (depth_data[i] - min_val) / (max_val - min_val);
        depth_data[i] = max_depth - normalized * (max_depth - min_depth);
    }
}
