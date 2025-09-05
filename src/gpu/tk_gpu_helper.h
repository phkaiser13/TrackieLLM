/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_gpu_helper.h
*
* This header file defines common data structures and types used across all
* GPU backends (CUDA, Vulkan, Metal, etc.). This ensures that the high-level
* application logic can interact with the GPU abstraction layer using a
* consistent and backend-agnostic API.
*
* The primary contents of this file are the parameter structs for the various
* compute kernels. By centralizing these definitions, we avoid code duplication
* and ensure that all backends conform to the same interface contract.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TRACKIELLM_GPU_TK_GPU_HELPER_H
#define TRACKIELLM_GPU_TK_GPU_HELPER_H

#include <stdint.h>

// Define a basic float3 vector type that can be shared.
// In a real project, this would be part of a more extensive math library.
typedef struct {
    float x, y, z;
} TkFloat3;


//------------------------------------------------------------------------------
// Kernel: Image Pre-processing for ONNX Models
//------------------------------------------------------------------------------

/**
 * @struct tk_preprocess_params_t
 * @brief Parameters for the image pre-processing kernel.
 *
 * This structure is backend-agnostic. The pointers `d_input_image` and
 * `d_output_tensor` are interpreted by each backend as pointers to their
 * native buffer types (e.g., `void*` in CUDA, bound via a handle).
 */
typedef struct {
    // --- Input ---
    const void* d_input_image; /**< DEVICE pointer/handle to the source image. */
    uint32_t input_width;
    uint32_t input_height;
    uint32_t input_stride_bytes;

    // --- Output ---
    void* d_output_tensor;     /**< DEVICE pointer/handle to the output planar tensor. */
    uint32_t output_width;
    uint32_t output_height;

    // --- Normalization Parameters ---
    TkFloat3 mean;
    TkFloat3 std_dev;
} tk_preprocess_params_t;


//------------------------------------------------------------------------------
// Kernel: Depth Map Post-processing
//------------------------------------------------------------------------------

/**
 * @struct tk_postprocess_depth_params_t
 * @brief Parameters for the depth map post-processing kernel.
 */
typedef struct {
    // --- Input ---
    const void* d_raw_depth_map; /**< DEVICE pointer/handle to the raw depth map. */
    uint32_t width;
    uint32_t height;

    // --- Output ---
    void* d_metric_depth_map;    /**< DEVICE pointer/handle to the output depth map. */

    // --- Conversion Parameters ---
    float scale;
    float shift;
} tk_postprocess_depth_params_t;


//------------------------------------------------------------------------------
// Kernel: Depth Map to 3D Point Cloud
//------------------------------------------------------------------------------

/**
 * @struct tk_depth_to_points_params_t
 * @brief Parameters for the depth-to-point-cloud kernel.
 */
typedef struct {
    // --- Input ---
    const void* d_metric_depth_map; /**< DEVICE pointer/handle to the metric depth map. */
    uint32_t width;
    uint32_t height;

    // --- Output ---
    void* d_point_cloud;         /**< DEVICE pointer/handle to the output point cloud buffer. */

    // --- Camera Intrinsics ---
    float fx;
    float fy;
    float cx;
    float cy;
} tk_depth_to_points_params_t;


#endif // TRACKIELLM_GPU_TK_GPU_HELPER_H
