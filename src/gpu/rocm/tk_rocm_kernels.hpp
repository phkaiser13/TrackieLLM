/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_rocm_kernels.hpp
 *
 * This header file defines the C-callable interface for the core HIP kernels
 * designed for the ROCm (AMD GPU) backend. It serves as a stable ABI contract
 * between the C/C++ dispatch layer and the HIP C++ (`.cpp`) implementation file.
 * This ensures a clean separation of concerns, where the dispatcher handles the
 * "what" and "when" of execution, while the kernel implementation handles the "how".
 *
 * Key Engineering Principles:
 *   1.  **Abstraction of Complexity**: Hides the `__global__` kernel launch syntax
 *       and grid/block calculation from the dispatcher.
 *   2.  **Asynchronous by Design**: All kernel wrappers accept a `hipStream_t`,
 *       enabling the dispatcher to orchestrate a fully asynchronous execution
 *       pipeline, overlapping compute and memory operations for peak performance.
 *   3.  **Stable and Extensible API**: Kernel parameters are encapsulated in
 *       dedicated structs. This design choice prevents brittle function signatures
 *       and allows new parameters to be added without breaking existing code.
 *
 * This file is the public-facing API for the GPU's computational workhorse functions.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#ifndef TRACKIELLM_GPU_ROCM_TK_ROCM_KERNELS_HPP
#define TRACKIELLM_GPU_ROCM_TK_ROCM_KERNELS_HPP

#include <hip/hip_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "gpu/rocm/tk_rocm_math_helpers.hpp" // For math types and helpers.

// The `TkFloat3` type from the original CUDA code is assumed to be a struct
// compatible with HIP's `float3`. We define our own `tk_float3` to ensure
// compatibility and maintain consistency. This struct can be passed directly
// to kernels that expect a `float3`.
struct tk_float3 {
    float x, y, z;
};

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Kernel: Image Pre-processing for ONNX Models
//------------------------------------------------------------------------------

/**
 * @struct tk_preprocess_params_t
 * @brief Parameters for the image pre-processing kernel. This struct is designed
 *        to be passed by value to the kernel launcher.
 */
typedef struct {
    // --- Input Buffer ---
    const unsigned char* d_input_image; /**< DEVICE pointer to the source image (e.g., uint8 RGB interleaved). */
    uint32_t input_width;               /**< Width of the source image in pixels. */
    uint32_t input_height;              /**< Height of the source image in pixels. */
    uint32_t input_stride_bytes;        /**< Full row pitch of the source image in bytes. */

    // --- Output Buffer ---
    float* d_output_tensor;             /**< DEVICE pointer to the output planar tensor (e.g., float32 NCHW). */
    uint32_t output_width;              /**< Width of the output tensor. */
    uint32_t output_height;             /**< Height of the output tensor. */

    // --- Normalization & Transformation Parameters ---
    // This allows for flexible pre-processing adapted to various neural network requirements.
    // The transformation is: output = (sample(input) * scale - mean) / std_dev
    tk_float3 mean;                     /**< Mean values for R, G, B channels to be subtracted. */
    tk_float3 std_dev;                  /**< Standard deviation values for R, G, B channels for division. */
    float scale;                        /**< A scaling factor, typically 1/255.0f for uint8 inputs. */
} tk_preprocess_params_t;

/**
 * @brief Launches a highly optimized kernel to pre-process an image for a neural network.
 *
 * This kernel is a fusion of multiple operations into a single GPU pass:
 * 1. Resizing (using high-quality bilinear interpolation).
 * 2. Data Type Conversion (e.g., uint8 to float32).
 * 3. Layout Conversion (interleaved HWC to planar CHW).
 * 4. Normalization (scale, mean subtraction, and standard deviation division).
 *
 * Fusing these operations drastically reduces memory bandwidth requirements and
 * kernel launch overhead, leading to significant performance gains.
 *
 * @param[in] params A pointer to the parameter structure containing all necessary data.
 * @param[in] stream The HIP stream on which to enqueue the kernel launch.
 *
 * @return TK_SUCCESS if the kernel was successfully launched.
 * @return TK_ERROR_GPU_KERNEL_LAUNCH_FAILED on failure.
 */
TK_NODISCARD tk_error_code_t tk_kernels_preprocess_image(const tk_preprocess_params_t* params, hipStream_t stream);

//------------------------------------------------------------------------------
// Kernel: Depth Map Post-processing
//------------------------------------------------------------------------------

/**
 * @struct tk_postprocess_depth_params_t
 * @brief Parameters for the depth map post-processing kernel.
 */
typedef struct {
    // --- Input Buffer ---
    const float* d_raw_depth_map; /**< DEVICE pointer to the raw, single-channel output from a depth estimation model. */
    uint32_t width;
    uint32_t height;

    // --- Output Buffer ---
    float* d_metric_depth_map;    /**< DEVICE pointer to the output depth map, converted to meaningful units (e.g., meters). */

    // --- Conversion Parameters ---
    // Linear transformation: output = input * scale + shift
    float scale;                  /**< Scale factor to apply to the raw depth values. */
    float shift;                  /**< Shift factor to apply after scaling. */
} tk_postprocess_depth_params_t;

/**
 * @brief Launches a kernel to convert a raw model output into a metric depth map.
 *
 * This is a fast, element-wise kernel that applies a linear transformation to each
 * pixel of the raw depth map. It's a common step in computer vision pipelines to
 * make the output of a neural network interpretable in real-world units.
 *
 * @param[in] params A pointer to the parameter structure.
 * @param[in] stream The HIP stream on which to launch the kernel.
 *
 * @return TK_SUCCESS if the kernel was successfully launched.
 */
TK_NODISCARD tk_error_code_t tk_kernels_postprocess_depth_map(const tk_postprocess_depth_params_t* params, hipStream_t stream);

//------------------------------------------------------------------------------
// Kernel: Depth Map to 3D Point Cloud
//------------------------------------------------------------------------------

/**
 * @struct tk_depth_to_points_params_t
 * @brief Parameters for the depth-to-point-cloud kernel.
 */
typedef struct {
    // --- Input Buffer ---
    const float* d_metric_depth_map; /**< DEVICE pointer to the metric depth map (values are distances in meters). */
    uint32_t width;
    uint32_t height;

    // --- Output Buffer ---
    tk_float3* d_point_cloud;        /**< DEVICE pointer to the output point cloud buffer. Must have space for width * height points. */

    // --- Camera Intrinsic Parameters ---
    // These define the projective geometry of the camera.
    float fx; /**< Focal length in x. */
    float fy; /**< Focal length in y. */
    float cx; /**< Principal point offset in x. */
    float cy; /**< Principal point offset in y. */
} tk_depth_to_points_params_t;

/**
 * @brief Launches a kernel to unproject a 2D depth map into a 3D point cloud.
 *
 * This kernel performs the fundamental geometric transformation from image space
 * to 3D world space. Each GPU thread processes one pixel (u, v) with its
 * corresponding depth value (d) and computes the 3D point (x, y, z) using the
 * camera's intrinsic parameters via the pinhole camera model equations:
 *   x = (u - cx) * d / fx
 *   y = (v - cy) * d / fy
 *   z = d
 *
 * @param[in] params A pointer to the parameter structure.
 * @param[in] stream The HIP stream on which to launch the kernel.
 *
 * @return TK_SUCCESS if the kernel was successfully launched.
 */
TK_NODISCARD tk_error_code_t tk_kernels_depth_to_point_cloud(const tk_depth_to_points_params_t* params, hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_GPU_ROCM_TK_ROCM_KERNELS_HPP
