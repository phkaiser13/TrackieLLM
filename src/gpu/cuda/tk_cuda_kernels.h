/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_cuda_kernels.h
*
* This header file defines the C-callable interface for the core CUDA kernels
* used in the TrackieLLM project. It acts as a strict contract between the
* C/C++ dispatch layer and the CUDA C++ (`.cu`) implementation file.
*
* The design philosophy is based on several key engineering principles:
*   1. Separation of Concerns: The dispatcher knows *what* to run and *when*,
*      but not *how*. This header abstracts the `__global__` kernel launch syntax.
*   2. Asynchronous Execution: Every kernel wrapper accepts a `cudaStream_t`,
*      enabling the dispatcher to manage concurrent operations and overlap
*      data transfers with computation for maximum throughput.
*   3. Interface Stability: Kernel parameters are passed via dedicated structs.
*      This prevents function signatures from becoming unwieldy and allows for
*      parameters to be added in the future without breaking the API.
*
* This is the public API for the GPU's computational workhorse functions.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TRACKIELLM_GPU_CUDA_TK_CUDA_KERNELS_H
#define TRACKIELLM_GPU_CUDA_TK_CUDA_KERNELS_H

#include <cuda_runtime.h> // For cudaStream_t

#include "utils/tk_error_handling.h"
#include "gpu/tk_gpu_helper.h" // Common GPU data structures

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Launches a kernel to pre-process an image for a neural network.
 *
 * This kernel handles resizing (using bilinear interpolation), data type
 * conversion (uint8 to float32), layout conversion (interleaved to planar),
* and normalization, all in a single, highly parallel pass.
 *
 * @param[in] params A pointer to the parameter structure. The device pointers
 *                   within are cast to their appropriate CUDA types inside the
 *                   implementation.
 * @param[in] stream The CUDA stream on which to launch the kernel.
 *
 * @return TK_SUCCESS if the kernel was successfully launched.
 */
TK_NODISCARD tk_error_code_t tk_kernels_preprocess_image(const tk_preprocess_params_t* params, cudaStream_t stream);

/**
 * @brief Launches a kernel to convert a raw model output into a metric depth map.
 *
 * This typically involves a simple element-wise operation (e.g., value * scale + shift)
 * to transform the network's output range into real-world units (meters).
 *
 * @param[in] params A pointer to the parameter structure.
 * @param[in] stream The CUDA stream on which to launch the kernel.
 *
 * @return TK_SUCCESS if the kernel was successfully launched.
 */
TK_NODISCARD tk_error_code_t tk_kernels_postprocess_depth_map(const tk_postprocess_depth_params_t* params, cudaStream_t stream);

/**
 * @brief Launches a kernel to unproject a 2D depth map into a 3D point cloud.
 *
 * Each thread processes one pixel (u, v) with depth (d) and computes the
 * corresponding 3D point (x, y, z) using the camera's intrinsic parameters.
 * This is a fundamental step for 3D geometric analysis.
 *
 * @param[in] params A pointer to the parameter structure.
 * @param[in] stream The CUDA stream on which to launch the kernel.
 *
 * @return TK_SUCCESS if the kernel was successfully launched.
 */
TK_NODISCARD tk_error_code_t tk_kernels_depth_to_point_cloud(const tk_depth_to_points_params_t* params, cudaStream_t stream);


/**
 * @brief Launches a kernel to compute the softmax function on a 2D tensor.
 *
 * This operation is performed row-wise. It's a key component for attention
 * layers in transformer models.
 *
 * @param[in] params A pointer to the softmax parameter structure.
 * @param[in] stream The CUDA stream on which to launch the kernel.
 *
 * @return TK_SUCCESS if the kernel was successfully launched.
 */
TK_NODISCARD tk_error_code_t tk_kernels_softmax(const tk_softmax_params_t* params, cudaStream_t stream);


#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_GPU_CUDA_TK_CUDA_KERNELS_H