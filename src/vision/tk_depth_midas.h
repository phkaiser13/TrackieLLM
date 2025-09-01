/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_depth_midas.h
*
* This header file defines the public API for the Monocular Depth Estimation
* Engine. This is a specialized sub-component of the TrackieLLM Vision Pipeline,
* dedicated to inferring 3D depth information from a single 2D image using the
* MiDaS (DPT-SwinV2-Tiny) model.
*
* The primary objective of this engine is to produce a dense depth map, where
* each pixel's value corresponds to an estimated distance from the camera. This
* output is a critical input for navigation, obstacle avoidance, and providing
* rich spatial context to the Cortex.
*
* Similar to the object detector, this engine is designed to be instantiated and
* managed by the `tk_vision_pipeline`, ensuring a clean, hierarchical design.
* The API abstracts the specifics of the ONNX Runtime, model-specific pre- and
* post-processing, and memory management for the large output tensors.
*
* SPDX-License-Identifier:
*/

#ifndef TRACKIELLM_VISION_TK_DEPTH_MIDAS_H
#define TRACKIELLM_VISION_TK_DEPTH_MIDAS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "internal_tools/tk_file_manager.h"
#include "vision/tk_vision_pipeline.h" // For shared types

// Forward-declare the primary estimator object as an opaque type.
typedef struct tk_depth_estimator_s tk_depth_estimator_t;

/**
 * @struct tk_depth_estimator_config_t
 * @brief Configuration for initializing the Depth Estimation Engine.
 */
typedef struct {
    tk_vision_backend_e backend;            /**< The desired inference backend (CPU, CUDA, etc.). */
    int                 gpu_device_id;      /**< The ID of the GPU to use (if applicable). */
    tk_path_t*          model_path;         /**< Path to the ONNX model file. */

    // Model-specific parameters
    uint32_t            input_width;        /**< The width the model expects for its input tensor (e.g., 256). */
    uint32_t            input_height;       /**< The height the model expects for its input tensor (e.g., 256). */
} tk_depth_estimator_config_t;


#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Engine Lifecycle Management
//------------------------------------------------------------------------------

/**
 * @brief Creates and initializes a new Depth Estimation Engine instance.
 *
 * This function loads the MiDaS ONNX model, creates an ONNX Runtime inference
 * session for the specified backend, and pre-allocates internal buffers.
 *
 * @param[out] out_estimator Pointer to receive the address of the new engine instance.
 * @param[in] config The configuration for the engine.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL or config is invalid.
 * @return TK_ERROR_OUT_OF_MEMORY on memory allocation failure.
 * @return TK_ERROR_MODEL_LOAD_FAILED if the ONNX model cannot be loaded or parsed.
 */
TK_NODISCARD tk_error_code_t tk_depth_estimator_create(tk_depth_estimator_t** out_estimator, const tk_depth_estimator_config_t* config);

/**
 * @brief Destroys a Depth Estimation Engine instance.
 *
 * @param[in,out] estimator Pointer to the engine instance to be destroyed.
 */
void tk_depth_estimator_destroy(tk_depth_estimator_t** estimator);

//------------------------------------------------------------------------------
// Core Inference Function
//------------------------------------------------------------------------------

/**
 * @brief Estimates the depth map for a single video frame.
 *
 * This function performs the full depth estimation pipeline:
 * 1. Pre-processes the input frame (resize, normalize, convert to tensor).
 * 2. Runs the ONNX model to get a raw inverse depth tensor.
 * 3. Post-processes the raw output to convert it into a metric depth map (meters).
 *
 * @param[in] estimator The depth estimator instance.
 * @param[in] video_frame The raw video frame to process.
 * @param[out] out_depth_map Pointer to a dynamically allocated depth map structure.
 *                         The caller assumes ownership of this structure and
 *                         MUST free it using `tk_depth_estimator_free_map`.
 *                         The `data` buffer within the structure is part of this
 *                         single allocation.
 *
 * @return TK_SUCCESS on successful estimation.
 * @return TK_ERROR_INVALID_ARGUMENT if required pointers are NULL.
 * @return TK_ERROR_INFERENCE_FAILED if the ONNX Runtime session fails.
 * @return TK_ERROR_OUT_OF_MEMORY if result allocation fails.
 *
 * @par Thread-Safety
 * This function is NOT thread-safe. It modifies the internal state of the
 * estimator and should only be called from one thread at a time.
 */
TK_NODISCARD tk_error_code_t tk_depth_estimator_estimate(
    tk_depth_estimator_t* estimator,
    const tk_video_frame_t* video_frame,
    tk_vision_depth_map_t** out_depth_map
);

//------------------------------------------------------------------------------
// Result Data Management
//------------------------------------------------------------------------------

/**
 * @brief Frees the memory allocated for a depth map result.
 *
 * This function must be called on the `out_depth_map` pointer returned by
 * `tk_depth_estimator_estimate` to prevent memory leaks.
 *
 * @param[in,out] depth_map Pointer to the depth map structure to be freed.
 */
void tk_depth_estimator_free_map(tk_vision_depth_map_t** depth_map);


#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_VISION_TK_DEPTH_MIDAS_H