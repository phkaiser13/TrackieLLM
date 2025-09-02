/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_object_detector.h
*
* This header file defines the public API for the Object Detection Engine, a
* specialized sub-component within the TrackieLLM Vision Pipeline. Its sole
* responsibility is to execute a trained object detection model (e.g., YOLOv5nu)
* on a video frame and return a list of detected objects with high fidelity.
*
* This module is not intended for direct use by the Cortex. Instead, it is
* designed to be instantiated and managed by the `tk_vision_pipeline`, which
* acts as the orchestrator for all vision-related tasks. This hierarchical
* design promotes separation of concerns and modularity.
*
* The API provides granular control over the detection process, including
* inference backend selection and post-processing parameters like confidence
* and Non-Max Suppression (NMS) thresholds, which are critical for tuning
* performance and accuracy in real-world scenarios.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TRACKIELLM_VISION_TK_OBJECT_DETECTOR_H
#define TRACKIELLM_VISION_TK_OBJECT_DETECTOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "internal_tools/tk_file_manager.h"
#include "vision/tk_vision_pipeline.h" // For shared types like tk_vision_backend_e

// Forward-declare the primary detector object as an opaque type.
typedef struct tk_object_detector_s tk_object_detector_t;

/**
 * @struct tk_object_detector_config_t
 * @brief Configuration for initializing the Object Detection Engine.
 */
typedef struct {
    tk_vision_backend_e backend;            /**< The desired inference backend (CPU, CUDA, etc.). */
    int                 gpu_device_id;      /**< The ID of the GPU to use (if applicable). */
    tk_path_t*          model_path;         /**< Path to the ONNX model file. */

    // Model-specific parameters
    uint32_t            input_width;        /**< The width the model expects for its input tensor (e.g., 640). */
    uint32_t            input_height;       /**< The height the model expects for its input tensor (e.g., 640). */
    const char**        class_labels;       /**< A null-terminated array of strings representing the class names.
                                                 The order must match the model's output indices. */
    size_t              class_count;        /**< The total number of classes. */

    // Post-processing parameters
    float               confidence_threshold; /**< Minimum confidence score to consider a detection valid. */
    float               iou_threshold;      /**< Intersection-over-Union (IoU) threshold for Non-Max Suppression. */
} tk_object_detector_config_t;

/**
 * @struct tk_detection_result_t
 * @brief Represents a single object detected by the engine.
 *
 * This is a low-level result structure. The vision pipeline will consume this
 * and transform it into the higher-level `tk_vision_object_t`.
 */
typedef struct {
    uint32_t    class_id;       /**< The integer ID of the object's class. */
    const char* label;          /**< The human-readable string label (pointer to config data). */
    float       confidence;     /**< The model's confidence in the detection (0.0 to 1.0). */
    tk_rect_t   bbox;           /**< The bounding box in the original frame's coordinates. */
} tk_detection_result_t;


#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Engine Lifecycle Management
//------------------------------------------------------------------------------

/**
 * @brief Creates and initializes a new Object Detection Engine instance.
 *
 * This function loads the ONNX model, creates an ONNX Runtime inference session
 * configured for the specified backend, and pre-allocates internal buffers for
 * pre- and post-processing.
 *
 * @param[out] out_detector Pointer to receive the address of the new engine instance.
 * @param[in] config The configuration for the engine. The `class_labels` array
 *                   is not copied; its lifetime must be managed by the caller
 *                   and must exceed the lifetime of the detector object.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL or config is invalid.
 * @return TK_ERROR_OUT_OF_MEMORY on memory allocation failure.
 * @return TK_ERROR_MODEL_LOAD_FAILED if the ONNX model cannot be loaded or parsed.
 */
TK_NODISCARD tk_error_code_t tk_object_detector_create(tk_object_detector_t** out_detector, const tk_object_detector_config_t* config);

/**
 * @brief Destroys an Object Detection Engine instance.
 *
 * @param[in,out] detector Pointer to the engine instance to be destroyed.
 */
void tk_object_detector_destroy(tk_object_detector_t** detector);

//------------------------------------------------------------------------------
// Core Inference Function
//------------------------------------------------------------------------------

/**
 * @brief Runs object detection on a single video frame.
 *
 * This function performs the full detection pipeline:
 * 1. Pre-processes the input frame (resize, pad, normalize, convert to tensor).
 * 2. Runs the ONNX model to get raw output tensors.
 * 3. Post-processes the raw output (applies confidence threshold, performs NMS).
 * 4. Converts the final detections back to the original frame's coordinate space.
 *
 * @param[in] detector The object detector instance.
 * @param[in] video_frame The raw video frame to process.
 * @param[out] out_results Pointer to a dynamically allocated array of results.
 *                         The caller assumes ownership of this array and MUST
 *                         free it using `tk_object_detector_free_results`.
 * @param[out] out_result_count Pointer to a size_t that will hold the number of
 *                            detections in the `out_results` array.
 *
 * @return TK_SUCCESS on successful detection.
 * @return TK_ERROR_INVALID_ARGUMENT if required pointers are NULL.
 * @return TK_ERROR_INFERENCE_FAILED if the ONNX Runtime session fails.
 * @return TK_ERROR_OUT_OF_MEMORY if result allocation fails.
 *
 * @par Thread-Safety
 * This function is NOT thread-safe. It modifies the internal state of the
 * detector and should only be called from one thread at a time.
 */
TK_NODISCARD tk_error_code_t tk_object_detector_detect(
    tk_object_detector_t* detector,
    const tk_video_frame_t* video_frame,
    tk_detection_result_t** out_results,
    size_t* out_result_count
);

//------------------------------------------------------------------------------
// Result Data Management
//------------------------------------------------------------------------------

/**
 * @brief Frees the memory allocated for an array of detection results.
 *
 * This function must be called on the `out_results` pointer returned by
 * `tk_object_detector_detect` to prevent memory leaks.
 *
 * @param[in,out] results Pointer to the array of results to be freed.
 */
void tk_object_detector_free_results(tk_detection_result_t** results);


#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_VISION_TK_OBJECT_DETECTOR_H