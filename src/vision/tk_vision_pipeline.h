/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_vision_pipeline.h
*
* This header file defines the public API for the TrackieLLM Vision Pipeline.
* This subsystem is a cornerstone of the project, responsible for transforming
* raw video frames into a structured, semantic understanding of the environment.
*
* The pipeline is architected as a stateful, high-throughput processing engine.
* It encapsulates multiple AI models and computer vision algorithms:
*   1. Object Detection (YOLOv5nu): To identify and locate objects.
*   2. Monocular Depth Estimation (MiDaS): To perceive distances.
*   3. Optical Character Recognition (Tesseract): To read text.
*   4. Sensor Fusion Logic: To combine the outputs of the above models into
*      actionable insights (e.g., attaching a distance to each detected object).
*
* The API is designed around an opaque handle, `tk_vision_pipeline_t`, to hide
* the immense complexity of model loading, backend selection (CPU/CUDA), memory
* management, and inference execution. The primary interaction occurs via the
* `tk_vision_pipeline_process_frame` function, which is a synchronous, blocking
* call designed to be executed within the Cortex's main real-time loop.
*
* Ownership of the complex result data is managed explicitly via a dedicated
* `tk_vision_result_destroy` function, promoting a clear and safe memory model.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TRACKIELLM_VISION_TK_VISION_PIPELINE_H
#define TRACKIELLM_VISION_TK_VISION_PIPELINE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "internal_tools/tk_file_manager.h" // For tk_path_t
#include "event_bus/event_bus_ffi.h" // For tk_event_bus_t

// Forward-declare the primary pipeline and result objects as opaque types.
typedef struct tk_vision_pipeline_s tk_vision_pipeline_t;
typedef struct tk_vision_result_s tk_vision_result_t;

/**
 * @enum tk_vision_backend_e
 * @brief Enumerates the supported inference backends for the vision models.
 *
 * The selection of a backend determines where the neural network computations
 * will be executed.
 */
typedef enum {
    TK_VISION_BACKEND_CPU,  /**< Use the CPU for all inference tasks. Fallback option. */
    TK_VISION_BACKEND_CUDA, /**< Use an NVIDIA GPU via the CUDA backend. */
    TK_VISION_BACKEND_METAL,/**< Use an Apple GPU via the Metal Performance Shaders. (Future) */
    TK_VISION_BACKEND_ROCM  /**< Use an AMD GPU via the ROCm/HIP backend. (Future) */
} tk_vision_backend_e;

/**
 * @typedef tk_vision_analysis_flags_t
 * @brief A bitmask to control which analyses are performed on a frame.
 *
 * This allows the Cortex to dynamically request only the necessary information,
 * saving significant computational resources. For example, OCR is expensive and
 * should only be run when explicitly needed.
 */
typedef uint32_t tk_vision_analysis_flags_t;
enum {
    TK_VISION_ANALYZE_NONE                 = 0,
    TK_VISION_ANALYZE_OBJECT_DETECTION     = 1 << 0, /**< Enable YOLO object detection. */
    TK_VISION_ANALYZE_DEPTH_ESTIMATION     = 1 << 1, /**< Enable MiDaS depth estimation. */
    TK_VISION_ANALYZE_OCR                  = 1 << 2, /**< Enable Tesseract OCR. */
    
    /**
     * @brief A meta-flag to enable the fusion of detection and depth data.
     * Requires both OBJECT_DETECTION and DEPTH_ESTIMATION to be active.
     * This calculates the distance for each detected object.
     */
    TK_VISION_ANALYZE_FUSION_DISTANCE      = 1 << 3,

    /**
     * @brief A common preset for general environmental awareness.
     */
    TK_VISION_PRESET_ENVIRONMENT_AWARENESS = TK_VISION_ANALYZE_OBJECT_DETECTION |
                                             TK_VISION_ANALYZE_DEPTH_ESTIMATION |
                                             TK_VISION_ANALYZE_FUSION_DISTANCE
};

/**
 * @struct tk_vision_pipeline_config_t
 * @brief Comprehensive configuration for initializing the vision pipeline.
 */
typedef struct {
    tk_event_bus_t*     event_bus;          /**< Handle to the central event bus. */
    tk_vision_backend_e backend;            /**< The desired inference backend. */
    int                 gpu_device_id;      /**< The ID of the GPU to use (if applicable). */
    tk_path_t*          object_detection_model_path; /**< Path to the YOLO ONNX model. */
    tk_path_t*          depth_estimation_model_path; /**< Path to the MiDaS ONNX model. */
    tk_path_t*          tesseract_data_path;/**< Path to the Tesseract 'tessdata' directory. */
    float               object_confidence_threshold; /**< Minimum confidence to report a detected object (0.0 to 1.0). */
    uint32_t            max_detected_objects; /**< Maximum number of objects to report per frame. */
    float               focal_length_x;     /**< Camera focal length in x direction (pixels). */
    float               focal_length_y;     /**< Camera focal length in y direction (pixels). */
} tk_vision_pipeline_config_t;

/**
 * @struct tk_rect_t
 * @brief A simple integer rectangle structure for bounding boxes.
 */
typedef struct {
    int x; /**< X-coordinate of the top-left corner. */
    int y; /**< Y-coordinate of the top-left corner. */
    int w; /**< Width of the rectangle. */
    int h; /**< Height of the rectangle. */
} tk_rect_t;

/**
 * @struct tk_vision_object_t
 * @brief Represents a single detected object in the scene.
 */
typedef struct {
    uint32_t    class_id;       /**< The integer ID of the object's class. */
    const char* label;          /**< The human-readable string label for the class. */
    float       confidence;     /**< The model's confidence in the detection (0.0 to 1.0). */
    tk_rect_t   bbox;           /**< The bounding box of the object in pixel coordinates. */
    float       distance_meters;/**< The estimated distance to the object in meters. Populated by fusion. */
    float       width_meters;   /**< The estimated width of the object in meters. Populated by fusion. */
    float       height_meters;  /**< The estimated height of the object in meters. Populated by fusion. */
} tk_vision_object_t;

/**
 * @struct tk_vision_text_block_t
 * @brief Represents a block of recognized text.
 */
typedef struct {
    char*       text;           /**< The recognized text string (UTF-8). Owned by the result object. */
    float       confidence;     /**< The OCR engine's confidence in the recognition. */
    tk_rect_t   bbox;           /**< The bounding box of the text block. */
} tk_vision_text_block_t;

/**
 * @struct tk_vision_depth_map_t
 * @brief Represents the dense depth map of the scene.
 */
typedef struct {
    uint32_t    width;          /**< Width of the depth map. */
    uint32_t    height;         /**< Height of the depth map. */
    float*      data;           /**< Pointer to the raw depth data (in meters). Owned by the result object. */
} tk_vision_depth_map_t;

/**
 * @struct tk_vision_result_s
 * @brief A comprehensive container for all vision analysis results from a single frame.
 *
 * This structure and all its contents are allocated by `tk_vision_pipeline_process_frame`
 * and must be freed by the caller using `tk_vision_result_destroy`.
 */
struct tk_vision_result_s {
    uint64_t                source_frame_timestamp_ns; /**< Timestamp of the input frame for synchronization. */
    
    size_t                  object_count;   /**< Number of valid objects in the `objects` array. */
    tk_vision_object_t*     objects;        /**< Dynamically allocated array of detected objects. */

    size_t                  text_block_count; /**< Number of valid blocks in the `text_blocks` array. */
    tk_vision_text_block_t* text_blocks;    /**< Dynamically allocated array of recognized text blocks. */

    tk_vision_depth_map_t*  depth_map;      /**< Pointer to the depth map data. May be NULL if not requested. */
};

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Pipeline Lifecycle Management
//------------------------------------------------------------------------------

/**
 * @brief Creates and initializes a new vision pipeline instance.
 *
 * This function performs all heavy lifting: loading models into memory,
 * initializing the selected inference backend (e.g., ONNX Runtime with CUDA),
 * setting up Tesseract, and pre-allocating necessary memory buffers.
 *
 * @param[out] out_pipeline A pointer to a tk_vision_pipeline_t* that will receive
 *                          the address of the newly created pipeline instance.
 * @param[in] config A pointer to the configuration structure. The pipeline
 *                   maintains its own copy of necessary configuration data.
 *
 * @return TK_SUCCESS on successful creation and initialization.
 * @return TK_ERROR_INVALID_ARGUMENT if out_pipeline or config is NULL, or if paths are invalid.
 * @return TK_ERROR_OUT_OF_MEMORY if memory allocation fails.
 * @return TK_ERROR_MODEL_LOAD_FAILED if any of the specified AI models cannot be loaded.
 * @return TK_ERROR_GPU_DEVICE_NOT_FOUND if the specified GPU backend/device is unavailable.
 */
TK_NODISCARD tk_error_code_t tk_vision_pipeline_create(tk_vision_pipeline_t** out_pipeline, const tk_vision_pipeline_config_t* config);

/**
 * @brief Destroys a vision pipeline instance and frees all associated resources.
 *
 * Gracefully releases all models, inference sessions, GPU memory, and other
 * allocated resources.
 *
 * @param[in,out] pipeline A pointer to the tk_vision_pipeline_t* to be destroyed.
 *                         The pointer is set to NULL after destruction.
 */
void tk_vision_pipeline_destroy(tk_vision_pipeline_t** pipeline);

//------------------------------------------------------------------------------
// Core Processing Function
//------------------------------------------------------------------------------

/**
 * @brief Processes a single video frame and produces a structured result.
 *
 * This is the core function of the vision pipeline. It is a synchronous,
 * blocking, and computationally intensive call. It takes a raw video frame,
 * performs all requested analyses (as specified by the flags), and returns a
* comprehensive result object.
 *
 * @param[in] pipeline The vision pipeline instance.
 * @param[in] video_frame The raw video frame data to be processed. The pixel
 *                        format must be compatible (e.g., RGB8).
 * @param[in] analysis_flags A bitmask specifying which analyses to perform.
 * @param[in] timestamp_ns A nanosecond-resolution timestamp for the frame.
 * @param[out] out_result A pointer to a tk_vision_result_t* that will receive
 *                        the address of the newly allocated result object. The
 *                        caller assumes ownership of this object and MUST free
 *                        it using `tk_vision_result_destroy`.
 *
 * @return TK_SUCCESS on successful processing.
 * @return TK_ERROR_INVALID_ARGUMENT if pipeline, video_frame, or out_result is NULL.
 * @return TK_ERROR_INFERENCE_FAILED if a model fails to run.
 * @return TK_ERROR_OUT_OF_MEMORY if allocation for the result object fails.
 *
 * @par Thread-Safety
 * This function is NOT thread-safe with respect to the pipeline object. Only
 * one thread should call this function on a given pipeline instance at a time.
 */
TK_NODISCARD tk_error_code_t tk_vision_pipeline_process_frame(
    tk_vision_pipeline_t* pipeline,
    const tk_video_frame_t* video_frame,
    tk_vision_analysis_flags_t analysis_flags,
    uint64_t timestamp_ns,
    tk_vision_result_t** out_result
);

//------------------------------------------------------------------------------
// Result Data Management
//------------------------------------------------------------------------------

/**
 * @brief Destroys a vision result object and all its associated data.
 *
 * This function must be called by the consumer of the vision results to prevent
 * memory leaks. It frees the main result struct, its arrays of objects and
 * text blocks, and the depth map data.
 *
 * @param[in,out] result A pointer to the tk_vision_result_t* to be destroyed.
 *                       The pointer is set to NULL after destruction.
 */
void tk_vision_result_destroy(tk_vision_result_t** result);

#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_VISION_TK_VISION_PIPELINE_H
