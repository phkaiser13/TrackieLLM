/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_cuda_image_ops.h
 *
 * This header file defines the public API for high-level CUDA-accelerated image
 * processing operations used in the TrackieLLM vision pipeline. These functions
 * provide optimized implementations for common computer vision tasks such as
 * image filtering, edge detection, morphological operations, and color space
 * conversions that are essential for preprocessing and feature extraction.
 *
 * The design philosophy emphasizes:
 *   1. Performance: Operations are implemented using optimized CUDA kernels
 *      with memory coalescing and shared memory utilization.
 *   2. Flexibility: Functions support various image formats (grayscale, RGB,
 *      RGBA) and data types (uint8, float32) through template-like parameterization.
 *   3. Integration: Seamless integration with the existing CUDA dispatcher and
 *      buffer management system for zero-copy operations.
 *   4. Robustness: Comprehensive error handling and input validation to prevent
 *      GPU memory corruption and ensure system stability.
 *
 * These operations are specifically optimized for real-time processing of
 * video streams from mobile and embedded devices with limited computational
 * resources.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#ifndef TRACKIELLM_GPU_EXTENSIONS_CUDA_TK_CUDA_IMAGE_OPS_H
#define TRACKIELLM_GPU_EXTENSIONS_CUDA_TK_CUDA_IMAGE_OPS_H

#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "gpu/cuda/tk_cuda_dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Image Format Definitions
//------------------------------------------------------------------------------

/**
 * @enum tk_image_format_t
 * @brief Enumeration of supported image formats.
 */
typedef enum {
    TK_IMAGE_FORMAT_GRAYSCALE_U8 = 0,  /**< Grayscale 8-bit unsigned integer */
    TK_IMAGE_FORMAT_RGB_U8,            /**< RGB 8-bit unsigned integer */
    TK_IMAGE_FORMAT_RGBA_U8,           /**< RGBA 8-bit unsigned integer */
    TK_IMAGE_FORMAT_GRAYSCALE_F32,     /**< Grayscale 32-bit floating point */
    TK_IMAGE_FORMAT_RGB_F32,           /**< RGB 32-bit floating point */
    TK_IMAGE_FORMAT_RGBA_F32           /**< RGBA 32-bit floating point */
} tk_image_format_t;

/**
 * @struct tk_image_descriptor_t
 * @brief Descriptor for image properties and GPU buffer handle.
 */
typedef struct {
    tk_gpu_buffer_t buffer;      /**< GPU buffer handle containing image data */
    uint32_t width;              /**< Image width in pixels */
    uint32_t height;             /**< Image height in pixels */
    tk_image_format_t format;    /**< Image format enumeration */
    size_t stride_bytes;         /**< Row stride in bytes (may include padding) */
    size_t data_size_bytes;      /**< Total size of image data in bytes */
} tk_image_descriptor_t;

//------------------------------------------------------------------------------
// Convolution and Filtering Operations
//------------------------------------------------------------------------------

/**
 * @struct tk_convolution_params_t
 * @brief Parameters for convolution operations.
 */
typedef struct {
    const float* kernel;         /**< Pointer to convolution kernel coefficients */
    uint32_t kernel_width;       /**< Width of the convolution kernel */
    uint32_t kernel_height;      /**< Height of the convolution kernel */
    float divisor;               /**< Normalization divisor for kernel coefficients */
    float offset;                /**< Constant offset added to result */
    uint32_t iterations;         /**< Number of times to apply the filter */
} tk_convolution_params_t;

/**
 * @brief Apply a separable convolution filter to an image.
 *
 * This function applies a separable convolution filter which is computationally
 * more efficient than a full 2D convolution. It's commonly used for Gaussian
 * blur and other separable filters.
 *
 * @param[in] dispatcher The CUDA dispatcher instance.
 * @param[in] input Image descriptor for the input image.
 * @param[out] output Image descriptor for the output image.
 * @param[in] params Convolution parameters including kernel and normalization.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_cuda_image_separable_convolution(
    tk_cuda_dispatcher_t* dispatcher,
    const tk_image_descriptor_t* input,
    tk_image_descriptor_t* output,
    const tk_convolution_params_t* params
);

/**
 * @brief Apply a Sobel edge detection filter to an image.
 *
 * This function computes gradient magnitude using Sobel operators in X and Y
 * directions, producing an edge-enhanced output image.
 *
 * @param[in] dispatcher The CUDA dispatcher instance.
 * @param[in] input Image descriptor for the input grayscale image.
 * @param[out] output Image descriptor for the output edge map.
 * @param[in] threshold Minimum gradient magnitude to consider as edge.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_cuda_image_sobel_edge_detection(
    tk_cuda_dispatcher_t* dispatcher,
    const tk_image_descriptor_t* input,
    tk_image_descriptor_t* output,
    float threshold
);

/**
 * @brief Apply a bilateral filter for noise reduction while preserving edges.
 *
 * This advanced filter reduces noise while maintaining important edge information
 * by considering both spatial proximity and intensity similarity.
 *
 * @param[in] dispatcher The CUDA dispatcher instance.
 * @param[in] input Image descriptor for the input image.
 * @param[out] output Image descriptor for the output filtered image.
 * @param[in] spatial_sigma Standard deviation for spatial kernel.
 * @param[in] intensity_sigma Standard deviation for intensity kernel.
 * @param[in] kernel_radius Radius of the bilateral filter kernel.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_cuda_image_bilateral_filter(
    tk_cuda_dispatcher_t* dispatcher,
    const tk_image_descriptor_t* input,
    tk_image_descriptor_t* output,
    float spatial_sigma,
    float intensity_sigma,
    uint32_t kernel_radius
);

//------------------------------------------------------------------------------
// Morphological Operations
//------------------------------------------------------------------------------

/**
 * @enum tk_morphology_operation_t
 * @brief Enumeration of morphological operations.
 */
typedef enum {
    TK_MORPHOLOGY_ERODE = 0,     /**< Erosion operation */
    TK_MORPHOLOGY_DILATE,        /**< Dilation operation */
    TK_MORPHOLOGY_OPEN,          /**< Opening (erode then dilate) */
    TK_MORPHOLOGY_CLOSE          /**< Closing (dilate then erode) */
} tk_morphology_operation_t;

/**
 * @struct tk_morphology_params_t
 * @brief Parameters for morphological operations.
 */
typedef struct {
    tk_morphology_operation_t operation;  /**< Type of morphological operation */
    uint32_t kernel_width;                /**< Width of structuring element */
    uint32_t kernel_height;               /**< Height of structuring element */
    uint32_t iterations;                  /**< Number of times to apply operation */
} tk_morphology_params_t;

/**
 * @brief Apply morphological operation to a binary or grayscale image.
 *
 * This function performs morphological operations using configurable structuring
 * elements for tasks such as noise removal, gap filling, and shape analysis.
 *
 * @param[in] dispatcher The CUDA dispatcher instance.
 * @param[in] input Image descriptor for the input image.
 * @param[out] output Image descriptor for the output image.
 * @param[in] params Morphological operation parameters.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_cuda_image_morphology_operation(
    tk_cuda_dispatcher_t* dispatcher,
    const tk_image_descriptor_t* input,
    tk_image_descriptor_t* output,
    const tk_morphology_params_t* params
);

//------------------------------------------------------------------------------
// Color Space Conversions
//------------------------------------------------------------------------------

/**
 * @enum tk_color_space_t
 * @brief Enumeration of supported color spaces.
 */
typedef enum {
    TK_COLOR_SPACE_RGB = 0,      /**< Red-Green-Blue color space */
    TK_COLOR_SPACE_HSV,          /**< Hue-Saturation-Value color space */
    TK_COLOR_SPACE_LAB,          /**< CIELAB color space */
    TK_COLOR_SPACE_YUV           /**< Luminance-Chrominance color space */
} tk_color_space_t;

/**
 * @brief Convert image from one color space to another.
 *
 * This function performs color space conversion using optimized GPU kernels
 * that leverage parallel processing for real-time performance.
 *
 * @param[in] dispatcher The CUDA dispatcher instance.
 * @param[in] input Image descriptor for the input image.
 * @param[out] output Image descriptor for the output image.
 * @param[in] target_space Target color space for conversion.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_cuda_image_color_space_conversion(
    tk_cuda_dispatcher_t* dispatcher,
    const tk_image_descriptor_t* input,
    tk_image_descriptor_t* output,
    tk_color_space_t target_space
);

//------------------------------------------------------------------------------
// Histogram Operations
//------------------------------------------------------------------------------

/**
 * @struct tk_histogram_params_t
 * @brief Parameters for histogram operations.
 */
typedef struct {
    uint32_t num_bins;           /**< Number of histogram bins */
    float min_value;             /**< Minimum value for histogram range */
    float max_value;             /**< Maximum value for histogram range */
    int compute_cumulative;      /**< Flag to compute cumulative histogram */
} tk_histogram_params_t;

/**
 * @brief Compute histogram of image intensities.
 *
 * This function computes the intensity histogram of an image using atomic
 * operations for thread-safe bin updates. Supports both regular and
 * cumulative histograms.
 *
 * @param[in] dispatcher The CUDA dispatcher instance.
 * @param[in] input Image descriptor for the input image.
 * @param[out] histogram_output GPU buffer to store histogram results.
 * @param[in] params Histogram computation parameters.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_cuda_image_compute_histogram(
    tk_cuda_dispatcher_t* dispatcher,
    const tk_image_descriptor_t* input,
    tk_gpu_buffer_t histogram_output,
    const tk_histogram_params_t* params
);

/**
 * @brief Apply histogram equalization to enhance image contrast.
 *
 * This function enhances image contrast by redistributing intensity values
 * according to the cumulative distribution function of the histogram.
 *
 * @param[in] dispatcher The CUDA dispatcher instance.
 * @param[in] input Image descriptor for the input image.
 * @param[out] output Image descriptor for the output equalized image.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_cuda_image_histogram_equalization(
    tk_cuda_dispatcher_t* dispatcher,
    const tk_image_descriptor_t* input,
    tk_image_descriptor_t* output
);

//------------------------------------------------------------------------------
// Geometric Transformations
//------------------------------------------------------------------------------

/**
 * @struct tk_geometric_transform_t
 * @brief Parameters for geometric transformations.
 */
typedef struct {
    float rotation_angle;        /**< Rotation angle in degrees */
    float scale_x;               /**< Scaling factor in X direction */
    float scale_y;               /**< Scaling factor in Y direction */
    float translate_x;           /**< Translation in X direction (pixels) */
    float translate_y;           /**< Translation in Y direction (pixels) */
    int interpolation_method;    /**< Interpolation method (0=nearest, 1=bilinear, 2=bicubic) */
} tk_geometric_transform_t;

/**
 * @brief Apply geometric transformation to an image.
 *
 * This function applies affine transformations including rotation, scaling,
 * and translation with configurable interpolation methods for quality control.
 *
 * @param[in] dispatcher The CUDA dispatcher instance.
 * @param[in] input Image descriptor for the input image.
 * @param[out] output Image descriptor for the output transformed image.
 * @param[in] transform Geometric transformation parameters.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_cuda_image_geometric_transform(
    tk_cuda_dispatcher_t* dispatcher,
    const tk_image_descriptor_t* input,
    tk_image_descriptor_t* output,
    const tk_geometric_transform_t* transform
);

//------------------------------------------------------------------------------
// Feature Detection
//------------------------------------------------------------------------------

/**
 * @struct tk_feature_detection_params_t
 * @brief Parameters for feature detection algorithms.
 */
typedef struct {
    float threshold;             /**< Detection threshold for feature response */
    uint32_t max_features;       /**< Maximum number of features to detect */
    int non_max_suppression;     /**< Enable non-maximum suppression */
    float min_distance;          /**< Minimum distance between features */
} tk_feature_detection_params_t;

/**
 * @brief Detect Harris corners in an image.
 *
 * This function implements the Harris corner detection algorithm using GPU
 * acceleration for real-time feature point extraction.
 *
 * @param[in] dispatcher The CUDA dispatcher instance.
 * @param[in] input Image descriptor for the input grayscale image.
 * @param[out] features_output GPU buffer to store detected feature coordinates.
 * @param[out] num_features_output Pointer to store number of detected features.
 * @param[in] params Feature detection parameters.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_cuda_image_harris_corner_detection(
    tk_cuda_dispatcher_t* dispatcher,
    const tk_image_descriptor_t* input,
    tk_gpu_buffer_t features_output,
    uint32_t* num_features_output,
    const tk_feature_detection_params_t* params
);

/**
 * @brief Detect FAST features in an image.
 *
 * This function implements the FAST (Features from Accelerated Segment Test)
 * corner detection algorithm optimized for GPU execution.
 *
 * @param[in] dispatcher The CUDA dispatcher instance.
 * @param[in] input Image descriptor for the input grayscale image.
 * @param[out] features_output GPU buffer to store detected feature coordinates.
 * @param[out] num_features_output Pointer to store number of detected features.
 * @param[in] params Feature detection parameters.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_cuda_image_fast_feature_detection(
    tk_cuda_dispatcher_t* dispatcher,
    const tk_image_descriptor_t* input,
    tk_gpu_buffer_t features_output,
    uint32_t* num_features_output,
    const tk_feature_detection_params_t* params
);

#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_GPU_EXTENSIONS_CUDA_TK_CUDA_IMAGE_OPS_H
