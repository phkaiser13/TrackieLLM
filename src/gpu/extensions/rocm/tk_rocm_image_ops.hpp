/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_rocm_image_ops.hpp
 *
 * This header file defines the public API for a comprehensive suite of HIP-accelerated
 * image processing operations. These functions are highly optimized for the ROCm
 * platform (AMD GPUs) and provide the building blocks for advanced computer vision
 * pipelines, including filtering, feature extraction, and geometric transformations.
 *
 * The design philosophy is rooted in modern GPGPU engineering practices:
 *   1.  **High-Performance Kernels**: Each operation is backed by a meticulously
 *       optimized HIP kernel that leverages architectural features of AMD GPUs,
 *       such as efficient use of the LDS (Local Data Store) for shared memory tasks.
 *   2.  **Type and Format Agnostic**: The API is designed to be flexible, supporting
 *       various image formats (grayscale, RGB, RGBA) and data types through a
 *       unified descriptor-based approach (`tk_image_descriptor_t`).
 *   3.  **Seamless Integration**: The functions integrate directly with the ROCm
 *       dispatcher, using its stream and buffer management systems to ensure
 *       asynchronous execution and zero-copy data flow where possible.
 *   4.  **Robustness and Safety**: All functions perform extensive input validation
 *       and error handling to guarantee system stability and provide clear diagnostics.
 *
 * These operations are engineered for demanding real-time applications.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#ifndef TRACKIELLM_GPU_EXTENSIONS_ROCM_TK_ROCM_IMAGE_OPS_HPP
#define TRACKIELLM_GPU_EXTENSIONS_ROCM_TK_ROCM_IMAGE_OPS_HPP

#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "gpu/rocm/tk_rocm_dispatch.hpp" // For tk_rocm_dispatcher_t and tk_gpu_buffer_t

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Image Format and Descriptor Definitions
//------------------------------------------------------------------------------

/**
 * @enum tk_image_format_t
 * @brief Enumeration of supported image formats for processing.
 */
typedef enum {
    TK_IMAGE_FORMAT_GRAYSCALE_U8 = 0,  /**< Single-channel, 8-bit unsigned integer. */
    TK_IMAGE_FORMAT_RGB_U8,            /**< 3-channel, 8-bit unsigned integer, interleaved. */
    TK_IMAGE_FORMAT_RGBA_U8,           /**< 4-channel, 8-bit unsigned integer, interleaved. */
    TK_IMAGE_FORMAT_GRAYSCALE_F32,     /**< Single-channel, 32-bit floating point. */
    TK_IMAGE_FORMAT_RGB_F32,           /**< 3-channel, 32-bit floating point, interleaved. */
    TK_IMAGE_FORMAT_RGBA_F32           /**< 4-channel, 32-bit floating point, interleaved. */
} tk_image_format_t;

/**
 * @struct tk_image_descriptor_t
 * @brief A comprehensive descriptor for an image stored in GPU memory.
 *        This struct provides all the necessary metadata for a kernel to
 *        correctly interpret the image data.
 */
typedef struct {
    tk_gpu_buffer_t buffer;      /**< Opaque handle to the GPU buffer containing the image data. */
    uint32_t width;              /**< Image width in pixels. */
    uint32_t height;             /**< Image height in pixels. */
    tk_image_format_t format;    /**< The format of the pixel data. */
    size_t stride_bytes;         /**< Row stride in bytes, accounting for potential padding for memory alignment. */
    size_t data_size_bytes;      /**< Total size of the image data in bytes. */
} tk_image_descriptor_t;

//------------------------------------------------------------------------------
// Convolution and Filtering Operations
//------------------------------------------------------------------------------

/**
 * @struct tk_convolution_params_t
 * @brief Parameters for custom 2D convolution operations.
 */
typedef struct {
    tk_gpu_buffer_t kernel_buffer; /**< GPU buffer containing the convolution kernel coefficients. */
    uint32_t kernel_width;         /**< Width of the convolution kernel. */
    uint32_t kernel_height;        /**< Height of the convolution kernel. */
    float divisor;                 /**< Normalization factor to divide the result by. */
    float offset;                  /**< A constant offset to add to the result. */
} tk_convolution_params_t;

/**
 * @brief Applies a general 2D convolution to an image using a specified kernel.
 *
 * This function is highly optimized, using shared memory (LDS) to cache parts of
 * the input image, drastically reducing global memory reads.
 *
 * @param[in] dispatcher The ROCm dispatcher instance.
 * @param[in] input      A descriptor for the input image.
 * @param[out] output    A descriptor for the output image.
 * @param[in] params     Parameters defining the convolution kernel and its properties.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_rocm_image_convolution_2d(
    tk_rocm_dispatcher_t* dispatcher,
    const tk_image_descriptor_t* input,
    tk_image_descriptor_t* output,
    const tk_convolution_params_t* params
);

/**
 * @brief Applies a Sobel edge detection filter to a grayscale image.
 *
 * This function computes the gradient magnitude at each pixel using the 3x3 Sobel
 * operators, producing a powerful edge-enhanced representation of the image.
 *
 * @param[in] dispatcher The ROCm dispatcher instance.
 * @param[in] input      A descriptor for the input grayscale image (must be F32 format).
 * @param[out] output    A descriptor for the output edge map (F32 format).
 * @param[in] threshold  A minimum gradient magnitude to be considered an edge. Values below are set to 0.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_rocm_image_sobel_edge_detection(
    tk_rocm_dispatcher_t* dispatcher,
    const tk_image_descriptor_t* input,
    tk_image_descriptor_t* output,
    float threshold
);

//------------------------------------------------------------------------------
// Morphological Operations
//------------------------------------------------------------------------------

/**
 * @enum tk_morphology_op_t
 * @brief Enumeration of fundamental morphological operations.
 */
typedef enum {
    TK_MORPHOLOGY_OP_ERODE = 0,     /**< Erosion: shrinks bright regions. */
    TK_MORPHOLOGY_OP_DILATE,        /**< Dilation: expands bright regions. */
} tk_morphology_op_t;

/**
 * @struct tk_morphology_params_t
 * @brief Parameters for morphological operations.
 */
typedef struct {
    tk_morphology_op_t operation;   /**< The type of morphological operation to perform. */
    uint32_t kernel_width;          /**< Width of the structuring element (e.g., 3 for a 3x3 kernel). */
    uint32_t kernel_height;         /**< Height of the structuring element. */
    uint32_t iterations;            /**< The number of times to apply the operation. */
} tk_morphology_params_t;

/**
 * @brief Applies a morphological operation (Erode or Dilate) to an image.
 *
 * These operations are fundamental in image processing for noise removal,
 * object separation, and feature detection.
 *
 * @param[in] dispatcher The ROCm dispatcher instance.
 * @param[in] input      A descriptor for the input image.
 * @param[out] output    A descriptor for the output image.
 * @param[in] params     Parameters defining the operation and structuring element.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_rocm_image_morphology_operation(
    tk_rocm_dispatcher_t* dispatcher,
    const tk_image_descriptor_t* input,
    tk_image_descriptor_t* output,
    const tk_morphology_params_t* params
);

//------------------------------------------------------------------------------
// Color Space Conversions
//------------------------------------------------------------------------------

/**
 * @enum tk_color_conversion_t
 * @brief Enumeration of supported color space conversions.
 */
typedef enum {
    TK_COLOR_CONV_RGB_TO_GRAY = 0,
    TK_COLOR_CONV_RGB_TO_HSV,
    TK_COLOR_CONV_HSV_TO_RGB,
    TK_COLOR_CONV_RGB_TO_YUV,
    TK_COLOR_CONV_YUV_TO_RGB,
} tk_color_conversion_t;

/**
 * @brief Converts an image from one color space to another.
 *
 * @param[in] dispatcher  The ROCm dispatcher instance.
 * @param[in] input       A descriptor for the input image.
 * @param[out] output     A descriptor for the output image.
 * @param[in] conversion  The specific color space conversion to perform.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_rocm_image_color_space_conversion(
    tk_rocm_dispatcher_t* dispatcher,
    const tk_image_descriptor_t* input,
    tk_image_descriptor_t* output,
    tk_color_conversion_t conversion
);

//------------------------------------------------------------------------------
// Histogram Operations
//------------------------------------------------------------------------------

/**
 * @brief Computes the intensity histogram of a grayscale image.
 *
 * This function uses an efficient, two-stage approach. A first kernel calculates
 * partial histograms in shared memory, which are then atomically aggregated into
 * a global histogram buffer. This minimizes atomic contention on global memory.
 *
 * @param[in] dispatcher        The ROCm dispatcher instance.
 * @param[in] input             A descriptor for the input grayscale image.
 * @param[out] histogram_buffer A GPU buffer to store the resulting histogram (must be of size 256 * sizeof(uint32_t)).
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_rocm_image_compute_histogram(
    tk_rocm_dispatcher_t* dispatcher,
    const tk_image_descriptor_t* input,
    tk_gpu_buffer_t histogram_buffer
);

/**
 * @brief Applies histogram equalization to enhance the contrast of a grayscale image.
 *
 * This function first computes the histogram, then calculates the cumulative
 * distribution function (CDF), and finally remaps the image intensities based on the CDF.
 *
 * @param[in] dispatcher The ROCm dispatcher instance.
 * @param[in] input      A descriptor for the input grayscale image.
 * @param[out] output    A descriptor for the output contrast-enhanced image.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_rocm_image_histogram_equalization(
    tk_rocm_dispatcher_t* dispatcher,
    const tk_image_descriptor_t* input,
    tk_image_descriptor_t* output
);

#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_GPU_EXTENSIONS_ROCM_TK_ROCM_IMAGE_OPS_HPP
