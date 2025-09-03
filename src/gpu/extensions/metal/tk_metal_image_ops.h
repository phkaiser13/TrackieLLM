/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_metal_image_ops.h
 *
 * Public C‑style API for GPU‑accelerated image processing on Apple‑metal
 * (Metal backend).  The functions declared here are thin wrappers that forward
 * the heavy lifting to Metal compute shaders via the Metal dispatcher
 * (tk_metal_dispatch).  All data structures are deliberately layout‑compatible
 * with the CUDA and ROCm back‑ends so that the higher‑level vision pipeline can
 * remain completely agnostic of the underlying hardware.
 *
 * Dependencies:
 *   - tk_gpu.h                (opaque GPU buffer handle)
 *   - tk_image.h              (image descriptor & format enums)
 *   - tk_metal_dispatch.h     (dispatcher API used by the implementation)
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#ifndef TK_METAL_IMAGE_OPS_H_
#define TK_METAL_IMAGE_OPS_H_

#include <stddef.h>
#include <stdint.h>
#include "tk_gpu.h"          /* tk_gpu_buffer_t */
#include "tk_image.h"        /* tk_image_format_t, tk_image_descriptor_t */
#include "tk_metal_dispatch.h"

/* -------------------------------------------------------------------------
 *  Status codes returned by every public function.
 * ------------------------------------------------------------------------- */
typedef enum {
    TK_STATUS_OK = 0,               /**< Operation completed successfully. */
    TK_STATUS_INVALID_ARGUMENT,     /**< One or more arguments are malformed. */
    TK_STATUS_ALLOCATION_FAILURE,   /**< GPU buffer allocation failed. */
    TK_STATUS_DISPATCH_FAILURE,     /**< Failed to enqueue a compute command. */
    TK_STATUS_SHADER_NOT_FOUND,     /**< Requested Metal kernel could not be loaded. */
    TK_STATUS_INTERNAL_ERROR        /**< Unexpected condition (assertion, etc.). */
} tk_status_t;

/* -------------------------------------------------------------------------
 *  Convolution parameters – used by separable and 2‑D convolutions.
 * ------------------------------------------------------------------------- */
typedef struct {
    const float *kernel;            /**< Pointer to kernel coefficients in host RAM. */
    uint32_t      kernel_width;    /**< Width of the kernel (in elements). */
    uint32_t      kernel_height;   /**< Height of the kernel (in elements). */
    float         divisor;         /**< Normalisation factor (usually sum of kernel). */
    float         offset;          /**< Constant added to each output pixel. */
} tk_convolution_params_t;

/* -------------------------------------------------------------------------
 *  Morphology operation type.
 * ------------------------------------------------------------------------- */
typedef enum {
    TK_MORPHOLOGY_ERODE = 0,
    TK_MORPHOLOGY_DILATE,
    TK_MORPHOLOGY_OPEN,
    TK_MORPHOLOGY_CLOSE
} tk_morphology_op_t;

/* -------------------------------------------------------------------------
 *  Morphology parameters – a simple struct for now (future extensions may
 *  include structuring‑element shape, radius, etc.).
 * ------------------------------------------------------------------------- */
typedef struct {
    tk_morphology_op_t op;          /**< Erode, Dilate, … */
    uint32_t           radius;      /**< Radius of the structuring element. */
} tk_morphology_params_t;

/* -------------------------------------------------------------------------
 *  Color‑space conversion direction.
 * ------------------------------------------------------------------------- */
typedef enum {
    TK_COLOR_CONVERT_RGB_TO_GRAYSCALE = 0,
    TK_COLOR_CONVERT_RGB_TO_HSV,
    TK_COLOR_CONVERT_GRAYSCALE_TO_RGB
} tk_color_conversion_t;

/* -------------------------------------------------------------------------
 *  Geometric transform – only affine transforms are required for the current
 *  pipeline.  The matrix is stored in row‑major order.
 * ------------------------------------------------------------------------- */
typedef struct {
    float matrix[3][3];             /**< 3×3 affine matrix (last row = {0,0,1}). */
    int   interpolation;            /**< 0 = nearest, 1 = bilinear, … */
    int   border_mode;              /**< 0 = clamp, 1 = repeat, 2 = constant. */
} tk_geometric_transform_t;

/* -------------------------------------------------------------------------
 *  Public API – each function returns a tk_status_t and never throws
 *  exceptions.  All pointers must point to valid objects that live at least
 *  until the function returns.
 * ------------------------------------------------------------------------- */

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 *  Separable convolution (horizontal + vertical).  The source image is read,
 *  the destination image is written.  Both images must have the same format,
 *  width and height.  The kernel coefficients are supplied in host memory.
 * ------------------------------------------------------------------------- */
tk_status_t tk_metal_image_separable_convolution(
    const tk_image_descriptor_t *src,
    const tk_image_descriptor_t *dst,
    const tk_convolution_params_t *horiz,
    const tk_convolution_params_t *vert);

/* -------------------------------------------------------------------------
 *  Sobel edge detection – produces a single‑channel gradient magnitude
 *  image (format must be GRAYSCALE_F32).  The source must be a single‑channel
 *  image; the destination must be allocated with the same dimensions.
 * ------------------------------------------------------------------------- */
tk_status_t tk_metal_image_sobel_edge_detection(
    const tk_image_descriptor_t *src,
    const tk_image_descriptor_t *dst);

/* -------------------------------------------------------------------------
 *  Bilateral filter – edge‑preserving smoothing.  The sigma values are
 *  encoded in the `kernel` field of the supplied tk_convolution_params_t:
 *  kernel[0] = spatial sigma, kernel[1] = intensity sigma.
 * ------------------------------------------------------------------------- */
tk_status_t tk_metal_image_bilateral_filter(
    const tk_image_descriptor_t *src,
    const tk_image_descriptor_t *dst,
    const tk_convolution_params_t *params);

/* -------------------------------------------------------------------------
 *  Morphology operation – erode, dilate, open or close.
 * ------------------------------------------------------------------------- */
tk_status_t tk_metal_image_morphology_operation(
    const tk_image_descriptor_t *src,
    const tk_image_descriptor_t *dst,
    const tk_morphology_params_t *params);

/* -------------------------------------------------------------------------
 *  Color‑space conversion – e.g. RGB → Grayscale.
 * ------------------------------------------------------------------------- */
tk_status_t tk_metal_image_color_space_conversion(
    const tk_image_descriptor_t *src,
    const tk_image_descriptor_t *dst,
    tk_color_conversion_t conversion);

/* -------------------------------------------------------------------------
 *  Compute histogram – the destination buffer must be a GPU buffer large enough
 *  to hold `num_bins * sizeof(uint32_t)` bytes.  The function writes the
 *  histogram in host‑order (little‑endian) regardless of the GPU’s native
 *  byte order.
 * ------------------------------------------------------------------------- */
tk_status_t tk_metal_image_compute_histogram(
    const tk_image_descriptor_t *src,
    tk_gpu_buffer_t histogram_buffer,
    uint32_t num_bins);

/* -------------------------------------------------------------------------
 *  Histogram equalization – reads a pre‑computed histogram (as produced by the
 *  previous function) and writes the equalised image to `dst`.
 * ------------------------------------------------------------------------- */
tk_status_t tk_metal_image_histogram_equalization(
    const tk_image_descriptor_t *src,
    const tk_image_descriptor_t *dst,
    tk_gpu_buffer_t histogram_buffer,
    uint32_t num_bins);

/* -------------------------------------------------------------------------
 *  Geometric transform – applies an affine matrix to the source image.
 * ------------------------------------------------------------------------- */
tk_status_t tk_metal_image_geometric_transform(
    const tk_image_descriptor_t *src,
    const tk_image_descriptor_t *dst,
    const tk_geometric_transform_t *transform);

/* -------------------------------------------------------------------------
 *  Harris corner detection – writes a single‑channel response map (GRAYSCALE_F32)
 *  to `dst`.  The `params` structure contains the usual Harris constants.
 * ------------------------------------------------------------------------- */
typedef struct {
    float k;            /**< Harris free‑parameter (typically 0.04–0.06). */
    float threshold;   /**< Minimum response to be considered a corner. */
    uint32_t block_size; /**< Neighborhood size for gradient covariance. */
} tk_harris_params_t;

tk_status_t tk_metal_image_harris_corner_detection(
    const tk_image_descriptor_t *src,
    const tk_image_descriptor_t *dst,
    const tk_harris_params_t *params);

#ifdef __cplusplus
}
#endif

#endif /* TK_METAL_IMAGE_OPS_H_ */
