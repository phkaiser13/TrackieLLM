/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_metal_tensor_ops.h
 *
 * Public C‑style API for GPU‑accelerated tensor primitives on Apple‑metal
 * (Metal backend).  The functions declared here are thin wrappers that forward
 * the work to Metal compute kernels (or to Metal Performance Shaders where
 * appropriate) via the Metal dispatcher (`tk_metal_dispatch`).  All data
 * structures are layout‑compatible with the CUDA and ROCm back‑ends so that the
 * higher‑level AI stack can remain completely agnostic of the underlying
 * hardware.
 *
 * Dependencies:
 *   - tk_gpu.h                (opaque GPU buffer handle)
 *   - tk_tensor.h             (tensor descriptor & data‑type enum)
 *   - tk_status.h             (tk_status_t – unified error codes)
 *   - tk_metal_dispatch.h     (dispatcher API used by the implementation)
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#ifndef TK_METAL_TENSOR_OPS_H_
#define TK_METAL_TENSOR_OPS_H_

/* -------------------------------------------------------------------------
 *  Standard includes – only the minimal set required for the public API.
 * ------------------------------------------------------------------------- */
#include <stddef.h>
#include <stdint.h>

#include "tk_gpu.h"          /* tk_gpu_buffer_t */
#include "tk_tensor.h"       /* tk_tensor_descriptor_t, tk_data_type_t */
#include "tk_status.h"       /* tk_status_t */
#include "tk_metal_dispatch.h"

/* -------------------------------------------------------------------------
 *  Forward declarations for structures that are defined in other headers.
 * ------------------------------------------------------------------------- */
typedef struct tk_tensor_descriptor_t tk_tensor_descriptor_t;

/* -------------------------------------------------------------------------
 *  Tensor‑wise scalar parameters – used by the “*_scalar” family of
 *  functions.  Keeping the struct separate makes the API extensible without
 *  breaking binary compatibility.
 * ------------------------------------------------------------------------- */
typedef struct {
    float value;               /**< Scalar value (float32). */
} tk_scalar_f32_t;

/* -------------------------------------------------------------------------
 *  Convolution‑style parameters for pooling – kernel size and stride are
 *  expressed in pixels (or elements) and must be > 0.
 * ------------------------------------------------------------------------- */
typedef struct {
    uint32_t kernel_width;     /**< Width of the pooling window. */
    uint32_t kernel_height;    /**< Height of the pooling window. */
    uint32_t stride_x;         /**< Horizontal stride. */
    uint32_t stride_y;         /**< Vertical stride. */
    uint32_t padding_x;        /**< Horizontal padding (0 = no padding). */
    uint32_t padding_y;        /**< Vertical padding (0 = no padding). */
} tk_pooling_params_t;

/* -------------------------------------------------------------------------
 *  Layer‑normalization parameters – epsilon prevents division by zero.
 * ------------------------------------------------------------------------- */
typedef struct {
    float epsilon;             /**< Small constant added to variance. */
} tk_layer_norm_params_t;

/* -------------------------------------------------------------------------
 *  Concatenation parameters – the axis along which tensors are concatenated.
 * ------------------------------------------------------------------------- */
typedef struct {
    uint32_t axis;             /**< Axis index (0 = batch, 1 = channel, …). */
} tk_concat_params_t;

/* -------------------------------------------------------------------------
 *  Public API – all functions return a tk_status_t and never throw C++
 *  exceptions.  The caller must guarantee that the supplied descriptors
 *  remain valid for the duration of the call.
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 *  Element‑wise arithmetic (float32 tensors only).  The source and destination
 *  tensors must have identical shapes, data type and element count.
 * ------------------------------------------------------------------------- */
tk_status_t tk_metal_tensor_add(const tk_tensor_descriptor_t *a,
                                const tk_tensor_descriptor_t *b,
                                const tk_tensor_descriptor_t *out);

tk_status_t tk_metal_tensor_subtract(const tk_tensor_descriptor_t *a,
                                     const tk_tensor_descriptor_t *b,
                                     const tk_tensor_descriptor_t *out);

tk_status_t tk_metal_tensor_multiply(const tk_tensor_descriptor_t *a,
                                     const tk_tensor_descriptor_t *b,
                                     const tk_tensor_descriptor_t *out);

tk_status_t tk_metal_tensor_divide(const tk_tensor_descriptor_t *a,
                                   const tk_tensor_descriptor_t *b,
                                   const tk_tensor_descriptor_t *out);

/* -------------------------------------------------------------------------
 *  Element‑wise scalar operations – the scalar is supplied in a tiny struct
 *  to keep the C ABI stable.
 * ------------------------------------------------------------------------- */
tk_status_t tk_metal_tensor_add_scalar(const tk_tensor_descriptor_t *src,
                                       const tk_scalar_f32_t *scalar,
                                       const tk_tensor_descriptor_t *out);

tk_status_t tk_metal_tensor_mul_scalar(const tk_tensor_descriptor_t *src,
                                       const tk_scalar_f32_t *scalar,
                                       const tk_tensor_descriptor_t *out);

/* -------------------------------------------------------------------------
 *  Matrix multiplication – the most performance‑critical primitive.  The
 *  tensors are interpreted as 2‑D matrices (row‑major).  The dispatcher will
 *  prefer Metal Performance Shaders (MPSMatrixMultiplication) when available.
 *
 *  Dimensions:
 *      A: (M x K)
 *      B: (K x N)
 *      out: (M x N)
 *
 *  The caller must ensure that the descriptors describe buffers of the
 *  appropriate size and that the data type is TK_DATA_TYPE_FLOAT32.
 * ------------------------------------------------------------------------- */
tk_status_t tk_metal_tensor_matmul(const tk_tensor_descriptor_t *a,
                                   const tk_tensor_descriptor_t *b,
                                   const tk_tensor_descriptor_t *out);

/* -------------------------------------------------------------------------
 *  Transpose – swaps the two innermost dimensions of a 2‑D tensor.
 * ------------------------------------------------------------------------- */
tk_status_t tk_metal_tensor_transpose(const tk_tensor_descriptor_t *src,
                                      const tk_tensor_descriptor_t *out);

/* -------------------------------------------------------------------------
 *  Activation functions – operate element‑wise on a single tensor.
 * ------------------------------------------------------------------------- */
tk_status_t tk_metal_tensor_relu(const tk_tensor_descriptor_t *src,
                                 const tk_tensor_descriptor_t *out);

tk_status_t tk_metal_tensor_sigmoid(const tk_tensor_descriptor_t *src,
                                    const tk_tensor_descriptor_t *out);

tk_status_t tk_metal_tensor_tanh(const tk_tensor_descriptor_t *src,
                                 const tk_tensor_descriptor_t *out);

tk_status_t tk_metal_tensor_softmax(const tk_tensor_descriptor_t *src,
                                   const tk_tensor_descriptor_t *out);

/* -------------------------------------------------------------------------
 *  Pooling – max and average pooling for 2‑D feature maps.
 * ------------------------------------------------------------------------- */
tk_status_t tk_metal_tensor_max_pool(const tk_tensor_descriptor_t *src,
                                     const tk_pooling_params_t *params,
                                     const tk_tensor_descriptor_t *out);

tk_status_t tk_metal_tensor_avg_pool(const tk_tensor_descriptor_t *src,
                                     const tk_pooling_params_t *params,
                                     const tk_tensor_descriptor_t *out);

/* -------------------------------------------------------------------------
 *  Layer Normalization – normalizes across the channel dimension.
 * ------------------------------------------------------------------------- */
tk_status_t tk_metal_tensor_layer_norm(const tk_tensor_descriptor_t *src,
                                       const tk_layer_norm_params_t *params,
                                       const tk_tensor_descriptor_t *out);

/* -------------------------------------------------------------------------
 *  Reshape – changes the logical dimensions without moving data.  The total
 *  element count must stay identical.
 * ------------------------------------------------------------------------- */
tk_status_t tk_metal_tensor_reshape(const tk_tensor_descriptor_t *src,
                                    const tk_tensor_descriptor_t *out);

/* -------------------------------------------------------------------------
 *  Concatenation – joins an array of tensors along a given axis.  The caller
 *  supplies a pointer to an array of descriptors and the count of tensors.
 * ------------------------------------------------------------------------- */
tk_status_t tk_metal_tensor_concat(const tk_tensor_descriptor_t * const *tensors,
                                   uint32_t tensor_count,
                                   const tk_concat_params_t *params,
                                   const tk_tensor_descriptor_t *out);

/* -------------------------------------------------------------------------
 *  End of public C interface.
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
}
#endif

#endif /* TK_METAL_TENSOR_OPS_H_ */
