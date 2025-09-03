/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_cuda_math_helpers.h
 *
 * This file contains device-side mathematical helper functions used across CUDA kernels
 * in the TrackieLLM vision processing pipeline. These functions are optimized for GPU
 * execution and provide common operations like activation functions, clamping, and
 * interpolation that are frequently used in image processing and neural network inference.
 *
 * Dependencies:
 *   - CUDA Runtime API
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#ifndef TK_CUDA_MATH_HELPERS_H
#define TK_CUDA_MATH_HELPERS_H

#ifdef __cplusplus
extern "C" {
#endif

// Include CUDA headers
#include <cuda_runtime.h>
#include <math_constants.h>

// Device function declarations

__device__ __forceinline__ float tk_cuda_sigmoidf(float x);
__device__ __forceinline__ float tk_cuda_clampf(float value, float min_val, float max_val);
__device__ __forceinline__ float tk_cuda_lerpf(float a, float b, float t);
__device__ __forceinline__ int tk_cuda_clampi(int value, int min_val, int max_val);
__device__ __forceinline__ float tk_cuda_relu(float x);
__device__ __forceinline__ float tk_cuda_tanhf_approx(float x);
__device__ __forceinline__ float tk_cuda_normalize_to_range(
    float value, 
    float in_min, 
    float in_max, 
    float out_min, 
    float out_max
);
__device__ __forceinline__ float tk_cuda_fast_sqrt(float x);
__device__ __forceinline__ float tk_cuda_fast_inverse_sqrt(float x);
__device__ __forceinline__ float tk_cuda_absf(float x);
__device__ __forceinline__ int tk_cuda_absi(int x);
__device__ __forceinline__ float tk_cuda_roundf(float x);
__device__ __forceinline__ float tk_cuda_floorf(float x);
__device__ __forceinline__ float tk_cuda_ceilf(float x);
__device__ __forceinline__ float tk_cuda_fmodf(float x, float y);
__device__ __forceinline__ float tk_cuda_powf(float base, float exponent);
__device__ __forceinline__ float tk_cuda_expf(float x);
__device__ __forceinline__ float tk_cuda_logf(float x);
__device__ __forceinline__ float tk_cuda_degrees_to_radians(float degrees);
__device__ __forceinline__ float tk_cuda_radians_to_degrees(float radians);
__device__ __forceinline__ float tk_cuda_smoothstep(float edge0, float edge1, float x);
__device__ __forceinline__ float tk_cuda_luminance_from_rgb(float r, float g, float b);
__device__ __forceinline__ float tk_cuda_distance_2d(float x1, float y1, float x2, float y2);
__device__ __forceinline__ float tk_cuda_distance_3d(float x1, float y1, float z1, float x2, float y2, float z2);
__device__ __forceinline__ float tk_cuda_dot_product_2d(float x1, float y1, float x2, float y2);
__device__ __forceinline__ float tk_cuda_dot_product_3d(float x1, float y1, float z1, float x2, float y2, float z2);
__device__ __forceinline__ float tk_cuda_magnitude_2d(float x, float y);
__device__ __forceinline__ float tk_cuda_magnitude_3d(float x, float y, float z);
__device__ __forceinline__ void tk_cuda_normalize_vector_2d(float* x, float* y);
__device__ __forceinline__ void tk_cuda_normalize_vector_3d(float* x, float* y, float* z);
__device__ __forceinline__ float tk_cuda_cross_product_2d(float x1, float y1, float x2, float y2);
__device__ __forceinline__ float tk_cuda_angle_between_vectors_2d(float x1, float y1, float x2, float y2);
__device__ __forceinline__ float tk_cuda_wrap_angle(float angle);
__device__ __forceinline__ float tk_cuda_interpolate_bilinear(
    float x, float y,
    const float* data,
    int width, int height
);
__device__ __forceinline__ float tk_cuda_gaussian_weight(float distance, float sigma);
__device__ __forceinline__ float tk_cuda_softplus(float x);
__device__ __forceinline__ float tk_cuda_elu(float x, float alpha);
__device__ __forceinline__ float tk_cuda_swish(float x);
__device__ __forceinline__ float tk_cuda_mish(float x);
__device__ __forceinline__ float tk_cuda_gelu(float x);
__device__ __forceinline__ float tk_cuda_hard_sigmoid(float x);
__device__ __forceinline__ float tk_cuda_hard_tanh(float x);
__device__ __forceinline__ float tk_cuda_leaky_relu(float x, float alpha);
__device__ __forceinline__ float tk_cuda_prelu(float x, float alpha);
__device__ __forceinline__ float tk_cuda_selu(float x);
__device__ __forceinline__ float tk_cuda_threshold(float x, float threshold, float value);
__device__ __forceinline__ float tk_cuda_soft_threshold(float x, float threshold);
__device__ __forceinline__ float tk_cuda_hard_shrink(float x, float lambda);
__device__ __forceinline__ float tk_cuda_soft_shrink(float x, float lambda);
__device__ __forceinline__ float tk_cuda_normalize_to_unit_vector(float* vec, int size);
__device__ __forceinline__ float tk_cuda_cosine_similarity(const float* a, const float* b, int size);
__device__ __forceinline__ float tk_cuda_euclidean_distance(const float* a, const float* b, int size);
__device__ __forceinline__ float tk_cuda_manhattan_distance(const float* a, const float* b, int size);
__device__ __forceinline__ float tk_cuda_chebyshev_distance(const float* a, const float* b, int size);
__device__ __forceinline__ float tk_cuda_minkowski_distance(const float* a, const float* b, int size, float p);
__device__ __forceinline__ void tk_cuda_matrix_multiply_2x2(
    const float a[4], const float b[4], float result[4]
);
__device__ __forceinline__ void tk_cuda_matrix_multiply_3x3(
    const float a[9], const float b[9], float result[9]
);
__device__ __forceinline__ float tk_cuda_determinant_2x2(const float m[4]);
__device__ __forceinline__ float tk_cuda_determinant_3x3(const float m[9]);
__device__ __forceinline__ void tk_cuda_transpose_2x2(const float m[4], float result[4]);
__device__ __forceinline__ void tk_cuda_transpose_3x3(const float m[9], float result[9]);
__device__ __forceinline__ void tk_cuda_inverse_2x2(const float m[4], float result[4]);
__device__ __forceinline__ void tk_cuda_inverse_3x3(const float m[9], float result[9]);

#ifdef __cplusplus
}
#endif

#endif // TK_CUDA_MATH_HELPERS_H
