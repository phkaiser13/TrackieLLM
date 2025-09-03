/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_cuda_math_helpers.cu
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

#include "tk_cuda_math_helpers.h"

// Device function implementations

__device__ __forceinline__ float tk_cuda_sigmoidf(float x) {
    /*
     * Compute sigmoid function: 1 / (1 + exp(-x))
     * Optimized version using fast math intrinsics for better performance on GPU
     */
    return __frcp_rn(__fadd_rn(1.0f, __expf(-x)));
}

__device__ __forceinline__ float tk_cuda_clampf(float value, float min_val, float max_val) {
    /*
     * Clamp a floating point value between min and max bounds
     * Uses CUDA's built-in fminf/fmaxf for optimal GPU performance
     */
    return fminf(fmaxf(value, min_val), max_val);
}

__device__ __forceinline__ float tk_cuda_lerpf(float a, float b, float t) {
    /*
     * Linear interpolation between two values
     * Formula: a + t * (b - a)
     */
    return __fmaf_rn(t, __fsub_rn(b, a), a);
}

__device__ __forceinline__ int tk_cuda_clampi(int value, int min_val, int max_val) {
    /*
     * Clamp an integer value between min and max bounds
     */
    return min(max(value, min_val), max_val);
}

__device__ __forceinline__ float tk_cuda_relu(float x) {
    /*
     * Rectified Linear Unit activation function
     * Returns max(0, x)
     */
    return fmaxf(0.0f, x);
}

__device__ __forceinline__ float tk_cuda_tanhf_approx(float x) {
    /*
     * Fast approximation of hyperbolic tangent function
     * Uses polynomial approximation for improved performance
     * Accurate in range [-2, 2]
     */
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

__device__ __forceinline__ float tk_cuda_normalize_to_range(
    float value, 
    float in_min, 
    float in_max, 
    float out_min, 
    float out_max
) {
    /*
     * Normalize a value from one range to another
     * Formula: out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)
     */
    float in_range = in_max - in_min;
    float out_range = out_max - out_min;
    
    // Avoid division by zero
    if (in_range == 0.0f) {
        return out_min;
    }
    
    float normalized = (value - in_min) / in_range;
    return out_min + normalized * out_range;
}

__device__ __forceinline__ float tk_cuda_fast_sqrt(float x) {
    /*
     * Fast square root using CUDA intrinsic
     * Provides good balance between speed and accuracy
     */
    return sqrtf(x);
}

__device__ __forceinline__ float tk_cuda_fast_inverse_sqrt(float x) {
    /*
     * Fast inverse square root using CUDA intrinsic
     * Commonly used in normalization operations
     */
    return rsqrtf(x);
}

__device__ __forceinline__ float tk_cuda_absf(float x) {
    /*
     * Absolute value function for floats
     */
    return fabsf(x);
}

__device__ __forceinline__ int tk_cuda_absi(int x) {
    /*
     * Absolute value function for integers
     */
    return abs(x);
}

__device__ __forceinline__ float tk_cuda_roundf(float x) {
    /*
     * Round to nearest integer value
     */
    return rintf(x);
}

__device__ __forceinline__ float tk_cuda_floorf(float x) {
    /*
     * Floor function - largest integer not greater than x
     */
    return floorf(x);
}

__device__ __forceinline__ float tk_cuda_ceilf(float x) {
    /*
     * Ceiling function - smallest integer not less than x
     */
    return ceilf(x);
}

__device__ __forceinline__ float tk_cuda_fmodf(float x, float y) {
    /*
     * Floating point modulo operation
     */
    return fmodf(x, y);
}

__device__ __forceinline__ float tk_cuda_powf(float base, float exponent) {
    /*
     * Power function using CUDA intrinsic
     */
    return powf(base, exponent);
}

__device__ __forceinline__ float tk_cuda_expf(float x) {
    /*
     * Exponential function using CUDA intrinsic
     */
    return expf(x);
}

__device__ __forceinline__ float tk_cuda_logf(float x) {
    /*
     * Natural logarithm function using CUDA intrinsic
     */
    return logf(x);
}

__device__ __forceinline__ float tk_cuda_degrees_to_radians(float degrees) {
    /*
     * Convert degrees to radians
     * Formula: degrees * PI / 180
     */
    return degrees * 0.017453292519943295f; // PI / 180
}

__device__ __forceinline__ float tk_cuda_radians_to_degrees(float radians) {
    /*
     * Convert radians to degrees
     * Formula: radians * 180 / PI
     */
    return radians * 57.29577951308232f; // 180 / PI
}

__device__ __forceinline__ float tk_cuda_smoothstep(float edge0, float edge1, float x) {
    /*
     * Smoothstep function for smooth interpolation
     * Returns 0 if x <= edge0, 1 if x >= edge1, and smoothly interpolates in between
     */
    float t = tk_cuda_clampf((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

__device__ __forceinline__ float tk_cuda_luminance_from_rgb(float r, float g, float b) {
    /*
     * Convert RGB to luminance using standard weights
     * Based on ITU-R BT.709 standard
     */
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

__device__ __forceinline__ float tk_cuda_distance_2d(float x1, float y1, float x2, float y2) {
    /*
     * Calculate Euclidean distance between two 2D points
     */
    float dx = x2 - x1;
    float dy = y2 - y1;
    return tk_cuda_fast_sqrt(dx * dx + dy * dy);
}

__device__ __forceinline__ float tk_cuda_distance_3d(float x1, float y1, float z1, float x2, float y2, float z2) {
    /*
     * Calculate Euclidean distance between two 3D points
     */
    float dx = x2 - x1;
    float dy = y2 - y1;
    float dz = z2 - z1;
    return tk_cuda_fast_sqrt(dx * dx + dy * dy + dz * dz);
}

__device__ __forceinline__ float tk_cuda_dot_product_2d(float x1, float y1, float x2, float y2) {
    /*
     * Calculate dot product of two 2D vectors
     */
    return x1 * x2 + y1 * y2;
}

__device__ __forceinline__ float tk_cuda_dot_product_3d(float x1, float y1, float z1, float x2, float y2, float z2) {
    /*
     * Calculate dot product of two 3D vectors
     */
    return x1 * x2 + y1 * y2 + z1 * z2;
}

__device__ __forceinline__ float tk_cuda_magnitude_2d(float x, float y) {
    /*
     * Calculate magnitude of a 2D vector
     */
    return tk_cuda_fast_sqrt(x * x + y * y);
}

__device__ __forceinline__ float tk_cuda_magnitude_3d(float x, float y, float z) {
    /*
     * Calculate magnitude of a 3D vector
     */
    return tk_cuda_fast_sqrt(x * x + y * y + z * z);
}

__device__ __forceinline__ void tk_cuda_normalize_vector_2d(float* x, float* y) {
    /*
     * Normalize a 2D vector to unit length
     */
    float mag = tk_cuda_magnitude_2d(*x, *y);
    if (mag > 0.0f) {
        *x /= mag;
        *y /= mag;
    }
}

__device__ __forceinline__ void tk_cuda_normalize_vector_3d(float* x, float* y, float* z) {
    /*
     * Normalize a 3D vector to unit length
     */
    float mag = tk_cuda_magnitude_3d(*x, *y, *z);
    if (mag > 0.0f) {
        *x /= mag;
        *y /= mag;
        *z /= mag;
    }
}

__device__ __forceinline__ float tk_cuda_cross_product_2d(float x1, float y1, float x2, float y2) {
    /*
     * Calculate 2D cross product (z-component of 3D cross product)
     */
    return x1 * y2 - y1 * x2;
}

__device__ __forceinline__ float tk_cuda_angle_between_vectors_2d(float x1, float y1, float x2, float y2) {
    /*
     * Calculate angle between two 2D vectors in radians
     */
    float dot = tk_cuda_dot_product_2d(x1, y1, x2, y2);
    float mag1 = tk_cuda_magnitude_2d(x1, y1);
    float mag2 = tk_cuda_magnitude_2d(x2, y2);
    
    if (mag1 == 0.0f || mag2 == 0.0f) {
        return 0.0f;
    }
    
    float cos_theta = tk_cuda_clampf(dot / (mag1 * mag2), -1.0f, 1.0f);
    return acosf(cos_theta);
}

__device__ __forceinline__ float tk_cuda_wrap_angle(float angle) {
    /*
     * Wrap angle to [-PI, PI] range
     */
    while (angle > M_PI_F) {
        angle -= 2.0f * M_PI_F;
    }
    while (angle < -M_PI_F) {
        angle += 2.0f * M_PI_F;
    }
    return angle;
}

__device__ __forceinline__ float tk_cuda_interpolate_bilinear(
    float x, float y,
    const float* data,
    int width, int height
) {
    /*
     * Bilinear interpolation for 2D data array
     */
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    // Clamp to valid indices
    x0 = tk_cuda_clampi(x0, 0, width - 1);
    y0 = tk_cuda_clampi(y0, 0, height - 1);
    x1 = tk_cuda_clampi(x1, 0, width - 1);
    y1 = tk_cuda_clampi(y1, 0, height - 1);
    
    // Get values at four corners
    float v00 = data[y0 * width + x0];
    float v01 = data[y0 * width + x1];
    float v10 = data[y1 * width + x0];
    float v11 = data[y1 * width + x1];
    
    // Calculate interpolation weights
    float wx = x - x0;
    float wy = y - y0;
    
    // Bilinear interpolation
    float v0 = tk_cuda_lerpf(v00, v01, wx);
    float v1 = tk_cuda_lerpf(v10, v11, wx);
    return tk_cuda_lerpf(v0, v1, wy);
}

__device__ __forceinline__ float tk_cuda_gaussian_weight(float distance, float sigma) {
    /*
     * Calculate Gaussian weight for a given distance
     */
    return expf(-(distance * distance) / (2.0f * sigma * sigma));
}

__device__ __forceinline__ float tk_cuda_softplus(float x) {
    /*
     * Softplus activation function: log(1 + exp(x))
     * Smooth approximation of ReLU
     */
    return logf(1.0f + expf(x));
}

__device__ __forceinline__ float tk_cuda_elu(float x, float alpha) {
    /*
     * Exponential Linear Unit activation function
     * alpha * (exp(x) - 1) for x < 0, x for x >= 0
     */
    return x >= 0.0f ? x : alpha * (expf(x) - 1.0f);
}

__device__ __forceinline__ float tk_cuda_swish(float x) {
    /*
     * Swish activation function: x * sigmoid(x)
     */
    return x * tk_cuda_sigmoidf(x);
}

__device__ __forceinline__ float tk_cuda_mish(float x) {
    /*
     * Mish activation function: x * tanh(softplus(x))
     */
    return x * tanhf(logf(1.0f + expf(x)));
}

__device__ __forceinline__ float tk_cuda_gelu(float x) {
    /*
     * Gaussian Error Linear Unit approximation
     * 0.5 * x * (1 + tanh(sqrt(2/PI) * (x + 0.044715 * x^3)))
     */
    float c = 0.7978845608028654f; // sqrt(2/PI)
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x3)));
}

__device__ __forceinline__ float tk_cuda_hard_sigmoid(float x) {
    /*
     * Hard sigmoid function: max(0, min(1, (x + 3) / 6))
     */
    return fmaxf(0.0f, fminf(1.0f, (x + 3.0f) / 6.0f));
}

__device__ __forceinline__ float tk_cuda_hard_tanh(float x) {
    /*
     * Hard tanh function: max(-1, min(1, x))
     */
    return fmaxf(-1.0f, fminf(1.0f, x));
}

__device__ __forceinline__ float tk_cuda_leaky_relu(float x, float alpha) {
    /*
     * Leaky ReLU activation function
     * x if x > 0, alpha * x if x <= 0
     */
    return x > 0.0f ? x : alpha * x;
}

__device__ __forceinline__ float tk_cuda_prelu(float x, float alpha) {
    /*
     * Parametric ReLU activation function
     * Same as Leaky ReLU but with learnable alpha parameter
     */
    return x > 0.0f ? x : alpha * x;
}

__device__ __forceinline__ float tk_cuda_selu(float x) {
    /*
     * Scaled Exponential Linear Unit
     * scale * (x if x > 0 else alpha * (exp(x) - 1))
     */
    const float alpha = 1.6732632423543772848170429916717f;
    const float scale = 1.0507009873554804934193349852946f;
    return scale * (x > 0.0f ? x : alpha * (expf(x) - 1.0f));
}

__device__ __forceinline__ float tk_cuda_threshold(float x, float threshold, float value) {
    /*
     * Threshold function: value if x > threshold else 0
     */
    return x > threshold ? value : 0.0f;
}

__device__ __forceinline__ float tk_cuda_soft_threshold(float x, float threshold) {
    /*
     * Soft thresholding function
     * x - threshold if x > threshold
     * x + threshold if x < -threshold
     * 0 otherwise
     */
    if (x > threshold) {
        return x - threshold;
    } else if (x < -threshold) {
        return x + threshold;
    } else {
        return 0.0f;
    }
}

__device__ __forceinline__ float tk_cuda_hard_shrink(float x, float lambda) {
    /*
     * Hard shrinkage function
     * x if x > lambda or x < -lambda, 0 otherwise
     */
    return (x > lambda || x < -lambda) ? x : 0.0f;
}

__device__ __forceinline__ float tk_cuda_soft_shrink(float x, float lambda) {
    /*
     * Soft shrinkage function
     * x - lambda if x > lambda
     * x + lambda if x < -lambda
     * 0 otherwise
     */
    if (x > lambda) {
        return x - lambda;
    } else if (x < -lambda) {
        return x + lambda;
    } else {
        return 0.0f;
    }
}

__device__ __forceinline__ float tk_cuda_normalize_to_unit_vector(float* vec, int size) {
    /*
     * Normalize a vector to unit length
     * Returns the original magnitude
     */
    float magnitude = 0.0f;
    
    // Calculate magnitude
    for (int i = 0; i < size; i++) {
        magnitude += vec[i] * vec[i];
    }
    magnitude = tk_cuda_fast_sqrt(magnitude);
    
    // Normalize if magnitude is non-zero
    if (magnitude > 0.0f) {
        for (int i = 0; i < size; i++) {
            vec[i] /= magnitude;
        }
    }
    
    return magnitude;
}

__device__ __forceinline__ float tk_cuda_cosine_similarity(const float* a, const float* b, int size) {
    /*
     * Calculate cosine similarity between two vectors
     */
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (int i = 0; i < size; i++) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    if (norm_a == 0.0f || norm_b == 0.0f) {
        return 0.0f;
    }
    
    return dot_product / (tk_cuda_fast_sqrt(norm_a) * tk_cuda_fast_sqrt(norm_b));
}

__device__ __forceinline__ float tk_cuda_euclidean_distance(const float* a, const float* b, int size) {
    /*
     * Calculate Euclidean distance between two vectors
     */
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return tk_cuda_fast_sqrt(sum);
}

__device__ __forceinline__ float tk_cuda_manhattan_distance(const float* a, const float* b, int size) {
    /*
     * Calculate Manhattan distance between two vectors
     */
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += tk_cuda_absf(a[i] - b[i]);
    }
    return sum;
}

__device__ __forceinline__ float tk_cuda_chebyshev_distance(const float* a, const float* b, int size) {
    /*
     * Calculate Chebyshev distance between two vectors
     */
    float max_diff = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = tk_cuda_absf(a[i] - b[i]);
        max_diff = fmaxf(max_diff, diff);
    }
    return max_diff;
}

__device__ __forceinline__ float tk_cuda_minkowski_distance(const float* a, const float* b, int size, float p) {
    /*
     * Calculate Minkowski distance between two vectors
     */
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += powf(tk_cuda_absf(a[i] - b[i]), p);
    }
    return powf(sum, 1.0f / p);
}

__device__ __forceinline__ void tk_cuda_matrix_multiply_2x2(
    const float a[4], const float b[4], float result[4]
) {
    /*
     * Multiply two 2x2 matrices
     * a = [a00 a01]  b = [b00 b01]
     *     [a10 a11]      [b10 b11]
     */
    result[0] = a[0] * b[0] + a[1] * b[2];  // a00*b00 + a01*b10
    result[1] = a[0] * b[1] + a[1] * b[3];  // a00*b01 + a01*b11
    result[2] = a[2] * b[0] + a[3] * b[2];  // a10*b00 + a11*b10
    result[3] = a[2] * b[1] + a[3] * b[3];  // a10*b01 + a11*b11
}

__device__ __forceinline__ void tk_cuda_matrix_multiply_3x3(
    const float a[9], const float b[9], float result[9]
) {
    /*
     * Multiply two 3x3 matrices
     * a = [a00 a01 a02]  b = [b00 b01 b02]
     *     [a10 a11 a12]      [b10 b11 b12]
     *     [a20 a21 a22]      [b20 b21 b22]
     */
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result[i * 3 + j] = 0.0f;
            for (int k = 0; k < 3; k++) {
                result[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
            }
        }
    }
}

__device__ __forceinline__ float tk_cuda_determinant_2x2(const float m[4]) {
    /*
     * Calculate determinant of 2x2 matrix
     * m = [m00 m01]
     *     [m10 m11]
     */
    return m[0] * m[3] - m[1] * m[2];
}

__device__ __forceinline__ float tk_cuda_determinant_3x3(const float m[9]) {
    /*
     * Calculate determinant of 3x3 matrix
     * m = [m00 m01 m02]
     *     [m10 m11 m12]
     *     [m20 m21 m22]
     */
    return m[0] * (m[4] * m[8] - m[5] * m[7]) -
           m[1] * (m[3] * m[8] - m[5] * m[6]) +
           m[2] * (m[3] * m[7] - m[4] * m[6]);
}

__device__ __forceinline__ void tk_cuda_transpose_2x2(const float m[4], float result[4]) {
    /*
     * Transpose 2x2 matrix
     */
    result[0] = m[0]; result[1] = m[2];
    result[2] = m[1]; result[3] = m[3];
}

__device__ __forceinline__ void tk_cuda_transpose_3x3(const float m[9], float result[9]) {
    /*
     * Transpose 3x3 matrix
     */
    result[0] = m[0]; result[1] = m[3]; result[2] = m[6];
    result[3] = m[1]; result[4] = m[4]; result[5] = m[7];
    result[6] = m[2]; result[7] = m[5]; result[8] = m[8];
}

__device__ __forceinline__ void tk_cuda_inverse_2x2(const float m[4], float result[4]) {
    /*
     * Calculate inverse of 2x2 matrix
     */
    float det = tk_cuda_determinant_2x2(m);
    if (det == 0.0f) {
        // Matrix is singular, return zero matrix
        result[0] = result[1] = result[2] = result[3] = 0.0f;
        return;
    }
    
    float inv_det = 1.0f / det;
    result[0] = m[3] * inv_det;
    result[1] = -m[1] * inv_det;
    result[2] = -m[2] * inv_det;
    result[3] = m[0] * inv_det;
}

__device__ __forceinline__ void tk_cuda_inverse_3x3(const float m[9], float result[9]) {
    /*
     * Calculate inverse of 3x3 matrix
     */
    float det = tk_cuda_determinant_3x3(m);
    if (det == 0.0f) {
        // Matrix is singular, return zero matrix
        for (int i = 0; i < 9; i++) {
            result[i] = 0.0f;
        }
        return;
    }
    
    float inv_det = 1.0f / det;
    
    result[0] = (m[4] * m[8] - m[5] * m[7]) * inv_det;
    result[1] = (m[2] * m[7] - m[1] * m[8]) * inv_det;
    result[2] = (m[1] * m[5] - m[2] * m[4]) * inv_det;
    result[3] = (m[5] * m[6] - m[3] * m[8]) * inv_det;
    result[4] = (m[0] * m[8] - m[2] * m[6]) * inv_det;
    result[5] = (m[2] * m[3] - m[0] * m[5]) * inv_det;
    result[6] = (m[3] * m[7] - m[4] * m[6]) * inv_det;
    result[7] = (m[1] * m[6] - m[0] * m[7]) * inv_det;
    result[8] = (m[0] * m[4] - m[1] * m[3]) * inv_det;
}
