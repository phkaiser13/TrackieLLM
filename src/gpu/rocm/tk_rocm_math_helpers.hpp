/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_rocm_math_helpers.hpp
 *
 * This header file provides a comprehensive library of device-side mathematical
 * helper functions, specifically optimized for the ROCm/HIP execution environment.
 * These functions are designed to be used within GPU kernels to perform common
 * and complex mathematical operations with maximum performance and precision.
 *
 * The library is engineered with the following principles:
 *   - Performance-First: All functions are declared as `__device__ __forceinline__`
 *     to eliminate function call overhead, ensuring that the assembly generated is
 *     as efficient as possible. Implementations leverage hardware-specific
 *     intrinsics where available.
 *   - High-Fidelity Porting: This is a direct, high-quality port of the original
 *     CUDA math helpers, ensuring functional parity while adapting to the nuances
 *     of the AMD GPU architecture.
 *   - Robustness and Clarity: Each function is extensively documented, explaining
 *     its purpose, algorithm, and parameter constraints. The code is written to be
 *     highly readable and maintainable.
 *   - C++ Namespace Encapsulation: All functions are encapsulated within the
 *     `tk::gpu::rocm::math` namespace to prevent symbol clashes and provide a
 *     clear, organized API structure, adhering to modern C++ practices.
 *
 * Dependencies:
 *   - HIP Runtime API (hip/hip_runtime.h)
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#ifndef TRACKIELLM_GPU_ROCM_TK_ROCM_MATH_HELPERS_HPP
#define TRACKIELLM_GPU_ROCM_TK_ROCM_MATH_HELPERS_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h> // For half-precision types if needed

// ROCm does not have a direct equivalent of <math_constants.h> in the same way
// as CUDA. We define essential constants manually within our namespace to ensure
// portability and self-containment.
namespace tk {
namespace gpu {
namespace rocm {
namespace math {

// --- Mathematical Constants ---
// High-precision constants are crucial for numerical stability in complex calculations.
constexpr float PI         = 3.14159265358979323846f;
constexpr float PI_INV     = 1.0f / PI;
constexpr float TWO_PI     = 2.0f * PI;
constexpr float PI_OVER_2  = PI / 2.0f;
constexpr float E          = 2.71828182845904523536f;
constexpr float SQRT2      = 1.41421356237309504880f;
constexpr float EPSILON    = 1e-6f; // Small value to prevent division by zero and for floating point comparisons

// --- Vector Types ---
// Re-aliasing HIP vector types for consistency within the project's type system.
// This abstraction layer allows for easier porting to other platforms in the future.
using float2 = float2;
using float3 = float3;
using float4 = float4;
using int2 = int2;
using int3 = int3;
using int4 = int4;

// --- Activation Functions & Non-linearities ---
// These are fundamental building blocks for neural networks.

/**
 * @brief Computes the sigmoid activation function.
 * @param x The input value.
 * @return The result of 1 / (1 + exp(-x)).
 */
__device__ __forceinline__ float sigmoidf(float x) {
    // Numerically stable implementation
    return 1.0f / (1.0f + expf(-x));
}

/**
 * @brief Computes the Rectified Linear Unit (ReLU) activation.
 * @param x The input value.
 * @return `x` if `x > 0`, otherwise `0`.
 */
__device__ __forceinline__ float relu(float x) {
    return fmaxf(0.0f, x);
}

/**
 * @brief Computes the Leaky ReLU activation.
 * @param x The input value.
 * @param alpha The slope for negative inputs (e.g., 0.01).
 * @return `x` if `x > 0`, otherwise `alpha * x`.
 */
__device__ __forceinline__ float leaky_relu(float x, float alpha) {
    return (x > 0.0f) ? x : alpha * x;
}

/**
 * @brief Computes an approximation of the hyperbolic tangent function.
 * This is faster than the standard `tanhf` but with slightly less precision.
 * @param x The input value.
 * @return An approximation of tanh(x).
 */
__device__ __forceinline__ float tanhf_approx(float x) {
    // A common and fast approximation using a rational function
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

/**
 * @brief Computes the Gaussian Error Linear Unit (GELU) activation.
 * This is a smooth approximation of ReLU used in modern transformers.
 * @param x The input value.
 * @return The GELU activation of x.
 */
__device__ __forceinline__ float gelu(float x) {
    // Using the erf-based formula, common in models like BERT.
    return 0.5f * x * (1.0f + erff(x / SQRT2));
}


// --- Clamping, Interpolation, and Normalization ---

/**
 * @brief Clamps a floating-point value to a specified range [min_val, max_val].
 * @param value The value to clamp.
 * @param min_val The minimum allowed value.
 * @param max_val The maximum allowed value.
 * @return The clamped value.
 */
__device__ __forceinline__ float clampf(float value, float min_val, float max_val) {
    return fminf(fmaxf(value, min_val), max_val);
}

/**
 * @brief Clamps an integer value to a specified range [min_val, max_val].
 * @param value The value to clamp.
 * @param min_val The minimum allowed value.
 * @param max_val The maximum allowed value.
 * @return The clamped value.
 */
__device__ __forceinline__ int clampi(int value, int min_val, int max_val) {
    return min(max(value, min_val), max_val);
}

/**
 * @brief Performs linear interpolation between two values.
 * @param a The start value (when t=0).
 * @param b The end value (when t=1).
 * @param t The interpolation factor, clamped to [0, 1].
 * @return The interpolated value: a + t * (b - a).
 */
__device__ __forceinline__ float lerpf(float a, float b, float t) {
    float clamped_t = clampf(t, 0.0f, 1.0f);
    return a + clamped_t * (b - a);
}

/**
 * @brief Normalizes a value from one range to another.
 * @param value The input value.
 * @param in_min The minimum of the input range.
 * @param in_max The maximum of the input range.
 * @param out_min The minimum of the output range.
 * @param out_max The maximum of the output range.
 * @return The value mapped to the new range.
 */
__device__ __forceinline__ float normalize_to_range(float value, float in_min, float in_max, float out_min, float out_max) {
    // Avoid division by zero
    if (fabsf(in_max - in_min) < EPSILON) {
        return out_min;
    }
    float normalized_value = (value - in_min) / (in_max - in_min);
    return out_min + normalized_value * (out_max - out_min);
}

/**
 * @brief Performs bilinear interpolation on a 2D grid.
 * Samples a texture-like data buffer at a fractional coordinate.
 * @param x The fractional x-coordinate.
 * @param y The fractional y-coordinate.
 * @param data Pointer to the 1D array representing the 2D grid (row-major).
 * @param width The width of the grid.
 * @param height The height of the grid.
 * @return The bilinearly interpolated value.
 */
__device__ __forceinline__ float interpolate_bilinear(const float* data, float x, float y, int width, int height) {
    // Clamp coordinates to be within the valid range
    x = clampf(x, 0.0f, width - 1.0f - EPSILON);
    y = clampf(y, 0.0f, height - 1.0f - EPSILON);

    int x0 = static_cast<int>(floorf(x));
    int y0 = static_cast<int>(floorf(y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Clamp indices to prevent out-of-bounds access
    x0 = clampi(x0, 0, width - 1);
    x1 = clampi(x1, 0, width - 1);
    y0 = clampi(y0, 0, height - 1);
    y1 = clampi(y1, 0, height - 1);

    float tx = x - floorf(x);
    float ty = y - floorf(y);

    float v00 = data[y0 * width + x0];
    float v10 = data[y0 * width + x1];
    float v01 = data[y1 * width + x0];
    float v11 = data[y1 * width + x1];

    float ix0 = lerpf(v00, v10, tx);
    float ix1 = lerpf(v01, v11, tx);

    return lerpf(ix0, ix1, ty);
}


// --- Vector & Geometry Operations ---

/**
 * @brief Computes the dot product of two 3D vectors.
 * @param v1 The first vector.
 * @param v2 The second vector.
 * @return The dot product.
 */
__device__ __forceinline__ float dot(const float3& v1, const float3& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

/**
 * @brief Computes the magnitude (length) of a 3D vector.
 * @param v The vector.
 * @return The magnitude of the vector.
 */
__device__ __forceinline__ float magnitude(const float3& v) {
    return sqrtf(dot(v, v));
}

/**
 * @brief Normalizes a 3D vector to unit length.
 * @param v The vector to normalize.
 * @return The normalized vector. Returns a zero vector if magnitude is zero.
 */
__device__ __forceinline__ float3 normalize(const float3& v) {
    float mag = magnitude(v);
    if (mag < EPSILON) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    float inv_mag = 1.0f / mag;
    return make_float3(v.x * inv_mag, v.y * inv_mag, v.z * inv_mag);
}

/**
 * @brief Computes the cross product of two 3D vectors.
 * @param v1 The first vector.
 * @param v2 The second vector.
 * @return The vector perpendicular to v1 and v2.
 */
__device__ __forceinline__ float3 cross(const float3& v1, const float3& v2) {
    return make_float3(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x
    );
}

/**
 * @brief Reflects an incident vector across a surface normal.
 * @param incident The incident vector.
 * @param normal The surface normal (must be a unit vector).
 * @return The reflected vector.
 */
__device__ __forceinline__ float3 reflect(const float3& incident, const float3& normal) {
    // Formula: R = I - 2 * N * dot(I, N)
    return incident - 2.0f * normal * dot(incident, normal);
}

// --- Color & Image Processing ---

/**
 * @brief Converts an RGB color to its luminance (grayscale) value.
 * Uses the standard NTSC/PAL formula.
 * @param rgb The RGB color vector.
 * @return The luminance value.
 */
__device__ __forceinline__ float luminance(const float3& rgb) {
    // Standard weights for human perception: 0.299*R + 0.587*G + 0.114*B
    return 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
}

/**
 * @brief Applies a gamma correction to a color channel.
 * @param value The color channel value.
 * @param gamma The gamma factor (e.g., 2.2).
 * @return The gamma-corrected value.
 */
__device__ __forceinline__ float gamma_correct(float value, float gamma) {
    return powf(value, 1.0f / gamma);
}

/**
 * @brief Applies a gamma correction to an RGB color.
 * @param rgb The RGB color.
 * @param gamma The gamma factor.
 * @return The gamma-corrected color.
 */
__device__ __forceinline__ float3 gamma_correct(const float3& rgb, float gamma) {
    float inv_gamma = 1.0f / gamma;
    return make_float3(powf(rgb.x, inv_gamma), powf(rgb.y, inv_gamma), powf(rgb.z, inv_gamma));
}

} // namespace math
} // namespace rocm
} // namespace gpu
} // namespace tk

#endif // TRACKIELLM_GPU_ROCM_TK_ROCM_MATH_HELPERS_HPP
