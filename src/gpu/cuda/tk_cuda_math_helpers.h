/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_cuda_math_helpers.h
*
* This header file provides a library of fundamental mathematical types and
* functions (vectors, matrices) designed for high-performance computing in the
* TrackieLLM project.
*
* CRITICAL DESIGN FEATURE: This header is "bilingual". It is written to be
* fully compatible with both standard C/C++ compilers (for host-side code) and
* the NVIDIA CUDA C++ compiler (NVCC, for device-side kernel code). This is
* achieved through preprocessor macros that abstract away CUDA-specific
* decorators like `__host__` and `__device__`.
*
* This unified approach is a cornerstone of robust GPGPU engineering, as it
* eliminates code duplication, ensures data structure compatibility between CPU
* and GPU, and provides a single, reliable mathematical foundation for all
* vision and navigation algorithms. All functions are aggressively inlined for
* maximum performance.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TRACKIELLM_GPU_CUDA_TK_CUDA_MATH_HELPERS_H
#define TRACKIELLM_GPU_CUDA_TK_CUDA_MATH_HELPERS_H

#include <vector_types.h> // From CUDA Toolkit for float2, float3, etc.
#include <vector_functions.h> // For make_float3, etc.
#include <math.h>

//------------------------------------------------------------------------------
// Compiler Abstraction Macros
//------------------------------------------------------------------------------

#ifdef __CUDACC__
    // Compiling with NVCC for CUDA
    #define TK_CUDA_HD __host__ __device__
    #define TK_CUDA_D __device__
    #define TK_INLINE __forceinline__
#else
    // Compiling with a standard C/C++ compiler
    #define TK_CUDA_HD
    #define TK_CUDA_D
    #define TK_INLINE inline
#endif

//------------------------------------------------------------------------------
// Type Definitions
//------------------------------------------------------------------------------

// Define our own structs that wrap the CUDA vector types. This provides a
// clear namespace and type safety.
typedef struct { int2 v; } TkInt2;
typedef struct { float2 v; } TkFloat2;
typedef struct { float3 v; } TkFloat3;
typedef struct { float4 v; } TkFloat4;

// Column-major matrix definitions
typedef struct {
    TkFloat3 col0;
    TkFloat3 col1;
    TkFloat3 col2;
} TkMatrix3x3;

typedef struct {
    TkFloat4 col0;
    TkFloat4 col1;
    TkFloat4 col2;
    TkFloat4 col3;
} TkMatrix4x4;


//------------------------------------------------------------------------------
// Constructors
//------------------------------------------------------------------------------

TK_CUDA_HD TK_INLINE TkInt2 make_tk_int2(int x, int y) {
    TkInt2 t;
    t.v = make_int2(x, y);
    return t;
}

TK_CUDA_HD TK_INLINE TkFloat2 make_tk_float2(float x, float y) {
    TkFloat2 t;
    t.v = make_float2(x, y);
    return t;
}

TK_CUDA_HD TK_INLINE TkFloat3 make_tk_float3(float x, float y, float z) {
    TkFloat3 t;
    t.v = make_float3(x, y, z);
    return t;
}

TK_CUDA_HD TK_INLINE TkFloat4 make_tk_float4(float x, float y, float z, float w) {
    TkFloat4 t;
    t.v = make_float4(x, y, z, w);
    return t;
}

TK_CUDA_HD TK_INLINE TkFloat4 make_tk_float4_from_float3(TkFloat3 v, float w) {
    TkFloat4 t;
    t.v = make_float4(v.v.x, v.v.y, v.v.z, w);
    return t;
}

//------------------------------------------------------------------------------
// Vector Operations
//------------------------------------------------------------------------------

// Addition
TK_CUDA_HD TK_INLINE TkFloat3 add_f3(TkFloat3 a, TkFloat3 b) {
    return make_tk_float3(a.v.x + b.v.x, a.v.y + b.v.y, a.v.z + b.v.z);
}

// Subtraction
TK_CUDA_HD TK_INLINE TkFloat3 sub_f3(TkFloat3 a, TkFloat3 b) {
    return make_tk_float3(a.v.x - b.v.x, a.v.y - b.v.y, a.v.z - b.v.z);
}

// Scalar Multiplication
TK_CUDA_HD TK_INLINE TkFloat3 mul_f3_s(TkFloat3 a, float s) {
    return make_tk_float3(a.v.x * s, a.v.y * s, a.v.z * s);
}

// Dot Product
TK_CUDA_HD TK_INLINE float dot_f3(TkFloat3 a, TkFloat3 b) {
    return a.v.x * b.v.x + a.v.y * b.v.y + a.v.z * b.v.z;
}

// Cross Product
TK_CUDA_HD TK_INLINE TkFloat3 cross_f3(TkFloat3 a, TkFloat3 b) {
    return make_tk_float3(a.v.y * b.v.z - a.v.z * b.v.y,
                          a.v.z * b.v.x - a.v.x * b.v.z,
                          a.v.x * b.v.y - a.v.y * b.v.x);
}

// Length
TK_CUDA_HD TK_INLINE float length_f3(TkFloat3 a) {
    return sqrtf(dot_f3(a, a));
}

// Normalize
TK_CUDA_HD TK_INLINE TkFloat3 normalize_f3(TkFloat3 a) {
    float len = length_f3(a);
    // Avoid division by zero
    if (len > 1e-6f) {
        float inv_len = 1.0f / len;
        return mul_f3_s(a, inv_len);
    }
    return make_tk_float3(0.0f, 0.0f, 0.0f);
}

// Linear Interpolation (Lerp)
TK_CUDA_HD TK_INLINE TkFloat3 lerp_f3(TkFloat3 a, TkFloat3 b, float t) {
    return add_f3(a, mul_f3_s(sub_f3(b, a), t));
}

//------------------------------------------------------------------------------
// Matrix Operations
//------------------------------------------------------------------------------

TK_CUDA_HD TK_INLINE TkMatrix4x4 make_identity_m4x4() {
    TkMatrix4x4 M;
    M.col0 = make_tk_float4(1.0f, 0.0f, 0.0f, 0.0f);
    M.col1 = make_tk_float4(0.0f, 1.0f, 0.0f, 0.0f);
    M.col2 = make_tk_float4(0.0f, 0.0f, 1.0f, 0.0f);
    M.col3 = make_tk_float4(0.0f, 0.0f, 0.0f, 1.0f);
    return M;
}

TK_CUDA_HD TK_INLINE TkMatrix4x4 make_translation_m4x4(float x, float y, float z) {
    TkMatrix4x4 M = make_identity_m4x4();
    M.col3 = make_tk_float4(x, y, z, 1.0f);
    return M;
}

// Matrix-Vector Multiplication (M * v)
TK_CUDA_HD TK_INLINE TkFloat4 mul_m4x4_v4(TkMatrix4x4 M, TkFloat4 v) {
    TkFloat4 res;
    res.v.x = M.col0.v.x * v.v.x + M.col1.v.x * v.v.y + M.col2.v.x * v.v.z + M.col3.v.x * v.v.w;
    res.v.y = M.col0.v.y * v.v.x + M.col1.v.y * v.v.y + M.col2.v.y * v.v.z + M.col3.v.y * v.v.w;
    res.v.z = M.col0.v.z * v.v.x + M.col1.v.z * v.v.y + M.col2.v.z * v.v.z + M.col3.v.z * v.v.w;
    res.v.w = M.col0.v.w * v.v.x + M.col1.v.w * v.v.y + M.col2.v.w * v.v.z + M.col3.v.w * v.v.w;
    return res;
}

// C++ specific operator overloads for better ergonomics
#ifdef __cplusplus
TK_CUDA_HD TK_INLINE TkFloat3 operator+(TkFloat3 a, TkFloat3 b) { return add_f3(a, b); }
TK_CUDA_HD TK_INLINE TkFloat3 operator-(TkFloat3 a, TkFloat3 b) { return sub_f3(a, b); }
TK_CUDA_HD TK_INLINE TkFloat3 operator*(TkFloat3 a, float s) { return mul_f3_s(a, s); }
TK_CUDA_HD TK_INLINE TkFloat3 operator*(float s, TkFloat3 a) { return mul_f3_s(a, s); }
TK_CUDA_HD TK_INLINE TkFloat4 operator*(TkMatrix4x4 M, TkFloat4 v) { return mul_m4x4_v4(M, v); }
#endif


#endif // TRACKIELLM_GPU_CUDA_TK_CUDA_MATH_HELPERS_H