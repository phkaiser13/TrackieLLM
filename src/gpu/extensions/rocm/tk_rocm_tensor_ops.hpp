/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_rocm_tensor_ops.hpp
 *
 * This header file defines the public API for high-performance, HIP-accelerated
 * tensor operations, forming the core of a neural network inference engine on
 * the ROCm platform. It provides a suite of functions for deep learning primitives,
 * including linear algebra, convolutions, and normalization.
 *
 * The design philosophy is centered on:
 *   1.  **Performance through Specialization**: Leverages AMD's high-performance
 *       libraries, such as `rocBLAS` for matrix multiplications and `MIOpen` for
 *       optimized convolutions, ensuring that computations are executed on the
 *       most efficient hardware paths available.
 *   2.  **Data-Centric Abstraction**: A powerful tensor descriptor (`tk_tensor_descriptor_t`)
 *       is used throughout the API to abstract away the complexities of memory
 *       layouts (NCHW/NHWC), data types (FP32/FP16), and dimensions.
 *   3.  **Asynchronous Execution Model**: All operations are designed to be
 *       asynchronous and are enqueued on HIP streams provided by the dispatcher,
 *       enabling the construction of sophisticated, high-throughput inference pipelines.
 *   4.  **Extreme Robustness**: The API enforces strict validation of all inputs,
 *       handles, and parameters to prevent common sources of GPU errors, ensuring
 *       stability in production environments.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#ifndef TRACKIELLM_GPU_EXTENSIONS_ROCM_TK_ROCM_TENSOR_OPS_HPP
#define TRACKIELLM_GPU_EXTENSIONS_ROCM_TK_ROCM_TENSOR_OPS_HPP

#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "gpu/rocm/tk_rocm_dispatch.hpp"

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Tensor Descriptor and Format Definitions
//------------------------------------------------------------------------------

/**
 * @enum tk_tensor_data_type_t
 * @brief Enumeration of supported tensor element data types.
 */
typedef enum {
    TK_TENSOR_TYPE_F32 = 0,      /**< 32-bit single-precision floating point. */
    TK_TENSOR_TYPE_F16,          /**< 16-bit half-precision floating point. */
    TK_TENSOR_TYPE_I32,          /**< 32-bit signed integer. */
    TK_TENSOR_TYPE_I8,           /**< 8-bit signed integer (for quantized models). */
} tk_tensor_data_type_t;

/**
 * @enum tk_tensor_layout_t
 * @brief Enumeration of supported tensor memory layouts.
 */
typedef enum {
    TK_TENSOR_LAYOUT_NCHW = 0,   /**< Batch, Channels, Height, Width (preferred by MIOpen). */
    TK_TENSOR_LAYOUT_NHWC,       /**< Batch, Height, Width, Channels. */
} tk_tensor_layout_t;

/**
 * @struct tk_tensor_descriptor_t
 * @brief A comprehensive descriptor for a tensor in GPU memory. It provides all
 *        the metadata required for kernels and libraries like MIOpen/rocBLAS to
 *        correctly interpret the tensor data.
 */
typedef struct {
    tk_gpu_buffer_t buffer;         /**< Opaque handle to the GPU buffer. */
    tk_tensor_data_type_t type;     /**< Data type of the tensor elements. */
    tk_tensor_layout_t layout;      /**< The memory layout of the tensor. */
    uint32_t dimensions;            /**< The number of dimensions (rank) of the tensor. */
    uint32_t shape[8];              /**< The size of each dimension. */
} tk_tensor_descriptor_t;

//------------------------------------------------------------------------------
// Element-wise and Activation Operations
//------------------------------------------------------------------------------

/**
 * @enum tk_activation_type_t
 * @brief Enumeration of supported element-wise activation functions.
 */
typedef enum {
    TK_ACTIVATION_RELU = 0,
    TK_ACTIVATION_SIGMOID,
    TK_ACTIVATION_TANH,
    TK_ACTIVATION_LEAKY_RELU,
} tk_activation_type_t;

/**
 * @brief Applies a non-linear activation function to each element of a tensor.
 *
 * @param[in] dispatcher The ROCm dispatcher instance.
 * @param[in] input      A descriptor for the input tensor.
 * @param[out] output    A descriptor for the output tensor.
 * @param[in] type       The type of activation function to apply.
 * @param[in] alpha      A parameter for certain activations (e.g., slope for Leaky ReLU).
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_rocm_tensor_activation(
    tk_rocm_dispatcher_t* dispatcher,
    const tk_tensor_descriptor_t* input,
    tk_tensor_descriptor_t* output,
    tk_activation_type_t type,
    float alpha
);

/**
 * @brief Performs an element-wise binary operation: output = alpha * input_a + beta * input_b.
 *
 * @param[in] dispatcher The ROCm dispatcher instance.
 * @param[in] alpha      Scalar multiplier for the first input tensor.
 * @param[in] input_a    A descriptor for the first input tensor.
 * @param[in] beta       Scalar multiplier for the second input tensor.
 * @param[in] input_b    A descriptor for the second input tensor.
 * @param[out] output    A descriptor for the output tensor.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_rocm_tensor_add(
    tk_rocm_dispatcher_t* dispatcher,
    float alpha,
    const tk_tensor_descriptor_t* input_a,
    float beta,
    const tk_tensor_descriptor_t* input_b,
    tk_tensor_descriptor_t* output
);

//------------------------------------------------------------------------------
// Linear Algebra (GEMM)
//------------------------------------------------------------------------------

/**
 * @struct tk_tensor_gemm_params_t
 * @brief Parameters for the General Matrix Multiply (GEMM) operation.
 */
typedef struct {
    float alpha;                 /**< Scalar multiplier for the product of A and B. */
    float beta;                  /**< Scalar multiplier for the input matrix C. */
    int trans_a;                 /**< Flag to transpose matrix A (0 for no, 1 for yes). */
    int trans_b;                 /**< Flag to transpose matrix B (0 for no, 1 for yes). */
} tk_tensor_gemm_params_t;

/**
 * @brief Performs General Matrix Multiplication (GEMM): C = alpha * op(A) * op(B) + beta * C.
 *
 * This function is a wrapper around the highly optimized `rocBLAS` library,
 * ensuring maximum performance for matrix multiplication, a cornerstone of
 * most neural networks.
 *
 * @param[in] dispatcher The ROCm dispatcher instance.
 * @param[in] matrix_a   A descriptor for matrix A.
 * @param[in] matrix_b   A descriptor for matrix B.
 * @param[in,out] matrix_c A descriptor for matrix C, which is both an input and output.
 * @param[in] params     Parameters controlling the GEMM operation (alpha, beta, transpose flags).
 *
 * @return TK_SUCCESS on a successful call to rocBLAS.
 */
TK_NODISCARD tk_error_code_t tk_rocm_tensor_gemm(
    tk_rocm_dispatcher_t* dispatcher,
    const tk_tensor_descriptor_t* matrix_a,
    const tk_tensor_descriptor_t* matrix_b,
    tk_tensor_descriptor_t* matrix_c,
    const tk_tensor_gemm_params_t* params
);

//------------------------------------------------------------------------------
// Convolution Operations (via MIOpen)
//------------------------------------------------------------------------------

/**
 * @struct tk_tensor_conv_params_t
 * @brief Parameters defining a 2D convolution operation.
 */
typedef struct {
    uint32_t stride_h;           /**< Stride of the convolution in the height dimension. */
    uint32_t stride_w;           /**< Stride of the convolution in the width dimension. */
    uint32_t pad_h;              /**< Padding applied to the height dimension. */
    uint32_t pad_w;              /**< Padding applied to the width dimension. */
    uint32_t dilation_h;         /**< Dilation factor in the height dimension. */
    uint32_t dilation_w;         /**< Dilation factor in the width dimension. */
    int groups;                  /**< Number of groups for grouped convolution. */
} tk_tensor_conv_params_t;

/**
 * @brief Performs a 2D convolution.
 *
 * This function is a wrapper around the `MIOpen` library, which automatically
 * finds the best algorithm (e.g., Winograd, FFT, direct) for the given
 * convolution parameters and hardware, delivering optimal performance.
 *
 * @param[in] dispatcher The ROCm dispatcher instance.
 * @param[in] input      A descriptor for the input tensor (NCHW layout).
 * @param[in] weights    A descriptor for the convolution weights (filter).
 * @param[in] bias       An optional descriptor for a bias tensor to be added (can be NULL).
 * @param[out] output    A descriptor for the output tensor.
 * @param[in] params     Parameters defining the convolution geometry (stride, padding, etc.).
 *
 * @return TK_SUCCESS on a successful call to MIOpen.
 */
TK_NODISCARD tk_error_code_t tk_rocm_tensor_conv2d(
    tk_rocm_dispatcher_t* dispatcher,
    const tk_tensor_descriptor_t* input,
    const tk_tensor_descriptor_t* weights,
    const tk_tensor_descriptor_t* bias,
    tk_tensor_descriptor_t* output,
    const tk_tensor_conv_params_t* params
);

//------------------------------------------------------------------------------
// Pooling Operations
//------------------------------------------------------------------------------

/**
 * @enum tk_pooling_type_t
 * @brief Enumeration of supported pooling modes.
 */
typedef enum {
    TK_POOLING_MAX = 0,
    TK_POOLING_AVERAGE,
} tk_pooling_type_t;

/**
 * @struct tk_tensor_pooling_params_t
 * @brief Parameters defining a 2D pooling operation.
 */
typedef struct {
    tk_pooling_type_t type;      /**< The type of pooling to perform (Max or Average). */
    uint32_t kernel_h;           /**< Height of the pooling window. */
    uint32_t kernel_w;           /**< Width of the pooling window. */
    uint32_t stride_h;           /**< Stride in the height dimension. */
    uint32_t stride_w;           /**< Stride in the width dimension. */
    uint32_t pad_h;              /**< Padding in the height dimension. */
    uint32_t pad_w;              /**< Padding in the width dimension. */
} tk_tensor_pooling_params_t;

/**
 * @brief Performs a 2D pooling operation on a tensor.
 *
 * @param[in] dispatcher The ROCm dispatcher instance.
 * @param[in] input      A descriptor for the input tensor.
 * @param[out] output    A descriptor for the output tensor.
 * @param[in] params     Parameters defining the pooling operation.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_rocm_tensor_pooling(
    tk_rocm_dispatcher_t* dispatcher,
    const tk_tensor_descriptor_t* input,
    tk_tensor_descriptor_t* output,
    const tk_tensor_pooling_params_t* params
);

#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_GPU_EXTENSIONS_ROCM_TK_ROCM_TENSOR_OPS_HPP
