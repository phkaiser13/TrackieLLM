/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_cuda_tensor_ops.h
 *
 * This header file defines the public API for high-performance CUDA-accelerated
 * tensor operations used in the TrackieLLM neural network inference pipeline.
 * These functions provide optimized implementations for common deep learning
 * operations such as matrix multiplication, convolution, activation functions,
 * and normalization that are essential for running AI models efficiently on
 * embedded GPU hardware.
 *
 * The design philosophy emphasizes:
 *   1. Performance: Operations are implemented using optimized CUDA kernels
 *      with cuBLAS integration, shared memory utilization, and memory coalescing.
 *   2. Precision: Support for both FP32 and FP16 data types with automatic
 *      conversion and mixed-precision capabilities for optimal accuracy/performance.
 *   3. Integration: Seamless integration with the existing CUDA dispatcher and
 *      buffer management system for zero-copy operations and efficient memory
 *      management.
 *   4. Scalability: Support for batched operations and multi-dimensional tensors
 *      with automatic broadcasting and dimension handling.
 *   5. Robustness: Comprehensive error handling and input validation to prevent
 *      GPU memory corruption and ensure system stability.
 *
 * These operations are specifically optimized for real-time inference of
 * computer vision models (YOLO, MiDaS) and language models (Mistral-7B) on
 * resource-constrained embedded devices.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#ifndef TRACKIELLM_GPU_EXTENSIONS_CUDA_TK_CUDA_TENSOR_OPS_H
#define TRACKIELLM_GPU_EXTENSIONS_CUDA_TK_CUDA_TENSOR_OPS_H

#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "gpu/cuda/tk_cuda_dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Tensor Format Definitions
//------------------------------------------------------------------------------

/**
 * @enum tk_tensor_data_type_t
 * @brief Enumeration of supported tensor data types.
 */
typedef enum {
    TK_TENSOR_TYPE_F32 = 0,      /**< 32-bit floating point */
    TK_TENSOR_TYPE_F16,          /**< 16-bit floating point */
    TK_TENSOR_TYPE_I32,          /**< 32-bit signed integer */
    TK_TENSOR_TYPE_I8,           /**< 8-bit signed integer */
    TK_TENSOR_TYPE_U8            /**< 8-bit unsigned integer */
} tk_tensor_data_type_t;

/**
 * @enum tk_tensor_layout_t
 * @brief Enumeration of supported tensor memory layouts.
 */
typedef enum {
    TK_TENSOR_LAYOUT_NCHW = 0,   /**< Batch, Channel, Height, Width */
    TK_TENSOR_LAYOUT_NHWC,       /**< Batch, Height, Width, Channel */
    TK_TENSOR_LAYOUT_NC,         /**< Batch, Channel (for 1D tensors) */
    TK_TENSOR_LAYOUT_NCW         /**< Batch, Channel, Width (for 2D tensors) */
} tk_tensor_layout_t;

/**
 * @struct tk_tensor_descriptor_t
 * @brief Descriptor for tensor properties and GPU buffer handle.
 */
typedef struct {
    tk_gpu_buffer_t buffer;      /**< GPU buffer handle containing tensor data */
    tk_tensor_data_type_t type;  /**< Data type of tensor elements */
    tk_tensor_layout_t layout;   /**< Memory layout of tensor */
    uint32_t dimensions;         /**< Number of dimensions */
    uint32_t shape[8];           /**< Size of each dimension (up to 8D) */
    size_t stride[8];            /**< Stride for each dimension */
    size_t data_size_bytes;      /**< Total size of tensor data in bytes */
    uint32_t batch_size;         /**< Batch size for batched operations */
} tk_tensor_descriptor_t;

//------------------------------------------------------------------------------
// Basic Tensor Operations
//------------------------------------------------------------------------------

/**
 * @struct tk_tensor_elementwise_params_t
 * @brief Parameters for element-wise tensor operations.
 */
typedef struct {
    float alpha;                 /**< Scaling factor for first operand */
    float beta;                  /**< Scaling factor for second operand */
    float gamma;                 /**< Constant offset for operation */
    int operation;               /**< Operation type (0=add, 1=sub, 2=mul, 3=div) */
} tk_tensor_elementwise_params_t;

/**
 * @brief Perform element-wise operation on two tensors.
 *
 * This function performs element-wise operations such as addition, subtraction,
 * multiplication, and division on tensors with automatic broadcasting support.
 *
 * @param[in] dispatcher The CUDA dispatcher instance.
 * @param[in] input_a First input tensor descriptor.
 * @param[in] input_b Second input tensor descriptor.
 * @param[out] output Output tensor descriptor.
 * @param[in] params Element-wise operation parameters.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_cuda_tensor_elementwise_operation(
    tk_cuda_dispatcher_t* dispatcher,
    const tk_tensor_descriptor_t* input_a,
    const tk_tensor_descriptor_t* input_b,
    tk_tensor_descriptor_t* output,
    const tk_tensor_elementwise_params_t* params
);

/**
 * @brief Apply activation function to tensor elements.
 *
 * This function applies non-linear activation functions such as ReLU, sigmoid,
 * tanh, and others to tensor elements in parallel.
 *
 * @param[in] dispatcher The CUDA dispatcher instance.
 * @param[in] input Input tensor descriptor.
 * @param[out] output Output tensor descriptor.
 * @param[in] activation_type Type of activation function to apply.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_cuda_tensor_activation(
    tk_cuda_dispatcher_t* dispatcher,
    const tk_tensor_descriptor_t* input,
    tk_tensor_descriptor_t* output,
    int activation_type // 0=ReLU, 1=Sigmoid, 2=Tanh, 3=Softmax
);

/**
 * @brief Transpose tensor dimensions.
 *
 * This function transposes tensor dimensions according to a permutation array.
 * Supports arbitrary dimension reordering for flexible tensor manipulation.
 *
 * @param[in] dispatcher The CUDA dispatcher instance.
 * @param[in] input Input tensor descriptor.
 * @param[out] output Output tensor descriptor.
 * @param[in] perm Permutation array specifying new dimension order.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_cuda_tensor_transpose(
    tk_cuda_dispatcher_t* dispatcher,
    const tk_tensor_descriptor_t* input,
    tk_tensor_descriptor_t* output,
    const uint32_t* perm
);

//------------------------------------------------------------------------------
// Linear Algebra Operations
//------------------------------------------------------------------------------

/**
 * @struct tk_tensor_gemm_params_t
 * @brief Parameters for General Matrix Multiply (GEMM) operations.
 */
typedef struct {
    float alpha;                 /**< Scaling factor for A*B */
    float beta;                  /**< Scaling factor for C */
    int trans_a;                 /**< Transpose matrix A (0=no, 1=yes) */
    int trans_b;                 /**< Transpose matrix B (0=no, 1=yes) */
    int use_tensor_cores;        /**< Enable Tensor Core acceleration (if available) */
} tk_tensor_gemm_params_t;

/**
 * @brief Perform General Matrix Multiply (GEMM) operation.
 *
 * This function performs matrix multiplication C = alpha * A * B + beta * C
 * using optimized cuBLAS routines with support for mixed precision and
 * Tensor Core acceleration.
 *
 * @param[in] dispatcher The CUDA dispatcher instance.
 * @param[in] matrix_a First input matrix tensor.
 * @param[in] matrix_b Second input matrix tensor.
 * @param[in,out] matrix_c Output/input matrix tensor (accumulation).
 * @param[in] params GEMM operation parameters.
 *
 * @return TK_SUCCESS on successful operation.
 */
TK_NODISCARD tk_error_code_t tk_cuda_tensor_gemm(
    tk_cuda_dispatcher_t* dispatcher,
    const tk_tensor_descriptor_t* matrix_a,
    const tk_tensor_descriptor_t* matrix_b,
    tk_tensor_descriptor_t* matrix_c,
    const tk_tensor_gemm_params_t* params
);

/**
 * @brief Perform batched matrix multiplication.
 *
 * This function performs batched matrix multiplication for processing
 * multiple matrix pairs simultaneously, useful for batched inference.
 *
 * @param[in] dispatcher The CUDA dispatcher instance.
 * @param[in] batch_a Array of first input matrix tensors.
 * @param[in] batch_b Array of second input matrix tensors.
 * @param[out] batch_c Array of output matrix tensors.
 * @param[in] batch_count Number of matrix pairs to process.
 * @param[in] params GEMM operation parameters.
 *
 * @return TK_SUCCESS on successful operation.
 */
TK_NODISCARD tk_error_code_t tk_cuda_tensor_batch_gemm(
    tk_cuda_dispatcher_t* dispatcher,
    const tk_tensor_descriptor_t* const* batch_a,
    const tk_tensor_descriptor_t* const* batch_b,
    tk_tensor_descriptor_t* const* batch_c,
    uint32_t batch_count,
    const tk_tensor_gemm_params_t* params
);

//------------------------------------------------------------------------------
// Convolution Operations
//------------------------------------------------------------------------------

/**
 * @struct tk_tensor_conv_params_t
 * @brief Parameters for convolution operations.
 */
typedef struct {
    uint32_t kernel_h;           /**< Kernel height */
    uint32_t kernel_w;           /**< Kernel width */
    uint32_t stride_h;           /**< Stride in height dimension */
    uint32_t stride_w;           /**< Stride in width dimension */
    uint32_t pad_h;              /**< Padding in height dimension */
    uint32_t pad_w;              /**< Padding in width dimension */
    uint32_t dilation_h;         /**< Dilation in height dimension */
    uint32_t dilation_w;         /**< Dilation in width dimension */
    int groups;                  /**< Number of convolution groups */
    int use_winograd;            /**< Enable Winograd algorithm optimization */
} tk_tensor_conv_params_t;

/**
 * @brief Perform 2D convolution operation.
 *
 * This function performs 2D convolution with support for padding, striding,
 * dilation, and grouped convolutions. Optimized for computer vision models.
 *
 * @param[in] dispatcher The CUDA dispatcher instance.
 * @param[in] input Input tensor descriptor (NCHW layout).
 * @param[in] weights Weight tensor descriptor.
 * @param[in] bias Bias tensor descriptor (optional, can be NULL).
 * @param[out] output Output tensor descriptor.
 * @param[in] params Convolution parameters.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_cuda_tensor_conv2d(
    tk_cuda_dispatcher_t* dispatcher,
    const tk_tensor_descriptor_t* input,
    const tk_tensor_descriptor_t* weights,
    const tk_tensor_descriptor_t* bias,
    tk_tensor_descriptor_t* output,
    const tk_tensor_conv_params_t* params
);

/**
 * @brief Perform depthwise separable convolution.
 *
 * This function performs depthwise separable convolution which is more
 * efficient than standard convolution for mobile and embedded models.
 *
 * @param[in] dispatcher The CUDA dispatcher instance.
 * @param[in] input Input tensor descriptor.
 * @param[in] depthwise_weights Depthwise convolution weights.
 * @param[in] pointwise_weights Pointwise convolution weights.
 * @param[in] bias Bias tensor descriptor (optional).
 * @param[out] output Output tensor descriptor.
 * @param[in] params Convolution parameters.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_cuda_tensor_depthwise_conv2d(
    tk_cuda_dispatcher_t* dispatcher,
    const tk_tensor_descriptor_t* input,
    const tk_tensor_descriptor_t* depthwise_weights,
    const tk_tensor_descriptor_t* pointwise_weights,
    const tk_tensor_descriptor_t* bias,
    tk_tensor_descriptor_t* output,
    const tk_tensor_conv_params_t* params
);

//------------------------------------------------------------------------------
// Pooling Operations
//------------------------------------------------------------------------------

/**
 * @enum tk_pooling_type_t
 * @brief Enumeration of pooling operation types.
 */
typedef enum {
    TK_POOLING_MAX = 0,          /**< Max pooling */
    TK_POOLING_AVERAGE,          /**< Average pooling */
    TK_POOLING_ADAPTIVE_MAX,     /**< Adaptive max pooling */
    TK_POOLING_ADAPTIVE_AVERAGE  /**< Adaptive average pooling */
} tk_pooling_type_t;

/**
 * @struct tk_tensor_pooling_params_t
 * @brief Parameters for pooling operations.
 */
typedef struct {
    tk_pooling_type_t type;      /**< Type of pooling operation */
    uint32_t kernel_h;           /**< Pooling kernel height */
    uint32_t kernel_w;           /**< Pooling kernel width */
    uint32_t stride_h;           /**< Stride in height dimension */
    uint32_t stride_w;           /**< Stride in width dimension */
    uint32_t pad_h;              /**< Padding in height dimension */
    uint32_t pad_w;              /**< Padding in width dimension */
} tk_tensor_pooling_params_t;

/**
 * @brief Perform pooling operation on tensor.
 *
 * This function performs spatial pooling operations such as max pooling and
 * average pooling for feature map downsampling.
 *
 * @param[in] dispatcher The CUDA dispatcher instance.
 * @param[in] input Input tensor descriptor.
 * @param[out] output Output tensor descriptor.
 * @param[in] params Pooling parameters.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_cuda_tensor_pooling(
    tk_cuda_dispatcher_t* dispatcher,
    const tk_tensor_descriptor_t* input,
    tk_tensor_descriptor_t* output,
    const tk_tensor_pooling_params_t* params
);

//------------------------------------------------------------------------------
// Normalization Operations
//------------------------------------------------------------------------------

/**
 * @struct tk_tensor_norm_params_t
 * @brief Parameters for normalization operations.
 */
typedef struct {
    float epsilon;               /**< Small constant to prevent division by zero */
    int across_channels;         /**< Normalize across channels (1) or per channel (0) */
    int normalize_variance;      /**< Normalize variance (1) or just mean (0) */
} tk_tensor_norm_params_t;

/**
 * @brief Perform batch normalization on tensor.
 *
 * This function applies batch normalization using pre-computed mean and
 * variance statistics for inference-time normalization.
 *
 * @param[in] dispatcher The CUDA dispatcher instance.
 * @param[in] input Input tensor descriptor.
 * @param[in] scale Scale parameter tensor (gamma).
 * @param[in] bias Bias parameter tensor (beta).
 * @param[in] mean Mean tensor for normalization.
 * @param[in] variance Variance tensor for normalization.
 * @param[out] output Output tensor descriptor.
 * @param[in] params Normalization parameters.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_cuda_tensor_batch_norm(
    tk_cuda_dispatcher_t* dispatcher,
    const tk_tensor_descriptor_t* input,
    const tk_tensor_descriptor_t* scale,
    const tk_tensor_descriptor_t* bias,
    const tk_tensor_descriptor_t* mean,
    const tk_tensor_descriptor_t* variance,
    tk_tensor_descriptor_t* output,
    const tk_tensor_norm_params_t* params
);

/**
 * @brief Perform layer normalization on tensor.
 *
 * This function applies layer normalization across feature dimensions,
 * commonly used in transformer models and recurrent networks.
 *
 * @param[in] dispatcher The CUDA dispatcher instance.
 * @param[in] input Input tensor descriptor.
 * @param[in] scale Scale parameter tensor.
 * @param[in] bias Bias parameter tensor.
 * @param[out] output Output tensor descriptor.
 * @param[in] params Normalization parameters.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_cuda_tensor_layer_norm(
    tk_cuda_dispatcher_t* dispatcher,
    const tk_tensor_descriptor_t* input,
    const tk_tensor_descriptor_t* scale,
    const tk_tensor_descriptor_t* bias,
    tk_tensor_descriptor_t* output,
    const tk_tensor_norm_params_t* params
);

//------------------------------------------------------------------------------
// Reduction Operations
//------------------------------------------------------------------------------

/**
 * @enum tk_reduction_type_t
 * @brief Enumeration of reduction operation types.
 */
typedef enum {
    TK_REDUCTION_SUM = 0,        /**< Sum reduction */
    TK_REDUCTION_MEAN,           /**< Mean reduction */
    TK_REDUCTION_MAX,            /**< Maximum reduction */
    TK_REDUCTION_MIN,            /**< Minimum reduction */
    TK_REDUCTION_ARGMAX,         /**< Argmax reduction */
    TK_REDUCTION_ARGMIN          /**< Argmin reduction */
} tk_reduction_type_t;

/**
 * @struct tk_tensor_reduction_params_t
 * @brief Parameters for reduction operations.
 */
typedef struct {
    tk_reduction_type_t type;    /**< Type of reduction operation */
    uint32_t* axes;              /**< Array of axes to reduce */
    uint32_t num_axes;           /**< Number of axes to reduce */
    int keep_dims;               /**< Keep reduced dimensions (1) or remove (0) */
} tk_tensor_reduction_params_t;

/**
 * @brief Perform reduction operation on tensor.
 *
 * This function performs reduction operations such as sum, mean, max, min
 * along specified axes with optional dimension preservation.
 *
 * @param[in] dispatcher The CUDA dispatcher instance.
 * @param[in] input Input tensor descriptor.
 * @param[out] output Output tensor descriptor.
 * @param[in] params Reduction parameters.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_cuda_tensor_reduction(
    tk_cuda_dispatcher_t* dispatcher,
    const tk_tensor_descriptor_t* input,
    tk_tensor_descriptor_t* output,
    const tk_tensor_reduction_params_t* params
);

//------------------------------------------------------------------------------
// Specialized Operations for Computer Vision
//------------------------------------------------------------------------------

/**
 * @struct tk_tensor_upsample_params_t
 * @brief Parameters for upsampling operations.
 */
typedef struct {
    float scale_h;               /**< Scale factor for height dimension */
    float scale_w;               /**< Scale factor for width dimension */
    int mode;                    /**< Interpolation mode (0=nearest, 1=bilinear) */
} tk_tensor_upsample_params_t;

/**
 * @brief Perform upsampling operation on tensor.
 *
 * This function performs spatial upsampling for tasks such as image
 * super-resolution and segmentation mask upsampling.
 *
 * @param[in] dispatcher The CUDA dispatcher instance.
 * @param[in] input Input tensor descriptor.
 * @param[out] output Output tensor descriptor.
 * @param[in] params Upsampling parameters.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_cuda_tensor_upsample(
    tk_cuda_dispatcher_t* dispatcher,
    const tk_tensor_descriptor_t* input,
    tk_tensor_descriptor_t* output,
    const tk_tensor_upsample_params_t* params
);

/**
 * @brief Perform non-maximum suppression on detection results.
 *
 * This function filters overlapping bounding boxes by removing those
 * with lower confidence scores that have high IoU with higher scoring boxes.
 *
 * @param[in] dispatcher The CUDA dispatcher instance.
 * @param[in] boxes Input tensor containing bounding box coordinates.
 * @param[in] scores Input tensor containing detection scores.
 * @param[out] indices Output buffer for selected box indices.
 * @param[out] num_detections Output pointer for number of selected detections.
 * @param[in] max_detections Maximum number of detections to return.
 * @param[in] iou_threshold IoU threshold for suppression.
 * @param[in] score_threshold Minimum score threshold for detections.
 *
 * @return TK_SUCCESS on successful kernel launch.
 */
TK_NODISCARD tk_error_code_t tk_cuda_tensor_nms(
    tk_cuda_dispatcher_t* dispatcher,
    const tk_tensor_descriptor_t* boxes,
    const tk_tensor_descriptor_t* scores,
    tk_gpu_buffer_t indices,
    uint32_t* num_detections,
    uint32_t max_detections,
    float iou_threshold,
    float score_threshold
);

#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_GPU_EXTENSIONS_CUDA_TK_CUDA_TENSOR_OPS_H
