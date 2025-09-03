/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_rocm_tensor_ops.cpp
 *
 * This file provides the concrete implementation for the high-performance tensor
 * operations defined in `tk_rocm_tensor_ops.hpp`. It combines hand-written HIP
 * kernels for element-wise operations with deep integration into AMD's specialized
 * deep learning libraries, `rocBLAS` and `MIOpen`.
 *
 * Implementation Strategy & Engineering Decisions:
 *   1.  **Hybrid Approach**: For simple, memory-bound operations like activations,
 *       custom HIP kernels are written to provide maximum flexibility and avoid
 *       library overhead. For complex, compute-bound operations like GEMM and
 *       convolutions, we delegate to `rocBLAS` and `MIOpen`, as these libraries
 *       are highly optimized by AMD for their hardware and are unbeatable for
 *       performance in these domains.
 *   2.  **Library Abstraction**: The complexity of using `rocBLAS` and `MIOpen`
 *       (e.g., handle management, descriptor creation, workspace allocation) is
 *       completely abstracted away from the caller. The public API only deals with
 *       the high-level `tk_tensor_descriptor_t`.
 *   3.  **Stateful Dispatcher Integration**: The dispatcher (`tk_rocm_dispatcher_t`)
 *       will be extended in a real-world scenario to manage library handles
 *       (`rocblas_handle`, `miopenHandle_t`), ensuring they are created once and
 *       reused, which is critical for performance. For this implementation, we
 *       create them on-demand as a simplification.
 *   4.  **Error Handling**: Every call to the HIP runtime, `rocBLAS`, or `MIOpen`
 *       is wrapped in a macro or function that checks the return status and
 *       translates it into the project's `tk_error_code_t` system, providing
 *       a unified and robust error-handling mechanism.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "tk_rocm_tensor_ops.hpp"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <miopen/miopen.h>

#include <vector> // Used for MIOpen tensor dimensions

// Helper macro for checking HIP/rocBLAS/MIOpen API calls
#define ROCM_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "ROCM Error: %s in %s at line %d\n", hipGetErrorString(err), __FILE__, __LINE__); \
            return TK_ERROR_GPU_UNKNOWN; \
        } \
    } while(0)

#define ROCBLAS_CHECK(call) \
    do { \
        rocblas_status status = call; \
        if (status != rocblas_status_success) { \
            fprintf(stderr, "rocBLAS Error: %s in %s at line %d\n", rocblas_status_to_string(status), __FILE__, __LINE__); \
            return TK_ERROR_GPU_LIBRARY_FAILED; \
        } \
    } while(0)

#define MIOPEN_CHECK(call) \
    do { \
        miopenStatus_t status = call; \
        if (status != miopenStatusSuccess) { \
            fprintf(stderr, "MIOpen Error: %s in %s at line %d\n", miopenGetErrorString(status), __FILE__, __LINE__); \
            return TK_ERROR_GPU_LIBRARY_FAILED; \
        } \
    } while(0)


//------------------------------------------------------------------------------
// Element-wise and Activation Kernels
//------------------------------------------------------------------------------

template <typename T>
__global__ void activation_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    size_t total_elements,
    tk_activation_type_t type,
    float alpha
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    T val = input[idx];
    T result;

    switch (type) {
        case TK_ACTIVATION_RELU:
            result = (val > 0) ? val : 0;
            break;
        case TK_ACTIVATION_SIGMOID:
            result = 1 / (1 + expf(-val));
            break;
        case TK_ACTIVATION_TANH:
            result = tanhf(val);
            break;
        case TK_ACTIVATION_LEAKY_RELU:
            result = (val > 0) ? val : val * alpha;
            break;
        default:
            result = val;
    }
    output[idx] = result;
}

template <typename T>
__global__ void add_kernel(
    float alpha,
    const T* __restrict__ input_a,
    float beta,
    const T* __restrict__ input_b,
    T* __restrict__ output,
    size_t total_elements
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    output[idx] = alpha * input_a[idx] + beta * input_b[idx];
}


//------------------------------------------------------------------------------
// Public API Implementations
//------------------------------------------------------------------------------

TK_NODISCARD tk_error_code_t tk_rocm_tensor_activation(
    tk_rocm_dispatcher_t* dispatcher,
    const tk_tensor_descriptor_t* input,
    tk_tensor_descriptor_t* output,
    tk_activation_type_t type,
    float alpha
) {
    if (!dispatcher || !input || !output) return TK_ERROR_INVALID_ARGUMENT;
    if (input->type != TK_TENSOR_TYPE_F32) return TK_ERROR_UNSUPPORTED_TENSOR_TYPE;

    size_t total_elements = 1;
    for (uint32_t i = 0; i < input->dimensions; ++i) {
        total_elements *= input->shape[i];
    }

    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    hipLaunchKernelGGL(
        activation_kernel<float>,
        dim3(grid_size),
        dim3(block_size),
        0, 0, // sharedMem, stream
        (const float*)((tk_gpu_buffer_internal_t*)input->buffer)->d_ptr,
        (float*)((tk_gpu_buffer_internal_t*)output->buffer)->d_ptr,
        total_elements,
        type,
        alpha
    );
    ROCM_CHECK(hipGetLastError());
    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t tk_rocm_tensor_add(
    tk_rocm_dispatcher_t* dispatcher,
    float alpha,
    const tk_tensor_descriptor_t* input_a,
    float beta,
    const tk_tensor_descriptor_t* input_b,
    tk_tensor_descriptor_t* output
) {
    if (!dispatcher || !input_a || !input_b || !output) return TK_ERROR_INVALID_ARGUMENT;
    if (input_a->type != TK_TENSOR_TYPE_F32) return TK_ERROR_UNSUPPORTED_TENSOR_TYPE;

    size_t total_elements = 1;
    for (uint32_t i = 0; i < input_a->dimensions; ++i) {
        total_elements *= input_a->shape[i];
    }
    
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    hipLaunchKernelGGL(
        add_kernel<float>,
        dim3(grid_size),
        dim3(block_size),
        0, 0,
        alpha,
        (const float*)((tk_gpu_buffer_internal_t*)input_a->buffer)->d_ptr,
        beta,
        (const float*)((tk_gpu_buffer_internal_t*)input_b->buffer)->d_ptr,
        (float*)((tk_gpu_buffer_internal_t*)output->buffer)->d_ptr,
        total_elements
    );
    ROCM_CHECK(hipGetLastError());
    return TK_SUCCESS;
}


TK_NODISCARD tk_error_code_t tk_rocm_tensor_gemm(
    tk_rocm_dispatcher_t* dispatcher,
    const tk_tensor_descriptor_t* matrix_a,
    const tk_tensor_descriptor_t* matrix_b,
    tk_tensor_descriptor_t* matrix_c,
    const tk_tensor_gemm_params_t* params
) {
    if (!dispatcher || !matrix_a || !matrix_b || !matrix_c || !params) return TK_ERROR_INVALID_ARGUMENT;
    if (matrix_a->type != TK_TENSOR_TYPE_F32) return TK_ERROR_UNSUPPORTED_TENSOR_TYPE;

    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    const int m = matrix_c->shape[0];
    const int n = matrix_c->shape[1];
    const int k = matrix_a->shape[params->trans_a ? 0 : 1];

    const float* a_ptr = (const float*)((tk_gpu_buffer_internal_t*)matrix_a->buffer)->d_ptr;
    const float* b_ptr = (const float*)((tk_gpu_buffer_internal_t*)matrix_b->buffer)->d_ptr;
    float* c_ptr = (float*)((tk_gpu_buffer_internal_t*)matrix_c->buffer)->d_ptr;
    
    const rocblas_operation trans_a = params->trans_a ? rocblas_operation_transpose : rocblas_operation_none;
    const rocblas_operation trans_b = params->trans_b ? rocblas_operation_transpose : rocblas_operation_none;

    const int lda = params->trans_a ? m : k;
    const int ldb = params->trans_b ? k : n;
    const int ldc = n;

    ROCBLAS_CHECK(rocblas_sgemm(
        handle,
        trans_a, trans_b,
        m, n, k,
        &params->alpha,
        a_ptr, lda,
        b_ptr, ldb,
        &params->beta,
        c_ptr, ldc
    ));

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t tk_rocm_tensor_conv2d(
    tk_rocm_dispatcher_t* dispatcher,
    const tk_tensor_descriptor_t* input,
    const tk_tensor_descriptor_t* weights,
    const tk_tensor_descriptor_t* bias,
    tk_tensor_descriptor_t* output,
    const tk_tensor_conv_params_t* params
) {
    if (!dispatcher || !input || !weights || !output || !params) return TK_ERROR_INVALID_ARGUMENT;
    if (input->type != TK_TENSOR_TYPE_F32) return TK_ERROR_UNSUPPORTED_TENSOR_TYPE;

    miopenHandle_t handle;
    MIOPEN_CHECK(miopenCreate(&handle));

    // Input Tensor Descriptor
    miopenTensorDescriptor_t input_desc;
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&input_desc));
    std::vector<int> in_dims = {(int)input->shape[0], (int)input->shape[1], (int)input->shape[2], (int)input->shape[3]};
    MIOPEN_CHECK(miopenSet4dTensorDescriptor(input_desc, miopenFloat, in_dims[0], in_dims[1], in_dims[2], in_dims[3]));

    // Weights Tensor Descriptor
    miopenTensorDescriptor_t weights_desc;
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&weights_desc));
    std::vector<int> w_dims = {(int)weights->shape[0], (int)weights->shape[1], (int)weights->shape[2], (int)weights->shape[3]};
    MIOPEN_CHECK(miopenSet4dTensorDescriptor(weights_desc, miopenFloat, w_dims[0], w_dims[1], w_dims[2], w_dims[3]));

    // Convolution Descriptor
    miopenConvolutionDescriptor_t conv_desc;
    MIOPEN_CHECK(miopenCreateConvolutionDescriptor(&conv_desc));
    MIOPEN_CHECK(miopenInitConvolutionDescriptor(conv_desc, miopenConvolution, params->pad_h, params->pad_w, params->stride_h, params->stride_w, params->dilation_h, params->dilation_w));

    // Output Tensor Descriptor
    miopenTensorDescriptor_t output_desc;
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&output_desc));
    std::vector<int> out_dims = {(int)output->shape[0], (int)output->shape[1], (int)output->shape[2], (int)output->shape[3]};
    MIOPEN_CHECK(miopenSet4dTensorDescriptor(output_desc, miopenFloat, out_dims[0], out_dims[1], out_dims[2], out_dims[3]));
    
    // Find best algorithm
    miopenConvAlgoPerf_t perf_result;
    int returned_algo_count;
    MIOPEN_CHECK(miopenFindConvolutionForwardAlgorithm(handle, input_desc, ((tk_gpu_buffer_internal_t*)input->buffer)->d_ptr, weights_desc, ((tk_gpu_buffer_internal_t*)weights->buffer)->d_ptr, conv_desc, output_desc, ((tk_gpu_buffer_internal_t*)output->buffer)->d_ptr, 1, &returned_algo_count, &perf_result, nullptr, 0, false));

    // Workspace
    size_t workspace_size = 0;
    MIOPEN_CHECK(miopenConvolutionForwardGetWorkSpaceSize(handle, weights_desc, input_desc, conv_desc, output_desc, &workspace_size));
    tk_gpu_buffer_t workspace_buffer = nullptr;
    void* d_workspace_ptr = nullptr;
    if (workspace_size > 0) {
        tk_error_code_t malloc_err = tk_rocm_dispatch_malloc(dispatcher, &workspace_buffer, workspace_size);
        if (malloc_err != TK_SUCCESS) {
            // Cleanup MIOpen handles before returning
            miopenDestroyTensorDescriptor(input_desc);
            miopenDestroyTensorDescriptor(weights_desc);
            miopenDestroyTensorDescriptor(output_desc);
            miopenDestroyConvolutionDescriptor(conv_desc);
            miopenDestroy(&handle);
            return malloc_err;
        }
        d_workspace_ptr = ((tk_gpu_buffer_internal_t*)workspace_buffer)->d_ptr;
    }

    // Run convolution
    float alpha = 1.0f, beta = 0.0f;
    MIOPEN_CHECK(miopenConvolutionForward(handle, &alpha, input_desc, ((tk_gpu_buffer_internal_t*)input->buffer)->d_ptr, weights_desc, ((tk_gpu_buffer_internal_t*)weights->buffer)->d_ptr, conv_desc, perf_result.fwd_algo, &beta, output_desc, ((tk_gpu_buffer_internal_t*)output->buffer)->d_ptr, d_workspace_ptr, workspace_size));

    // Cleanup
    if (workspace_buffer) {
        tk_rocm_dispatch_free(dispatcher, &workspace_buffer);
    }
    MIOPEN_CHECK(miopenDestroyTensorDescriptor(input_desc));
    MIOPEN_CHECK(miopenDestroyTensorDescriptor(weights_desc));
    MIOPEN_CHECK(miopenDestroyTensorDescriptor(output_desc));
    MIOPEN_CHECK(miopenDestroyConvolutionDescriptor(conv_desc));
    MIOPEN_CHECK(miopenDestroy(&handle));

    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t tk_rocm_tensor_pooling(
    tk_rocm_dispatcher_t* dispatcher,
    const tk_tensor_descriptor_t* input,
    tk_tensor_descriptor_t* output,
    const tk_tensor_pooling_params_t* params
) {
    if (!dispatcher || !input || !output || !params) return TK_ERROR_INVALID_ARGUMENT;
    if (input->type != TK_TENSOR_TYPE_F32) return TK_ERROR_UNSUPPORTED_TENSOR_TYPE;

    miopenHandle_t handle;
    MIOPEN_CHECK(miopenCreate(&handle));

    // Input Tensor Descriptor
    miopenTensorDescriptor_t input_desc;
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&input_desc));
    std::vector<int> in_dims = {(int)input->shape[0], (int)input->shape[1], (int)input->shape[2], (int)input->shape[3]};
    MIOPEN_CHECK(miopenSet4dTensorDescriptor(input_desc, miopenFloat, in_dims[0], in_dims[1], in_dims[2], in_dims[3]));
    
    // Output Tensor Descriptor
    miopenTensorDescriptor_t output_desc;
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&output_desc));
    std::vector<int> out_dims = {(int)output->shape[0], (int)output->shape[1], (int)output->shape[2], (int)output->shape[3]};
    MIOPEN_CHECK(miopenSet4dTensorDescriptor(output_desc, miopenFloat, out_dims[0], out_dims[1], out_dims[2], out_dims[3]));

    // Pooling Descriptor
    miopenPoolingDescriptor_t pool_desc;
    MIOPEN_CHECK(miopenCreatePoolingDescriptor(&pool_desc));
    miopenPoolingMode_t mode = (params->type == TK_POOLING_MAX) ? miopenPoolingMax : miopenPoolingAverage;
    MIOPEN_CHECK(miopenSet2dPoolingDescriptor(pool_desc, mode, params->kernel_h, params->kernel_w, params->pad_h, params->pad_w, params->stride_h, params->stride_w));

    // Run pooling forward
    float alpha = 1.0f, beta = 0.0f;
    MIOPEN_CHECK(miopenPoolingForward(
        handle, 
        pool_desc, 
        &alpha, 
        input_desc, 
        ((tk_gpu_buffer_internal_t*)input->buffer)->d_ptr, 
        &beta, 
        output_desc, 
        ((tk_gpu_buffer_internal_t*)output->buffer)->d_ptr,
        false, // do_backward
        nullptr, // workspace
        0        // workspace_size
    ));

    // Cleanup
    MIOPEN_CHECK(miopenDestroyTensorDescriptor(input_desc));
    MIOPEN_CHECK(miopenDestroyTensorDescriptor(output_desc));
    MIOPEN_CHECK(miopenDestroyPoolingDescriptor(pool_desc));
    MIOPEN_CHECK(miopenDestroy(&handle));

    return TK_SUCCESS;
}
