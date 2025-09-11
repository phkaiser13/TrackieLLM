/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_cuda_kernels.cu
 *
 * This file contains the implementation of CUDA kernels for the TrackieLLM vision
 * processing pipeline. These kernels are optimized for high-performance image
 * processing, depth map conversion, and 3D point cloud generation on NVIDIA GPUs.
 *
 * The kernels are designed with the following principles:
 *   1. Memory Coalescing: Access patterns are optimized for maximum memory bandwidth.
 *   2. Occupancy Optimization: Thread block sizes are chosen to maximize GPU occupancy.
 *   3. Minimal Divergence: Branching is minimized and structured for warp efficiency.
 *   4. Numerical Stability: Floating-point operations use CUDA intrinsics for precision.
 *
 * Dependencies:
 *   - CUDA Runtime API
 *   - tk_cuda_math_helpers.h for device-side mathematical functions
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "tk_cuda_kernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

// Define block dimensions for optimal occupancy
#define TK_CUDA_PREPROCESS_BLOCK_SIZE_X 16
#define TK_CUDA_PREPROCESS_BLOCK_SIZE_Y 16
#define TK_CUDA_DEPTH_PROCESS_BLOCK_SIZE 256
#define TK_CUDA_POINT_CLOUD_BLOCK_SIZE 256

// Error checking macro for CUDA calls within kernels
#define CUDA_KERNEL_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        return TK_ERROR_CUDA_KERNEL_FAILURE; \
    } \
} while(0)

// Device function to convert RGB to normalized float values
__device__ __forceinline__ void tk_cuda_convert_rgb_to_normalized(
    const unsigned char* input_ptr,
    float* output_r,
    float* output_g,
    float* output_b,
    const TkFloat3 mean,
    const TkFloat3 std_dev
) {
    /*
     * Convert RGB pixel values to normalized floating point values
     * Formula: (pixel_value / 255.0 - mean) / std_dev
     */
    *output_r = (static_cast<float>(input_ptr[0]) * 0.00392156862745098f - mean.x) / std_dev.x;
    *output_g = (static_cast<float>(input_ptr[1]) * 0.00392156862745098f - mean.y) / std_dev.y;
    *output_b = (static_cast<float>(input_ptr[2]) * 0.00392156862745098f - mean.z) / std_dev.z;
}

// Device function for bilinear interpolation
__device__ __forceinline__ float tk_cuda_bilinear_interpolate(
    const unsigned char* src,
    int src_width,
    int src_height,
    int src_stride,
    float x,
    float y,
    int channel
) {
    /*
     * Perform bilinear interpolation for a given pixel coordinate
     * This is used for image resizing with sub-pixel accuracy
     */
    int x0 = static_cast<int>(floorf(x));
    int y0 = static_cast<int>(floorf(y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    // Clamp coordinates to valid range
    x0 = max(0, min(x0, src_width - 1));
    x1 = max(0, min(x1, src_width - 1));
    y0 = max(0, min(y0, src_height - 1));
    y1 = max(0, min(y1, src_height - 1));
    
    // Calculate interpolation weights
    float wx = x - x0;
    float wy = y - y0;
    
    // Get pixel values at four corners
    float p00 = static_cast<float>(src[(y0 * src_stride + x0 * 3 + channel)]);
    float p01 = static_cast<float>(src[(y0 * src_stride + x1 * 3 + channel)]);
    float p10 = static_cast<float>(src[(y1 * src_stride + x0 * 3 + channel)]);
    float p11 = static_cast<float>(src[(y1 * src_stride + x1 * 3 + channel)]);
    
    // Bilinear interpolation
    float p0 = p00 * (1.0f - wx) + p01 * wx;
    float p1 = p10 * (1.0f - wx) + p11 * wx;
    return p0 * (1.0f - wy) + p1 * wy;
}

// CUDA kernel for image preprocessing
__global__ void tk_cuda_preprocess_image_kernel(
    const unsigned char* d_input_image,
    uint32_t input_width,
    uint32_t input_height,
    uint32_t input_stride_bytes,
    float* d_output_tensor,
    uint32_t output_width,
    uint32_t output_height,
    TkFloat3 mean,
    TkFloat3 std_dev
) {
    /*
     * Preprocess image kernel that performs:
     * 1. Bilinear resizing from input dimensions to output dimensions
     * 2. RGB to normalized float conversion
     * 3. Channel reordering from interleaved (HWC) to planar (CHW)
     *
     * Each thread block processes a tile of the output tensor
     * Each thread processes one output pixel
     */
    
    // Calculate global thread indices
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds
    if (out_x >= output_width || out_y >= output_height) {
        return;
    }
    
    // Calculate scaling factors for resizing
    float scale_x = static_cast<float>(input_width) / static_cast<float>(output_width);
    float scale_y = static_cast<float>(input_height) / static_cast<float>(output_height);
    
    // Map output coordinates to input coordinates with half-pixel offset
    float in_x = (static_cast<float>(out_x) + 0.5f) * scale_x - 0.5f;
    float in_y = (static_cast<float>(out_y) + 0.5f) * scale_y - 0.5f;
    
    // Clamp input coordinates to valid range
    in_x = fmaxf(0.0f, fminf(in_x, static_cast<float>(input_width - 1)));
    in_y = fmaxf(0.0f, fminf(in_y, static_cast<float>(input_height - 1)));
    
    // Perform bilinear interpolation for each channel
    float r_val = tk_cuda_bilinear_interpolate(
        d_input_image, input_width, input_height, input_stride_bytes, in_x, in_y, 0
    );
    float g_val = tk_cuda_bilinear_interpolate(
        d_input_image, input_width, input_height, input_stride_bytes, in_x, in_y, 1
    );
    float b_val = tk_cuda_bilinear_interpolate(
        d_input_image, input_width, input_height, input_stride_bytes, in_x, in_y, 2
    );
    
    // Normalize values
    float r_norm = (r_val * 0.00392156862745098f - mean.x) / std_dev.x;
    float g_norm = (g_val * 0.00392156862745098f - mean.y) / std_dev.y;
    float b_norm = (b_val * 0.00392156862745098f - mean.z) / std_dev.z;
    
    // Calculate output tensor indices for planar format (CHW)
    int r_index = out_y * output_width + out_x;
    int g_index = r_index + output_width * output_height;
    int b_index = g_index + output_width * output_height;
    
    // Write normalized values to output tensor
    d_output_tensor[r_index] = r_norm;
    d_output_tensor[g_index] = g_norm;
    d_output_tensor[b_index] = b_norm;
}

// CUDA kernel for depth map post-processing
__global__ void tk_cuda_postprocess_depth_kernel(
    const float* d_raw_depth_map,
    float* d_metric_depth_map,
    uint32_t width,
    uint32_t height,
    float scale,
    float shift
) {
    /*
     * Convert raw depth values to metric depth values
     * Formula: metric_depth = raw_depth * scale + shift
     *
     * This kernel processes one pixel per thread
     */
    
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (idx >= width * height) {
        return;
    }
    
    // Apply linear transformation to convert to metric units
    float raw_value = d_raw_depth_map[idx];
    d_metric_depth_map[idx] = raw_value * scale + shift;
}

// CUDA kernel for converting depth map to point cloud
__global__ void tk_cuda_depth_to_point_cloud_kernel(
    const float* d_metric_depth_map,
    TkFloat3* d_point_cloud,
    uint32_t width,
    uint32_t height,
    float fx,
    float fy,
    float cx,
    float cy
) {
    /*
     * Convert depth map to 3D point cloud using camera intrinsics
     * For each pixel (u,v) with depth d:
     *   x = (u - cx) * d / fx
     *   y = (v - cy) * d / fy
     *   z = d
     *
     * This kernel processes one pixel per thread
     */
    
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (idx >= width * height) {
        return;
    }
    
    // Calculate 2D coordinates from linear index
    int y = idx / width;
    int x = idx % width;
    
    // Get depth value for this pixel
    float depth = d_metric_depth_map[idx];
    
    // Handle invalid depth values
    if (depth <= 0.0f || depth > 1000.0f) { // Arbitrary max depth threshold
        d_point_cloud[idx] = make_float3(0.0f, 0.0f, 0.0f);
        return;
    }
    
    // Calculate 3D coordinates using camera intrinsics
    float point_x = (static_cast<float>(x) - cx) * depth / fx;
    float point_y = (static_cast<float>(y) - cy) * depth / fy;
    float point_z = depth;
    
    // Store point in output array
    d_point_cloud[idx] = make_float3(point_x, point_y, point_z);
}

// Host function to launch image preprocessing kernel
TK_NODISCARD tk_error_code_t tk_kernels_preprocess_image(
    const tk_preprocess_params_t* params,
    cudaStream_t stream
) {
    /*
     * Launch the image preprocessing kernel with appropriate grid and block dimensions
     *
     * This function:
     * 1. Validates input parameters
     * 2. Calculates optimal grid dimensions
     * 3. Launches the kernel on the specified stream
     * 4. Checks for launch errors
     */
    
    // Validate input parameters
    if (!params || !params->d_input_image || !params->d_output_tensor) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    if (params->input_width == 0 || params->input_height == 0 ||
        params->output_width == 0 || params->output_height == 0) {
        return TK_ERROR_INVALID_DIMENSIONS;
    }
    
    // Calculate grid dimensions for 2D block layout
    dim3 block_dim(TK_CUDA_PREPROCESS_BLOCK_SIZE_X, TK_CUDA_PREPROCESS_BLOCK_SIZE_Y);
    dim3 grid_dim(
        (params->output_width + block_dim.x - 1) / block_dim.x,
        (params->output_height + block_dim.y - 1) / block_dim.y
    );
    
    // Launch kernel
    tk_cuda_preprocess_image_kernel<<<grid_dim, block_dim, 0, stream>>>(
        params->d_input_image,
        params->input_width,
        params->input_height,
        params->input_stride_bytes,
        params->d_output_tensor,
        params->output_width,
        params->output_height,
        params->mean,
        params->std_dev
    );
    
    // Check for kernel launch errors
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        return TK_ERROR_CUDA_KERNEL_LAUNCH_FAILURE;
    }
    
    return TK_SUCCESS;
}

// -----------------------------------------------------------------------------
// Softmax Kernel Implementation
// -----------------------------------------------------------------------------

// A numerically stable softmax kernel. Each thread block processes one row.
// This requires the number of columns to be less than or equal to the max
// thread block size (e.g., 1024).
__global__ void tk_cuda_softmax_kernel(
    const float* d_input,
    float* d_output,
    unsigned int num_cols
) {
    // Each block processes one row of the input tensor.
    // The row index is given by the block index.
    const unsigned int row_idx = blockIdx.x;
    const unsigned int thread_idx = threadIdx.x;

    // A shared memory buffer for the reduction operations within the block.
    extern __shared__ float s_data[];

    // 1. Find the maximum value in the row for numerical stability.
    // Each thread loads one value from global memory into shared memory.
    s_data[thread_idx] = d_input[row_idx * num_cols + thread_idx];
    __syncthreads();

    // Perform reduction in shared memory to find the max value.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (thread_idx < s) {
            s_data[thread_idx] = fmaxf(s_data[thread_idx], s_data[thread_idx + s]);
        }
        __syncthreads();
    }
    float max_val = s_data[0];
    __syncthreads();

    // 2. Compute exp(x - max) and the sum of exponents.
    // Each thread computes its own exponentiated value.
    float exp_val = expf(d_input[row_idx * num_cols + thread_idx] - max_val);
    s_data[thread_idx] = exp_val;
    __syncthreads();

    // Perform reduction in shared memory to find the sum of exponents.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (thread_idx < s) {
            s_data[thread_idx] += s_data[thread_idx + s];
        }
        __syncthreads();
    }
    float sum_exp = s_data[0];
    __syncthreads();

    // 3. Divide each exponentiated value by the sum.
    d_output[row_idx * num_cols + thread_idx] = exp_val / sum_exp;
}


// Host function to launch the softmax kernel.
TK_NODISCARD tk_error_code_t tk_kernels_softmax(
    const tk_softmax_params_t* params,
    cudaStream_t stream
) {
    if (!params || !params->d_input_tensor || !params->d_output_tensor) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    if (params->num_cols > 1024) {
        // This simple kernel requires the row size to fit in a block.
        return TK_ERROR_NOT_IMPLEMENTED;
    }

    // The grid dimension is the number of rows.
    dim3 grid_dim(params->num_rows);
    // The block dimension is the number of columns (size of the softmax dimension).
    dim3 block_dim(params->num_cols);
    // Shared memory size is one float per thread in the block.
    size_t shared_mem_size = params->num_cols * sizeof(float);

    tk_cuda_softmax_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(
        (const float*)params->d_input_tensor,
        (float*)params->d_output_tensor,
        params->num_cols
    );

    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        return TK_ERROR_CUDA_KERNEL_LAUNCH_FAILURE;
    }

    return TK_SUCCESS;
}

// Host function to launch depth map post-processing kernel
TK_NODISCARD tk_error_code_t tk_kernels_postprocess_depth_map(
    const tk_postprocess_depth_params_t* params,
    cudaStream_t stream
) {
    /*
     * Launch the depth map post-processing kernel
     *
     * This function:
     * 1. Validates input parameters
     * 2. Calculates optimal grid dimensions
     * 3. Launches the kernel on the specified stream
     * 4. Checks for launch errors
     */
    
    // Validate input parameters
    if (!params || !params->d_raw_depth_map || !params->d_metric_depth_map) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    if (params->width == 0 || params->height == 0) {
        return TK_ERROR_INVALID_DIMENSIONS;
    }
    
    // Calculate total number of pixels
    size_t total_pixels = static_cast<size_t>(params->width) * static_cast<size_t>(params->height);
    
    // Calculate grid dimensions for 1D block layout
    int block_size = TK_CUDA_DEPTH_PROCESS_BLOCK_SIZE;
    int grid_size = (total_pixels + block_size - 1) / block_size;
    
    // Launch kernel
    tk_cuda_postprocess_depth_kernel<<<grid_size, block_size, 0, stream>>>(
        params->d_raw_depth_map,
        params->d_metric_depth_map,
        params->width,
        params->height,
        params->scale,
        params->shift
    );
    
    // Check for kernel launch errors
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        return TK_ERROR_CUDA_KERNEL_LAUNCH_FAILURE;
    }
    
    return TK_SUCCESS;
}

// Host function to launch depth to point cloud kernel
TK_NODISCARD tk_error_code_t tk_kernels_depth_to_point_cloud(
    const tk_depth_to_points_params_t* params,
    cudaStream_t stream
) {
    /*
     * Launch the depth to point cloud conversion kernel
     *
     * This function:
     * 1. Validates input parameters
     * 2. Calculates optimal grid dimensions
     * 3. Launches the kernel on the specified stream
     * 4. Checks for launch errors
     */
    
    // Validate input parameters
    if (!params || !params->d_metric_depth_map || !params->d_point_cloud) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    if (params->width == 0 || params->height == 0) {
        return TK_ERROR_INVALID_DIMENSIONS;
    }
    
    // Validate camera intrinsics
    if (params->fx <= 0.0f || params->fy <= 0.0f) {
        return TK_ERROR_INVALID_CAMERA_PARAMETERS;
    }
    
    // Calculate total number of pixels
    size_t total_pixels = static_cast<size_t>(params->width) * static_cast<size_t>(params->height);
    
    // Calculate grid dimensions for 1D block layout
    int block_size = TK_CUDA_POINT_CLOUD_BLOCK_SIZE;
    int grid_size = (total_pixels + block_size - 1) / block_size;
    
    // Launch kernel
    tk_cuda_depth_to_point_cloud_kernel<<<grid_size, block_size, 0, stream>>>(
        params->d_metric_depth_map,
        params->d_point_cloud,
        params->width,
        params->height,
        params->fx,
        params->fy,
        params->cx,
        params->cy
    );
    
    // Check for kernel launch errors
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        return TK_ERROR_CUDA_KERNEL_LAUNCH_FAILURE;
    }
    
    return TK_SUCCESS;
}
