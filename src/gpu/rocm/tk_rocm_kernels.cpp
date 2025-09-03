/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_rocm_kernels.cpp
 *
 * This file contains the implementation of the core computational kernels for the
 * ROCm/HIP backend. Each kernel is a highly-optimized function designed to run on
 * AMD GPUs, performing a specific task within the vision processing pipeline.
 *
 * The engineering philosophy behind this implementation includes:
 *   1.  **Kernel Fusion**: Operations like resizing, normalization, and data layout
 *       conversion are fused into single kernels to minimize memory bandwidth usage
 *       and reduce kernel launch overhead, which are critical for performance.
 *   2.  **Architectural Awareness**: The choice of block sizes, memory access
 *       patterns (coalescing), and use of shared memory are designed to map
 *       efficiently to the AMD GCN/CDNA architecture.
 *   3.  **Defensive Programming**: Kernels include sanity checks (e.g., boundary
 *       checks) to prevent out-of-bounds access, ensuring stability and making
 *       debugging easier.
 *   4.  **Clarity and Maintainability**: Despite the complexity of GPU programming,
 *       the code is heavily commented to explain the "why" behind the "how,"
 *       covering algorithms, data structures, and performance trade-offs.
 *
 * Dependencies:
 *   - HIP Runtime for kernel launch and execution.
 *   - tk_rocm_kernels.hpp for the public-facing API contract.
 *   - tk_rocm_math_helpers.hpp for optimized device-side math functions.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "tk_rocm_kernels.hpp"
#include <hip/hip_runtime.h>

// Bring the ROCm math helpers into the current scope for convenience.
// This makes the kernel code cleaner and more readable.
using namespace tk::gpu::rocm::math;

//------------------------------------------------------------------------------
// Kernel Implementation: Image Pre-processing
//------------------------------------------------------------------------------

/**
 * @brief __global__ kernel for image pre-processing.
 *
 * This kernel performs a fused operation:
 * - Reads an interleaved uint8 RGB pixel from the source image.
 * - Uses bilinear interpolation to sample the image at the correct location,
 *   effectively resizing it.
 * - Converts the uint8 value to float and scales it (e.g., divides by 255.0).
 * - Normalizes the value using the provided mean and standard deviation.
 * - Writes the final float value to the output tensor in planar (CHW) format.
 *
 * Each thread in the grid processes one pixel of the *output* tensor.
 */
__global__ void preprocess_image_kernel(
    const tk_preprocess_params_t params
) {
    // Calculate the global thread ID for the output tensor (x, y coordinates).
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check: ensure the thread is within the output tensor dimensions.
    if (out_x >= params.output_width || out_y >= params.output_height) {
        return;
    }

    // --- Step 1: Map output coordinates to input coordinates ---
    // This determines which pixel to sample from the source image.
    const float in_x = (float)out_x * ((float)params.input_width / (float)params.output_width);
    const float in_y = (float)out_y * ((float)params.input_height / (float)params.output_height);

    // --- Step 2: Perform bilinear interpolation for RGB channels ---
    // We need to sample 3 channels (R, G, B). We can't use hardware samplers
    // directly on this generic buffer, so we implement it manually.

    // Get the integer and fractional parts of the input coordinates.
    const int x0 = static_cast<int>(floorf(in_x));
    const int y0 = static_cast<int>(floorf(in_y));
    const float tx = in_x - floorf(in_x);
    const float ty = in_y - floorf(in_y);

    // Get the coordinates of the 4 neighboring pixels.
    const int x1 = min(x0 + 1, (int)params.input_width - 1);
    const int y1 = min(y0 + 1, (int)params.input_height - 1);

    // Calculate indices for the 4 neighboring pixels (assuming 3 channels, RGB).
    const int idx00 = (y0 * params.input_stride_bytes) + x0 * 3;
    const int idx10 = (y0 * params.input_stride_bytes) + x1 * 3;
    const int idx01 = (y1 * params.input_stride_bytes) + x0 * 3;
    const int idx11 = (y1 * params.input_stride_bytes) + x1 * 3;

    // --- Step 3: Sample, Convert, and Interpolate ---
    // This is done per channel.
    float3 interpolated_color;

    // Red channel
    float r00 = (float)params.d_input_image[idx00 + 0];
    float r10 = (float)params.d_input_image[idx10 + 0];
    float r01 = (float)params.d_input_image[idx01 + 0];
    float r11 = (float)params.d_input_image[idx11 + 0];
    float r_interp = lerpf(lerpf(r00, r10, tx), lerpf(r01, r11, tx), ty);

    // Green channel
    float g00 = (float)params.d_input_image[idx00 + 1];
    float g10 = (float)params.d_input_image[idx10 + 1];
    float g01 = (float)params.d_input_image[idx01 + 1];
    float g11 = (float)params.d_input_image[idx11 + 1];
    float g_interp = lerpf(lerpf(g00, g10, tx), lerpf(g01, g11, tx), ty);

    // Blue channel
    float b00 = (float)params.d_input_image[idx00 + 2];
    float b10 = (float)params.d_input_image[idx10 + 2];
    float b01 = (float)params.d_input_image[idx01 + 2];
    float b11 = (float)params.d_input_image[idx11 + 2];
    float b_interp = lerpf(lerpf(b00, b10, tx), lerpf(b01, b11, tx), ty);
    
    interpolated_color = make_float3(r_interp, g_interp, b_interp);

    // --- Step 4: Scale and Normalize ---
    // output = (input * scale - mean) / std_dev
    float norm_r = ((interpolated_color.x * params.scale) - params.mean.x) / params.std_dev.x;
    float norm_g = ((interpolated_color.y * params.scale) - params.mean.y) / params.std_dev.y;
    float norm_b = ((interpolated_color.z * params.scale) - params.mean.z) / params.std_dev.z;

    // --- Step 5: Write to output tensor in planar format (CHW) ---
    const int channel_size = params.output_width * params.output_height;
    const int out_idx = out_y * params.output_width + out_x;

    params.d_output_tensor[out_idx + 0 * channel_size] = norm_r; // Red channel plane
    params.d_output_tensor[out_idx + 1 * channel_size] = norm_g; // Green channel plane
    params.d_output_tensor[out_idx + 2 * channel_size] = norm_b; // Blue channel plane
}


//------------------------------------------------------------------------------
// Kernel Implementation: Depth Map Post-processing
//------------------------------------------------------------------------------

/**
 * @brief __global__ kernel for depth map post-processing.
 *
 * This is a simple element-wise kernel that applies a scale and shift to each
 * pixel of the raw depth map.
 *
 * Each thread processes one pixel.
 */
__global__ void postprocess_depth_map_kernel(
    const tk_postprocess_depth_params_t params
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    const int idx = y * params.width + x;
    const float raw_depth = params.d_raw_depth_map[idx];

    // Apply the linear transformation.
    params.d_metric_depth_map[idx] = raw_depth * params.scale + params.shift;
}


//------------------------------------------------------------------------------
// Kernel Implementation: Depth to Point Cloud
//------------------------------------------------------------------------------

/**
 * @brief __global__ kernel to unproject a depth map to a 3D point cloud.
 *
 * Each thread reads one depth pixel and computes the corresponding 3D point
 * using the provided camera intrinsic parameters.
 */
__global__ void depth_to_point_cloud_kernel(
    const tk_depth_to_points_params_t params
) {
    const int u = blockIdx.x * blockDim.x + threadIdx.x; // Image coordinate x
    const int v = blockIdx.y * blockDim.y + threadIdx.y; // Image coordinate y

    if (u >= params.width || v >= params.height) {
        return;
    }

    const int idx = v * params.width + u;
    const float depth = params.d_metric_depth_map[idx];

    // Defensive check: if depth is non-positive, it's invalid.
    // Output a point at the origin or a specific NaN value.
    if (depth <= 0.0f) {
        params.d_point_cloud[idx] = make_float3(0.0f, 0.0f, 0.0f);
        return;
    }

    // Unprojection equations (pinhole camera model)
    const float x = (static_cast<float>(u) - params.cx) * depth / params.fx;
    const float y = (static_cast<float>(v) - params.cy) * depth / params.fy;
    const float z = depth;

    params.d_point_cloud[idx] = make_float3(x, y, z);
}


//==============================================================================
// KERNEL LAUNCHER WRAPPERS (C-Callable API)
//==============================================================================

// A helper function to calculate optimal launch parameters.
// This is a simplified version; a real-world engine might query device
// properties to determine optimal block sizes.
static void get_launch_dims(dim3& blocks, dim3& threads, uint32_t width, uint32_t height) {
    threads.x = 16;
    threads.y = 16;
    threads.z = 1;
    blocks.x = (width + threads.x - 1) / threads.x;
    blocks.y = (height + threads.y - 1) / threads.y;
    blocks.z = 1;
}

TK_NODISCARD tk_error_code_t tk_kernels_preprocess_image(
    const tk_preprocess_params_t* params, hipStream_t stream
) {
    if (!params) return TK_ERROR_INVALID_ARGUMENT;

    dim3 threads, blocks;
    get_launch_dims(blocks, threads, params->output_width, params->output_height);
    
    hipLaunchKernelGGL(
        preprocess_image_kernel,
        blocks,
        threads,
        0, // sharedMemBytes
        stream,
        *params
    );

    // Check for any asynchronous errors from the launch.
    hipError_t err = hipGetLastError();
    return (err == hipSuccess) ? TK_SUCCESS : TK_ERROR_GPU_KERNEL_LAUNCH_FAILED;
}

TK_NODISCARD tk_error_code_t tk_kernels_postprocess_depth_map(
    const tk_postprocess_depth_params_t* params, hipStream_t stream
) {
    if (!params) return TK_ERROR_INVALID_ARGUMENT;

    dim3 threads, blocks;
    get_launch_dims(blocks, threads, params->width, params->height);

    hipLaunchKernelGGL(
        postprocess_depth_map_kernel,
        blocks,
        threads,
        0,
        stream,
        *params
    );

    hipError_t err = hipGetLastError();
    return (err == hipSuccess) ? TK_SUCCESS : TK_ERROR_GPU_KERNEL_LAUNCH_FAILED;
}

TK_NODISCARD tk_error_code_t tk_kernels_depth_to_point_cloud(
    const tk_depth_to_points_params_t* params, hipStream_t stream
) {
    if (!params) return TK_ERROR_INVALID_ARGUMENT;

    dim3 threads, blocks;
    get_launch_dims(blocks, threads, params->width, params->height);

    hipLaunchKernelGGL(
        depth_to_point_cloud_kernel,
        blocks,
        threads,
        0,
        stream,
        *params
    );

    hipError_t err = hipGetLastError();
    return (err == hipSuccess) ? TK_SUCCESS : TK_ERROR_GPU_KERNEL_LAUNCH_FAILED;
}
