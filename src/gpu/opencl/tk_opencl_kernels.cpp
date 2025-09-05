/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_opencl_kernels.cpp
*
* This file implements the C-callable kernel dispatch wrappers for the OpenCL
* backend. These functions are responsible for setting kernel arguments and
* enqueuing the kernels for execution.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#include "tk_opencl_kernels.h"
#include "tk_opencl_dispatch.h" // For internal struct definitions
#include "tk_opencl_helpers.h"

// Placeholder for the internal dispatcher struct definition.
struct tk_opencl_dispatcher_s {
    cl_command_queue command_queue;
    cl_kernel preprocess_kernel;
    cl_kernel depth_to_points_kernel;
    // ... other OpenCL context info
};

// OpenCL buffer is just a cl_mem object, so we can cast directly.
typedef cl_mem tk_gpu_buffer_s;

tk_error_code_t tk_opencl_kernel_preprocess_image(
    struct tk_opencl_dispatcher_s* dispatcher,
    const tk_preprocess_params_t* params)
{
    cl_int err;
    cl_kernel kernel = dispatcher->preprocess_kernel;

    // Set kernel arguments
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), params->d_input_image);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), params->d_output_tensor);
    err |= clSetKernelArg(kernel, 2, sizeof(params->input_width), &params->input_width);
    err |= clSetKernelArg(kernel, 3, sizeof(params->input_height), &params->input_height);
    err |= clSetKernelArg(kernel, 4, sizeof(params->output_width), &params->output_width);
    err |= clSetKernelArg(kernel, 5, sizeof(params->output_height), &params->output_height);
    err |= clSetKernelArg(kernel, 6, sizeof(params->mean), &params->mean);
    err |= clSetKernelArg(kernel, 7, sizeof(params->std_dev), &params->std_dev);
    if (err != CL_SUCCESS) return tk::gpu::opencl::translate_cl_error(err);

    // Define the global and local work sizes
    size_t global_work_size[2] = { params->output_width, params->output_height };
    // size_t local_work_size[2] = { 16, 16 }; // Can be tuned

    // Enqueue the kernel
    err = clEnqueueNDRangeKernel(dispatcher->command_queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    return tk::gpu::opencl::translate_cl_error(err);
}

tk_error_code_t tk_opencl_kernel_depth_to_point_cloud(
    struct tk_opencl_dispatcher_s* dispatcher,
    const tk_depth_to_points_params_t* params)
{
    cl_int err;
    cl_kernel kernel = dispatcher->depth_to_points_kernel;

    // Set kernel arguments
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), params->d_metric_depth_map);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), params->d_point_cloud);
    err |= clSetKernelArg(kernel, 2, sizeof(params->width), &params->width);
    err |= clSetKernelArg(kernel, 3, sizeof(params->height), &params->height);
    err |= clSetKernelArg(kernel, 4, sizeof(params->fx), &params->fx);
    err |= clSetKernelArg(kernel, 5, sizeof(params->fy), &params->fy);
    err |= clSetKernelArg(kernel, 6, sizeof(params->cx), &params->cx);
    err |= clSetKernelArg(kernel, 7, sizeof(params->cy), &params->cy);
    if (err != CL_SUCCESS) return tk::gpu::opencl::translate_cl_error(err);

    // Define the global work size
    size_t global_work_size[2] = { params->width, params->height };

    // Enqueue the kernel
    err = clEnqueueNDRangeKernel(dispatcher->command_queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    return tk::gpu::opencl::translate_cl_error(err);
}
