/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_opencl_dispatch.cpp
*
* This file implements the OpenCL Dispatcher. It orchestrates the setup of
* the OpenCL environment and manages resources for compute tasks.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#include "tk_opencl_dispatch.h"
#include "tk_opencl_helpers.h"
#include "tk_opencl_kernels.h"

#include <vector>

// --- Internal Struct Definitions ---

// For OpenCL, the tk_gpu_buffer_s is just a cl_mem object.
// We can simply use a type alias.
typedef cl_mem tk_gpu_buffer_s;

struct tk_opencl_dispatcher_s {
    cl_context context;
    cl_command_queue command_queue;
    cl_program program; // Assume one program for all kernels for simplicity
    cl_kernel preprocess_kernel;
    cl_kernel depth_to_points_kernel;
    tk_opencl_dispatcher_config_t config;
};

// --- Public API Implementation ---

TK_NODISCARD tk_error_code_t tk_opencl_dispatch_create(tk_opencl_dispatcher_t** out_dispatcher, const tk_opencl_dispatcher_config_t* config) {
    tk_opencl_dispatcher_s* dispatcher = new (std::nothrow) tk_opencl_dispatcher_s();
    if (!dispatcher) return TK_ERROR_OUT_OF_MEMORY;
    dispatcher->config = *config;

    cl_int err;
    
    // 1. Get Platform and Device
    std::vector<cl_platform_id> platforms;
    // ... code to get all platforms ...
    cl_platform_id platform = platforms[config->platform_id];

    std::vector<cl_device_id> devices;
    // ... code to get all devices for the platform ...
    cl_device_id device = devices[config->device_id];
    
    // 2. Create Context
    dispatcher->context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) return tk::gpu::opencl::translate_cl_error(err);

    // 3. Create Command Queue
    dispatcher->command_queue = clCreateCommandQueue(dispatcher->context, device, 0, &err);
    if (err != CL_SUCCESS) return tk::gpu::opencl::translate_cl_error(err);

    // 4. Create and Build Program
    // This assumes all kernels are in one file, which is a simplification.
    // A real implementation might manage multiple cl_programs.
    tk_error_code_t tk_err = tk::gpu::opencl::create_and_build_program(dispatcher->context, device, "kernels.cl", &dispatcher->program);
    if (tk_err != TK_SUCCESS) return tk_err;

    // 5. Create Kernels
    dispatcher->preprocess_kernel = clCreateKernel(dispatcher->program, "preprocess_image", &err);
    if (err != CL_SUCCESS) return tk::gpu::opencl::translate_cl_error(err);
    
    dispatcher->depth_to_points_kernel = clCreateKernel(dispatcher->program, "depth_to_point_cloud", &err);
    if (err != CL_SUCCESS) return tk::gpu::opencl::translate_cl_error(err);

    *out_dispatcher = dispatcher;
    return TK_SUCCESS;
}

void tk_opencl_dispatch_destroy(tk_opencl_dispatcher_t** dispatcher) {
    if (!dispatcher || !*dispatcher) return;
    
    clFinish((*dispatcher)->command_queue);
    clReleaseKernel((*dispatcher)->preprocess_kernel);
    clReleaseKernel((*dispatcher)->depth_to_points_kernel);
    clReleaseProgram((*dispatcher)->program);
    clReleaseCommandQueue((*dispatcher)->command_queue);
    clReleaseContext((*dispatcher)->context);

    delete *dispatcher;
    *dispatcher = nullptr;
}

TK_NODISCARD tk_error_code_t tk_opencl_dispatch_malloc(tk_opencl_dispatcher_t* dispatcher, tk_gpu_buffer_t* out_buffer, size_t size_bytes, uint64_t flags) {
    cl_int err;
    *out_buffer = clCreateBuffer(dispatcher->context, flags, size_bytes, NULL, &err);
    if (err != CL_SUCCESS) {
        *out_buffer = nullptr;
        return tk::gpu::opencl::translate_cl_error(err);
    }
    return TK_SUCCESS;
}

void tk_opencl_dispatch_free(tk_opencl_dispatcher_t* dispatcher, tk_gpu_buffer_t* buffer) {
    if (!dispatcher || !buffer || !*buffer) return;
    clReleaseMemObject(*buffer);
    *buffer = nullptr;
}

TK_NODISCARD tk_error_code_t tk_opencl_dispatch_upload_async(tk_opencl_dispatcher_t* dispatcher, tk_gpu_buffer_t dst_buffer, const void* src_host_ptr, size_t size_bytes) {
    cl_int err = clEnqueueWriteBuffer(dispatcher->command_queue, dst_buffer, CL_FALSE, 0, size_bytes, src_host_ptr, 0, NULL, NULL);
    return tk::gpu::opencl::translate_cl_error(err);
}

TK_NODISCARD tk_error_code_t tk_opencl_dispatch_download_async(tk_opencl_dispatcher_t* dispatcher, void* dst_host_ptr, tk_gpu_buffer_t src_buffer, size_t size_bytes) {
    cl_int err = clEnqueueReadBuffer(dispatcher->command_queue, src_buffer, CL_FALSE, 0, size_bytes, dst_host_ptr, 0, NULL, NULL);
    return tk::gpu::opencl::translate_cl_error(err);
}

TK_NODISCARD tk_error_code_t tk_opencl_dispatch_synchronize(tk_opencl_dispatcher_t* dispatcher) {
    cl_int err = clFinish(dispatcher->command_queue);
    return tk::gpu::opencl::translate_cl_error(err);
}

TK_NODISCARD tk_error_code_t tk_opencl_dispatch_preprocess_image(tk_opencl_dispatcher_t* dispatcher, const tk_preprocess_params_t* params) {
    return tk_opencl_kernel_preprocess_image(dispatcher, params);
}

TK_NODISCARD tk_error_code_t tk_opencl_dispatch_depth_to_point_cloud(tk_opencl_dispatcher_t* dispatcher, const tk_depth_to_points_params_t* params) {
    return tk_opencl_kernel_depth_to_point_cloud(dispatcher, params);
}
