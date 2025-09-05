/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_opencl_helpers.cpp
*
* This file implements the helper functions for the OpenCL backend.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#include "tk_opencl_helpers.h"
#include <fstream>
#include <iostream>

namespace tk {
namespace gpu {
namespace opencl {

tk_error_code_t translate_cl_error(cl_int err) {
    switch (err) {
        case CL_SUCCESS:                         return TK_SUCCESS;
        case CL_DEVICE_NOT_FOUND:                return TK_ERROR_GPU_DEVICE_NOT_FOUND;
        case CL_DEVICE_NOT_AVAILABLE:            return TK_ERROR_GPU_DEVICE_NOT_FOUND;
        case CL_COMPILER_NOT_AVAILABLE:          return TK_ERROR_GPU_SHADER_COMPILE;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:   return TK_ERROR_GPU_MEMORY;
        case CL_OUT_OF_RESOURCES:                return TK_ERROR_GPU_MEMORY;
        case CL_OUT_OF_HOST_MEMORY:              return TK_ERROR_OUT_OF_MEMORY;
        case CL_PROFILING_INFO_NOT_AVAILABLE:    return TK_ERROR_GPU_UNKNOWN;
        case CL_MEM_COPY_OVERLAP:                return TK_ERROR_GPU_MEMORY;
        case CL_IMAGE_FORMAT_MISMATCH:           return TK_ERROR_GPU_UNKNOWN;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:      return TK_ERROR_GPU_UNKNOWN;
        case CL_BUILD_PROGRAM_FAILURE:           return TK_ERROR_GPU_SHADER_LINK;
        case CL_MAP_FAILURE:                     return TK_ERROR_GPU_MEMORY;
        case CL_INVALID_VALUE:                   return TK_ERROR_INVALID_ARGUMENT;
        // Add more specific translations as needed
        default:                                 return TK_ERROR_GPU_UNKNOWN;
    }
}


tk_error_code_t create_and_build_program(
    cl_context context,
    cl_device_id device,
    const std::string& filepath,
    cl_program* out_program)
{
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return TK_ERROR_FILESYSTEM;
    }
    std::string source_code((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    cl_int err;
    const char* source_ptr = source_code.c_str();
    size_t source_size = source_code.length();
    *out_program = clCreateProgramWithSource(context, 1, &source_ptr, &source_size, &err);
    if (err != CL_SUCCESS) return translate_cl_error(err);

    err = clBuildProgram(*out_program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(*out_program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(*out_program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        std::cerr << "OpenCL build error: " << log.data() << std::endl;
        return translate_cl_error(err);
    }

    return TK_SUCCESS;
}


} // namespace opencl
} // namespace gpu
} // namespace tk
