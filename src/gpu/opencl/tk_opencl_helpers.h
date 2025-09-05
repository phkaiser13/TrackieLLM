/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_opencl_helpers.h
*
* This header provides a C++ interface with helper functions to simplify
* interactions with the OpenCL API. It is for internal use by the OpenCL
* backend implementation.
*
* Key responsibilities include:
*   - Translating cl_int error codes to tk_error_code_t.
*   - Loading and building OpenCL C kernel source files.
*
* This file is NOT part of the public C-style API.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TRACKIELLM_GPU_OPENCL_TK_OPENCL_HELPERS_H
#define TRACKIELLM_GPU_OPENCL_TK_OPENCL_HELPERS_H

#include <CL/cl.h>
#include "utils/tk_error_handling.h"
#include <string>
#include <vector>

namespace tk {
namespace gpu {
namespace opencl {

/**
 * @brief Translates a cl_int error code to a tk_error_code_t.
 * @param err The OpenCL error code.
 * @return The corresponding tk_error_code_t.
 */
tk_error_code_t translate_cl_error(cl_int err);

/**
 * @brief Creates an OpenCL program from a source file and builds it.
 * @param context The OpenCL context.
 * @param device The device to build the program for.
 * @param filepath The path to the .cl source file.
 * @param out_program Pointer to store the created program object.
 * @return TK_SUCCESS on success.
 */
tk_error_code_t create_and_build_program(
    cl_context context,
    cl_device_id device,
    const std::string& filepath,
    cl_program* out_program
);

} // namespace opencl
} // namespace gpu
} // namespace tk

#endif // TRACKIELLM_GPU_OPENCL_TK_OPENCL_HELPERS_H
