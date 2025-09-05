/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_opencl_kernels.h
*
* This header file defines the C-callable interface for launching the OpenCL
* kernels used in the TrackieLLM project. It acts as a contract between the
* C++ dispatch layer and the kernel execution logic.
*
* This API abstracts the details of setting kernel arguments and enqueuing
* the kernel for execution on the command queue.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TRACKIELLM_GPU_OPENCL_TK_OPENCL_KERNELS_H
#define TRACKIELLM_GPU_OPENCL_TK_OPENCL_KERNELS_H

#include <CL/cl.h>
#include "gpu/tk_gpu_helper.h"
#include "utils/tk_error_handling.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward-declare internal dispatcher struct
struct tk_opencl_dispatcher_s;

/**
 * @brief Enqueues the image pre-processing kernel for execution.
 *
 * @param[in] dispatcher The OpenCL dispatcher, containing the cl_kernel object.
 * @param[in] params The parameters for the pre-processing kernel. The buffer
 *                   handles will be set as kernel arguments.
 * @return TK_SUCCESS if the kernel was successfully enqueued.
 */
TK_NODISCARD tk_error_code_t tk_opencl_kernel_preprocess_image(
    struct tk_opencl_dispatcher_s* dispatcher,
    const tk_preprocess_params_t* params
);

/**
 * @brief Enqueues the depth-to-point-cloud kernel for execution.
 *
 * @param[in] dispatcher The OpenCL dispatcher, containing the cl_kernel object.
 * @param[in] params The parameters for the conversion kernel.
 * @return TK_SUCCESS if the kernel was successfully enqueued.
 */
TK_NODISCARD tk_error_code_t tk_opencl_kernel_depth_to_point_cloud(
    struct tk_opencl_dispatcher_s* dispatcher,
    const tk_depth_to_points_params_t* params
);

#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_GPU_OPENCL_TK_OPENCL_KERNELS_H
