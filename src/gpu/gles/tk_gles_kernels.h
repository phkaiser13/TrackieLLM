/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_gles_kernels.h
*
* This header file defines the C-callable interface for the GLES compute
* shaders used in the TrackieLLM project. It acts as a contract between the
* C++ dispatch layer and the kernel execution logic.
*
* This API abstracts the details of setting uniforms, binding shader storage
* buffers (SSBOs), and dispatching the compute shader.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TRACKIELLM_GPU_GLES_TK_GLES_KERNELS_H
#define TRACKIELLM_GPU_GLES_TK_GLES_KERNELS_H

#include <GLES3/gl32.h>
#include "gpu/tk_gpu_helper.h"
#include "utils/tk_error_handling.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward-declare internal dispatcher and buffer structs
struct tk_gles_dispatcher_s;
struct tk_gpu_buffer_s;

/**
 * @brief Dispatches the image pre-processing compute shader.
 *
 * @param[in] dispatcher The GLES dispatcher, containing shader program handles.
 * @param[in] params The parameters for the pre-processing kernel. The buffer
 *                   handles within this struct will be used to bind the
 *                   appropriate SSBOs.
 * @return TK_SUCCESS if the shader was successfully dispatched.
 */
TK_NODISCARD tk_error_code_t tk_gles_kernel_preprocess_image(
    struct tk_gles_dispatcher_s* dispatcher,
    const tk_preprocess_params_t* params
);

/**
 * @brief Dispatches the depth-to-point-cloud compute shader.
 *
 * @param[in] dispatcher The GLES dispatcher, containing shader program handles.
 * @param[in] params The parameters for the conversion kernel.
 * @return TK_SUCCESS if the shader was successfully dispatched.
 */
TK_NODISCARD tk_error_code_t tk_gles_kernel_depth_to_point_cloud(
    struct tk_gles_dispatcher_s* dispatcher,
    const tk_depth_to_points_params_t* params
);


#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_GPU_GLES_TK_GLES_KERNELS_H
