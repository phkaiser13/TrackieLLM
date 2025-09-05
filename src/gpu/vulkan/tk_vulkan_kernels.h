/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_vulkan_kernels.h
*
* This header file defines the C-callable interface for the Vulkan compute
* pipelines used in the TrackieLLM project. It serves as a stable contract
* between the C++ dispatch layer and the kernel execution logic.
*
* This API abstracts the details of creating descriptor sets, binding resources,
* and dispatching compute commands.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TRACKIELLM_GPU_VULKAN_TK_VULKAN_KERNELS_H
#define TRACKIELLM_GPU_VULKAN_TK_VULKAN_KERNELS_H

#include <vulkan/vulkan.h>
#include "gpu/tk_gpu_helper.h" // For parameter structs like tk_preprocess_params_t
#include "utils/tk_error_handling.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward-declare internal dispatcher and buffer structs
// to avoid exposing Vulkan types in this public-facing header.
struct tk_vulkan_dispatcher_s;
struct tk_gpu_buffer_s;


// Note: The parameter structs (tk_preprocess_params_t, etc.) are assumed to be
// defined in a common header like "gpu/tk_gpu_helper.h" to be shared across backends.
// For this implementation, we will imagine this file exists. If it doesn't,
// these structs would need to be defined here or in a new shared header.


/**
 * @brief Records commands to execute the image pre-processing compute shader.
 *
 * @param[in] dispatcher The Vulkan dispatcher, containing pipelines and layouts.
 * @param[in] cmd_buffer The command buffer to record into.
 * @param[in] params The parameters for the pre-processing kernel, including
 *                   handles to the input and output GPU buffers.
 * @return TK_SUCCESS if commands were successfully recorded.
 */
TK_NODISCARD tk_error_code_t tk_vulkan_kernel_preprocess_image(
    struct tk_vulkan_dispatcher_s* dispatcher,
    VkCommandBuffer cmd_buffer,
    const tk_preprocess_params_t* params
);

/**
 * @brief Records commands to execute the depth-to-point-cloud compute shader.
 *
 * @param[in] dispatcher The Vulkan dispatcher, containing pipelines and layouts.
 * @param[in] cmd_buffer The command buffer to record into.
 * @param[in] params The parameters for the conversion kernel, including
 *                   handles to the input and output GPU buffers.
 * @return TK_SUCCESS if commands were successfully recorded.
 */
TK_NODISCARD tk_error_code_t tk_vulkan_kernel_depth_to_point_cloud(
    struct tk_vulkan_dispatcher_s* dispatcher,
    VkCommandBuffer cmd_buffer,
    const tk_depth_to_points_params_t* params
);


#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_GPU_VULKAN_TK_VULKAN_KERNELS_H
