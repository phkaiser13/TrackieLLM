/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_vulkan_kernels.cpp
*
* This file implements the C-callable kernel dispatch wrappers for the Vulkan
* backend. These functions are responsible for setting up the necessary Vulkan
* state to execute a compute shader, including:
*   - Creating and updating descriptor sets to bind data buffers.
*   - Binding the correct compute pipeline.
*   - Dispatching the compute work (`vkCmdDispatch`).
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#include "tk_vulkan_kernels.h"
#include "tk_vulkan_dispatch.h" // For the internal dispatcher struct definition
#include "tk_vulkan_helpers.h"  // For helper functions

// In a real implementation, the tk_vulkan_dispatcher_s struct would be defined
// in the tk_vulkan_dispatch.cpp file and we would need a way to access its
// internal state (like the VkDevice, pipelines, etc.). For this example,
// we'll assume we have a way to get the necessary Vulkan handles from the dispatcher.
// A common pattern is to have a private header or friend class.

// Let's define a placeholder for the internal dispatcher struct
// to allow the code to be illustrative.
struct tk_vulkan_dispatcher_s {
    VkDevice device;
    VkPipeline preprocess_pipeline;
    VkPipelineLayout preprocess_pipeline_layout;
    VkPipeline depth_to_points_pipeline;
    VkPipelineLayout depth_to_points_pipeline_layout;
    // ... other members like descriptor pool
};

// And a placeholder for the internal buffer struct
struct tk_gpu_buffer_s {
    VkBuffer buffer;
    VkDeviceMemory memory;
};


tk_error_code_t tk_vulkan_kernel_preprocess_image(
    struct tk_vulkan_dispatcher_s* dispatcher,
    VkCommandBuffer cmd_buffer,
    const tk_preprocess_params_t* params)
{
    // 1. Create Descriptor Set
    // In a real app, you'd likely use a descriptor set pool in the dispatcher.
    // This is a simplified placeholder.
    VkDescriptorSet descriptor_set;
    // ... code to allocate and update a descriptor set ...
    // Update descriptor set to bind the input and output buffers from params

    // 2. Bind Pipeline and Descriptor Set
    vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, dispatcher->preprocess_pipeline);
    vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, dispatcher->preprocess_pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

    // 3. Push Constants (if the shader uses them for parameters like width/height)
    // vkCmdPushConstants(...);

    // 4. Dispatch
    // Calculate workgroup counts based on output tensor size
    uint32_t group_count_x = (params->output_width + 15) / 16; // Assuming workgroup size of 16x16
    uint32_t group_count_y = (params->output_height + 15) / 16;
    vkCmdDispatch(cmd_buffer, group_count_x, group_count_y, 1);

    return TK_SUCCESS;
}

tk_error_code_t tk_vulkan_kernel_depth_to_point_cloud(
    struct tk_vulkan_dispatcher_s* dispatcher,
    VkCommandBuffer cmd_buffer,
    const tk_depth_to_points_params_t* params)
{
    // 1. Create and update descriptor set to bind the depth map and point cloud buffers.
    VkDescriptorSet descriptor_set;
    // ...

    // 2. Bind pipeline and descriptor set
    vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, dispatcher->depth_to_points_pipeline);
    vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, dispatcher->depth_to_points_pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

    // 3. Push constants for camera intrinsics
    // vkCmdPushConstants(cmd_buffer, dispatcher->depth_to_points_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params->fx) * 4, &params->fx);

    // 4. Dispatch
    uint32_t group_count_x = (params->width + 15) / 16;
    uint32_t group_count_y = (params->height + 15) / 16;
    vkCmdDispatch(cmd_buffer, group_count_x, group_count_y, 1);

    return TK_SUCCESS;
}
