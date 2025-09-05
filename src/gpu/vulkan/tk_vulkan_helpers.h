/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_vulkan_helpers.h
*
* This header provides a C++ interface with helper functions and classes to
* simplify interactions with the Vulkan API. It is intended for internal use by
* the Vulkan backend implementation (`.cpp` files) to abstract away much of
* the boilerplate associated with setting up and managing Vulkan resources.
*
* Key responsibilities of this module include:
*   - Finding a suitable compute-capable physical device.
*   - Creating a logical device and compute queue.
*   - Allocating and managing device memory.
*   - Creating command buffers and synchronization primitives.
*   - Loading SPIR-V shaders.
*   - Translating VkResult codes into the project's standard tk_error_code_t.
*
* This file is NOT part of the public C-style API.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TRACKIELLM_GPU_VULKAN_TK_VULKAN_HELPERS_H
#define TRACKIELLM_GPU_VULKAN_TK_VULKAN_HELPERS_H

#include <vulkan/vulkan.h>
#include <vector>

#include "utils/tk_error_handling.h"

namespace tk {
namespace gpu {
namespace vulkan {

//------------------------------------------------------------------------------
// Vulkan Resource Wrappers (RAII)
//------------------------------------------------------------------------------

// Note: A full implementation would use RAII wrappers for all Vulkan handles
// to ensure automatic cleanup. For this example, we'll keep it more direct,
// but in a production system, you would have classes like:
// class VulkanInstance { ... };
// class VulkanDevice { ... };
// etc.

//------------------------------------------------------------------------------
// Helper Functions
//------------------------------------------------------------------------------

/**
 * @brief Translates a VkResult error code to a tk_error_code_t.
 * @param result The Vulkan result code.
 * @return The corresponding tk_error_code_t.
 */
tk_error_code_t translate_vulkan_result(VkResult result);

/**
 * @brief Finds a suitable physical device that supports compute operations.
 * @param instance The Vulkan instance.
 * @param out_physical_device Pointer to store the selected physical device.
 * @param out_queue_family_index Pointer to store the index of the compute queue family.
 * @return TK_SUCCESS on success.
 */
tk_error_code_t find_physical_device(
    VkInstance instance,
    VkPhysicalDevice* out_physical_device,
    uint32_t* out_queue_family_index
);

/**
 * @brief Allocates memory on the device.
 * @param physical_device The physical device.
 * @param device The logical device.
 * @param size The allocation size.
 * @param memory_type_index The index of the memory type to use.
 * @param out_memory Pointer to store the allocated device memory.
 * @return TK_SUCCESS on success.
 */
tk_error_code_t allocate_device_memory(
    VkPhysicalDevice physical_device,
    VkDevice device,
    VkDeviceSize size,
    uint32_t memory_type_index,
    VkDeviceMemory* out_memory
);

/**
 * @brief Finds a suitable memory type index.
 * @param physical_device The physical device.
 * @param type_filter A bitmask of suitable memory types.
 * @param property_flags The required memory property flags.
 * @param out_type_index Pointer to store the found memory type index.
 * @return TK_SUCCESS if a suitable type is found.
 */
tk_error_code_t find_memory_type(
    VkPhysicalDevice physical_device,
    uint32_t type_filter,
    VkMemoryPropertyFlags property_flags,
    uint32_t* out_type_index
);

/**
 * @brief Creates a VkBuffer.
 * @param device The logical device.
 * @param size The buffer size.
 * @param usage The buffer usage flags.
 * @param out_buffer Pointer to store the created buffer.
 * @return TK_SUCCESS on success.
 */
tk_error_code_t create_buffer(
    VkDevice device,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkBuffer* out_buffer
);

/**
 * @brief Creates a compute pipeline.
 * @param device The logical device.
 * @param shader_module The compiled shader module.
 * @param pipeline_layout The pipeline layout.
 * @param out_pipeline Pointer to store the created compute pipeline.
 * @return TK_SUCCESS on success.
 */
tk_error_code_t create_compute_pipeline(
    VkDevice device,
    VkShaderModule shader_module,
    VkPipelineLayout pipeline_layout,
    VkPipeline* out_pipeline
);

/**
 * @brief Loads a SPIR-V shader from a file.
 * @param device The logical device.
 * @param filepath The path to the SPIR-V shader file.
 * @param out_shader_module Pointer to store the created shader module.
 * @return TK_SUCCESS on success.
 */
tk_error_code_t create_shader_module(
    VkDevice device,
    const std::vector<char>& code,
    VkShaderModule* out_shader_module
);

} // namespace vulkan
} // namespace gpu
} // namespace tk

#endif // TRACKIELLM_GPU_VULKAN_TK_VULKAN_HELPERS_H
