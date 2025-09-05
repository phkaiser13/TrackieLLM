/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_vulkan_helpers.cpp
*
* This file implements the helper functions for the Vulkan backend.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#include "tk_vulkan_helpers.h"

#include <fstream>
#include <iostream>

namespace tk {
namespace gpu {
namespace vulkan {

tk_error_code_t translate_vulkan_result(VkResult result) {
    switch (result) {
        case VK_SUCCESS:                         return TK_SUCCESS;
        case VK_ERROR_OUT_OF_HOST_MEMORY:        return TK_ERROR_OUT_OF_MEMORY;
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:      return TK_ERROR_GPU_MEMORY;
        case VK_ERROR_DEVICE_LOST:               return TK_ERROR_GPU_DEVICE_LOST;
        case VK_ERROR_INITIALIZATION_FAILED:     return TK_ERROR_GPU_INITIALIZATION_FAILED;
        // Add more specific translations as needed
        default:                                 return TK_ERROR_GPU_UNKNOWN;
    }
}

tk_error_code_t find_physical_device(
    VkInstance instance,
    VkPhysicalDevice* out_physical_device,
    uint32_t* out_queue_family_index)
{
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);

    if (device_count == 0) {
        return TK_ERROR_GPU_DEVICE_NOT_FOUND;
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

    for (const auto& device : devices) {
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);

        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

        for (uint32_t i = 0; i < queue_family_count; ++i) {
            if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                *out_physical_device = device;
                *out_queue_family_index = i;
                return TK_SUCCESS;
            }
        }
    }

    return TK_ERROR_GPU_DEVICE_NOT_FOUND;
}

tk_error_code_t find_memory_type(
    VkPhysicalDevice physical_device,
    uint32_t type_filter,
    VkMemoryPropertyFlags property_flags,
    uint32_t* out_type_index)
{
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && (mem_properties.memoryTypes[i].propertyFlags & property_flags) == property_flags) {
            *out_type_index = i;
            return TK_SUCCESS;
        }
    }

    return TK_ERROR_GPU_MEMORY;
}


tk_error_code_t allocate_device_memory(
    VkPhysicalDevice physical_device,
    VkDevice device,
    VkDeviceSize size,
    uint32_t memory_type_index,
    VkDeviceMemory* out_memory)
{
    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = size;
    alloc_info.memoryTypeIndex = memory_type_index;

    VkResult result = vkAllocateMemory(device, &alloc_info, nullptr, out_memory);
    return translate_vulkan_result(result);
}

tk_error_code_t create_buffer(
    VkDevice device,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkBuffer* out_buffer)
{
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateBuffer(device, &buffer_info, nullptr, out_buffer);
    return translate_vulkan_result(result);
}

tk_error_code_t create_shader_module(
    VkDevice device,
    const std::vector<char>& code,
    VkShaderModule* out_shader_module)
{
    VkShaderModuleCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = code.size();
    create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkResult result = vkCreateShaderModule(device, &create_info, nullptr, out_shader_module);
    return translate_vulkan_result(result);
}


tk_error_code_t create_compute_pipeline(
    VkDevice device,
    VkShaderModule shader_module,
    VkPipelineLayout pipeline_layout,
    VkPipeline* out_pipeline)
{
    VkPipelineShaderStageCreateInfo shader_stage_info = {};
    shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shader_stage_info.module = shader_module;
    shader_stage_info.pName = "main"; // Entry point of the shader

    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = shader_stage_info;
    pipeline_info.layout = pipeline_layout;

    VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, out_pipeline);
    return translate_vulkan_result(result);
}


} // namespace vulkan
} // namespace gpu
} // namespace tk
