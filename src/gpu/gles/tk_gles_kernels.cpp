/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_gles_kernels.cpp
*
* This file implements the C-callable kernel dispatch wrappers for the GLES
* backend. These functions are responsible for setting up the GLES state to
* execute a compute shader.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#include "tk_gles_kernels.h"
#include "tk_gles_dispatch.h" // For internal struct definitions
#include "tk_gles_helpers.h"

// Placeholder for the internal dispatcher struct definition.
// In a real implementation, this would be defined in tk_gles_dispatch.cpp
// and its members accessed via a private header or other means.
struct tk_gles_dispatcher_s {
    GLuint preprocess_program;
    GLuint depth_to_points_program;
    // ... other EGL/GLES context info
};

// Placeholder for the internal buffer struct.
// The handle tk_gpu_buffer_t points to this.
struct tk_gpu_buffer_s {
    GLuint buffer_id;
    size_t size;
};

tk_error_code_t tk_gles_kernel_preprocess_image(
    struct tk_gles_dispatcher_s* dispatcher,
    const tk_preprocess_params_t* params)
{
    glUseProgram(dispatcher->preprocess_program);

    // Bind input and output buffers as Shader Storage Buffers (SSBOs)
    const tk_gpu_buffer_s* input_buf = (const tk_gpu_buffer_s*)params->d_input_image;
    const tk_gpu_buffer_s* output_buf = (const tk_gpu_buffer_s*)params->d_output_tensor;

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, input_buf->buffer_id);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, output_buf->buffer_id);

    // Set uniforms for other parameters
    // This requires getting uniform locations first, which should be cached.
    // GLint width_loc = glGetUniformLocation(dispatcher->preprocess_program, "u_params.input_width");
    // glUniform1ui(width_loc, params->input_width);
    // ... set other uniforms ...

    // Calculate workgroup counts
    GLuint group_count_x = (params->output_width + 15) / 16; // Assuming workgroup size 16x16
    GLuint group_count_y = (params->output_height + 15) / 16;

    // Dispatch the compute shader
    glDispatchCompute(group_count_x, group_count_y, 1);

    // Add a memory barrier to ensure writes are visible for subsequent operations
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    return tk::gpu::gles::check_gl_error("tk_gles_kernel_preprocess_image");
}

tk_error_code_t tk_gles_kernel_depth_to_point_cloud(
    struct tk_gles_dispatcher_s* dispatcher,
    const tk_depth_to_points_params_t* params)
{
    glUseProgram(dispatcher->depth_to_points_program);

    // Bind buffers
    const tk_gpu_buffer_s* input_buf = (const tk_gpu_buffer_s*)params->d_metric_depth_map;
    const tk_gpu_buffer_s* output_buf = (const tk_gpu_buffer_s*)params->d_point_cloud;

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, input_buf->buffer_id);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, output_buf->buffer_id);

    // Set uniforms for camera intrinsics
    // ...

    // Calculate workgroup counts
    GLuint group_count_x = (params->width + 15) / 16;
    GLuint group_count_y = (params->height + 15) / 16;

    glDispatchCompute(group_count_x, group_count_y, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    return tk::gpu::gles::check_gl_error("tk_gles_kernel_depth_to_point_cloud");
}
