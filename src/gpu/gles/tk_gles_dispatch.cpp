/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_gles_dispatch.cpp
*
* This file implements the GLES Dispatcher. It orchestrates the setup of a
* headless EGL context and manages GLES resources for compute tasks.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#include "tk_gles_dispatch.h"
#include "tk_gles_helpers.h"
#include "tk_gles_kernels.h"

#include <iostream>

// --- Internal Struct Definitions ---

struct tk_gpu_buffer_s {
    GLuint buffer_id;
    size_t size;
};

struct tk_gles_dispatcher_s {
    EGLDisplay display;
    EGLConfig config;
    EGLContext context;
    EGLSurface surface; // Headless pbuffer surface

    // Shader Programs
    GLuint preprocess_program;
    GLuint depth_to_points_program;

    tk_gles_dispatcher_config_t dispatcher_config;
};

// --- Helper Functions ---

static tk_error_code_t create_shader_programs(tk_gles_dispatcher_s* dispatcher) {
    // In a real app, load from glsl files
    // For now, imagine we have the source code.
    std::string preprocess_glsl = "#version 310 es\n...";
    std::string depth_glsl = "#version 310 es\n...";

    tk_error_code_t err;
    err = tk::gpu::gles::create_compute_program(preprocess_glsl, &dispatcher->preprocess_program);
    if (err != TK_SUCCESS) return err;

    err = tk::gpu::gles::create_compute_program(depth_glsl, &dispatcher->depth_to_points_program);
    if (err != TK_SUCCESS) return err;

    return TK_SUCCESS;
}

// --- Public API Implementation ---

TK_NODISCARD tk_error_code_t tk_gles_dispatch_create(tk_gles_dispatcher_t** out_dispatcher, const tk_gles_dispatcher_config_t* config) {
    tk_gles_dispatcher_s* dispatcher = new (std::nothrow) tk_gles_dispatcher_s();
    if (!dispatcher) return TK_ERROR_OUT_OF_MEMORY;
    dispatcher->dispatcher_config = *config;

    // 1. Initialize EGL
    dispatcher->display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (dispatcher->display == EGL_NO_DISPLAY) return tk::gpu::gles::check_egl_error("eglGetDisplay");
    eglInitialize(dispatcher->display, NULL, NULL);

    // 2. Choose EGLConfig
    const EGLint config_attribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_BLUE_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_RED_SIZE, 8,
        EGL_DEPTH_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, // Request GLES 3.x
        EGL_NONE
    };
    EGLint num_configs;
    eglChooseConfig(dispatcher->display, config_attribs, &dispatcher->config, 1, &num_configs);

    // 3. Create Pbuffer Surface
    const EGLint pbuffer_attribs[] = { EGL_WIDTH, 9, EGL_HEIGHT, 9, EGL_NONE };
    dispatcher->surface = eglCreatePbufferSurface(dispatcher->display, dispatcher->config, pbuffer_attribs);
    if (dispatcher->surface == EGL_NO_SURFACE) return tk::gpu::gles::check_egl_error("eglCreatePbufferSurface");

    // 4. Create GLES Context
    const EGLint context_attribs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
    dispatcher->context = eglCreateContext(dispatcher->display, dispatcher->config, EGL_NO_CONTEXT, context_attribs);
    if (dispatcher->context == EGL_NO_CONTEXT) return tk::gpu::gles::check_egl_error("eglCreateContext");

    // 5. Make Context Current
    eglMakeCurrent(dispatcher->display, dispatcher->surface, dispatcher->surface, dispatcher->context);

    // 6. Create shader programs
    tk_error_code_t err = create_shader_programs(dispatcher);
    if (err != TK_SUCCESS) {
        // ... cleanup ...
        return err;
    }

    *out_dispatcher = dispatcher;
    return TK_SUCCESS;
}

void tk_gles_dispatch_destroy(tk_gles_dispatcher_t** dispatcher) {
    if (!dispatcher || !*dispatcher) return;

    eglMakeCurrent((*dispatcher)->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    glDeleteProgram((*dispatcher)->preprocess_program);
    glDeleteProgram((*dispatcher)->depth_to_points_program);
    eglDestroyContext((*dispatcher)->display, (*dispatcher)->context);
    eglDestroySurface((*dispatcher)->display, (*dispatcher)->surface);
    eglTerminate((*dispatcher)->display);

    delete *dispatcher;
    *dispatcher = nullptr;
}

TK_NODISCARD tk_error_code_t tk_gles_dispatch_malloc(tk_gles_dispatcher_t* dispatcher, tk_gpu_buffer_t* out_buffer, size_t size_bytes, uint32_t usage_hint) {
    tk_gpu_buffer_s* buf = new (std::nothrow) tk_gpu_buffer_s();
    if (!buf) return TK_ERROR_OUT_OF_MEMORY;
    buf->size = size_bytes;

    glGenBuffers(1, &buf->buffer_id);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf->buffer_id);
    glBufferData(GL_SHADER_STORAGE_BUFFER, size_bytes, NULL, usage_hint);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    *out_buffer = buf;
    return tk::gpu::gles::check_gl_error("tk_gles_dispatch_malloc");
}

void tk_gles_dispatch_free(tk_gles_dispatcher_t* dispatcher, tk_gpu_buffer_t* buffer) {
    if (!dispatcher || !buffer || !*buffer) return;
    glDeleteBuffers(1, &(*buffer)->buffer_id);
    delete *buffer;
    *buffer = nullptr;
}

TK_NODISCARD tk_error_code_t tk_gles_dispatch_upload_async(tk_gles_dispatcher_t* dispatcher, tk_gpu_buffer_t dst_buffer, const void* src_host_ptr, size_t size_bytes) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, dst_buffer->buffer_id);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, size_bytes, src_host_ptr);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return tk::gpu::gles::check_gl_error("tk_gles_dispatch_upload_async");
}

TK_NODISCARD tk_error_code_t tk_gles_dispatch_download_async(tk_gles_dispatcher_t* dispatcher, void* dst_host_ptr, tk_gpu_buffer_t src_buffer, size_t size_bytes) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, src_buffer->buffer_id);
    void* ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, size_bytes, GL_MAP_READ_BIT);
    if (!ptr) {
        return tk::gpu::gles::check_gl_error("glMapBufferRange");
    }
    memcpy(dst_host_ptr, ptr, size_bytes);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t tk_gles_dispatch_synchronize(tk_gles_dispatcher_t* dispatcher) {
    glFinish();
    return tk::gpu::gles::check_gl_error("glFinish");
}

TK_NODISCARD tk_error_code_t tk_gles_dispatch_preprocess_image(tk_gles_dispatcher_t* dispatcher, const tk_preprocess_params_t* params) {
    return tk_gles_kernel_preprocess_image(dispatcher, params);
}

TK_NODISCARD tk_error_code_t tk_gles_dispatch_depth_to_point_cloud(tk_gles_dispatcher_t* dispatcher, const tk_depth_to_points_params_t* params) {
    return tk_gles_kernel_depth_to_point_cloud(dispatcher, params);
}
