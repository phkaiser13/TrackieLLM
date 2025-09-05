/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_gles_helpers.cpp
*
* This file implements the helper functions for the GLES backend.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#include "tk_gles_helpers.h"
#include <fstream>
#include <iostream>
#include <vector>

namespace tk {
namespace gpu {
namespace gles {

tk_error_code_t check_gl_error(const char* location) {
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL Error at " << location << ": " << err << std::endl;
        return TK_ERROR_GPU_UNKNOWN;
    }
    return TK_SUCCESS;
}

tk_error_code_t check_egl_error(const char* location) {
    EGLint err = eglGetError();
    if (err != EGL_SUCCESS) {
        std::cerr << "EGL Error at " << location << ": " << err << std::endl;
        return TK_ERROR_GPU_INITIALIZATION_FAILED;
    }
    return TK_SUCCESS;
}

tk_error_code_t load_shader_source(const std::string& filepath, std::string& out_source) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return TK_ERROR_FILESYSTEM;
    }
    out_source = std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return TK_SUCCESS;
}

tk_error_code_t compile_shader(GLenum type, const std::string& source, GLuint* out_shader) {
    *out_shader = glCreateShader(type);
    const char* src_ptr = source.c_str();
    glShaderSource(*out_shader, 1, &src_ptr, NULL);
    glCompileShader(*out_shader);

    GLint compiled;
    glGetShaderiv(*out_shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        GLint info_len = 0;
        glGetShaderiv(*out_shader, GL_INFO_LOG_LENGTH, &info_len);
        if (info_len > 1) {
            std::vector<char> info_log(info_len);
            glGetShaderInfoLog(*out_shader, info_len, NULL, &info_log[0]);
            std::cerr << "Error compiling shader: " << &info_log[0] << std::endl;
        }
        glDeleteShader(*out_shader);
        return TK_ERROR_GPU_SHADER_COMPILE;
    }
    return TK_SUCCESS;
}

tk_error_code_t create_compute_program(const std::string& compute_shader_source, GLuint* out_program) {
    GLuint compute_shader;
    tk_error_code_t err = compile_shader(GL_COMPUTE_SHADER, compute_shader_source, &compute_shader);
    if (err != TK_SUCCESS) {
        return err;
    }

    *out_program = glCreateProgram();
    glAttachShader(*out_program, compute_shader);
    glLinkProgram(*out_program);

    glDeleteShader(compute_shader); // Can be deleted after linking

    GLint linked;
    glGetProgramiv(*out_program, GL_LINK_STATUS, &linked);
    if (!linked) {
        GLint info_len = 0;
        glGetProgramiv(*out_program, GL_INFO_LOG_LENGTH, &info_len);
        if (info_len > 1) {
            std::vector<char> info_log(info_len);
            glGetProgramInfoLog(*out_program, info_len, NULL, &info_log[0]);
            std::cerr << "Error linking program: " << &info_log[0] << std::endl;
        }
        glDeleteProgram(*out_program);
        return TK_ERROR_GPU_SHADER_LINK;
    }

    return TK_SUCCESS;
}

} // namespace gles
} // namespace gpu
} // namespace tk
