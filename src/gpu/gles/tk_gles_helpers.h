/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_gles_helpers.h
*
* This header provides a C++ interface with helper functions to simplify
* interactions with the OpenGL ES API. It is for internal use by the GLES
* backend to abstract away common, verbose tasks.
*
* Key responsibilities include:
*   - Loading and compiling GLSL shaders.
*   - Linking compute programs.
*   - Checking for and translating GL and EGL errors.
*
* This file is NOT part of the public C-style API.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TRACKIELLM_GPU_GLES_TK_GLES_HELPERS_H
#define TRACKIELLM_GPU_GLES_TK_GLES_HELPERS_H

#include <GLES3/gl32.h> // For GLES 3.2 types, needs compute shader support
#include <EGL/egl.h>

#include "utils/tk_error_handling.h"
#include <string>

namespace tk {
namespace gpu {
namespace gles {

/**
 * @brief Checks for a GLES error and logs it if found.
 * @param location A string identifying where the check is being made.
 * @return TK_SUCCESS if no error, TK_ERROR_GPU_UNKNOWN otherwise.
 */
tk_error_code_t check_gl_error(const char* location);

/**
 * @brief Checks for an EGL error and logs it if found.
 * @param location A string identifying where the check is being made.
 * @return TK_SUCCESS if no error, TK_ERROR_GPU_INITIALIZATION_FAILED otherwise.
 */
tk_error_code_t check_egl_error(const char* location);

/**
 * @brief Loads a shader source from a file.
 * @param filepath The path to the shader file.
 * @param out_source The string to store the loaded source code.
 * @return TK_SUCCESS on success.
 */
tk_error_code_t load_shader_source(const std::string& filepath, std::string& out_source);

/**
 * @brief Compiles a shader.
 * @param type The type of shader (e.g., GL_COMPUTE_SHADER).
 * @param source The GLSL source code.
 * @param out_shader Pointer to store the handle of the compiled shader.
 * @return TK_SUCCESS on success.
 */
tk_error_code_t compile_shader(GLenum type, const std::string& source, GLuint* out_shader);

/**
 * @brief Creates a compute program by linking a compute shader.
 * @param compute_shader_source The source code for the compute shader.
 * @param out_program Pointer to store the handle of the linked program.
 * @return TK_SUCCESS on success.
 */
tk_error_code_t create_compute_program(const std::string& compute_shader_source, GLuint* out_program);


} // namespace gles
} // namespace gpu
} // namespace tk

#endif // TRACKIELLM_GPU_GLES_TK_GLES_HELPERS_H
