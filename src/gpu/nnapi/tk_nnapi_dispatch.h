/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_nnapi_dispatch.h
*
* This header defines the public API for the Android Neural Networks API (NNAPI)
* Dispatcher. Unlike other GPU backends that execute individual kernels, the
* NNAPI backend is designed to offload entire neural network models to
* available accelerators (GPU, DSP, NPU) on Android devices.
*
* This module provides a high-level, C-style interface for loading models,
* setting inputs, executing inference, and retrieving outputs.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TRACKIELLM_GPU_NNAPI_TK_NNAPI_DISPATCH_H
#define TRACKIELLM_GPU_NNAPI_TK_NNAPI_DISPATCH_H

#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"

// Forward-declare the primary objects as opaque types.
typedef struct tk_nnapi_model_s tk_nnapi_model_t;
typedef struct tk_nnapi_execution_s tk_nnapi_execution_t;

/**
 * @struct tk_nnapi_buffer_t
 * @brief Represents an input or output buffer for an NNAPI execution.
 */
typedef struct {
    int32_t index;       /**< The operand index in the model. */
    void* data;          /**< Pointer to the data buffer. */
    size_t length;       /**< Length of the data buffer in bytes. */
} tk_nnapi_buffer_t;


#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Model Management
//------------------------------------------------------------------------------

/**
 * @brief Creates and compiles an NNAPI model from a file.
 * (Note: This is a simplification. NNAPI models are typically defined in code,
 * not loaded directly from a file like ONNX. A real implementation would have
 * a function like `tk_nnapi_create_yolo_model()` that hard-codes the network
 * structure using NNAPI operand and operation functions).
 *
 * @param[out] out_model Pointer to receive the handle to the compiled model.
 * @param[in] model_path Path to the model file (conceptual).
 * @return TK_SUCCESS on success.
 */
TK_NODISCARD tk_error_code_t tk_nnapi_model_create(tk_nnapi_model_t** out_model, const char* model_path);

/**
 * @brief Destroys an NNAPI model object.
 *
 * @param[in,out] model Pointer to the model handle to be destroyed.
 */
void tk_nnapi_model_destroy(tk_nnapi_model_t** model);


//------------------------------------------------------------------------------
// Inference Execution
//------------------------------------------------------------------------------

/**
 * @brief Creates an execution instance from a compiled model.
 *
 * @param[in] model The compiled model to execute.
 * @param[out] out_execution Pointer to receive the handle to the new execution instance.
 * @return TK_SUCCESS on success.
 */
TK_NODISCARD tk_error_code_t tk_nnapi_execution_create(tk_nnapi_model_t* model, tk_nnapi_execution_t** out_execution);

/**
 * @brief Sets the input buffers for an execution.
 *
 * @param[in] execution The execution instance.
 * @param[in] inputs An array of input buffers.
 * @param[in] num_inputs The number of input buffers.
 * @return TK_SUCCESS on success.
 */
TK_NODISCARD tk_error_code_t tk_nnapi_execution_set_inputs(tk_nnapi_execution_t* execution, const tk_nnapi_buffer_t* inputs, uint32_t num_inputs);

/**
 * @brief Sets the output buffers for an execution.
 *
 * @param[in] execution The execution instance.
 * @param[in] outputs An array of output buffers.
 * @param[in] num_outputs The number of output buffers.
 * @return TK_SUCCESS on success.
 */
TK_NODISCARD tk_error_code_t tk_nnapi_execution_set_outputs(tk_nnapi_execution_t* execution, const tk_nnapi_buffer_t* outputs, uint32_t num_outputs);

/**
 * @brief Executes the inference computation asynchronously.
 *
 * @param[in] execution The execution instance.
 * @return TK_SUCCESS if the execution was successfully started.
 */
TK_NODISCARD tk_error_code_t tk_nnapi_execute_async(tk_nnapi_execution_t* execution);

/**
 * @brief Waits for an asynchronous execution to complete.
 *
 * @param[in] execution The execution instance.
 * @return TK_SUCCESS if the execution completed successfully.
 */
TK_NODISCARD tk_error_code_t tk_nnapi_wait(tk_nnapi_execution_t* execution);

/**
 * @brief Destroys an NNAPI execution instance.
 *
 * @param[in,out] execution Pointer to the execution handle to be destroyed.
 */
void tk_nnapi_execution_destroy(tk_nnapi_execution_t** execution);


#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_GPU_NNAPI_TK_NNAPI_DISPATCH_H
