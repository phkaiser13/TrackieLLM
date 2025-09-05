/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_nnapi_dispatch.cpp
*
* This file implements the NNAPI Dispatcher for executing entire models.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#include "tk_nnapi_dispatch.h"
#include <android/neuralnetworks.h>
#include <iostream>

// --- Internal Struct Definitions ---

struct tk_nnapi_model_s {
    ANeuralNetworksModel* model;
    ANeuralNetworksCompilation* compilation;
};

struct tk_nnapi_execution_s {
    ANeuralNetworksExecution* execution;
    ANeuralNetworksEvent* event;
};

// --- Helper Functions ---

static tk_error_code_t translate_nnapi_result(int result_code, const char* location) {
    if (result_code == ANEURALNETWORKS_NO_ERROR) {
        return TK_SUCCESS;
    }
    std::cerr << "NNAPI Error at " << location << ": " << result_code << std::endl;
    // Map specific NNAPI errors to tk_error_code_t as needed
    switch (result_code) {
        case ANEURALNETWORKS_OUT_OF_MEMORY: return TK_ERROR_OUT_OF_MEMORY;
        case ANEURALNETWORKS_BAD_DATA:      return TK_ERROR_INVALID_ARGUMENT;
        case ANEURALNETWORKS_BAD_STATE:     return TK_ERROR_INVALID_STATE;
        default:                            return TK_ERROR_GPU_UNKNOWN;
    }
}

// --- Public API Implementation ---

TK_NODISCARD tk_error_code_t tk_nnapi_model_create(tk_nnapi_model_t** out_model, const char* model_path) {
    tk_nnapi_model_s* m = new (std::nothrow) tk_nnapi_model_s();
    if (!m) return TK_ERROR_OUT_OF_MEMORY;

    int err = ANeuralNetworksModel_create(&m->model);
    if (err != ANEURALNETWORKS_NO_ERROR) return translate_nnapi_result(err, "Model_create");

    // --- Model Definition ---
    // This is the most complex part. A real implementation would not load from a file.
    // Instead, it would define the entire graph here in code.
    // For example, for a simple Add operation:
    //
    // ANeuralNetworksOperandType tensor_type;
    // tensor_type.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    // tensor_type.scale = 0.0f;
    // tensor_type.zeroPoint = 0;
    // tensor_type.dimensionCount = 1;
    // uint32_t dims[] = {1};
    // tensor_type.dimensions = dims;
    //
    // ANeuralNetworksModel_addOperand(m->model, &tensor_type); // operand 0
    // ANeuralNetworksModel_addOperand(m->model, &tensor_type); // operand 1
    // ANeuralNetworksModel_addOperand(m->model, &tensor_type); // operand 2 (output)
    //
    // uint32_t inputs[] = {0, 1};
    // uint32_t outputs[] = {2};
    // ANeuralNetworksModel_addOperation(m->model, ANEURALNETWORKS_ADD, 2, inputs, 1, outputs);
    //
    // ANeuralNetworksModel_identifyInputsAndOutputs(m->model, ...);
    //
    // This process would be repeated for every layer in the desired network (e.g., YOLO).
    std::cout << "Placeholder: NNAPI model definition for '" << model_path << "' would happen here." << std::endl;


    err = ANeuralNetworksModel_finish(m->model);
    if (err != ANEURALNETWORKS_NO_ERROR) return translate_nnapi_result(err, "Model_finish");

    err = ANeuralNetworksCompilation_create(m->model, &m->compilation);
    if (err != ANEURALNETWORKS_NO_ERROR) return translate_nnapi_result(err, "Compilation_create");
    
    // Set compilation preferences, e.g., prefer low power or fast execution
    // ANeuralNetworksCompilation_setPreference(m->compilation, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);

    err = ANeuralNetworksCompilation_finish(m->compilation);
    if (err != ANEURALNETWORKS_NO_ERROR) return translate_nnapi_result(err, "Compilation_finish");

    *out_model = m;
    return TK_SUCCESS;
}

void tk_nnapi_model_destroy(tk_nnapi_model_t** model) {
    if (!model || !*model) return;
    ANeuralNetworksCompilation_free((*model)->compilation);
    ANeuralNetworksModel_free((*model)->model);
    delete *model;
    *model = nullptr;
}

TK_NODISCARD tk_error_code_t tk_nnapi_execution_create(tk_nnapi_model_t* model, tk_nnapi_execution_t** out_execution) {
    tk_nnapi_execution_s* exec = new (std::nothrow) tk_nnapi_execution_s();
    if (!exec) return TK_ERROR_OUT_OF_MEMORY;

    int err = ANeuralNetworksExecution_create(model->compilation, &exec->execution);
    if (err != ANEURALNETWORKS_NO_ERROR) {
        delete exec;
        return translate_nnapi_result(err, "Execution_create");
    }
    exec->event = nullptr;
    *out_execution = exec;
    return TK_SUCCESS;
}

void tk_nnapi_execution_destroy(tk_nnapi_execution_t** execution) {
    if (!execution || !*execution) return;
    if ((*execution)->event) {
        ANeuralNetworksEvent_free((*execution)->event);
    }
    ANeuralNetworksExecution_free((*execution)->execution);
    delete *execution;
    *execution = nullptr;
}

TK_NODISCARD tk_error_code_t tk_nnapi_execution_set_inputs(tk_nnapi_execution_t* execution, const tk_nnapi_buffer_t* inputs, uint32_t num_inputs) {
    for (uint32_t i = 0; i < num_inputs; ++i) {
        int err = ANeuralNetworksExecution_setInput(execution->execution, inputs[i].index, NULL, inputs[i].data, inputs[i].length);
        if (err != ANEURALNETWORKS_NO_ERROR) return translate_nnapi_result(err, "SetInput");
    }
    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t tk_nnapi_execution_set_outputs(tk_nnapi_execution_t* execution, const tk_nnapi_buffer_t* outputs, uint32_t num_outputs) {
    for (uint32_t i = 0; i < num_outputs; ++i) {
        int err = ANeuralNetworksExecution_setOutput(execution->execution, outputs[i].index, NULL, outputs[i].data, outputs[i].length);
        if (err != ANEURALNETWORKS_NO_ERROR) return translate_nnapi_result(err, "SetOutput");
    }
    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t tk_nnapi_execute_async(tk_nnapi_execution_t* execution) {
    if (execution->event) {
        ANeuralNetworksEvent_free(execution->event);
        execution->event = nullptr;
    }
    int err = ANeuralNetworksExecution_startCompute(execution->execution, &execution->event);
    return translate_nnapi_result(err, "startCompute");
}

TK_NODISCARD tk_error_code_t tk_nnapi_wait(tk_nnapi_execution_t* execution) {
    if (!execution->event) {
        return TK_ERROR_INVALID_STATE;
    }
    int err = ANeuralNetworksEvent_wait(execution->event);
    return translate_nnapi_result(err, "Event_wait");
}
