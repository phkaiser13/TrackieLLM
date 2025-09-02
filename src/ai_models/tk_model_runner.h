/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_model_runner.h
*
* This header file defines the public API for the LLM (Large Language Model)
* Runner. This component is the core reasoning engine of the Cortex, acting as
* a high-level interface to the underlying inference engine (e.g., llama.cpp).
*
* The design goes beyond simple text generation. It establishes a stateful
* conversational session (`tk_llm_runner_t`) and provides mechanisms for
* emulating "function calling" or "tool use" with a local LLM. This is achieved
* through structured prompt engineering and output parsing, all abstracted away
* from the Cortex.
*
* Key architectural features:
*   - Opaque handle (`tk_llm_runner_t`) for managing the LLM context and state.
*   - Structured definition of available "tools" that the LLM can request to use.
*   - Fusion of multimodal context (text from user, text describing vision)
*     into a single, coherent prompt.
*   - Structured output that differentiates between a textual response and a
*     request to execute a tool.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TRACKIELLM_AI_MODELS_TK_MODEL_RUNNER_H
#define TRACKIELLM_AI_MODELS_TK_MODEL_RUNNER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "internal_tools/tk_file_manager.h"

// Forward-declare the primary runner and result objects as opaque types.
typedef struct tk_llm_runner_s tk_llm_runner_t;
typedef struct tk_llm_result_s tk_llm_result_t;

/**
 * @struct tk_llm_config_t
 * @brief Configuration for initializing the LLM Runner.
 */
typedef struct {
    tk_path_t* model_path;          /**< Path to the GGUF model file. */
    uint32_t   context_size;        /**< The context window size for the model (e.g., 4096). */
    uint32_t   gpu_layers_offload;  /**< Number of model layers to offload to the GPU. 0 for CPU only. */
    const char* system_prompt;      /**< The initial, high-level instruction defining the AI's persona and goal. */
    uint32_t   random_seed;         /**< Seed for the random number generator for reproducible outputs. */
} tk_llm_config_t;

/**
 * @struct tk_llm_tool_definition_t
 * @brief Describes a single "tool" (a local C function) that the LLM can use.
 *
 * This information is used to construct a special section of the prompt that
 * instructs the LLM on how to request a function call.
 */
typedef struct {
    const char* name;               /**< The exact name of the function to be called (e.g., "locate_object"). */
    const char* description;        /**< A detailed, natural language description of what the tool does.
                                         This is critical for the LLM to understand its purpose. */
    const char* parameters_json_schema; /**< A JSON schema describing the parameters the tool accepts.
                                             Example: "{\"type\":\"object\",\"properties\":{\"object_name\":{\"type\":\"string\"}}}" */
} tk_llm_tool_definition_t;

/**
 * @struct tk_llm_prompt_context_t
 * @brief Encapsulates all contextual information for generating a single response.
 */
typedef struct {
    const char* user_transcription; /**< The text transcribed from the user's speech. */
    const char* vision_context;     /**< A textual summary of the current visual scene, generated from
                                         the vision pipeline results. Can be NULL if no visual data. */
} tk_llm_prompt_context_t;

/**
 * @enum tk_llm_result_type_e
 * @brief Discriminator for the type of result returned by the LLM runner.
 */
typedef enum {
    TK_LLM_RESULT_TYPE_UNKNOWN,
    TK_LLM_RESULT_TYPE_TEXT_RESPONSE, /**< The result is a natural language response for the user. */
    TK_LLM_RESULT_TYPE_TOOL_CALL      /**< The result is a request to execute a tool. */
} tk_llm_result_type_e;

/**
 * @struct tk_llm_tool_call_t
 * @brief Represents a request from the LLM to call a tool.
 */
typedef struct {
    char* name;                     /**< The name of the tool to call. Owned by the result object. */
    char* arguments_json;           /**< The arguments for the tool, formatted as a JSON string. Owned by the result object. */
} tk_llm_tool_call_t;

/**
 * @struct tk_llm_result_s
 * @brief A structured container for the LLM's output.
 *
 * This structure and all its contents are allocated by `tk_llm_runner_generate_response`
 * and must be freed by the caller using `tk_llm_result_destroy`.
 */
struct tk_llm_result_s {
    tk_llm_result_type_e type;
    union {
        char* text_response;        /**< Valid if type is TEXT_RESPONSE. Owned by the result object. */
        tk_llm_tool_call_t tool_call; /**< Valid if type is TOOL_CALL. */
    } data;
};

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Runner Lifecycle Management
//------------------------------------------------------------------------------

/**
 * @brief Creates and initializes a new LLM Runner instance.
 *
 * Loads the specified GGUF model, initializes the llama.cpp backend, allocates
 * the context, and prepares the runner for conversational inference.
 *
 * @param[out] out_runner Pointer to receive the address of the new runner instance.
 * @param[in] config The configuration for the runner.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL or config is invalid.
 * @return TK_ERROR_OUT_OF_MEMORY on memory allocation failure.
 * @return TK_ERROR_MODEL_LOAD_FAILED if the GGUF model cannot be loaded.
 */
TK_NODISCARD tk_error_code_t tk_llm_runner_create(tk_llm_runner_t** out_runner, const tk_llm_config_t* config);

/**
 * @brief Destroys an LLM Runner instance and frees all associated resources.
 *
 * @param[in,out] runner Pointer to the runner instance to be destroyed.
 */
void tk_llm_runner_destroy(tk_llm_runner_t** runner);

//------------------------------------------------------------------------------
// Conversational Inference
//------------------------------------------------------------------------------

/**
 * @brief Generates a structured response based on the current context and available tools.
 *
 * This is the core function of the runner. It performs the following steps:
 * 1. Constructs a complex prompt from the system prompt, conversation history,
 *    available tools, and the new multimodal context.
 * 2. Runs inference using the llama.cpp backend.
 * 3. Parses the raw text output from the LLM.
 * 4. Determines if the output is a direct response or a tool call request.
 * 5. Populates and returns a structured `tk_llm_result_t` object.
 *
 * @param[in] runner The LLM runner instance.
 * @param[in] prompt_context The new multimodal context for this turn.
 * @param[in] available_tools An array of tool definitions.
 * @param[in] tool_count The number of tools in the `available_tools` array.
 * @param[out] out_result Pointer to receive the newly allocated result object.
 *                        The caller assumes ownership and must free it with
 *                        `tk_llm_result_destroy`.
 *
 * @return TK_SUCCESS on successful generation.
 * @return TK_ERROR_INVALID_ARGUMENT if required pointers are NULL.
 * @return TK_ERROR_INFERENCE_FAILED if the backend fails during generation.
 * @return TK_ERROR_OUT_OF_MEMORY if result allocation fails.
 *
 * @par Thread-Safety
 * This function is NOT thread-safe. Only one thread should interact with a
 * given runner instance at a time.
 */
TK_NODISCARD tk_error_code_t tk_llm_runner_generate_response(
    tk_llm_runner_t* runner,
    const tk_llm_prompt_context_t* prompt_context,
    const tk_llm_tool_definition_t* available_tools,
    size_t tool_count,
    tk_llm_result_t** out_result
);

/**
 * @brief Adds the result of a tool call back into the conversation history.
 *
 * After the Cortex executes a tool requested by the LLM, it must call this
 * function to inform the LLM of the outcome. This allows the LLM to use the
 * tool's output to formulate its final response to the user.
 *
 * @param[in] runner The LLM runner instance.
 * @param[in] tool_name The name of the tool that was executed.
 * @param[in] tool_output The textual result from the tool's execution.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_INVALID_ARGUMENT if any arguments are NULL.
 *
 * @par Thread-Safety
 * This function is NOT thread-safe.
 */
TK_NODISCARD tk_error_code_t tk_llm_runner_add_tool_response(tk_llm_runner_t* runner, const char* tool_name, const char* tool_output);

/**
 * @brief Resets the conversation history of the LLM runner.
 *
 * Clears the entire context, except for the initial system prompt.
 *
 * @param[in] runner The LLM runner instance.
 * @return TK_SUCCESS on success.
 */
TK_NODISCARD tk_error_code_t tk_llm_runner_reset_context(tk_llm_runner_t* runner);

//------------------------------------------------------------------------------
// Result Data Management
//------------------------------------------------------------------------------

/**
 * @brief Destroys an LLM result object and all its associated data.
 *
 * @param[in,out] result A pointer to the result object to be destroyed.
 */
void tk_llm_result_destroy(tk_llm_result_t** result);

#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_AI_MODELS_TK_MODEL_RUNNER_H