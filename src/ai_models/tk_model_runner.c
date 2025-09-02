/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_model_runner.c
 *
 * This source file implements the LLM (Large Language Model) Runner module.
 * This component is the core reasoning engine of the Cortex, acting as a
 * high-level interface to the underlying inference engine (llama.cpp).
 *
 * The implementation provides a stateful conversational session and mechanisms
 * for emulating "function calling" or "tool use" with a local LLM through
 * structured prompt engineering and output parsing.
 *
 * Key architectural features:
 *   - Opaque handle for managing the LLM context and state.
 *   - Structured definition of available "tools" that the LLM can request to use.
 *   - Fusion of multimodal context into a single, coherent prompt.
 *   - Structured output that differentiates between a textual response and a
 *     request to execute a tool.
 *
 * Dependencies:
 *   - llama.cpp (https://github.com/ggerganov/llama.cpp)
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "ai_models/tk_model_runner.h"
#include "utils/tk_logging.h"
#include "utils/tk_error_handling.h"

// Include llama.cpp headers
#include "llama.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>

// Maximum length for prompts and responses
#define MAX_PROMPT_LENGTH 8192
#define MAX_RESPONSE_LENGTH 4096
#define MAX_HISTORY_LENGTH 1024
#define MAX_TOOL_CALL_LENGTH 1024

// Maximum number of conversation history entries
#define MAX_HISTORY_ENTRIES 32

// Maximum number of tools
#define MAX_TOOLS 64

// Internal structures for conversation history
typedef struct {
    char* role;      // "user", "assistant", or "tool"
    char* content;   // Content of the message
} tk_llm_history_entry_t;

// Internal structure for LLM runner context
struct tk_llm_runner_s {
    struct llama_context* ctx;          // llama.cpp context
    struct llama_model* model;          // llama.cpp model
    tk_llm_config_t config;             // Configuration copy
    char* system_prompt;                // System prompt
    tk_llm_history_entry_t* history;    // Conversation history
    size_t history_count;               // Number of history entries
    size_t history_capacity;            // Maximum number of history entries
    char* last_response;                // Last response from the model
    bool is_processing;                 // Whether the runner is currently processing
    uint32_t seed;                      // Random seed for generation
};

// Internal structure for prompt building
typedef struct {
    char* buffer;
    size_t capacity;
    size_t length;
} tk_prompt_builder_t;

// Internal structure for JSON parsing
typedef struct {
    const char* json_str;
    size_t pos;
    size_t len;
} tk_json_parser_t;

//------------------------------------------------------------------------------
// Private helper functions
//------------------------------------------------------------------------------

/**
 * @brief Converts llama.cpp error codes to Trackie error codes
 */
static tk_error_code_t convert_llama_error(int llama_error) {
    switch (llama_error) {
        case 0:
            return TK_SUCCESS;
        case -1:
            return TK_ERROR_MODEL_LOAD_FAILED;
        case -2:
            return TK_ERROR_INFERENCE_FAILED;
        case -3:
            return TK_ERROR_OUT_OF_MEMORY;
        default:
            return TK_ERROR_INTERNAL;
    }
}

/**
 * @brief Allocates and copies a string
 */
static char* duplicate_string(const char* src) {
    if (!src) return NULL;
    
    size_t len = strlen(src);
    char* dup = malloc(len + 1);
    if (!dup) return NULL;
    
    memcpy(dup, src, len + 1);
    return dup;
}

/**
 * @brief Frees memory allocated for a string
 */
static void free_string(char* str) {
    if (str) {
        free(str);
    }
}

/**
 * @brief Initializes a prompt builder
 */
static tk_prompt_builder_t* prompt_builder_create(size_t initial_capacity) {
    tk_prompt_builder_t* builder = calloc(1, sizeof(tk_prompt_builder_t));
    if (!builder) return NULL;
    
    builder->buffer = malloc(initial_capacity);
    if (!builder->buffer) {
        free(builder);
        return NULL;
    }
    
    builder->capacity = initial_capacity;
    builder->length = 0;
    builder->buffer[0] = '\0';
    
    return builder;
}

/**
 * @brief Destroys a prompt builder
 */
static void prompt_builder_destroy(tk_prompt_builder_t* builder) {
    if (!builder) return;
    
    if (builder->buffer) {
        free(builder->buffer);
    }
    
    free(builder);
}

/**
 * @brief Appends text to a prompt builder
 */
static tk_error_code_t prompt_builder_append(tk_prompt_builder_t* builder, const char* text) {
    if (!builder || !text) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    size_t text_len = strlen(text);
    size_t needed_capacity = builder->length + text_len + 1;
    
    // Resize buffer if needed
    if (needed_capacity > builder->capacity) {
        size_t new_capacity = builder->capacity * 2;
        if (new_capacity < needed_capacity) {
            new_capacity = needed_capacity;
        }
        
        char* new_buffer = realloc(builder->buffer, new_capacity);
        if (!new_buffer) {
            return TK_ERROR_OUT_OF_MEMORY;
        }
        
        builder->buffer = new_buffer;
        builder->capacity = new_capacity;
    }
    
    // Append text
    memcpy(builder->buffer + builder->length, text, text_len);
    builder->length += text_len;
    builder->buffer[builder->length] = '\0';
    
    return TK_SUCCESS;
}

/**
 * @brief Appends formatted text to a prompt builder
 */
static tk_error_code_t prompt_builder_appendf(tk_prompt_builder_t* builder, const char* format, ...) {
    if (!builder || !format) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    va_list args;
    va_start(args, format);
    
    // Calculate needed space
    int needed = vsnprintf(NULL, 0, format, args);
    va_end(args);
    
    if (needed < 0) {
        return TK_ERROR_INTERNAL;
    }
    
    // Allocate temporary buffer
    char* temp_buffer = malloc(needed + 1);
    if (!temp_buffer) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Format the string
    va_start(args, format);
    vsnprintf(temp_buffer, needed + 1, format, args);
    va_end(args);
    
    // Append to builder
    tk_error_code_t result = prompt_builder_append(builder, temp_buffer);
    
    // Clean up
    free(temp_buffer);
    
    return result;
}

/**
 * @brief Initializes a JSON parser
 */
static void json_parser_init(tk_json_parser_t* parser, const char* json_str) {
    if (!parser || !json_str) return;
    
    parser->json_str = json_str;
    parser->pos = 0;
    parser->len = strlen(json_str);
}

/**
 * @brief Skips whitespace in JSON parser
 */
static void json_parser_skip_whitespace(tk_json_parser_t* parser) {
    if (!parser) return;
    
    while (parser->pos < parser->len && 
           (parser->json_str[parser->pos] == ' ' || 
            parser->json_str[parser->pos] == '\t' || 
            parser->json_str[parser->pos] == '\n' || 
            parser->json_str[parser->pos] == '\r')) {
        parser->pos++;
    }
}

/**
 * @brief Parses a JSON string value
 */
static bool json_parser_parse_string(tk_json_parser_t* parser, char** out_string) {
    if (!parser || !out_string) return false;
    
    json_parser_skip_whitespace(parser);
    
    // Check for opening quote
    if (parser->pos >= parser->len || parser->json_str[parser->pos] != '"') {
        return false;
    }
    
    parser->pos++; // Skip opening quote
    
    // Find closing quote
    size_t start = parser->pos;
    while (parser->pos < parser->len && parser->json_str[parser->pos] != '"') {
        parser->pos++;
    }
    
    if (parser->pos >= parser->len) {
        return false; // No closing quote
    }
    
    // Extract string content
    size_t length = parser->pos - start;
    *out_string = malloc(length + 1);
    if (!*out_string) {
        return false;
    }
    
    memcpy(*out_string, parser->json_str + start, length);
    (*out_string)[length] = '\0';
    
    parser->pos++; // Skip closing quote
    return true;
}

/**
 * @brief Parses a JSON object
 */
static bool json_parser_parse_object(tk_json_parser_t* parser, char** out_json) {
    if (!parser || !out_json) return false;
    
    json_parser_skip_whitespace(parser);
    
    // Check for opening brace
    if (parser->pos >= parser->len || parser->json_str[parser->pos] != '{') {
        return false;
    }
    
    size_t start = parser->pos;
    parser->pos++; // Skip opening brace
    
    int brace_count = 1;
    while (parser->pos < parser->len && brace_count > 0) {
        if (parser->json_str[parser->pos] == '{') {
            brace_count++;
        } else if (parser->json_str[parser->pos] == '}') {
            brace_count--;
        }
        parser->pos++;
    }
    
    if (brace_count != 0) {
        return false; // Mismatched braces
    }
    
    // Extract object content
    size_t length = parser->pos - start;
    *out_json = malloc(length + 1);
    if (!*out_json) {
        return false;
    }
    
    memcpy(*out_json, parser->json_str + start, length);
    (*out_json)[length] = '\0';
    
    return true;
}

/**
 * @brief Finds a key in a JSON object and extracts its value
 */
static bool json_extract_value(const char* json_str, const char* key, char** out_value) {
    if (!json_str || !key || !out_value) return false;
    
    // Create search pattern
    size_t key_len = strlen(key);
    char* pattern = malloc(key_len + 4); // "key":
    if (!pattern) return false;
    
    sprintf(pattern, "\"%s\":", key);
    
    // Find the key in the JSON
    char* key_pos = strstr(json_str, pattern);
    if (!key_pos) {
        free(pattern);
        return false;
    }
    
    // Move past the key and colon
    key_pos += strlen(pattern);
    
    // Skip whitespace
    while (*key_pos == ' ' || *key_pos == '\t') {
        key_pos++;
    }
    
    // Parse the value based on its type
    bool result = false;
    if (*key_pos == '"') {
        // String value
        tk_json_parser_t parser = {0};
        parser.json_str = key_pos;
        parser.len = strlen(key_pos);
        result = json_parser_parse_string(&parser, out_value);
    } else if (*key_pos == '{' || *key_pos == '[') {
        // Object or array value
        tk_json_parser_t parser = {0};
        parser.json_str = key_pos;
        parser.len = strlen(key_pos);
        result = json_parser_parse_object(&parser, out_value);
    } else {
        // Other value types (number, boolean, null)
        char* end_pos = key_pos;
        while (*end_pos != ',' && *end_pos != '}' && *end_pos != '\0') {
            end_pos++;
        }
        
        size_t value_len = end_pos - key_pos;
        *out_value = malloc(value_len + 1);
        if (*out_value) {
            memcpy(*out_value, key_pos, value_len);
            (*out_value)[value_len] = '\0';
            result = true;
        }
    }
    
    free(pattern);
    return result;
}

/**
 * @brief Initializes conversation history
 */
static tk_error_code_t init_history(tk_llm_runner_t* runner) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    runner->history_capacity = MAX_HISTORY_ENTRIES;
    runner->history = calloc(runner->history_capacity, sizeof(tk_llm_history_entry_t));
    if (!runner->history) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    runner->history_count = 0;
    return TK_SUCCESS;
}

/**
 * @brief Adds an entry to conversation history
 */
static tk_error_code_t add_history_entry(
    tk_llm_runner_t* runner,
    const char* role,
    const char* content
) {
    if (!runner || !role || !content) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Check if we need to resize history
    if (runner->history_count >= runner->history_capacity) {
        size_t new_capacity = runner->history_capacity * 2;
        tk_llm_history_entry_t* new_history = realloc(
            runner->history,
            new_capacity * sizeof(tk_llm_history_entry_t)
        );
        
        if (!new_history) {
            return TK_ERROR_OUT_OF_MEMORY;
        }
        
        runner->history = new_history;
        runner->history_capacity = new_capacity;
    }
    
    // Add new entry
    size_t index = runner->history_count;
    runner->history[index].role = duplicate_string(role);
    runner->history[index].content = duplicate_string(content);
    
    if (!runner->history[index].role || !runner->history[index].content) {
        free_string(runner->history[index].role);
        free_string(runner->history[index].content);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    runner->history_count++;
    return TK_SUCCESS;
}

/**
 * @brief Clears conversation history
 */
static void clear_history(tk_llm_runner_t* runner) {
    if (!runner) return;
    
    for (size_t i = 0; i < runner->history_count; i++) {
        free_string(runner->history[i].role);
        free_string(runner->history[i].content);
    }
    
    runner->history_count = 0;
}

/**
 * @brief Builds a prompt with system instructions, history, and context
 */
static tk_error_code_t build_prompt(
    tk_llm_runner_t* runner,
    const tk_llm_prompt_context_t* prompt_context,
    const tk_llm_tool_definition_t* available_tools,
    size_t tool_count,
    tk_prompt_builder_t* builder
) {
    if (!runner || !builder) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Start with system prompt
    if (runner->system_prompt) {
        TK_CHECK_ERROR(prompt_builder_append(builder, runner->system_prompt));
        TK_CHECK_ERROR(prompt_builder_append(builder, "\n\n"));
    }
    
    // Add tool definitions if any
    if (available_tools && tool_count > 0) {
        TK_CHECK_ERROR(prompt_builder_append(builder, "You have access to the following tools:\n"));
        
        for (size_t i = 0; i < tool_count; i++) {
            TK_CHECK_ERROR(prompt_builder_appendf(
                builder,
                "- %s: %s\n",
                available_tools[i].name,
                available_tools[i].description
            ));
        }
        
        TK_CHECK_ERROR(prompt_builder_append(builder, 
            "To use a tool, respond with a JSON object in this format: "
            "{\"tool_call\": {\"name\": \"tool_name\", \"arguments\": {\"param1\": \"value1\"}}}\n"
            "After using a tool, you will receive its output to formulate your final response.\n\n"
        ));
    }
    
    // Add conversation history
    for (size_t i = 0; i < runner->history_count; i++) {
        TK_CHECK_ERROR(prompt_builder_appendf(
            builder,
            "%s: %s\n",
            runner->history[i].role,
            runner->history[i].content
        ));
    }
    
    // Add current context
    if (prompt_context) {
        TK_CHECK_ERROR(prompt_builder_append(builder, "user: "));
        
        if (prompt_context->user_transcription) {
            TK_CHECK_ERROR(prompt_builder_append(builder, prompt_context->user_transcription));
        }
        
        if (prompt_context->vision_context) {
            if (prompt_context->user_transcription) {
                TK_CHECK_ERROR(prompt_builder_append(builder, " "));
            }
            TK_CHECK_ERROR(prompt_builder_append(builder, prompt_context->vision_context));
        }
        
        TK_CHECK_ERROR(prompt_builder_append(builder, "\nassistant: "));
    }
    
    return TK_SUCCESS;
}

/**
 * @brief Runs inference using llama.cpp
 */
static tk_error_code_t run_inference(
    tk_llm_runner_t* runner,
    const char* prompt,
    char* response_buffer,
    size_t response_buffer_size
) {
    if (!runner || !prompt || !response_buffer) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Initialize llama sampling parameters
    struct llama_sampling_params sparams = {
        .n_prev = 64,
        .n_probs = 0,
        .top_k = 40,
        .top_p = 0.95f,
        .min_p = 0.05f,
        .temp = 0.7f,
        .repeat_penalty = 1.1f,
        .presence_penalty = 0.0f,
        .frequency_penalty = 0.0f,
        .penalty_last_n = 64,
        .penalty_repeat = 1.1f,
        .penalty_freq = 0.0f,
        .penalty_present = 0.0f,
        .mirostat = 0,
        .mirostat_tau = 5.0f,
        .mirostat_eta = 0.1f,
        .penalize_nl = true,
        .ignore_eos = false,
        .typical_p = 1.0f,
        .dynatemp_range = 0.0f,
        .dynatemp_exponent = 1.0f,
        .grammar = "",
        .n_seq_max = 1,
        .n_seq_keep = 1,
        .n_seq_discard = 0,
        .n_seq_min_keep = 1,
        .n_seq_min_discard = 0,
        .n_seq_min_keep_ratio = 0.5f,
        .n_seq_min_discard_ratio = 0.1f,
        .n_seq_min_keep_count = 1,
        .n_seq_min_discard_count = 1,
        .n_seq_min_keep_percentage = 50,
        .n_seq_min_discard_percentage = 10,
        .n_seq_min_keep_threshold = 0.5f,
        .n_seq_min_discard_threshold = 0.1f,
        .n_seq_min_keep_factor = 1.0f,
        .n_seq_min_discard_factor = 1.0f,
        .n_seq_min_keep_offset = 0,
        .n_seq_min_discard_offset = 0,
        .n_seq_min_keep_scale = 1.0f,
        .n_seq_min_discard_scale = 1.0f,
        .n_seq_min_keep_bias = 0.0f,
        .n_seq_min_discard_bias = 0.0f,
        .n_seq_min_keep_weight = 1.0f,
        .n_seq_min_discard_weight = 1.0f,
        .n_seq_min_keep_alpha = 1.0f,
        .n_seq_min_discard_alpha = 1.0f,
        .n_seq_min_keep_beta = 1.0f,
        .n_seq_min_discard_beta = 1.0f,
        .n_seq_min_keep_gamma = 1.0f,
        .n_seq_min_discard_gamma = 1.0f,
        .n_seq_min_keep_delta = 1.0f,
        .n_seq_min_discard_delta = 1.0f,
        .n_seq_min_keep_epsilon = 1.0f,
        .n_seq_min_discard_epsilon = 1.0f,
        .n_seq_min_keep_zeta = 1.0f,
        .n_seq_min_discard_zeta = 1.0f,
        .n_seq_min_keep_eta = 1.0f,
        .n_seq_min_discard_eta = 1.0f,
        .n_seq_min_keep_theta = 1.0f,
        .n_seq_min_discard_theta = 1.0f,
        .n_seq_min_keep_iota = 1.0f,
        .n_seq_min_discard_iota = 1.0f,
        .n_seq_min_keep_kappa = 1.0f,
        .n_seq_min_discard_kappa = 1.0f,
        .n_seq_min_keep_lambda = 1.0f,
        .n_seq_min_discard_lambda = 1.0f,
        .n_seq_min_keep_mu = 1.0f,
        .n_seq_min_discard_mu = 1.0f,
        .n_seq_min_keep_nu = 1.0f,
        .n_seq_min_discard_nu = 1.0f,
        .n_seq_min_keep_xi = 1.0f,
        .n_seq_min_discard_xi = 1.0f,
        .n_seq_min_keep_omicron = 1.0f,
        .n_seq_min_discard_omicron = 1.0f,
        .n_seq_min_keep_pi = 1.0f,
        .n_seq_min_discard_pi = 1.0f,
        .n_seq_min_keep_rho = 1.0f,
        .n_seq_min_discard_rho = 1.0f,
        .n_seq_min_keep_sigma = 1.0f,
        .n_seq_min_discard_sigma = 1.0f,
        .n_seq_min_keep_tau = 1.0f,
        .n_seq_min_discard_tau = 1.0f,
        .n_seq_min_keep_upsilon = 1.0f,
        .n_seq_min_discard_upsilon = 1.0f,
        .n_seq_min_keep_phi = 1.0f,
        .n_seq_min_discard_phi = 1.0f,
        .n_seq_min_keep_chi = 1.0f,
        .n_seq_min_discard_chi = 1.0f,
        .n_seq_min_keep_psi = 1.0f,
        .n_seq_min_discard_psi = 1.0f,
        .n_seq_min_keep_omega = 1.0f,
        .n_seq_min_discard_omega = 1.0f,
    };
    
    // Create sampling context
    struct llama_sampling_context* sctx = llama_sampling_init(sparams);
    if (!sctx) {
        return TK_ERROR_INFERENCE_FAILED;
    }
    
    // Tokenize the prompt
    std::vector<llama_token> prompt_tokens = ::llama_tokenize(runner->ctx, prompt, true);
    
    // Evaluate the prompt
    for (size_t i = 0; i < prompt_tokens.size(); i++) {
        if (llama_decode(runner->ctx, llama_batch_get_one(&prompt_tokens[i], 1, i, 0))) {
            llama_sampling_free(sctx);
            return TK_ERROR_INFERENCE_FAILED;
        }
    }
    
    // Generate response
    response_buffer[0] = '\0';
    size_t response_len = 0;
    
    for (int i = 0; i < 512; i++) { // Max 512 tokens
        const llama_token id = llama_sampling_sample(sctx, runner->ctx, NULL, 0);
        
        // Check if we've reached the end of generation
        if (id == llama_token_eos(runner->model)) {
            break;
        }
        
        // Add token to response
        char token_str[32];
        int token_len = llama_token_to_piece(runner->ctx, id, token_str, sizeof(token_str), 0, true);
        if (token_len > 0 && response_len + token_len < response_buffer_size - 1) {
            strncat(response_buffer, token_str, response_buffer_size - response_len - 1);
            response_len += token_len;
        }
        
        // Evaluate next token
        if (llama_decode(runner->ctx, llama_batch_get_one(&id, 1, prompt_tokens.size() + i, 0))) {
            llama_sampling_free(sctx);
            return TK_ERROR_INFERENCE_FAILED;
        }
    }
    
    llama_sampling_free(sctx);
    return TK_SUCCESS;
}

/**
 * @brief Parses the LLM response to determine if it's a tool call
 */
static bool parse_tool_call(const char* response, tk_llm_tool_call_t* out_tool_call) {
    if (!response || !out_tool_call) return false;
    
    // Look for tool_call pattern
    const char* tool_call_start = strstr(response, "{\"tool_call\":");
    if (!tool_call_start) return false;
    
    // Parse the JSON object
    tk_json_parser_t parser = {0};
    json_parser_init(&parser, tool_call_start);
    
    // Skip to the tool_call object
    json_parser_skip_whitespace(&parser);
    if (parser.pos >= parser.len || parser.json_str[parser.pos] != '{') {
        return false;
    }
    
    parser.pos++; // Skip opening brace
    json_parser_skip_whitespace(&parser);
    
    // Look for "tool_call" key
    if (parser.pos + 12 > parser.len || strncmp(parser.json_str + parser.pos, "\"tool_call\"", 11) != 0) {
        return false;
    }
    
    parser.pos += 11; // Skip "tool_call"
    json_parser_skip_whitespace(&parser);
    
    // Skip colon
    if (parser.pos >= parser.len || parser.json_str[parser.pos] != ':') {
        return false;
    }
    
    parser.pos++; // Skip colon
    json_parser_skip_whitespace(&parser);
    
    // Parse the tool_call object
    char* tool_call_json = NULL;
    if (!json_parser_parse_object(&parser, &tool_call_json)) {
        return false;
    }
    
    // Extract name and arguments
    char* name = NULL;
    char* arguments = NULL;
    
    bool success = json_extract_value(tool_call_json, "name", &name) &&
                   json_extract_value(tool_call_json, "arguments", &arguments);
    
    free(tool_call_json);
    
    if (success) {
        out_tool_call->name = name;
        out_tool_call->arguments_json = arguments;
        return true;
    } else {
        free_string(name);
        free_string(arguments);
        return false;
    }
}

//------------------------------------------------------------------------------
// Public API Implementation
//------------------------------------------------------------------------------

tk_error_code_t tk_llm_runner_create(
    tk_llm_runner_t** out_runner,
    const tk_llm_config_t* config
) {
    if (!out_runner || !config || !config->model_path) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_runner = NULL;
    
    // Allocate runner structure
    tk_llm_runner_t* runner = calloc(1, sizeof(tk_llm_runner_t));
    if (!runner) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Copy configuration
    runner->config = *config;
    runner->config.model_path = NULL; // We don't copy the path object
    
    // Copy system prompt
    if (config->system_prompt) {
        runner->system_prompt = duplicate_string(config->system_prompt);
        if (!runner->system_prompt) {
            free(runner);
            return TK_ERROR_OUT_OF_MEMORY;
        }
    }
    
    // Set seed
    runner->seed = config->random_seed;
    
    // Initialize llama.cpp backend
    llama_backend_init(false);
    
    // Load model
    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config->gpu_layers_offload;
    
    const char* model_path_str = config->model_path->path_str;
    runner->model = llama_load_model_from_file(model_path_str, model_params);
    if (!runner->model) {
        TK_LOG_ERROR("Failed to load LLM model from: %s", model_path_str);
        free_string(runner->system_prompt);
        free(runner);
        return TK_ERROR_MODEL_LOAD_FAILED;
    }
    
    // Initialize context
    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config->context_size;
    ctx_params.seed = config->random_seed;
    ctx_params.n_batch = 512;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    
    runner->ctx = llama_new_context_with_model(runner->model, ctx_params);
    if (!runner->ctx) {
        TK_LOG_ERROR("Failed to create LLM context");
        llama_free_model(runner->model);
        free_string(runner->system_prompt);
        free(runner);
        return TK_ERROR_MODEL_LOAD_FAILED;
    }
    
    // Initialize conversation history
    tk_error_code_t result = init_history(runner);
    if (result != TK_SUCCESS) {
        llama_free(runner->ctx);
        llama_free_model(runner->model);
        free_string(runner->system_prompt);
        free(runner);
        return result;
    }
    
    // Initialize last response
    runner->last_response = NULL;
    runner->is_processing = false;
    
    *out_runner = runner;
    return TK_SUCCESS;
}

void tk_llm_runner_destroy(tk_llm_runner_t** runner) {
    if (!runner || !*runner) return;
    
    tk_llm_runner_t* r = *runner;
    
    // Free llama.cpp resources
    if (r->ctx) {
        llama_free(r->ctx);
    }
    
    if (r->model) {
        llama_free_model(r->model);
    }
    
    // Free system prompt
    free_string(r->system_prompt);
    
    // Clear and free history
    clear_history(r);
    if (r->history) {
        free(r->history);
    }
    
    // Free last response
    free_string(r->last_response);
    
    // Free runner itself
    free(r);
    *runner = NULL;
    
    // Clean up llama.cpp backend
    llama_backend_free();
}

tk_error_code_t tk_llm_runner_generate_response(
    tk_llm_runner_t* runner,
    const tk_llm_prompt_context_t* prompt_context,
    const tk_llm_tool_definition_t* available_tools,
    size_t tool_count,
    tk_llm_result_t** out_result
) {
    if (!runner || !out_result) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_result = NULL;
    
    // Check if already processing
    if (runner->is_processing) {
        return TK_ERROR_INVALID_STATE;
    }
    
    runner->is_processing = true;
    
    // Allocate result structure
    tk_llm_result_t* result = calloc(1, sizeof(tk_llm_result_t));
    if (!result) {
        runner->is_processing = false;
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Build prompt
    tk_prompt_builder_t* prompt_builder = prompt_builder_create(MAX_PROMPT_LENGTH);
    if (!prompt_builder) {
        free(result);
        runner->is_processing = false;
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    tk_error_code_t error = build_prompt(
        runner,
        prompt_context,
        available_tools,
        tool_count,
        prompt_builder
    );
    
    if (error != TK_SUCCESS) {
        prompt_builder_destroy(prompt_builder);
        free(result);
        runner->is_processing = false;
        return error;
    }
    
    // Run inference
    char response_buffer[MAX_RESPONSE_LENGTH] = {0};
    error = run_inference(runner, prompt_builder->buffer, response_buffer, sizeof(response_buffer));
    prompt_builder_destroy(prompt_builder);
    
    if (error != TK_SUCCESS) {
        free(result);
        runner->is_processing = false;
        return error;
    }
    
    // Store response
    free_string(runner->last_response);
    runner->last_response = duplicate_string(response_buffer);
    
    // Add user prompt to history
    if (prompt_context && prompt_context->user_transcription) {
        add_history_entry(runner, "user", prompt_context->user_transcription);
    }
    
    // Parse response to determine type
    tk_llm_tool_call_t tool_call = {0};
    if (parse_tool_call(response_buffer, &tool_call)) {
        // This is a tool call
        result->type = TK_LLM_RESULT_TYPE_TOOL_CALL;
        result->data.tool_call.name = tool_call.name;
        result->data.tool_call.arguments_json = tool_call.arguments_json;
        
        // Add tool call to history
        add_history_entry(runner, "assistant", response_buffer);
    } else {
        // This is a text response
        result->type = TK_LLM_RESULT_TYPE_TEXT_RESPONSE;
        result->data.text_response = duplicate_string(response_buffer);
        
        if (!result->data.text_response) {
            free(result);
            runner->is_processing = false;
            return TK_ERROR_OUT_OF_MEMORY;
        }
        
        // Add text response to history
        add_history_entry(runner, "assistant", response_buffer);
    }
    
    runner->is_processing = false;
    *out_result = result;
    return TK_SUCCESS;
}

tk_error_code_t tk_llm_runner_add_tool_response(
    tk_llm_runner_t* runner,
    const char* tool_name,
    const char* tool_output
) {
    if (!runner || !tool_name || !tool_output) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Create tool response message
    size_t message_len = strlen(tool_name) + strlen(tool_output) + 16;
    char* message = malloc(message_len);
    if (!message) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    snprintf(message, message_len, "Tool %s output: %s", tool_name, tool_output);
    
    // Add to history
    tk_error_code_t result = add_history_entry(runner, "tool", message);
    
    free(message);
    return result;
}

tk_error_code_t tk_llm_runner_reset_context(tk_llm_runner_t* runner) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    clear_history(runner);
    free_string(runner->last_response);
    runner->last_response = NULL;
    
    return TK_SUCCESS;
}

void tk_llm_result_destroy(tk_llm_result_t** result) {
    if (!result || !*result) return;
    
    tk_llm_result_t* r = *result;
    
    if (r->type == TK_LLM_RESULT_TYPE_TEXT_RESPONSE) {
        free_string(r->data.text_response);
    } else if (r->type == TK_LLM_RESULT_TYPE_TOOL_CALL) {
        free_string(r->data.tool_call.name);
        free_string(r->data.tool_call.arguments_json);
    }
    
    free(r);
    *result = NULL;
}

//------------------------------------------------------------------------------
// Additional helper functions for advanced features
//------------------------------------------------------------------------------

/**
 * @brief Sets a custom system prompt
 */
tk_error_code_t tk_llm_runner_set_system_prompt(tk_llm_runner_t* runner, const char* system_prompt) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    free_string(runner->system_prompt);
    
    if (system_prompt) {
        runner->system_prompt = duplicate_string(system_prompt);
        if (!runner->system_prompt) {
            return TK_ERROR_OUT_OF_MEMORY;
        }
    } else {
        runner->system_prompt = NULL;
    }
    
    return TK_SUCCESS;
}

/**
 * @brief Gets information about the loaded model
 */
tk_error_code_t tk_llm_runner_get_model_info(
    tk_llm_runner_t* runner,
    uint32_t* out_context_size,
    uint32_t* out_embedding_size,
    uint32_t* out_vocab_size
) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    if (out_context_size) {
        *out_context_size = llama_n_ctx(runner->ctx);
    }
    
    if (out_embedding_size) {
        *out_embedding_size = llama_n_embd(runner->model);
    }
    
    if (out_vocab_size) {
        *out_vocab_size = llama_n_vocab(runner->model);
    }
    
    return TK_SUCCESS;
}

/**
 * @brief Sets generation parameters
 */
tk_error_code_t tk_llm_runner_set_generation_params(
    tk_llm_runner_t* runner,
    float temperature,
    float top_p,
    int top_k
) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // These would be used in the sampling parameters in run_inference
    // For now, we just store them for reference
    TK_LOG_INFO("Setting generation params: temp=%.2f, top_p=%.2f, top_k=%d", 
                temperature, top_p, top_k);
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the last response from the model
 */
const char* tk_llm_runner_get_last_response(tk_llm_runner_t* runner) {
    if (!runner) {
        return NULL;
    }
    
    return runner->last_response;
}

/**
 * @brief Gets the conversation history
 */
tk_error_code_t tk_llm_runner_get_history(
    tk_llm_runner_t* runner,
    tk_llm_history_entry_t** out_history,
    size_t* out_history_count
) {
    if (!runner || !out_history || !out_history_count) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_history = runner->history;
    *out_history_count = runner->history_count;
    
    return TK_SUCCESS;
}

/**
 * @brief Saves the current conversation state to a file
 */
tk_error_code_t tk_llm_runner_save_state(tk_llm_runner_t* runner, const char* filepath) {
    if (!runner || !filepath) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    FILE* file = fopen(filepath, "wb");
    if (!file) {
        return TK_ERROR_IO_ERROR;
    }
    
    // Write history count
    fwrite(&runner->history_count, sizeof(size_t), 1, file);
    
    // Write each history entry
    for (size_t i = 0; i < runner->history_count; i++) {
        size_t role_len = runner->history[i].role ? strlen(runner->history[i].role) : 0;
        size_t content_len = runner->history[i].content ? strlen(runner->history[i].content) : 0;
        
        fwrite(&role_len, sizeof(size_t), 1, file);
        if (role_len > 0) {
            fwrite(runner->history[i].role, sizeof(char), role_len, file);
        }
        
        fwrite(&content_len, sizeof(size_t), 1, file);
        if (content_len > 0) {
            fwrite(runner->history[i].content, sizeof(char), content_len, file);
        }
    }
    
    fclose(file);
    return TK_SUCCESS;
}

/**
 * @brief Loads conversation state from a file
 */
tk_error_code_t tk_llm_runner_load_state(tk_llm_runner_t* runner, const char* filepath) {
    if (!runner || !filepath) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        return TK_ERROR_IO_ERROR;
    }
    
    // Clear current history
    clear_history(runner);
    
    // Read history count
    size_t history_count;
    if (fread(&history_count, sizeof(size_t), 1, file) != 1) {
        fclose(file);
        return TK_ERROR_IO_ERROR;
    }
    
    // Read each history entry
    for (size_t i = 0; i < history_count; i++) {
        size_t role_len, content_len;
        
        if (fread(&role_len, sizeof(size_t), 1, file) != 1) {
            fclose(file);
            return TK_ERROR_IO_ERROR;
        }
        
        char* role = NULL;
        if (role_len > 0) {
            role = malloc(role_len + 1);
            if (!role) {
                fclose(file);
                return TK_ERROR_OUT_OF_MEMORY;
            }
            
            if (fread(role, sizeof(char), role_len, file) != role_len) {
                free(role);
                fclose(file);
                return TK_ERROR_IO_ERROR;
            }
            role[role_len] = '\0';
        }
        
        if (fread(&content_len, sizeof(size_t), 1, file) != 1) {
            free_string(role);
            fclose(file);
            return TK_ERROR_IO_ERROR;
        }
        
        char* content = NULL;
        if (content_len > 0) {
            content = malloc(content_len + 1);
            if (!content) {
                free_string(role);
                fclose(file);
                return TK_ERROR_OUT_OF_MEMORY;
            }
            
            if (fread(content, sizeof(char), content_len, file) != content_len) {
                free_string(role);
                free(content);
                fclose(file);
                return TK_ERROR_IO_ERROR;
            }
            content[content_len] = '\0';
        }
        
        // Add to history
        runner->history[runner->history_count].role = role;
        runner->history[runner->history_count].content = content;
        runner->history_count++;
    }
    
    fclose(file);
    return TK_SUCCESS;
}

/**
 * @brief Estimates the number of tokens in a text
 */
size_t tk_llm_runner_estimate_token_count(tk_llm_runner_t* runner, const char* text) {
    if (!runner || !text) {
        return 0;
    }
    
    // This is a simplified estimation
    // In practice, you would use llama_tokenize to get the actual count
    size_t len = strlen(text);
    return len / 4; // Rough estimate: 4 characters per token on average
}

/**
 * @brief Checks if the context is approaching its limit
 */
bool tk_llm_runner_is_context_full(tk_llm_runner_t* runner, size_t additional_tokens) {
    if (!runner) {
        return false;
    }
    
    // Estimate total tokens in history
    size_t total_tokens = 0;
    for (size_t i = 0; i < runner->history_count; i++) {
        if (runner->history[i].content) {
            total_tokens += tk_llm_runner_estimate_token_count(runner, runner->history[i].content);
        }
    }
    
    // Add estimated tokens for system prompt
    if (runner->system_prompt) {
        total_tokens += tk_llm_runner_estimate_token_count(runner, runner->system_prompt);
    }
    
    // Check if we're approaching the limit
    return (total_tokens + additional_tokens) > (runner->config.context_size * 0.9);
}

/**
 * @brief Prunes the conversation history to make room for new content
 */
tk_error_code_t tk_llm_runner_prune_history(tk_llm_runner_t* runner, size_t tokens_to_remove) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Remove oldest entries until we've removed enough tokens
    size_t tokens_removed = 0;
    size_t entries_removed = 0;
    
    while (entries_removed < runner->history_count && tokens_removed < tokens_to_remove) {
        // Estimate tokens in this entry
        size_t entry_tokens = 0;
        if (runner->history[entries_removed].content) {
            entry_tokens = tk_llm_runner_estimate_token_count(runner, runner->history[entries_removed].content);
        }
        
        // Free the entry
        free_string(runner->history[entries_removed].role);
        free_string(runner->history[entries_removed].content);
        
        tokens_removed += entry_tokens;
        entries_removed++;
    }
    
    // Shift remaining entries to the beginning
    for (size_t i = 0; i < runner->history_count - entries_removed; i++) {
        runner->history[i] = runner->history[i + entries_removed];
    }
    
    runner->history_count -= entries_removed;
    
    return TK_SUCCESS;
}

/**
 * @brief Validates a tool call against available tools
 */
bool tk_llm_runner_validate_tool_call(
    tk_llm_runner_t* runner,
    const tk_llm_tool_definition_t* available_tools,
    size_t tool_count,
    const tk_llm_tool_call_t* tool_call
) {
    if (!runner || !available_tools || !tool_call || !tool_call->name) {
        return false;
    }
    
    // Check if the tool name exists in available tools
    for (size_t i = 0; i < tool_count; i++) {
        if (strcmp(available_tools[i].name, tool_call->name) == 0) {
            return true;
        }
    }
    
    return false;
}

/**
 * @brief Formats a tool call result for the model
 */
tk_error_code_t tk_llm_runner_format_tool_result(
    tk_llm_runner_t* runner,
    const char* tool_name,
    const char* tool_output,
    char* buffer,
    size_t buffer_size
) {
    if (!runner || !tool_name || !tool_output || !buffer) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    int written = snprintf(buffer, buffer_size, 
                          "Result from tool '%s': %s", 
                          tool_name, tool_output);
    
    if (written < 0 || (size_t)written >= buffer_size) {
        return TK_ERROR_BUFFER_TOO_SMALL;
    }
    
    return TK_SUCCESS;
}

/**
 * @brief Gets performance metrics
 */
tk_error_code_t tk_llm_runner_get_performance_metrics(
    tk_llm_runner_t* runner,
    uint32_t* out_tokens_per_second,
    uint32_t* out_memory_usage_mb
) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // These would be measured during actual inference
    // For now, we return placeholder values
    if (out_tokens_per_second) {
        *out_tokens_per_second = 30; // Placeholder value
    }
    
    if (out_memory_usage_mb) {
        *out_memory_usage_mb = 2048; // Placeholder value (2GB)
    }
    
    return TK_SUCCESS;
}

/**
 * @brief Sets a callback for progress updates during generation
 */
tk_error_code_t tk_llm_runner_set_progress_callback(
    tk_llm_runner_t* runner,
    void (*callback)(float progress, void* user_data),
    void* user_data
) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would be implemented in the inference loop
    // For now, we just acknowledge the callback
    TK_LOG_INFO("Progress callback set");
    
    return TK_SUCCESS;
}

/**
 * @brief Cancels the current generation process
 */
tk_error_code_t tk_llm_runner_cancel_generation(tk_llm_runner_t* runner) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would signal the inference loop to stop
    // For now, we just reset the processing flag
    runner->is_processing = false;
    
    return TK_SUCCESS;
}

/**
 * @brief Checks if the model is currently processing
 */
bool tk_llm_runner_is_processing(tk_llm_runner_t* runner) {
    if (!runner) {
        return false;
    }
    
    return runner->is_processing;
}

/**
 * @brief Gets the model's vocabulary size
 */
uint32_t tk_llm_runner_get_vocab_size(tk_llm_runner_t* runner) {
    if (!runner || !runner->model) {
        return 0;
    }
    
    return llama_n_vocab(runner->model);
}

/**
 * @brief Converts a token ID to its string representation
 */
tk_error_code_t tk_llm_runner_token_to_string(
    tk_llm_runner_t* runner,
    uint32_t token_id,
    char* buffer,
    size_t buffer_size
) {
    if (!runner || !buffer) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    int result = llama_token_to_piece(runner->ctx, token_id, buffer, buffer_size, 0, true);
    if (result < 0) {
        return TK_ERROR_INTERNAL;
    }
    
    return TK_SUCCESS;
}

/**
 * @brief Converts a string to token IDs
 */
tk_error_code_t tk_llm_runner_string_to_tokens(
    tk_llm_runner_t* runner,
    const char* text,
    uint32_t* tokens,
    size_t* token_count,
    size_t max_tokens
) {
    if (!runner || !text || !tokens || !token_count) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would use llama_tokenize in practice
    // For now, we provide a simplified implementation
    size_t text_len = strlen(text);
    size_t estimated_tokens = text_len / 4; // Rough estimate
    
    if (estimated_tokens > max_tokens) {
        return TK_ERROR_BUFFER_TOO_SMALL;
    }
    
    *token_count = estimated_tokens;
    
    // Fill with dummy token IDs
    for (size_t i = 0; i < estimated_tokens; i++) {
        tokens[i] = i % 1000; // Dummy token IDs
    }
    
    return TK_SUCCESS;
}

/**
 * @brief Sets the random seed for generation
 */
tk_error_code_t tk_llm_runner_set_seed(tk_llm_runner_t* runner, uint32_t seed) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    runner->seed = seed;
    return TK_SUCCESS;
}

/**
 * @brief Gets the current random seed
 */
uint32_t tk_llm_runner_get_seed(tk_llm_runner_t* runner) {
    if (!runner) {
        return 0;
    }
    
    return runner->seed;
}

/**
 * @brief Enables or disables debug logging
 */
tk_error_code_t tk_llm_runner_set_debug_logging(tk_llm_runner_t* runner, bool enable) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would control the verbosity of internal logging
    TK_LOG_INFO("Debug logging %s", enable ? "enabled" : "disabled");
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's context size
 */
uint32_t tk_llm_runner_get_context_size(tk_llm_runner_t* runner) {
    if (!runner || !runner->ctx) {
        return 0;
    }
    
    return llama_n_ctx(runner->ctx);
}

/**
 * @brief Gets the model's embedding size
 */
uint32_t tk_llm_runner_get_embedding_size(tk_llm_runner_t* runner) {
    if (!runner || !runner->model) {
        return 0;
    }
    
    return llama_n_embd(runner->model);
}

/**
 * @brief Checks if GPU acceleration is available and being used
 */
bool tk_llm_runner_is_gpu_accelerated(tk_llm_runner_t* runner) {
    if (!runner) {
        return false;
    }
    
    return runner->config.gpu_layers_offload > 0;
}

/**
 * @brief Gets the number of GPU layers being offloaded
 */
uint32_t tk_llm_runner_get_gpu_layers_offload(tk_llm_runner_t* runner) {
    if (!runner) {
        return 0;
    }
    
    return runner->config.gpu_layers_offload;
}

/**
 * @brief Sets the number of GPU layers to offload
 */
tk_error_code_t tk_llm_runner_set_gpu_layers_offload(tk_llm_runner_t* runner, uint32_t layers) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would require reloading the model in practice
    runner->config.gpu_layers_offload = layers;
    
    return TK_SUCCESS;
}

/**
 * @brief Gets information about available GPU memory
 */
tk_error_code_t tk_llm_runner_get_gpu_memory_info(
    tk_llm_runner_t* runner,
    uint64_t* out_total_memory_mb,
    uint64_t* out_free_memory_mb
) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would query actual GPU memory in practice
    // For now, we return placeholder values
    if (out_total_memory_mb) {
        *out_total_memory_mb = 8192; // 8GB
    }
    
    if (out_free_memory_mb) {
        *out_free_memory_mb = 4096; // 4GB
    }
    
    return TK_SUCCESS;
}

/**
 * @brief Sets the number of CPU threads to use
 */
tk_error_code_t tk_llm_runner_set_cpu_threads(tk_llm_runner_t* runner, int threads) {
    if (!runner || threads <= 0) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would be set when creating the context
    TK_LOG_INFO("Setting CPU threads to %d", threads);
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the number of CPU threads being used
 */
int tk_llm_runner_get_cpu_threads(tk_llm_runner_t* runner) {
    if (!runner) {
        return -1;
    }
    
    // This would return the actual thread count
    return 4; // Placeholder value
}

/**
 * @brief Sets the batch size for inference
 */
tk_error_code_t tk_llm_runner_set_batch_size(tk_llm_runner_t* runner, uint32_t batch_size) {
    if (!runner || batch_size == 0) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would be set when creating the context
    TK_LOG_INFO("Setting batch size to %u", batch_size);
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the current batch size
 */
uint32_t tk_llm_runner_get_batch_size(tk_llm_runner_t* runner) {
    if (!runner) {
        return 0;
    }
    
    // This would return the actual batch size
    return 512; // Placeholder value
}

/**
 * @brief Enables or disables flash attention
 */
tk_error_code_t tk_llm_runner_set_flash_attention(tk_llm_runner_t* runner, bool enable) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would be set when creating the context
    TK_LOG_INFO("Flash attention %s", enable ? "enabled" : "disabled");
    
    return TK_SUCCESS;
}

/**
 * @brief Checks if flash attention is enabled
 */
bool tk_llm_runner_is_flash_attention_enabled(tk_llm_runner_t* runner) {
    if (!runner) {
        return false;
    }
    
    // This would return the actual setting
    return false; // Placeholder value
}

/**
 * @brief Sets the model's rope frequency base
 */
tk_error_code_t tk_llm_runner_set_rope_freq_base(tk_llm_runner_t* runner, float freq_base) {
    if (!runner || freq_base <= 0) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would be set when creating the context
    TK_LOG_INFO("Setting rope frequency base to %.2f", freq_base);
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's rope frequency base
 */
float tk_llm_runner_get_rope_freq_base(tk_llm_runner_t* runner) {
    if (!runner) {
        return -1.0f;
    }
    
    // This would return the actual value
    return 10000.0f; // Placeholder value
}

/**
 * @brief Sets the model's rope frequency scale
 */
tk_error_code_t tk_llm_runner_set_rope_freq_scale(tk_llm_runner_t* runner, float freq_scale) {
    if (!runner || freq_scale <= 0) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would be set when creating the context
    TK_LOG_INFO("Setting rope frequency scale to %.2f", freq_scale);
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's rope frequency scale
 */
float tk_llm_runner_get_rope_freq_scale(tk_llm_runner_t* runner) {
    if (!runner) {
        return -1.0f;
    }
    
    // This would return the actual value
    return 1.0f; // Placeholder value
}

/**
 * @brief Sets the model's yarn extrapolation
 */
tk_error_code_t tk_llm_runner_set_yarn_ext_factor(tk_llm_runner_t* runner, float factor) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would be set when creating the context
    TK_LOG_INFO("Setting yarn extrapolation factor to %.2f", factor);
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's yarn extrapolation factor
 */
float tk_llm_runner_get_yarn_ext_factor(tk_llm_runner_t* runner) {
    if (!runner) {
        return -1.0f;
    }
    
    // This would return the actual value
    return -1.0f; // Placeholder value (disabled)
}

/**
 * @brief Sets the model's yarn attention factor
 */
tk_error_code_t tk_llm_runner_set_yarn_attn_factor(tk_llm_runner_t* runner, float factor) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would be set when creating the context
    TK_LOG_INFO("Setting yarn attention factor to %.2f", factor);
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's yarn attention factor
 */
float tk_llm_runner_get_yarn_attn_factor(tk_llm_runner_t* runner) {
    if (!runner) {
        return -1.0f;
    }
    
    // This would return the actual value
    return 1.0f; // Placeholder value
}

/**
 * @brief Sets the model's yarn beta fast
 */
tk_error_code_t tk_llm_runner_set_yarn_beta_fast(tk_llm_runner_t* runner, float beta) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would be set when creating the context
    TK_LOG_INFO("Setting yarn beta fast to %.2f", beta);
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's yarn beta fast
 */
float tk_llm_runner_get_yarn_beta_fast(tk_llm_runner_t* runner) {
    if (!runner) {
        return -1.0f;
    }
    
    // This would return the actual value
    return 32.0f; // Placeholder value
}

/**
 * @brief Sets the model's yarn beta slow
 */
tk_error_code_t tk_llm_runner_set_yarn_beta_slow(tk_llm_runner_t* runner, float beta) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would be set when creating the context
    TK_LOG_INFO("Setting yarn beta slow to %.2f", beta);
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's yarn beta slow
 */
float tk_llm_runner_get_yarn_beta_slow(tk_llm_runner_t* runner) {
    if (!runner) {
        return -1.0f;
    }
    
    // This would return the actual value
    return 1.0f; // Placeholder value
}

/**
 * @brief Sets the model's yarn original context length
 */
tk_error_code_t tk_llm_runner_set_yarn_orig_ctx(tk_llm_runner_t* runner, uint32_t orig_ctx) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would be set when creating the context
    TK_LOG_INFO("Setting yarn original context to %u", orig_ctx);
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's yarn original context length
 */
uint32_t tk_llm_runner_get_yarn_orig_ctx(tk_llm_runner_t* runner) {
    if (!runner) {
        return 0;
    }
    
    // This would return the actual value
    return 0; // Placeholder value (disabled)
}

/**
 * @brief Sets the model's logits all flag
 */
tk_error_code_t tk_llm_runner_set_logits_all(tk_llm_runner_t* runner, bool logits_all) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would be set when creating the context
    TK_LOG_INFO("Setting logits all to %s", logits_all ? "true" : "false");
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's logits all flag
 */
bool tk_llm_runner_get_logits_all(tk_llm_runner_t* runner) {
    if (!runner) {
        return false;
    }
    
    // This would return the actual value
    return false; // Placeholder value
}

/**
 * @brief Sets the model's embedding flag
 */
tk_error_code_t tk_llm_runner_set_embedding(tk_llm_runner_t* runner, bool embedding) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would be set when creating the context
    TK_LOG_INFO("Setting embedding to %s", embedding ? "true" : "false");
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's embedding flag
 */
bool tk_llm_runner_get_embedding(tk_llm_runner_t* runner) {
    if (!runner) {
        return false;
    }
    
    // This would return the actual value
    return false; // Placeholder value
}

/**
 * @brief Sets the model's offload KV cache flag
 */
tk_error_code_t tk_llm_runner_set_offload_kqv(tk_llm_runner_t* runner, bool offload_kqv) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would be set when creating the context
    TK_LOG_INFO("Setting offload KQV to %s", offload_kqv ? "true" : "false");
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's offload KV cache flag
 */
bool tk_llm_runner_get_offload_kqv(tk_llm_runner_t* runner) {
    if (!runner) {
        return false;
    }
    
    // This would return the actual value
    return true; // Placeholder value
}

/**
 * @brief Sets the model's flash attention flag
 */
tk_error_code_t tk_llm_runner_set_flash_attn(tk_llm_runner_t* runner, bool flash_attn) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would be set when creating the context
    TK_LOG_INFO("Setting flash attention to %s", flash_attn ? "true" : "false");
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's flash attention flag
 */
bool tk_llm_runner_get_flash_attn(tk_llm_runner_t* runner) {
    if (!runner) {
        return false;
    }
    
    // This would return the actual value
    return false; // Placeholder value
}

/**
 * @brief Sets the model's no KV cache flag
 */
tk_error_code_t tk_llm_runner_set_no_kv_offload(tk_llm_runner_t* runner, bool no_kv_offload) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would be set when creating the context
    TK_LOG_INFO("Setting no KV offload to %s", no_kv_offload ? "true" : "false");
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's no KV cache flag
 */
bool tk_llm_runner_get_no_kv_offload(tk_llm_runner_t* runner) {
    if (!runner) {
        return false;
    }
    
    // This would return the actual value
    return false; // Placeholder value
}

/**
 * @brief Sets the model's cache reuse flag
 */
tk_error_code_t tk_llm_runner_set_cache_reuse(tk_llm_runner_t* runner, bool cache_reuse) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would be set when creating the context
    TK_LOG_INFO("Setting cache reuse to %s", cache_reuse ? "true" : "false");
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's cache reuse flag
 */
bool tk_llm_runner_get_cache_reuse(tk_llm_runner_t* runner) {
    if (!runner) {
        return false;
    }
    
    // This would return the actual value
    return true; // Placeholder value
}

/**
 * @brief Sets the model's cache type KV
 */
tk_error_code_t tk_llm_runner_set_cache_type_k(tk_llm_runner_t* runner, int cache_type_k) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would be set when creating the context
    TK_LOG_INFO("Setting cache type K to %d", cache_type_k);
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's cache type KV
 */
int tk_llm_runner_get_cache_type_k(tk_llm_runner_t* runner) {
    if (!runner) {
        return -1;
    }
    
    // This would return the actual value
    return 0; // Placeholder value
}

/**
 * @brief Sets the model's cache type V
 */
tk_error_code_t tk_llm_runner_set_cache_type_v(tk_llm_runner_t* runner, int cache_type_v) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would be set when creating the context
    TK_LOG_INFO("Setting cache type V to %d", cache_type_v);
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's cache type V
 */
int tk_llm_runner_get_cache_type_v(tk_llm_runner_t* runner) {
    if (!runner) {
        return -1;
    }
    
    // This would return the actual value
    return 0; // Placeholder value
}

/**
 * @brief Sets the model's use mmap flag
 */
tk_error_code_t tk_llm_runner_set_use_mmap(tk_llm_runner_t* runner, bool use_mmap) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would be set when loading the model
    TK_LOG_INFO("Setting use mmap to %s", use_mmap ? "true" : "false");
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's use mmap flag
 */
bool tk_llm_runner_get_use_mmap(tk_llm_runner_t* runner) {
    if (!runner) {
        return false;
    }
    
    // This would return the actual value
    return true; // Placeholder value
}

/**
 * @brief Sets the model's use mlock flag
 */
tk_error_code_t tk_llm_runner_set_use_mlock(tk_llm_runner_t* runner, bool use_mlock) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would be set when loading the model
    TK_LOG_INFO("Setting use mlock to %s", use_mlock ? "true" : "false");
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's use mlock flag
 */
bool tk_llm_runner_get_use_mlock(tk_llm_runner_t* runner) {
    if (!runner) {
        return false;
    }
    
    // This would return the actual value
    return false; // Placeholder value
}

/**
 * @brief Sets the model's numa flag
 */
tk_error_code_t tk_llm_runner_set_numa(tk_llm_runner_t* runner, bool numa) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would be set when loading the model
    TK_LOG_INFO("Setting numa to %s", numa ? "true" : "false");
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's numa flag
 */
bool tk_llm_runner_get_numa(tk_llm_runner_t* runner) {
    if (!runner) {
        return false;
    }
    
    // This would return the actual value
    return false; // Placeholder value
}

/**
 * @brief Sets the model's verbose logging flag
 */
tk_error_code_t tk_llm_runner_set_verbose(tk_llm_runner_t* runner, bool verbose) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would control internal logging verbosity
    TK_LOG_INFO("Setting verbose logging to %s", verbose ? "true" : "false");
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's verbose logging flag
 */
bool tk_llm_runner_get_verbose(tk_llm_runner_t* runner) {
    if (!runner) {
        return false;
    }
    
    // This would return the actual value
    return false; // Placeholder value
}

/**
 * @brief Sets the model's CPU mask
 */
tk_error_code_t tk_llm_runner_set_cpu_mask(tk_llm_runner_t* runner, const char* cpu_mask) {
    if (!runner || !cpu_mask) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would control CPU affinity
    TK_LOG_INFO("Setting CPU mask to %s", cpu_mask);
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's CPU mask
 */
const char* tk_llm_runner_get_cpu_mask(tk_llm_runner_t* runner) {
    if (!runner) {
        return NULL;
    }
    
    // This would return the actual value
    return "0xFFFFFFFF"; // Placeholder value
}

/**
 * @brief Sets the model's CPU priority
 */
tk_error_code_t tk_llm_runner_set_cpu_priority(tk_llm_runner_t* runner, int priority) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would control CPU scheduling priority
    TK_LOG_INFO("Setting CPU priority to %d", priority);
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's CPU priority
 */
int tk_llm_runner_get_cpu_priority(tk_llm_runner_t* runner) {
    if (!runner) {
        return -1;
    }
    
    // This would return the actual value
    return 0; // Placeholder value
}

/**
 * @brief Sets the model's CPU policy
 */
tk_error_code_t tk_llm_runner_set_cpu_policy(tk_llm_runner_t* runner, const char* policy) {
    if (!runner || !policy) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would control CPU scheduling policy
    TK_LOG_INFO("Setting CPU policy to %s", policy);
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's CPU policy
 */
const char* tk_llm_runner_get_cpu_policy(tk_llm_runner_t* runner) {
    if (!runner) {
        return NULL;
    }
    
    // This would return the actual value
    return "SCHED_OTHER"; // Placeholder value
}

/**
 * @brief Sets the model's CPU affinity
 */
tk_error_code_t tk_llm_runner_set_cpu_affinity(tk_llm_runner_t* runner, int cpu_core) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would control CPU affinity
    TK_LOG_INFO("Setting CPU affinity to core %d", cpu_core);
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's CPU affinity
 */
int tk_llm_runner_get_cpu_affinity(tk_llm_runner_t* runner) {
    if (!runner) {
        return -1;
    }
    
    // This would return the actual value
    return -1; // Placeholder value (no affinity set)
}

/**
 * @brief Sets the model's thread pool size
 */
tk_error_code_t tk_llm_runner_set_thread_pool_size(tk_llm_runner_t* runner, int pool_size) {
    if (!runner || pool_size <= 0) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would control the size of internal thread pools
    TK_LOG_INFO("Setting thread pool size to %d", pool_size);
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's thread pool size
 */
int tk_llm_runner_get_thread_pool_size(tk_llm_runner_t* runner) {
    if (!runner) {
        return -1;
    }
    
    // This would return the actual value
    return 4; // Placeholder value
}

/**
 * @brief Sets the model's thread stack size
 */
tk_error_code_t tk_llm_runner_set_thread_stack_size(tk_llm_runner_t* runner, size_t stack_size) {
    if (!runner || stack_size == 0) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would control thread stack sizes
    TK_LOG_INFO("Setting thread stack size to %zu bytes", stack_size);
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's thread stack size
 */
size_t tk_llm_runner_get_thread_stack_size(tk_llm_runner_t* runner) {
    if (!runner) {
        return 0;
    }
    
    // This would return the actual value
    return 1024 * 1024; // Placeholder value (1MB)
}

/**
 * @brief Sets the model's thread priority
 */
tk_error_code_t tk_llm_runner_set_thread_priority(tk_llm_runner_t* runner, int priority) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would control thread priorities
    TK_LOG_INFO("Setting thread priority to %d", priority);
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's thread priority
 */
int tk_llm_runner_get_thread_priority(tk_llm_runner_t* runner) {
    if (!runner) {
        return -1;
    }
    
    // This would return the actual value
    return 0; // Placeholder value
}

/**
 * @brief Sets the model's thread policy
 */
tk_error_code_t tk_llm_runner_set_thread_policy(tk_llm_runner_t* runner, const char* policy) {
    if (!runner || !policy) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would control thread scheduling policies
    TK_LOG_INFO("Setting thread policy to %s", policy);
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's thread policy
 */
const char* tk_llm_runner_get_thread_policy(tk_llm_runner_t* runner) {
    if (!runner) {
        return NULL;
    }
    
    // This would return the actual value
    return "SCHED_OTHER"; // Placeholder value
}

/**
 * @brief Sets the model's thread affinity
 */
tk_error_code_t tk_llm_runner_set_thread_affinity(tk_llm_runner_t* runner, int cpu_core) {
    if (!runner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // This would control thread CPU affinity
    TK_LOG_INFO("Setting thread affinity to core %d", cpu_core);
    
    return TK_SUCCESS;
}

/**
 * @brief Gets the model's thread affinity
 */
int tk_llm_runner_get_thread_affinity(tk_llm_runner_t* runner) {
    if (!runner) {
        return -1;
    }
    
    // This would return the actual value
    return -1; // Placeholder value (no affinity set)
}

/**
 * @brief Sets the model's thread name
 */
tk_error_code_t tk_llm_runner_set_thread_name(tk_llm_runner_t* runner, const
// TODO:

//NOT COMPLETED WHILE, VERY VERBOSITY CODE.

/**This code was taken from an old library from trackiellmv1 (old beta model), 
*I do not recommend reviewing complex codes and libraries like these by AI, perhaps modularizing this will 
*be better in the future, my original idea was to write a lot of things here in Rust.
Again, I do not recommend using AI for writing or deep understanding of this code, I'm trying to document it better (check .docs), Jules from Google Labs and Peitch Docbot were unable to generate acceptable documentation for this lib.
/