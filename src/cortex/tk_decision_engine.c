/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_decision_engine.c
 *
 * Implementation of the Decision Engine for TrackieLLM Cortex.
 * This module interprets LLM responses and executes corresponding actions
 * in a safe, prioritized, and context-aware manner.
 *
 * Key features:
 * - Structured LLM response parsing with validation using cJSON
 * - Priority-based action execution queue
 * - Context-aware action validation
 * - Thread-safe operation with timeout handling
 * - Emergency response capabilities
 *
 * Dependencies:
 * - cJSON library for JSON parsing
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "tk_decision_engine.h"
#include "utils/tk_logging.h"
#include "profiling/tk_memory_profiler.h"
#include "utils/tk_error_handling.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <stdatomic.h>
#include <cjson/cjson.h> // Include cJSON for JSON parsing

// Internal action queue structure
typedef struct {
    tk_action_t* actions;
    size_t capacity;
    size_t count;
    pthread_mutex_t mutex;
} action_queue_t;

// Main decision engine structure
struct tk_decision_engine_s {
    tk_decision_config_t config;
    tk_decision_callbacks_t callbacks;
    
    // Action queues for different states
    action_queue_t pending_queue;
    action_queue_t executing_queue;
    action_queue_t completed_queue;
    
    // Statistics tracking
    struct {
        size_t total_actions_queued;
        size_t total_actions_completed;
        size_t total_actions_failed;
        size_t total_actions_cancelled;
    } stats;
    
    // Thread safety
    pthread_mutex_t engine_mutex;
    bool initialized;
};

// Global atomic counter for action IDs
static atomic_uint_fast64_t g_action_counter = ATOMIC_VAR_INIT(1);

//-----------------------------------------------------------------------------
// Internal Helper Functions
//-----------------------------------------------------------------------------

static tk_error_code_t initialize_action_queue(action_queue_t* queue, size_t capacity);
static void destroy_action_queue(action_queue_t* queue);
static tk_error_code_t queue_action(action_queue_t* queue, const tk_action_t* action);
static tk_error_code_t dequeue_action(action_queue_t* queue, uint64_t action_id, tk_action_t* out_action);
static tk_error_code_t find_action_in_queue(const action_queue_t* queue, uint64_t action_id, size_t* out_index);
static tk_error_code_t update_action_status(action_queue_t* queue, uint64_t action_id, tk_action_status_e status, const char* error_msg);
static uint64_t generate_action_id(void);
static bool validate_action_params(const tk_action_params_t* params, const tk_context_summary_t* context);
static tk_error_code_t execute_single_action(tk_decision_engine_t* engine, tk_action_t* action, void* audio_ctx, void* nav_ctx, void* reasoner_ctx);
static void free_action_resources(tk_action_t* action);
static void free_action_params_resources(tk_action_params_t* params);
static tk_error_code_t parse_llm_response_text(const char* text, tk_llm_response_t** out_response);
static void log_action_execution(const tk_action_t* action);
static tk_error_code_t deep_copy_action(tk_action_t* dst, const tk_action_t* src);
static tk_error_code_t deep_copy_action_params(tk_action_params_t* dst, const tk_action_params_t* src);

//-----------------------------------------------------------------------------
// Public API Implementation
//-----------------------------------------------------------------------------

tk_error_code_t tk_decision_engine_create(
    tk_decision_engine_t** out_engine,
    const tk_decision_config_t* config,
    tk_decision_callbacks_t callbacks) {
    
    if (!out_engine || !config) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Validate config parameters
    if (config->max_concurrent_actions == 0) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    tk_log_info("Creating Decision Engine");
    
    // Allocate engine structure
    tk_decision_engine_t* engine = calloc(1, sizeof(tk_decision_engine_t));
    if (!engine) {
        tk_log_error("Failed to allocate decision engine memory");
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Copy configuration
    engine->config = *config;
    engine->callbacks = callbacks;
    
    // Initialize mutex
    if (pthread_mutex_init(&engine->engine_mutex, NULL) != 0) {
        tk_log_error("Failed to initialize engine mutex");
        free(engine);
        return TK_ERROR_SYSTEM_ERROR;
    }
    
    // Initialize action queues
    const size_t queue_capacity = config->max_concurrent_actions * 4; // Allow some buffer
    
    tk_error_code_t result = initialize_action_queue(&engine->pending_queue, queue_capacity);
    if (result != TK_SUCCESS) {
        tk_log_error("Failed to initialize pending queue");
        pthread_mutex_destroy(&engine->engine_mutex);
        free(engine);
        return result;
    }
    
    result = initialize_action_queue(&engine->executing_queue, config->max_concurrent_actions);
    if (result != TK_SUCCESS) {
        tk_log_error("Failed to initialize executing queue");
        destroy_action_queue(&engine->pending_queue);
        pthread_mutex_destroy(&engine->engine_mutex);
        free(engine);
        return result;
    }
    
    result = initialize_action_queue(&engine->completed_queue, queue_capacity);
    if (result != TK_SUCCESS) {
        tk_log_error("Failed to initialize completed queue");
        destroy_action_queue(&engine->pending_queue);
        destroy_action_queue(&engine->executing_queue);
        pthread_mutex_destroy(&engine->engine_mutex);
        free(engine);
        return result;
    }
    
    engine->initialized = true;
    *out_engine = engine;
    
    tk_log_info("Decision Engine created successfully");
    return TK_SUCCESS;
}

void tk_decision_engine_destroy(tk_decision_engine_t** engine) {
    if (!engine || !*engine) {
        return;
    }
    
    tk_log_info("Destroying Decision Engine");
    
    tk_decision_engine_t* e = *engine;
    
    // Acquire lock to ensure thread safety during destruction
    pthread_mutex_lock(&e->engine_mutex);
    
    // Cancel all pending and executing actions
    for (size_t i = 0; i < e->pending_queue.count; i++) {
        free_action_resources(&e->pending_queue.actions[i]);
    }
    
    for (size_t i = 0; i < e->executing_queue.count; i++) {
        free_action_resources(&e->executing_queue.actions[i]);
    }
    
    // Clean up completed actions
    for (size_t i = 0; i < e->completed_queue.count; i++) {
        free_action_resources(&e->completed_queue.actions[i]);
    }
    
    // Destroy action queues
    destroy_action_queue(&e->pending_queue);
    destroy_action_queue(&e->executing_queue);
    destroy_action_queue(&e->completed_queue);
    
    pthread_mutex_unlock(&e->engine_mutex);
    pthread_mutex_destroy(&e->engine_mutex);
    
    free(e);
    *engine = NULL;
    
    tk_log_info("Decision Engine destroyed");
}

tk_error_code_t tk_decision_engine_process_llm_response(
    tk_decision_engine_t* engine,
    const char* llm_output_text,
    const tk_context_summary_t* context_summary,
    tk_llm_response_t** out_parsed_response) {
    
    if (!engine || !llm_output_text || !out_parsed_response) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    pthread_mutex_lock(&engine->engine_mutex);
    
    if (!engine->initialized) {
        pthread_mutex_unlock(&engine->engine_mutex);
        return TK_ERROR_INVALID_STATE;
    }
    
    tk_log_debug("Processing LLM response: '%.100s'...", llm_output_text);
    
    // Parse the LLM response text
    tk_llm_response_t* response = NULL;
    tk_error_code_t result = parse_llm_response_text(llm_output_text, &response);
    
    if (result != TK_SUCCESS) {
        tk_log_error("Failed to parse LLM response: %d", result);
        pthread_mutex_unlock(&engine->engine_mutex);
        return result;
    }
    
    // Validate each action against current context
    for (size_t i = 0; i < response->action_count; i++) {
        tk_action_params_t* params = &response->actions[i];
        
        // Validate action parameters
        if (!validate_action_params(params, context_summary)) {
            tk_log_warning("Action validation failed for action type %d", params->type);
            
            // Set confidence to zero to prevent execution
            params->confidence = 0.0f;
            
            // Create error message
            free(params->params.speak.text);
            params->params.speak.text = strdup("Action validation failed - parameters invalid");
            if (!params->params.speak.text) {
                tk_decision_engine_free_response(&response);
                pthread_mutex_unlock(&engine->engine_mutex);
                return TK_ERROR_OUT_OF_MEMORY;
            }
        }
    }
    
    *out_parsed_response = response;
    pthread_mutex_unlock(&engine->engine_mutex);
    
    tk_log_debug("LLM response processed successfully with %zu actions", response->action_count);
    return TK_SUCCESS;
}

tk_error_code_t tk_decision_engine_execute_response(
    tk_decision_engine_t* engine,
    const tk_llm_response_t* parsed_response) {
    
    if (!engine || !parsed_response) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    pthread_mutex_lock(&engine->engine_mutex);
    
    if (!engine->initialized) {
        pthread_mutex_unlock(&engine->engine_mutex);
        return TK_ERROR_INVALID_STATE;
    }
    
    tk_log_debug("Executing response with %zu actions", parsed_response->action_count);
    
    // Queue all actions from the parsed response
    for (size_t i = 0; i < parsed_response->action_count; i++) {
        const tk_action_params_t* params = &parsed_response->actions[i];
        
        // Skip actions with low confidence
        if (params->confidence < engine->config.action_confidence_threshold) {
            tk_log_debug("Skipping action with low confidence (%.2f < %.2f)", 
                        params->confidence, engine->config.action_confidence_threshold);
            continue;
        }
        
        // Create action structure
        tk_action_t action = {0};
        tk_error_code_t result = deep_copy_action(&action, &(tk_action_t){
            .action_id = generate_action_id(),
            .params = *params,
            .status = TK_ACTION_STATUS_PENDING,
            .created_timestamp_ns = get_current_time_ns()
        });
        
        if (result != TK_SUCCESS) {
            tk_log_error("Failed to create action: %d", result);
            free_action_resources(&action);
            pthread_mutex_unlock(&engine->engine_mutex);
            return result;
        }
        
        // Queue the action
        result = queue_action(&engine->pending_queue, &action);
        if (result != TK_SUCCESS) {
            tk_log_error("Failed to queue action: %d", result);
            free_action_resources(&action);
            pthread_mutex_unlock(&engine->engine_mutex);
            return result;
        }
        
        // Free the temporary action since queue made its own copy
        free_action_resources(&action);
        engine->stats.total_actions_queued++;
        tk_log_debug("Action %llu queued successfully", (unsigned long long)action.action_id);
    }
    
    // If there's a response text, notify via callback
    if (parsed_response->response_text && engine->callbacks.on_response_ready) {
        // Make a copy of the response text for the callback
        char* response_copy = strdup(parsed_response->response_text);
        if (!response_copy) {
            tk_log_error("Failed to allocate response copy for callback");
            pthread_mutex_unlock(&engine->engine_mutex);
            return TK_ERROR_OUT_OF_MEMORY;
        }
        
        tk_response_priority_e priority = parsed_response->priority;
        void* user_data = engine->callbacks.user_data;
        
        pthread_mutex_unlock(&engine->engine_mutex);
        
        // Call callback outside of lock
        engine->callbacks.on_response_ready(response_copy, priority, user_data);
        
        free(response_copy);
        pthread_mutex_lock(&engine->engine_mutex);
    } else {
        pthread_mutex_unlock(&engine->engine_mutex);
    }
    
    return TK_SUCCESS;
}

tk_error_code_t tk_decision_engine_queue_action(
    tk_decision_engine_t* engine,
    const tk_action_params_t* action_params,
    uint64_t* out_action_id) {
    
    if (!engine || !action_params || !out_action_id) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    pthread_mutex_lock(&engine->engine_mutex);
    
    if (!engine->initialized) {
        pthread_mutex_unlock(&engine->engine_mutex);
        return TK_ERROR_INVALID_STATE;
    }
    
    // Validate action parameters
    // Note: We don't have context here, so we skip context-aware validation
    // This is acceptable for manually queued actions
    
    // Create action structure
    tk_action_t action = {0};
    action.action_id = generate_action_id();
    action.status = TK_ACTION_STATUS_PENDING;
    action.created_timestamp_ns = get_current_time_ns();
    action.started_timestamp_ns = 0;
    action.completed_timestamp_ns = 0;
    action.error_message = NULL;
    
    // Deep copy action parameters
    tk_error_code_t result = deep_copy_action_params(&action.params, action_params);
    if (result != TK_SUCCESS) {
        tk_log_error("Failed to copy action parameters: %d", result);
        free_action_resources(&action);
        pthread_mutex_unlock(&engine->engine_mutex);
        return result;
    }
    
    // Queue the action
    result = queue_action(&engine->pending_queue, &action);
    if (result != TK_SUCCESS) {
        tk_log_error("Failed to queue manual action: %d", result);
        free_action_resources(&action);
        pthread_mutex_unlock(&engine->engine_mutex);
        return result;
    }
    
    // Free the temporary action since queue made its own copy
    free_action_resources(&action);
    engine->stats.total_actions_queued++;
    *out_action_id = action.action_id;
    
    tk_log_debug("Manual action %llu queued successfully", (unsigned long long)action.action_id);
    pthread_mutex_unlock(&engine->engine_mutex);
    
    return TK_SUCCESS;
}

tk_error_code_t tk_decision_engine_cancel_action(
    tk_decision_engine_t* engine,
    uint64_t action_id) {
    
    if (!engine) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    pthread_mutex_lock(&engine->engine_mutex);
    
    if (!engine->initialized) {
        pthread_mutex_unlock(&engine->engine_mutex);
        return TK_ERROR_INVALID_STATE;
    }
    
    tk_log_debug("Cancelling action %llu", (unsigned long long)action_id);
    
    // Try to find and remove from pending queue
    tk_action_t action;
    tk_error_code_t result = dequeue_action(&engine->pending_queue, action_id, &action);
    
    if (result == TK_SUCCESS) {
        // Action was in pending queue, mark as cancelled
        action.status = TK_ACTION_STATUS_CANCELLED;
        action.completed_timestamp_ns = get_current_time_ns();
        
        // Move to completed queue
        tk_action_t completed_action;
        result = deep_copy_action(&completed_action, &action);
        if (result != TK_SUCCESS) {
            free_action_resources(&action);
            pthread_mutex_unlock(&engine->engine_mutex);
            return result;
        }
        
        result = queue_action(&engine->completed_queue, &completed_action);
        if (result == TK_SUCCESS) {
            engine->stats.total_actions_cancelled++;
        } else {
            // Failed to move to completed queue, clean up
            free_action_resources(&completed_action);
        }
        
        free_action_resources(&action);
        pthread_mutex_unlock(&engine->engine_mutex);
        return result;
    }
    
    // Try to find in executing queue
    result = dequeue_action(&engine->executing_queue, action_id, &action);
    
    if (result == TK_SUCCESS) {
        // Action was executing, mark as cancelled
        action.status = TK_ACTION_STATUS_CANCELLED;
        action.completed_timestamp_ns = get_current_time_ns();
        
        // Move to completed queue
        tk_action_t completed_action;
        result = deep_copy_action(&completed_action, &action);
        if (result != TK_SUCCESS) {
            free_action_resources(&action);
            pthread_mutex_unlock(&engine->engine_mutex);
            return result;
        }
        
        result = queue_action(&engine->completed_queue, &completed_action);
        if (result == TK_SUCCESS) {
            engine->stats.total_actions_cancelled++;
        } else {
            // Failed to move to completed queue, clean up
            free_action_resources(&completed_action);
        }
        
        free_action_resources(&action);
        pthread_mutex_unlock(&engine->engine_mutex);
        return result;
    }
    
    // Action not found in active queues
    pthread_mutex_unlock(&engine->engine_mutex);
    return TK_ERROR_NOT_FOUND;
}

tk_error_code_t tk_decision_engine_process_actions(
    tk_decision_engine_t* engine,
    uint64_t current_time_ns,
    void* audio_ctx,
    void* nav_ctx,
    void* reasoner_ctx) {
    
    if (!engine) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    pthread_mutex_lock(&engine->engine_mutex);
    
    if (!engine->initialized) {
        pthread_mutex_unlock(&engine->engine_mutex);
        return TK_ERROR_INVALID_STATE;
    }
    
    // Process executing actions for timeouts
    for (size_t i = 0; i < engine->executing_queue.count; ) {
        tk_action_t* action = &engine->executing_queue.actions[i];
        
        // Check for timeout
        uint32_t timeout_ms = action->params.timeout_ms > 0 ? 
                              action->params.timeout_ms : 
                              engine->config.action_timeout_ms;
        
        uint64_t elapsed_ms = (current_time_ns - action->started_timestamp_ns) / 1000000;
        
        if (elapsed_ms > timeout_ms) {
            tk_log_warning("Action %llu timed out after %llu ms", 
                          (unsigned long long)action->action_id, 
                          (unsigned long long)elapsed_ms);
            
            // Mark as timed out
            action->status = TK_ACTION_STATUS_TIMEOUT;
            action->completed_timestamp_ns = current_time_ns;
            
            // Move to completed queue
            tk_action_t completed_action;
            tk_error_code_t result = deep_copy_action(&completed_action, action);
            
            if (result == TK_SUCCESS) {
                result = queue_action(&engine->completed_queue, &completed_action);
            }
            
            if (result == TK_SUCCESS) {
                engine->stats.total_actions_failed++;
                
                // Remove from executing queue
                memmove(&engine->executing_queue.actions[i], 
                        &engine->executing_queue.actions[i+1], 
                        (engine->executing_queue.count - i - 1) * sizeof(tk_action_t));
                engine->executing_queue.count--;
                
                // Notify callback if provided
                if (engine->callbacks.on_action_completed) {
                    // Unlock mutex before calling callback
                    pthread_mutex_unlock(&engine->engine_mutex);
                    engine->callbacks.on_action_completed(&completed_action, engine->callbacks.user_data);
                    pthread_mutex_lock(&engine->engine_mutex);
                }
                
                free_action_resources(&completed_action);
            } else {
                // Failed to queue in completed, keep in executing (will retry next time)
                i++;
            }
        } else {
            i++;
        }
    }
    
    // Move pending actions to executing queue if space available
    while (engine->pending_queue.count > 0 && 
           engine->executing_queue.count < engine->config.max_concurrent_actions) {
        
        // Dequeue from pending
        tk_action_t action;
        tk_error_code_t result = dequeue_action(&engine->pending_queue, 0, &action); // 0 means first item
        if (result != TK_SUCCESS) {
            break;
        }
        
        // Update status and timestamps
        action.status = TK_ACTION_STATUS_EXECUTING;
        action.started_timestamp_ns = current_time_ns;
        
        // Queue to executing
        result = queue_action(&engine->executing_queue, &action);
        if (result != TK_SUCCESS) {
            tk_log_error("Failed to move action to executing queue: %d", result);
            free_action_resources(&action);
            break;
        }
        
        // Free the temporary action since queue made its own copy
        free_action_resources(&action);
        
        // Execute the action
        result = execute_single_action(engine, &engine->executing_queue.actions[engine->executing_queue.count - 1], audio_ctx, nav_ctx, reasoner_ctx);
        if (result != TK_SUCCESS) {
            tk_log_error("Failed to execute action %llu: %d", 
                        (unsigned long long)action.action_id, result);
        }
    }
    
    pthread_mutex_unlock(&engine->engine_mutex);
    return TK_SUCCESS;
}

tk_error_code_t tk_decision_engine_get_action_status(
    tk_decision_engine_t* engine,
    uint64_t action_id,
    tk_action_t* out_action) {
    
    if (!engine || !out_action) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    pthread_mutex_lock(&engine->engine_mutex);
    
    if (!engine->initialized) {
        pthread_mutex_unlock(&engine->engine_mutex);
        return TK_ERROR_INVALID_STATE;
    }
    
    // Search in all queues
    size_t index;
    tk_error_code_t result;
    
    // Check pending queue
    result = find_action_in_queue(&engine->pending_queue, action_id, &index);
    if (result == TK_SUCCESS) {
        tk_error_code_t copy_rc = deep_copy_action(out_action, &engine->pending_queue.actions[index]);
        pthread_mutex_unlock(&engine->engine_mutex);
        return copy_rc;
    }
    
    // Check executing queue
    result = find_action_in_queue(&engine->executing_queue, action_id, &index);
    if (result == TK_SUCCESS) {
        tk_error_code_t copy_rc = deep_copy_action(out_action, &engine->executing_queue.actions[index]);
        pthread_mutex_unlock(&engine->engine_mutex);
        return copy_rc;
    }
    
    // Check completed queue
    result = find_action_in_queue(&engine->completed_queue, action_id, &index);
    if (result == TK_SUCCESS) {
        tk_error_code_t copy_rc = deep_copy_action(out_action, &engine->completed_queue.actions[index]);
        pthread_mutex_unlock(&engine->engine_mutex);
        return copy_rc;
    }
    
    pthread_mutex_unlock(&engine->engine_mutex);
    return TK_ERROR_NOT_FOUND;
}

tk_error_code_t tk_decision_engine_get_queue_stats(
    tk_decision_engine_t* engine,
    size_t* out_pending_count,
    size_t* out_executing_count,
    size_t* out_completed_count,
    size_t* out_failed_count) {
    
    if (!engine) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    pthread_mutex_lock(&engine->engine_mutex);
    
    if (!engine->initialized) {
        pthread_mutex_unlock(&engine->engine_mutex);
        return TK_ERROR_INVALID_STATE;
    }
    
    if (out_pending_count) *out_pending_count = engine->pending_queue.count;
    if (out_executing_count) *out_executing_count = engine->executing_queue.count;
    if (out_completed_count) *out_completed_count = engine->completed_queue.count;
    if (out_failed_count) *out_failed_count = engine->stats.total_actions_failed;
    
    pthread_mutex_unlock(&engine->engine_mutex);
    return TK_SUCCESS;
}

tk_error_code_t tk_decision_engine_emergency_stop(
    tk_decision_engine_t* engine) {
    
    if (!engine) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    pthread_mutex_lock(&engine->engine_mutex);
    
    if (!engine->initialized) {
        pthread_mutex_unlock(&engine->engine_mutex);
        return TK_ERROR_INVALID_STATE;
    }
    
    tk_log_warning("Emergency stop requested - cancelling all actions");
    
    // Cancel all pending actions
    for (size_t i = 0; i < engine->pending_queue.count; i++) {
        tk_action_t* action = &engine->pending_queue.actions[i];
        action->status = TK_ACTION_STATUS_CANCELLED;
        action->completed_timestamp_ns = get_current_time_ns();
        
        // Move to completed queue
        tk_action_t completed_action;
        tk_error_code_t result = deep_copy_action(&completed_action, action);
        if (result != TK_SUCCESS) {
            continue;
        }
        
        result = queue_action(&engine->completed_queue, &completed_action);
        if (result == TK_SUCCESS) {
            engine->stats.total_actions_cancelled++;
            
            // Notify callback if provided
            if (engine->callbacks.on_action_completed) {
                // Unlock mutex before calling callback
                pthread_mutex_unlock(&engine->engine_mutex);
                engine->callbacks.on_action_completed(&completed_action, engine->callbacks.user_data);
                pthread_mutex_lock(&engine->engine_mutex);
            }
            
            free_action_resources(&completed_action);
        }
    }
    
    // Clear pending queue
    engine->pending_queue.count = 0;
    
    // Cancel all executing actions
    for (size_t i = 0; i < engine->executing_queue.count; i++) {
        tk_action_t* action = &engine->executing_queue.actions[i];
        action->status = TK_ACTION_STATUS_CANCELLED;
        action->completed_timestamp_ns = get_current_time_ns();
        
        // Move to completed queue
        tk_action_t completed_action;
        tk_error_code_t result = deep_copy_action(&completed_action, action);
        if (result != TK_SUCCESS) {
            continue;
        }
        
        result = queue_action(&engine->completed_queue, &completed_action);
        if (result == TK_SUCCESS) {
            engine->stats.total_actions_cancelled++;
            
            // Notify callback if provided
            if (engine->callbacks.on_action_completed) {
                // Unlock mutex before calling callback
                pthread_mutex_unlock(&engine->engine_mutex);
                engine->callbacks.on_action_completed(&completed_action, engine->callbacks.user_data);
                pthread_mutex_lock(&engine->engine_mutex);
            }
            
            free_action_resources(&completed_action);
        }
    }
    
    // Clear executing queue
    engine->executing_queue.count = 0;
    
    pthread_mutex_unlock(&engine->engine_mutex);
    return TK_SUCCESS;
}

tk_error_code_t tk_decision_engine_emergency_response(
    tk_decision_engine_t* engine,
    const char* emergency_text,
    uint32_t repeat_count) {

    if (!engine || !emergency_text) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    // Check engine state under lock, then release to avoid deadlock
    pthread_mutex_lock(&engine->engine_mutex);
    if (!engine->initialized) {
        pthread_mutex_unlock(&engine->engine_mutex);
        return TK_ERROR_INVALID_STATE;
    }
    pthread_mutex_unlock(&engine->engine_mutex);

    tk_log_warning("Emergency response: '%s'", emergency_text);

    // Prepare params (local). Validate strdup success.
    tk_action_params_t params;
    memset(&params, 0, sizeof(params));
    params.type = TK_ACTION_TYPE_EMERGENCY_ALERT;
    params.confidence = 1.0f;
    params.timeout_ms = 5000; // 5s

    params.params.emergency_alert.alert_message = strdup(emergency_text);
    if (!params.params.emergency_alert.alert_message) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    params.params.emergency_alert.repeat_alert = (repeat_count > 0);
    params.params.emergency_alert.repeat_interval_ms = 2000;

    // Call public API (it will lock engine_mutex internally)
    uint64_t action_id;
    tk_error_code_t result = tk_decision_engine_queue_action(engine, &params, &action_id);

    // cleanup local copy
    free(params.params.emergency_alert.alert_message);

    if (result == TK_SUCCESS) {
        tk_log_info("Emergency response queued with action ID %llu", (unsigned long long)action_id);
    } else {
        tk_log_error("Failed to queue emergency response: %d", result);
    }

    return result;
}

void tk_decision_engine_free_response(tk_llm_response_t** response) {
    if (!response || !*response) {
        return;
    }
    
    tk_llm_response_t* r = *response;
    
    free(r->response_text);
    
    for (size_t i = 0; i < r->action_count; i++) {
        free_action_params_resources(&r->actions[i]);
    }
    
    free(r->actions);
    free(r);
    *response = NULL;
}

void tk_decision_engine_free_action_params(tk_action_params_t* params) {
    if (!params) {
        return;
    }
    
    free_action_params_resources(params);
}

//-----------------------------------------------------------------------------
// Internal Helper Implementations
//-----------------------------------------------------------------------------

static tk_error_code_t initialize_action_queue(action_queue_t* queue, size_t capacity) {
    if (!queue) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    queue->actions = calloc(capacity, sizeof(tk_action_t));
    if (!queue->actions) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    queue->capacity = capacity;
    queue->count = 0;
    
    if (pthread_mutex_init(&queue->mutex, NULL) != 0) {
        free(queue->actions);
        queue->actions = NULL;
        return TK_ERROR_SYSTEM_ERROR;
    }
    
    return TK_SUCCESS;
}

static void destroy_action_queue(action_queue_t* queue) {
    if (!queue) {
        return;
    }
    
    for (size_t i = 0; i < queue->count; i++) {
        free_action_resources(&queue->actions[i]);
    }
    
    free(queue->actions);
    queue->actions = NULL;
    queue->capacity = 0;
    queue->count = 0;
    
    pthread_mutex_destroy(&queue->mutex);
}

static tk_error_code_t queue_action(action_queue_t* queue, const tk_action_t* action) {
    if (!queue || !action) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    pthread_mutex_lock(&queue->mutex);

    if (queue->count >= queue->capacity) {
        pthread_mutex_unlock(&queue->mutex);
        return TK_ERROR_QUEUE_FULL;
    }

    tk_error_code_t rc = deep_copy_action(&queue->actions[queue->count], action);
    if (rc != TK_SUCCESS) {
        // deep_copy_action already guarantees cleanup on failure
        pthread_mutex_unlock(&queue->mutex);
        return rc;
    }

    queue->count++;
    pthread_mutex_unlock(&queue->mutex);
    return TK_SUCCESS;
}

static tk_error_code_t dequeue_action(action_queue_t* queue, uint64_t action_id, tk_action_t* out_action) {
    if (!queue || !out_action) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    pthread_mutex_lock(&queue->mutex);
    
    size_t index = 0;
    bool found = false;
    
    // If action_id is 0, dequeue the first item
    if (action_id == 0) {
        if (queue->count == 0) {
            pthread_mutex_unlock(&queue->mutex);
            return TK_ERROR_NOT_FOUND;
        }
        index = 0;
        found = true;
    } else {
        // Find specific action
        for (size_t i = 0; i < queue->count; i++) {
            if (queue->actions[i].action_id == action_id) {
                index = i;
                found = true;
                break;
            }
        }
    }
    
    if (!found) {
        pthread_mutex_unlock(&queue->mutex);
        return TK_ERROR_NOT_FOUND;
    }
    
    // Copy the action
    *out_action = queue->actions[index];
    
    // Remove from queue
    memmove(&queue->actions[index], 
            &queue->actions[index+1], 
            (queue->count - index - 1) * sizeof(tk_action_t));
    queue->count--;
    
    pthread_mutex_unlock(&queue->mutex);
    return TK_SUCCESS;
}

static tk_error_code_t find_action_in_queue(const action_queue_t* queue, uint64_t action_id, size_t* out_index) {
    if (!queue || !out_index) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    pthread_mutex_lock((pthread_mutex_t*)&queue->mutex);
    
    for (size_t i = 0; i < queue->count; i++) {
        if (queue->actions[i].action_id == action_id) {
            *out_index = i;
            pthread_mutex_unlock((pthread_mutex_t*)&queue->mutex);
            return TK_SUCCESS;
        }
    }
    
    pthread_mutex_unlock((pthread_mutex_t*)&queue->mutex);
    return TK_ERROR_NOT_FOUND;
}

static tk_error_code_t update_action_status(action_queue_t* queue, uint64_t action_id, tk_action_status_e status, const char* error_msg) {
    if (!queue) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    pthread_mutex_lock(&queue->mutex);
    
    for (size_t i = 0; i < queue->count; i++) {
        if (queue->actions[i].action_id == action_id) {
            queue->actions[i].status = status;
            if (error_msg) {
                free(queue->actions[i].error_message);
                queue->actions[i].error_message = strdup(error_msg);
                if (!queue->actions[i].error_message) {
                    pthread_mutex_unlock(&queue->mutex);
                    return TK_ERROR_OUT_OF_MEMORY;
                }
            }
            pthread_mutex_unlock(&queue->mutex);
            return TK_SUCCESS;
        }
    }
    
    pthread_mutex_unlock(&queue->mutex);
    return TK_ERROR_NOT_FOUND;
}

static uint64_t generate_action_id(void) {
    return atomic_fetch_add(&g_action_counter, 1);
}

static bool validate_action_params(const tk_action_params_t* params, const tk_context_summary_t* context) {
    if (!params) {
        return false;
    }
    
    // Basic validation based on action type
    switch (params->type) {
        case TK_ACTION_TYPE_SPEAK:
            if (!params->params.speak.text || strlen(params->params.speak.text) == 0) {
                return false;
            }
            break;
            
        case TK_ACTION_TYPE_NAVIGATE_GUIDE:
            if (!params->params.navigate_guide.instruction || 
                strlen(params->params.navigate_guide.instruction) == 0) {
                return false;
            }
            // Validate direction range
            if (params->params.navigate_guide.direction_deg < -180.0f || 
                params->params.navigate_guide.direction_deg > 180.0f) {
                return false;
            }
            break;
            
        case TK_ACTION_TYPE_NAVIGATE_WARN:
            if (!params->params.navigate_warn.warning_text || 
                strlen(params->params.navigate_warn.warning_text) == 0) {
                return false;
            }
            break;
            
        case TK_ACTION_TYPE_DESCRIBE_OBJECT:
            if (params->params.describe_object.object_id == 0 && 
                (!params->params.describe_object.object_label || 
                 strlen(params->params.describe_object.object_label) == 0)) {
                return false;
            }
            break;
            
        case TK_ACTION_TYPE_READ_TEXT:
            if (!params->params.read_text.text_content || 
                strlen(params->params.read_text.text_content) == 0) {
                return false;
            }
            break;
            
        case TK_ACTION_TYPE_SYSTEM_SETTING:
            if (!params->params.system_setting.setting_name || 
                strlen(params->params.system_setting.setting_name) == 0 ||
                !params->params.system_setting.setting_value) {
                return false;
            }
            break;
            
        case TK_ACTION_TYPE_USER_QUERY_RESPONSE:
            if (!params->params.user_query_response.response_text || 
                strlen(params->params.user_query_response.response_text) == 0) {
                return false;
            }
            break;
            
        case TK_ACTION_TYPE_EMERGENCY_ALERT:
            if (!params->params.emergency_alert.alert_message || 
                strlen(params->params.emergency_alert.alert_message) == 0) {
                return false;
            }
            break;
            
        default:
            // For other action types, basic validation passes
            break;
    }
    
    // Context-aware validation (if context is provided)
    if (context) {
        // Example: Check if navigation actions are valid given current context
        if (params->type == TK_ACTION_TYPE_NAVIGATE_GUIDE) {
            // Only allow navigation guidance if there's a clear path
            if (!context->has_clear_path) {
                return false;
            }
        }
        
        // Example: Check if object description is valid
        if (params->type == TK_ACTION_TYPE_DESCRIBE_OBJECT) {
            bool object_found = false;
            for (size_t i = 0; i < context->visible_object_count; i++) {
                if (context->visible_objects[i].id == params->params.describe_object.object_id) {
                    object_found = true;
                    break;
                }
            }
            
            if (!object_found) {
                return false;
            }
        }
    }
    
    return true;
}

static tk_error_code_t execute_single_action(tk_decision_engine_t* engine, tk_action_t* action, void* audio_ctx, void* nav_ctx, void* reasoner_ctx) {
    if (!engine || !action) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    log_action_execution(action);
    
    // Execute action based on its type
    switch (action->params.type) {
        case TK_ACTION_TYPE_SPEAK: {
            if (!audio_ctx) {
                tk_log_error("Audio context is NULL for SPEAK action");
                action->status = TK_ACTION_STATUS_FAILED;
                free(action->error_message);
                action->error_message = strdup("Audio context is NULL");
                if (!action->error_message) {
                    return TK_ERROR_OUT_OF_MEMORY;
                }
                return TK_ERROR_INVALID_ARGUMENT;
            }
            
            tk_log_info("Executing SPEAK action: '%s'", action->params.params.speak.text);
            
            // Call the audio pipeline function
            tk_error_code_t result = tk_audio_pipeline_synthesize_text(audio_ctx, action->params.params.speak.text);
            if (result != TK_SUCCESS) {
                tk_log_error("Failed to synthesize text: %d", result);
                action->status = TK_ACTION_STATUS_FAILED;
                free(action->error_message);
                action->error_message = strdup("Failed to synthesize text");
                if (!action->error_message) {
                    return TK_ERROR_OUT_OF_MEMORY;
                }
                return result;
            }
            break;
        }
            
        case TK_ACTION_TYPE_NAVIGATE_GUIDE: {
            if (!nav_ctx) {
                tk_log_error("Navigation context is NULL for NAVIGATE_GUIDE action");
                action->status = TK_ACTION_STATUS_FAILED;
                free(action->error_message);
                action->error_message = strdup("Navigation context is NULL");
                if (!action->error_message) {
                    return TK_ERROR_OUT_OF_MEMORY;
                }
                return TK_ERROR_INVALID_ARGUMENT;
            }
            
            tk_log_info("Executing NAVIGATE_GUIDE action: '%s'", 
                       action->params.params.navigate_guide.instruction);
            
            // Call the navigation engine function
            tk_error_code_t result = tk_navigation_engine_provide_guidance(
                nav_ctx, 
                action->params.params.navigate_guide.instruction,
                action->params.params.navigate_guide.direction_deg
            );
            if (result != TK_SUCCESS) {
                tk_log_error("Failed to provide navigation guidance: %d", result);
                action->status = TK_ACTION_STATUS_FAILED;
                free(action->error_message);
                action->error_message = strdup("Failed to provide navigation guidance");
                if (!action->error_message) {
                    return TK_ERROR_OUT_OF_MEMORY;
                }
                return result;
            }
            break;
        }
            
        case TK_ACTION_TYPE_NAVIGATE_WARN: {
            if (!nav_ctx) {
                tk_log_error("Navigation context is NULL for NAVIGATE_WARN action");
                action->status = TK_ACTION_STATUS_FAILED;
                free(action->error_message);
                action->error_message = strdup("Navigation context is NULL");
                if (!action->error_message) {
                    return TK_ERROR_OUT_OF_MEMORY;
                }
                return TK_ERROR_INVALID_ARGUMENT;
            }
            
            tk_log_info("Executing NAVIGATE_WARN action: '%s'", 
                       action->params.params.navigate_warn.warning_text);
            
            // Call the navigation engine function
            tk_error_code_t result = tk_navigation_engine_issue_warning(
                nav_ctx,
                action->params.params.navigate_warn.warning_text,
                action->params.params.navigate_warn.obstacle_id
            );
            if (result != TK_SUCCESS) {
                tk_log_error("Failed to issue navigation warning: %d", result);
                action->status = TK_ACTION_STATUS_FAILED;
                free(action->error_message);
                action->error_message = strdup("Failed to issue navigation warning");
                if (!action->error_message) {
                    return TK_ERROR_OUT_OF_MEMORY;
                }
                return result;
            }
            break;
        }
            
        case TK_ACTION_TYPE_DESCRIBE_OBJECT: {
            if (!audio_ctx || !reasoner_ctx) {
                tk_log_error("Audio or reasoner context is NULL for DESCRIBE_OBJECT action");
                action->status = TK_ACTION_STATUS_FAILED;
                free(action->error_message);
                action->error_message = strdup("Audio or reasoner context is NULL");
                if (!action->error_message) {
                    return TK_ERROR_OUT_OF_MEMORY;
                }
                return TK_ERROR_INVALID_ARGUMENT;
            }
            
            tk_log_info("Executing DESCRIBE_OBJECT action for object ID %u", 
                       action->params.params.describe_object.object_id);
            
            // Create a buffer for the object description
            char description[512];
            
            // Call the contextual reasoner to get object details
            tk_error_code_t result = tk_contextual_reasoner_get_object_details(
                reasoner_ctx,
                action->params.params.describe_object.object_id,
                description,
                sizeof(description)
            );
            
            if (result != TK_SUCCESS) {
                tk_log_error("Failed to get object details: %d", result);
                action->status = TK_ACTION_STATUS_FAILED;
                free(action->error_message);
                action->error_message = strdup("Failed to get object details");
                if (!action->error_message) {
                    return TK_ERROR_OUT_OF_MEMORY;
                }
                return result;
            }
            
            // Now speak the description using the audio pipeline
            result = tk_audio_pipeline_synthesize_text(audio_ctx, description);
            if (result != TK_SUCCESS) {
                tk_log_error("Failed to synthesize object description: %d", result);
                action->status = TK_ACTION_STATUS_FAILED;
                free(action->error_message);
                action->error_message = strdup("Failed to synthesize object description");
                if (!action->error_message) {
                    return TK_ERROR_OUT_OF_MEMORY;
                }
                return result;
            }
            break;
        }
            
        case TK_ACTION_TYPE_READ_TEXT: {
            if (!audio_ctx) {
                tk_log_error("Audio context is NULL for READ_TEXT action");
                action->status = TK_ACTION_STATUS_FAILED;
                free(action->error_message);
                action->error_message = strdup("Audio context is NULL");
                if (!action->error_message) {
                    return TK_ERROR_OUT_OF_MEMORY;
                }
                return TK_ERROR_INVALID_ARGUMENT;
            }
            
            tk_log_info("Executing READ_TEXT action: '%.50s...'", 
                       action->params.params.read_text.text_content);
            
            // Call the audio pipeline function
            tk_error_code_t result = tk_audio_pipeline_synthesize_text(audio_ctx, action->params.params.read_text.text_content);
            if (result != TK_SUCCESS) {
                tk_log_error("Failed to synthesize text: %d", result);
                action->status = TK_ACTION_STATUS_FAILED;
                free(action->error_message);
                action->error_message = strdup("Failed to synthesize text");
                if (!action->error_message) {
                    return TK_ERROR_OUT_OF_MEMORY;
                }
                return result;
            }
            break;
        }
            
        case TK_ACTION_TYPE_SYSTEM_MODE_CHANGE:
            tk_log_info("Executing SYSTEM_MODE_CHANGE action");
            // In real implementation: change_system_mode(...)
            break;
            
        case TK_ACTION_TYPE_SYSTEM_SETTING:
            tk_log_info("Executing SYSTEM_SETTING action: %s = %s", 
                       action->params.params.system_setting.setting_name,
                       action->params.params.system_setting.setting_value);
            // In real implementation: update_system_setting(...)
            break;
            
        case TK_ACTION_TYPE_USER_QUERY_RESPONSE: {
            if (!audio_ctx) {
                tk_log_error("Audio context is NULL for USER_QUERY_RESPONSE action");
                action->status = TK_ACTION_STATUS_FAILED;
                free(action->error_message);
                action->error_message = strdup("Audio context is NULL");
                if (!action->error_message) {
                    return TK_ERROR_OUT_OF_MEMORY;
                }
                return TK_ERROR_INVALID_ARGUMENT;
            }
            
            tk_log_info("Executing USER_QUERY_RESPONSE action: '%.50s...'", 
                       action->params.params.user_query_response.response_text);
            
            // Call the audio pipeline function
            tk_error_code_t result = tk_audio_pipeline_synthesize_text(audio_ctx, action->params.params.user_query_response.response_text);
            if (result != TK_SUCCESS) {
                tk_log_error("Failed to synthesize user query response: %d", result);
                action->status = TK_ACTION_STATUS_FAILED;
                free(action->error_message);
                action->error_message = strdup("Failed to synthesize user query response");
                if (!action->error_message) {
                    return TK_ERROR_OUT_OF_MEMORY;
                }
                return result;
            }
            break;
        }
            
        case TK_ACTION_TYPE_EMERGENCY_ALERT: {
            if (!audio_ctx) {
                tk_log_error("Audio context is NULL for EMERGENCY_ALERT action");
                action->status = TK_ACTION_STATUS_FAILED;
                free(action->error_message);
                action->error_message = strdup("Audio context is NULL");
                if (!action->error_message) {
                    return TK_ERROR_OUT_OF_MEMORY;
                }
                return TK_ERROR_INVALID_ARGUMENT;
            }
            
            tk_log_warning("Executing EMERGENCY_ALERT action: '%s'", 
                          action->params.params.emergency_alert.alert_message);
            
            // Call the audio pipeline function
            // TODO: Consider implementing a higher priority alert function in the audio pipeline
            tk_error_code_t result = tk_audio_pipeline_synthesize_text(audio_ctx, action->params.params.emergency_alert.alert_message);
            if (result != TK_SUCCESS) {
                tk_log_error("Failed to synthesize emergency alert: %d", result);
                action->status = TK_ACTION_STATUS_FAILED;
                free(action->error_message);
                action->error_message = strdup("Failed to synthesize emergency alert");
                if (!action->error_message) {
                    return TK_ERROR_OUT_OF_MEMORY;
                }
                return result;
            }
            break;
        }
            
        default:
            tk_log_warning("Unknown action type: %d", action->params.type);
            action->status = TK_ACTION_STATUS_FAILED;
            free(action->error_message);
            action->error_message = strdup("Unknown action type");
            if (!action->error_message) {
                return TK_ERROR_OUT_OF_MEMORY;
            }
            return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Mark action as completed
    action->status = TK_ACTION_STATUS_COMPLETED;
    action->completed_timestamp_ns = get_current_time_ns();
    
    // Move to completed queue
    tk_action_t completed_action;
    tk_error_code_t result = deep_copy_action(&completed_action, action);
    
    if (result == TK_SUCCESS) {
        result = queue_action(&engine->completed_queue, &completed_action);
    }
    
    if (result == TK_SUCCESS) {
        engine->stats.total_actions_completed++;
        
        // Remove from executing queue
        for (size_t i = 0; i < engine->executing_queue.count; i++) {
            if (engine->executing_queue.actions[i].action_id == action->action_id) {
                memmove(&engine->executing_queue.actions[i], 
                        &engine->executing_queue.actions[i+1], 
                        (engine->executing_queue.count - i - 1) * sizeof(tk_action_t));
                engine->executing_queue.count--;
                break;
            }
        }
        
        // Notify callback if provided
        if (engine->callbacks.on_action_completed) {
            // Unlock mutex before calling callback
            pthread_mutex_unlock(&engine->engine_mutex);
            engine->callbacks.on_action_completed(&completed_action, engine->callbacks.user_data);
            pthread_mutex_lock(&engine->engine_mutex);
        }
        
        free_action_resources(&completed_action);
    } else {
        tk_log_error("Failed to move action to completed queue: %d", result);
        action->status = TK_ACTION_STATUS_FAILED;
        free(action->error_message);
        action->error_message = strdup("Failed to queue completion");
        if (!action->error_message) {
            return TK_ERROR_OUT_OF_MEMORY;
        }
        return result;
    }
    
    return TK_SUCCESS;
}

static void free_action_resources(tk_action_t* action) {
    if (!action) {
        return;
    }
    
    free_action_params_resources(&action->params);
    free(action->error_message);
    memset(action, 0, sizeof(tk_action_t));
}

static void free_action_params_resources(tk_action_params_t* params) {
    if (!params) {
        return;
    }
    
    switch (params->type) {
        case TK_ACTION_TYPE_SPEAK:
            free(params->params.speak.text);
            break;
            
        case TK_ACTION_TYPE_NAVIGATE_GUIDE:
            free(params->params.navigate_guide.instruction);
            break;
            
        case TK_ACTION_TYPE_NAVIGATE_WARN:
            free(params->params.navigate_warn.warning_text);
            break;
            
        case TK_ACTION_TYPE_DESCRIBE_OBJECT:
            free(params->params.describe_object.object_label);
            break;
            
        case TK_ACTION_TYPE_READ_TEXT:
            free(params->params.read_text.text_content);
            break;
            
        case TK_ACTION_TYPE_SYSTEM_SETTING:
            free(params->params.system_setting.setting_name);
            free(params->params.system_setting.setting_value);
            break;
            
        case TK_ACTION_TYPE_USER_QUERY_RESPONSE:
            free(params->params.user_query_response.response_text);
            break;
            
        case TK_ACTION_TYPE_EMERGENCY_ALERT:
            free(params->params.emergency_alert.alert_message);
            break;
            
        default:
            // No additional resources to free
            break;
    }
    
    memset(params, 0, sizeof(tk_action_params_t));
}

static tk_error_code_t parse_llm_response_text(const char* text, tk_llm_response_t** out_response) {
    if (!text || !out_response) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Parse the JSON text
    cJSON* json = cJSON_Parse(text);
    if (!json) {
        tk_log_error("Failed to parse JSON response from LLM");
        return TK_ERROR_INVALID_FORMAT;
    }
    
    // Allocate response structure
    tk_llm_response_t* response = calloc(1, sizeof(tk_llm_response_t));
    if (!response) {
        cJSON_Delete(json);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Get response_text
    cJSON* response_text_json = cJSON_GetObjectItemCaseSensitive(json, "response_text");
    if (cJSON_IsString(response_text_json) && response_text_json->valuestring) {
        response->response_text = strdup(response_text_json->valuestring);
        if (!response->response_text) {
            free(response);
            cJSON_Delete(json);
            return TK_ERROR_OUT_OF_MEMORY;
        }
    } else {
        // If no response_text, use empty string
        response->response_text = strdup("");
        if (!response->response_text) {
            free(response);
            cJSON_Delete(json);
            return TK_ERROR_OUT_OF_MEMORY;
        }
    }
    
    // Get priority
    cJSON* priority_json = cJSON_GetObjectItemCaseSensitive(json, "priority");
    if (cJSON_IsString(priority_json) && priority_json->valuestring) {
        if (strcmp(priority_json->valuestring, "high") == 0) {
            response->priority = TK_RESPONSE_PRIORITY_HIGH;
        } else if (strcmp(priority_json->valuestring, "critical") == 0) {
            response->priority = TK_RESPONSE_PRIORITY_CRITICAL;
        } else {
            response->priority = TK_RESPONSE_PRIORITY_NORMAL; // Default to normal
        }
    } else {
        response->priority = TK_RESPONSE_PRIORITY_NORMAL; // Default to normal
    }
    
    // Get actions array
    cJSON* actions_json = cJSON_GetObjectItemCaseSensitive(json, "actions");
    if (!cJSON_IsArray(actions_json)) {
        tk_log_error("Invalid or missing 'actions' array in JSON response");
        free(response->response_text);
        free(response);
        cJSON_Delete(json);
        return TK_ERROR_INVALID_FORMAT;
    }
    
    size_t action_count = cJSON_GetArraySize(actions_json);
    if (action_count == 0) {
        tk_log_warning("No actions found in LLM response");
        response->action_count = 0;
        response->actions = NULL;
        *out_response = response;
        cJSON_Delete(json);
        return TK_SUCCESS;
    }
    
    // Allocate actions array
    response->actions = calloc(action_count, sizeof(tk_action_params_t));
    if (!response->actions) {
        free(response->response_text);
        free(response);
        cJSON_Delete(json);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    response->action_count = action_count;
    
    // Parse each action
    for (size_t i = 0; i < action_count; i++) {
        cJSON* action_json = cJSON_GetArrayItem(actions_json, i);
        if (!action_json) {
            tk_log_error("Failed to get action %zu from array", i);
            // Clean up already parsed actions
            for (size_t j = 0; j < i; j++) {
                free_action_params_resources(&response->actions[j]);
            }
            free(response->actions);
            free(response->response_text);
            free(response);
            cJSON_Delete(json);
            return TK_ERROR_INVALID_FORMAT;
        }
        
        tk_action_params_t* action = &response->actions[i];
        
        // Get action type
        cJSON* type_json = cJSON_GetObjectItemCaseSensitive(action_json, "type");
        if (!cJSON_IsString(type_json) || !type_json->valuestring) {
            tk_log_error("Missing or invalid 'type' in action %zu", i);
            // Clean up already parsed actions
            for (size_t j = 0; j < i; j++) {
                free_action_params_resources(&response->actions[j]);
            }
            free(response->actions);
            free(response->response_text);
            free(response);
            cJSON_Delete(json);
            return TK_ERROR_INVALID_FORMAT;
        }
        
        // Map string type to enum
        if (strcmp(type_json->valuestring, "SPEAK") == 0) {
            action->type = TK_ACTION_TYPE_SPEAK;
        } else if (strcmp(type_json->valuestring, "NAVIGATE_GUIDE") == 0) {
            action->type = TK_ACTION_TYPE_NAVIGATE_GUIDE;
        } else if (strcmp(type_json->valuestring, "NAVIGATE_WARN") == 0) {
            action->type = TK_ACTION_TYPE_NAVIGATE_WARN;
        } else if (strcmp(type_json->valuestring, "DESCRIBE_OBJECT") == 0) {
            action->type = TK_ACTION_TYPE_DESCRIBE_OBJECT;
        } else if (strcmp(type_json->valuestring, "READ_TEXT") == 0) {
            action->type = TK_ACTION_TYPE_READ_TEXT;
        } else if (strcmp(type_json->valuestring, "SYSTEM_MODE_CHANGE") == 0) {
            action->type = TK_ACTION_TYPE_SYSTEM_MODE_CHANGE;
        } else if (strcmp(type_json->valuestring, "SYSTEM_SETTING") == 0) {
            action->type = TK_ACTION_TYPE_SYSTEM_SETTING;
        } else if (strcmp(type_json->valuestring, "USER_QUERY_RESPONSE") == 0) {
            action->type = TK_ACTION_TYPE_USER_QUERY_RESPONSE;
        } else if (strcmp(type_json->valuestring, "EMERGENCY_ALERT") == 0) {
            action->type = TK_ACTION_TYPE_EMERGENCY_ALERT;
        } else {
            tk_log_error("Unknown action type '%s' in action %zu", type_json->valuestring, i);
            // Clean up already parsed actions
            for (size_t j = 0; j < i; j++) {
                free_action_params_resources(&response->actions[j]);
            }
            free(response->actions);
            free(response->response_text);
            free(response);
            cJSON_Delete(json);
            return TK_ERROR_INVALID_FORMAT;
        }
        
        // Get confidence
        cJSON* confidence_json = cJSON_GetObjectItemCaseSensitive(action_json, "confidence");
        if (cJSON_IsNumber(confidence_json)) {
            action->confidence = (float)confidence_json->valuedouble;
        } else {
            action->confidence = 0.0f; // Default to 0 if not provided
        }
        
        // Get params object
        cJSON* params_json = cJSON_GetObjectItemCaseSensitive(action_json, "params");
        if (!cJSON_IsObject(params_json)) {
            tk_log_error("Missing or invalid 'params' object in action %zu", i);
            // Clean up already parsed actions
            for (size_t j = 0; j < i; j++) {
                free_action_params_resources(&response->actions[j]);
            }
            free(response->actions);
            free(response->response_text);
            free(response);
            cJSON_Delete(json);
            return TK_ERROR_INVALID_FORMAT;
        }
        
        // Parse params based on action type
        switch (action->type) {
            case TK_ACTION_TYPE_SPEAK: {
                cJSON* text_json = cJSON_GetObjectItemCaseSensitive(params_json, "text");
                if (cJSON_IsString(text_json) && text_json->valuestring) {
                    action->params.speak.text = strdup(text_json->valuestring);
                    if (!action->params.speak.text) {
                        // Clean up already parsed actions
                        for (size_t j = 0; j < i; j++) {
                            free_action_params_resources(&response->actions[j]);
                        }
                        free(response->actions);
                        free(response->response_text);
                        free(response);
                        cJSON_Delete(json);
                        return TK_ERROR_OUT_OF_MEMORY;
                    }
                } else {
                    action->params.speak.text = strdup(""); // Default to empty string
                    if (!action->params.speak.text) {
                        // Clean up already parsed actions
                        for (size_t j = 0; j < i; j++) {
                            free_action_params_resources(&response->actions[j]);
                        }
                        free(response->actions);
                        free(response->response_text);
                        free(response);
                        cJSON_Delete(json);
                        return TK_ERROR_OUT_OF_MEMORY;
                    }
                }
                action->params.speak.priority = TK_RESPONSE_PRIORITY_NORMAL; // Default
                action->params.speak.volume_multiplier = 1.0f; // Default
                break;
            }
                
            case TK_ACTION_TYPE_NAVIGATE_GUIDE: {
                cJSON* instruction_json = cJSON_GetObjectItemCaseSensitive(params_json, "instruction");
                if (cJSON_IsString(instruction_json) && instruction_json->valuestring) {
                    action->params.navigate_guide.instruction = strdup(instruction_json->valuestring);
                    if (!action->params.navigate_guide.instruction) {
                        // Clean up already parsed actions
                        for (size_t j = 0; j < i; j++) {
                            free_action_params_resources(&response->actions[j]);
                        }
                        free(response->actions);
                        free(response->response_text);
                        free(response);
                        cJSON_Delete(json);
                        return TK_ERROR_OUT_OF_MEMORY;
                    }
                } else {
                    action->params.navigate_guide.instruction = strdup(""); // Default to empty string
                    if (!action->params.navigate_guide.instruction) {
                        // Clean up already parsed actions
                        for (size_t j = 0; j < i; j++) {
                            free_action_params_resources(&response->actions[j]);
                        }
                        free(response->actions);
                        free(response->response_text);
                        free(response);
                        cJSON_Delete(json);
                        return TK_ERROR_OUT_OF_MEMORY;
                    }
                }
                
                cJSON* direction_json = cJSON_GetObjectItemCaseSensitive(params_json, "direction_deg");
                if (cJSON_IsNumber(direction_json)) {
                    action->params.navigate_guide.direction_deg = (float)direction_json->valuedouble;
                } else {
                    action->params.navigate_guide.direction_deg = 0.0f; // Default to 0
                }
                break;
            }
                
            case TK_ACTION_TYPE_NAVIGATE_WARN: {
                cJSON* warning_text_json = cJSON_GetObjectItemCaseSensitive(params_json, "warning_text");
                if (cJSON_IsString(warning_text_json) && warning_text_json->valuestring) {
                    action->params.navigate_warn.warning_text = strdup(warning_text_json->valuestring);
                    if (!action->params.navigate_warn.warning_text) {
                        // Clean up already parsed actions
                        for (size_t j = 0; j < i; j++) {
                            free_action_params_resources(&response->actions[j]);
                        }
                        free(response->actions);
                        free(response->response_text);
                        free(response);
                        cJSON_Delete(json);
                        return TK_ERROR_OUT_OF_MEMORY;
                    }
                } else {
                    action->params.navigate_warn.warning_text = strdup(""); // Default to empty string
                    if (!action->params.navigate_warn.warning_text) {
                        // Clean up already parsed actions
                        for (size_t j = 0; j < i; j++) {
                            free_action_params_resources(&response->actions[j]);
                        }
                        free(response->actions);
                        free(response->response_text);
                        free(response);
                        cJSON_Delete(json);
                        return TK_ERROR_OUT_OF_MEMORY;
                    }
                }
                
                cJSON* obstacle_id_json = cJSON_GetObjectItemCaseSensitive(params_json, "obstacle_id");
                if (cJSON_IsNumber(obstacle_id_json)) {
                    action->params.navigate_warn.obstacle_id = (uint32_t)obstacle_id_json->valueint;
                } else {
                    action->params.navigate_warn.obstacle_id = 0; // Default to 0
                }
                break;
            }
                
            case TK_ACTION_TYPE_DESCRIBE_OBJECT: {
                cJSON* object_id_json = cJSON_GetObjectItemCaseSensitive(params_json, "object_id");
                if (cJSON_IsNumber(object_id_json)) {
                    action->params.describe_object.object_id = (uint32_t)object_id_json->valueint;
                } else {
                    action->params.describe_object.object_id = 0; // Default to 0
                }
                
                cJSON* object_label_json = cJSON_GetObjectItemCaseSensitive(params_json, "object_label");
                if (cJSON_IsString(object_label_json) && object_label_json->valuestring) {
                    action->params.describe_object.object_label = strdup(object_label_json->valuestring);
                    if (!action->params.describe_object.object_label) {
                        // Clean up already parsed actions
                        for (size_t j = 0; j < i; j++) {
                            free_action_params_resources(&response->actions[j]);
                        }
                        free(response->actions);
                        free(response->response_text);
                        free(response);
                        cJSON_Delete(json);
                        return TK_ERROR_OUT_OF_MEMORY;
                    }
                } else {
                    action->params.describe_object.object_label = NULL; // Optional
                }
                break;
            }
                
            case TK_ACTION_TYPE_READ_TEXT: {
                cJSON* text_content_json = cJSON_GetObjectItemCaseSensitive(params_json, "text_content");
                if (cJSON_IsString(text_content_json) && text_content_json->valuestring) {
                    action->params.read_text.text_content = strdup(text_content_json->valuestring);
                    if (!action->params.read_text.text_content) {
                        // Clean up already parsed actions
                        for (size_t j = 0; j < i; j++) {
                            free_action_params_resources(&response->actions[j]);
                        }
                        free(response->actions);
                        free(response->response_text);
                        free(response);
                        cJSON_Delete(json);
                        return TK_ERROR_OUT_OF_MEMORY;
                    }
                } else {
                    action->params.read_text.text_content = strdup(""); // Default to empty string
                    if (!action->params.read_text.text_content) {
                        // Clean up already parsed actions
                        for (size_t j = 0; j < i; j++) {
                            free_action_params_resources(&response->actions[j]);
                        }
                        free(response->actions);
                        free(response->response_text);
                        free(response);
                        cJSON_Delete(json);
                        return TK_ERROR_OUT_OF_MEMORY;
                    }
                }
                break;
            }
                
            case TK_ACTION_TYPE_SYSTEM_SETTING: {
                cJSON* setting_name_json = cJSON_GetObjectItemCaseSensitive(params_json, "setting_name");
                if (cJSON_IsString(setting_name_json) && setting_name_json->valuestring) {
                    action->params.system_setting.setting_name = strdup(setting_name_json->valuestring);
                    if (!action->params.system_setting.setting_name) {
                        // Clean up already parsed actions
                        for (size_t j = 0; j < i; j++) {
                            free_action_params_resources(&response->actions[j]);
                        }
                        free(response->actions);
                        free(response->response_text);
                        free(response);
                        cJSON_Delete(json);
                        return TK_ERROR_OUT_OF_MEMORY;
                    }
                } else {
                    action->params.system_setting.setting_name = strdup(""); // Default to empty string
                    if (!action->params.system_setting.setting_name) {
                        // Clean up already parsed actions
                        for (size_t j = 0; j < i; j++) {
                            free_action_params_resources(&response->actions[j]);
                        }
                        free(response->actions);
                        free(response->response_text);
                        free(response);
                        cJSON_Delete(json);
                        return TK_ERROR_OUT_OF_MEMORY;
                    }
                }
                
                cJSON* setting_value_json = cJSON_GetObjectItemCaseSensitive(params_json, "setting_value");
                if (cJSON_IsString(setting_value_json) && setting_value_json->valuestring) {
                    action->params.system_setting.setting_value = strdup(setting_value_json->valuestring);
                    if (!action->params.system_setting.setting_value) {
                        // Clean up already parsed actions and setting_name
                        free(action->params.system_setting.setting_name);
                        for (size_t j = 0; j < i; j++) {
                            free_action_params_resources(&response->actions[j]);
                        }
                        free(response->actions);
                        free(response->response_text);
                        free(response);
                        cJSON_Delete(json);
                        return TK_ERROR_OUT_OF_MEMORY;
                    }
                } else {
                    action->params.system_setting.setting_value = strdup(""); // Default to empty string
                    if (!action->params.system_setting.setting_value) {
                        // Clean up already parsed actions and setting_name
                        free(action->params.system_setting.setting_name);
                        for (size_t j = 0; j < i; j++) {
                            free_action_params_resources(&response->actions[j]);
                        }
                        free(response->actions);
                        free(response->response_text);
                        free(response);
                        cJSON_Delete(json);
                        return TK_ERROR_OUT_OF_MEMORY;
                    }
                }
                break;
            }
                
            case TK_ACTION_TYPE_USER_QUERY_RESPONSE: {
                cJSON* response_text_json = cJSON_GetObjectItemCaseSensitive(params_json, "response_text");
                if (cJSON_IsString(response_text_json) && response_text_json->valuestring) {
                    action->params.user_query_response.response_text = strdup(response_text_json->valuestring);
                    if (!action->params.user_query_response.response_text) {
                        // Clean up already parsed actions
                        for (size_t j = 0; j < i; j++) {
                            free_action_params_resources(&response->actions[j]);
                        }
                        free(response->actions);
                        free(response->response_text);
                        free(response);
                        cJSON_Delete(json);
                        return TK_ERROR_OUT_OF_MEMORY;
                    }
                } else {
                    action->params.user_query_response.response_text = strdup(""); // Default to empty string
                    if (!action->params.user_query_response.response_text) {
                        // Clean up already parsed actions
                        for (size_t j = 0; j < i; j++) {
                            free_action_params_resources(&response->actions[j]);
                        }
                        free(response->actions);
                        free(response->response_text);
                        free(response);
                        cJSON_Delete(json);
                        return TK_ERROR_OUT_OF_MEMORY;
                    }
                }
                break;
            }
                
            case TK_ACTION_TYPE_EMERGENCY_ALERT: {
                cJSON* alert_message_json = cJSON_GetObjectItemCaseSensitive(params_json, "alert_message");
                if (cJSON_IsString(alert_message_json) && alert_message_json->valuestring) {
                    action->params.emergency_alert.alert_message = strdup(alert_message_json->valuestring);
                    if (!action->params.emergency_alert.alert_message) {
                        // Clean up already parsed actions
                        for (size_t j = 0; j < i; j++) {
                            free_action_params_resources(&response->actions[j]);
                        }
                        free(response->actions);
                        free(response->response_text);
                        free(response);
                        cJSON_Delete(json);
                        return TK_ERROR_OUT_OF_MEMORY;
                    }
                } else {
                    action->params.emergency_alert.alert_message = strdup(""); // Default to empty string
                    if (!action->params.emergency_alert.alert_message) {
                        // Clean up already parsed actions
                        for (size_t j = 0; j < i; j++) {
                            free_action_params_resources(&response->actions[j]);
                        }
                        free(response->actions);
                        free(response->response_text);
                        free(response);
                        cJSON_Delete(json);
                        return TK_ERROR_OUT_OF_MEMORY;
                    }
                }
                
                // Optional parameters
                cJSON* repeat_alert_json = cJSON_GetObjectItemCaseSensitive(params_json, "repeat_alert");
                if (cJSON_IsBool(repeat_alert_json)) {
                    action->params.emergency_alert.repeat_alert = cJSON_IsTrue(repeat_alert_json);
                } else {
                    action->params.emergency_alert.repeat_alert = false; // Default
                }
                
                cJSON* repeat_interval_ms_json = cJSON_GetObjectItemCaseSensitive(params_json, "repeat_interval_ms");
                if (cJSON_IsNumber(repeat_interval_ms_json)) {
                    action->params.emergency_alert.repeat_interval_ms = (uint32_t)repeat_interval_ms_json->valueint;
                } else {
                    action->params.emergency_alert.repeat_interval_ms = 0; // Default
                }
                break;
            }
                
            default:
                // For other action types, no additional params to parse
                break;
        }
    }
    
    // Clean up JSON
    cJSON_Delete(json);
    
    *out_response = response;
    return TK_SUCCESS;
}

static void log_action_execution(const tk_action_t* action) {
    if (!action) {
        return;
    }
    
    const char* type_names[] = {
        "SPEAK",
        "NAVIGATE_GUIDE",
        "NAVIGATE_WARN",
        "DESCRIBE_ENVIRONMENT",
        "DESCRIBE_OBJECT",
        "READ_TEXT",
        "SYSTEM_MODE_CHANGE",
        "SYSTEM_SETTING",
        "USER_QUERY_RESPONSE",
        "EMERGENCY_ALERT"
    };
    
    const char* type_name = "UNKNOWN";
    if (action->params.type >= 0 && action->params.type < TK_ACTION_TYPE_EMERGENCY_ALERT) {
        type_name = type_names[action->params.type];
    }
    
    tk_log_debug("Executing action %llu (type: %s, confidence: %.2f)", 
                (unsigned long long)action->action_id, type_name, action->params.confidence);
}

static uint64_t get_current_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static tk_error_code_t deep_copy_action(tk_action_t* dst, const tk_action_t* src) {
    if (!dst || !src) return TK_ERROR_INVALID_ARGUMENT;

    memset(dst, 0, sizeof(*dst));
    *dst = *src; // shallow copy of scalars and pointers

    tk_error_code_t rc = deep_copy_action_params(&dst->params, &src->params);
    if (rc != TK_SUCCESS) {
        memset(dst, 0, sizeof(*dst));
        return rc;
    }

    if (src->error_message) {
        dst->error_message = strdup(src->error_message);
        if (!dst->error_message) {
            free_action_params_resources(&dst->params);
            memset(dst, 0, sizeof(*dst));
            return TK_ERROR_OUT_OF_MEMORY;
        }
    }

    return TK_SUCCESS;
}

static tk_error_code_t deep_copy_action_params(tk_action_params_t* dst, const tk_action_params_t* src) {
    if (!dst || !src) return TK_ERROR_INVALID_ARGUMENT;

    // zero dst to make free_action_params_resources safe on failure
    memset(dst, 0, sizeof(*dst));
    *dst = *src; // shallow copy; we'll overwrite pointers below

    tk_error_code_t rc = TK_SUCCESS;

    switch (src->type) {
        case TK_ACTION_TYPE_SPEAK:
            if (src->params.speak.text) {
                dst->params.speak.text = strdup(src->params.speak.text);
                if (!dst->params.speak.text) rc = TK_ERROR_OUT_OF_MEMORY;
            }
            break;

        case TK_ACTION_TYPE_NAVIGATE_GUIDE:
            if (src->params.navigate_guide.instruction) {
                dst->params.navigate_guide.instruction = strdup(src->params.navigate_guide.instruction);
                if (!dst->params.navigate_guide.instruction) rc = TK_ERROR_OUT_OF_MEMORY;
            }
            break;

        case TK_ACTION_TYPE_NAVIGATE_WARN:
            if (src->params.navigate_warn.warning_text) {
                dst->params.navigate_warn.warning_text = strdup(src->params.navigate_warn.warning_text);
                if (!dst->params.navigate_warn.warning_text) rc = TK_ERROR_OUT_OF_MEMORY;
            }
            break;

        case TK_ACTION_TYPE_DESCRIBE_OBJECT:
            if (src->params.describe_object.object_label) {
                dst->params.describe_object.object_label = strdup(src->params.describe_object.object_label);
                if (!dst->params.describe_object.object_label) rc = TK_ERROR_OUT_OF_MEMORY;
            }
            break;

        case TK_ACTION_TYPE_READ_TEXT:
            if (src->params.read_text.text_content) {
                dst->params.read_text.text_content = strdup(src->params.read_text.text_content);
                if (!dst->params.read_text.text_content) rc = TK_ERROR_OUT_OF_MEMORY;
            }
            break;

        case TK_ACTION_TYPE_SYSTEM_SETTING:
            if (src->params.system_setting.setting_name) {
                dst->params.system_setting.setting_name = strdup(src->params.system_setting.setting_name);
                if (!dst->params.system_setting.setting_name) { rc = TK_ERROR_OUT_OF_MEMORY; break; }
            }
            if (src->params.system_setting.setting_value) {
                dst->params.system_setting.setting_value = strdup(src->params.system_setting.setting_value);
                if (!dst->params.system_setting.setting_value) rc = TK_ERROR_OUT_OF_MEMORY;
            }
            break;

        case TK_ACTION_TYPE_USER_QUERY_RESPONSE:
            if (src->params.user_query_response.response_text) {
                dst->params.user_query_response.response_text = strdup(src->params.user_query_response.response_text);
                if (!dst->params.user_query_response.response_text) rc = TK_ERROR_OUT_OF_MEMORY;
            }
            break;

        case TK_ACTION_TYPE_EMERGENCY_ALERT:
            if (src->params.emergency_alert.alert_message) {
                dst->params.emergency_alert.alert_message = strdup(src->params.emergency_alert.alert_message);
                if (!dst->params.emergency_alert.alert_message) rc = TK_ERROR_OUT_OF_MEMORY;
            }
            break;

        default:
            /* nothing dynamic to copy */
            break;
    }

    if (rc != TK_SUCCESS) {
        // rollback partial allocations
        free_action_params_resources(dst);
    }

    return rc;
}
