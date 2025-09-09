/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_decision_engine.h
*
* This header file defines the public API for the Decision Engine, the executive
* control center of the TrackieLLM Cortex. This module is responsible for
* interpreting the Large Language Model's responses and translating them into
* concrete actions that the system can execute.
*
* The Decision Engine implements a sophisticated action parsing and execution
* framework that allows the LLM to "call functions" by generating structured
* text responses. This enables the AI to interact with the real world through
* a controlled, safe interface.
*
* Key responsibilities include:
*   1. LLM Response Parsing: Analyzing the text output from the language model
*      to identify intended actions, parameters, and response content.
*   2. Action Validation: Ensuring that requested actions are safe, valid,
*      and within the system's capabilities.
*   3. Action Execution: Coordinating with other system components to carry
*      out the requested actions.
*   4. Response Generation: Formatting and prioritizing responses back to the user.
*   5. State Management: Tracking the execution state of multi-step actions.
*
* The engine supports various action categories:
*   - Navigation Actions (provide directions, warn of hazards)
*   - Communication Actions (speak responses, adjust volume)
*   - Information Actions (describe environment, read text)
*   - System Actions (change settings, enter specific modes)
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TRACKIELLM_CORTEX_TK_DECISION_ENGINE_H
#define TRACKIELLM_CORTEX_TK_DECISION_ENGINE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "cortex/tk_contextual_reasoner.h"  // For context types

// Forward-declare the primary engine object as an opaque type.
typedef struct tk_decision_engine_s tk_decision_engine_t;

/**
 * @struct tk_decision_config_t
 * @brief Configuration for initializing the Decision Engine.
 */
typedef struct {
    float action_confidence_threshold;      /**< Minimum confidence required to execute actions. */
    size_t max_concurrent_actions;          /**< Maximum number of actions that can be executing simultaneously. */
    uint32_t action_timeout_ms;             /**< Default timeout for action execution. */
    bool enable_safety_constraints;        /**< Whether to enforce safety constraints on actions. */
    float response_priority_threshold;      /**< Minimum priority for responses to be spoken. */
} tk_decision_config_t;

/**
 * @enum tk_action_type_e
 * @brief Types of actions the decision engine can execute.
 */
typedef enum {
    TK_ACTION_TYPE_SPEAK,                   /**< Generate speech output to user. */
    TK_ACTION_TYPE_NAVIGATE_GUIDE,          /**< Provide navigation guidance. */
    TK_ACTION_TYPE_NAVIGATE_WARN,           /**< Issue navigation warning/alert. */
    TK_ACTION_TYPE_DESCRIBE_ENVIRONMENT,    /**< Describe current environment. */
    TK_ACTION_TYPE_DESCRIBE_OBJECT,         /**< Describe specific detected object. */
    TK_ACTION_TYPE_READ_TEXT,               /**< Read detected text aloud. */
    TK_ACTION_TYPE_SYSTEM_MODE_CHANGE,      /**< Change system operational mode. */
    TK_ACTION_TYPE_SYSTEM_SETTING,          /**< Modify system settings. */
    TK_ACTION_TYPE_USER_QUERY_RESPONSE,     /**< Respond to user question/request. */
    TK_ACTION_TYPE_EMERGENCY_ALERT          /**< Issue emergency alert. */
} tk_action_type_e;

/**
 * @enum tk_action_status_e
 * @brief Status of action execution.
 */
typedef enum {
    TK_ACTION_STATUS_PENDING,               /**< Action is queued for execution. */
    TK_ACTION_STATUS_EXECUTING,             /**< Action is currently being executed. */
    TK_ACTION_STATUS_COMPLETED,             /**< Action completed successfully. */
    TK_ACTION_STATUS_FAILED,                /**< Action failed to execute. */
    TK_ACTION_STATUS_TIMEOUT,               /**< Action timed out. */
    TK_ACTION_STATUS_CANCELLED              /**< Action was cancelled. */
} tk_action_status_e;

/**
 * @enum tk_response_priority_e
 * @brief Priority levels for system responses.
 */
typedef enum {
    TK_RESPONSE_PRIORITY_EMERGENCY = 0,     /**< Immediate safety alerts. */
    TK_RESPONSE_PRIORITY_HIGH = 1,          /**< Navigation warnings, direct responses. */
    TK_RESPONSE_PRIORITY_NORMAL = 2,        /**< General conversation, descriptions. */
    TK_RESPONSE_PRIORITY_LOW = 3,           /**< Background information, confirmations. */
    TK_RESPONSE_PRIORITY_COUNT
} tk_response_priority_e;

/**
 * @struct tk_action_params_t
 * @brief Parameters for action execution (union-like structure).
 */
typedef struct {
    tk_action_type_e type;                  /**< The type of action to execute. */
    float confidence;                       /**< Confidence in the action decision. */
    uint32_t timeout_ms;                    /**< Timeout for this specific action. */
    
    union {
        struct {
            char* text;                     /**< Text to speak. */
            tk_response_priority_e priority; /**< Priority of the speech. */
            float volume_multiplier;        /**< Volume adjustment (1.0 = normal). */
        } speak;
        
        struct {
            float direction_deg;            /**< Direction to guide towards (-90 to +90). */
            float distance_m;               /**< Distance of clear path. */
            char* instruction;              /**< Verbal instruction to provide. */
        } navigate_guide;
        
        struct {
            tk_context_priority_e urgency;  /**< Urgency level of the warning. */
            char* warning_text;             /**< Warning message to speak. */
            float hazard_distance_m;        /**< Distance to the hazard. */
            float hazard_direction_deg;     /**< Direction of the hazard. */
        } navigate_warn;
        
        struct {
            bool include_objects;           /**< Whether to include object descriptions. */
            bool include_hazards;           /**< Whether to include hazard information. */
            bool include_navigation;        /**< Whether to include navigation options. */
            float detail_level;             /**< Level of detail (0.0 = brief, 1.0 = detailed). */
        } describe_environment;
        
        struct {
            uint32_t object_id;             /**< ID of the object to describe (from vision). */
            char* object_label;             /**< Label of the object. */
            float distance_m;               /**< Distance to the object. */
        } describe_object;
        
        struct {
            char* text_content;             /**< The text to read aloud. */
            float reading_speed;            /**< Speed multiplier for reading (1.0 = normal). */
        } read_text;
        
        struct {
            char* setting_name;             /**< Name of the setting to change. */
            char* setting_value;            /**< New value for the setting. */
        } system_setting;
        
        struct {
            char* response_text;            /**< Response to user query. */
            bool requires_context;          /**< Whether response needs current context. */
        } user_query_response;
        
        struct {
            char* alert_message;            /**< Emergency alert message. */
            bool repeat_alert;              /**< Whether to repeat the alert. */
            uint32_t repeat_interval_ms;    /**< Interval between repeats. */
        } emergency_alert;
    } params;
} tk_action_params_t;

/**
 * @struct tk_action_t
 * @brief Represents a single action in the execution queue.
 */
typedef struct {
    uint64_t action_id;                     /**< Unique identifier for this action. */
    tk_action_params_t params;              /**< Action parameters. */
    tk_action_status_e status;              /**< Current execution status. */
    uint64_t created_timestamp_ns;          /**< When the action was created. */
    uint64_t started_timestamp_ns;          /**< When execution started (0 if not started). */
    uint64_t completed_timestamp_ns;        /**< When execution completed (0 if not completed). */
    char* error_message;                    /**< Error message if execution failed. */
} tk_action_t;

/**
 * @struct tk_llm_response_t
 * @brief Represents a parsed response from the LLM.
 */
typedef struct {
    char* response_text;                    /**< The main response text for the user. */
    tk_response_priority_e priority;        /**< Priority of the response. */
    size_t action_count;                    /**< Number of actions requested. */
    tk_action_params_t* actions;            /**< Array of actions to execute. */
} tk_llm_response_t;

/**
 * @brief Callback function for action completion notification.
 * @param action The completed action with updated status.
 * @param user_data User data provided during engine creation.
 */
typedef void (*tk_on_action_completed_cb)(const tk_action_t* action, void* user_data);

/**
 * @brief Callback function for response delivery to the user.
 * @param response_text The text to be spoken to the user.
 * @param priority The priority level of the response.
 * @param user_data User data provided during engine creation.
 */
typedef void (*tk_on_response_ready_cb)(const char* response_text, tk_response_priority_e priority, void* user_data);

/**
 * @struct tk_decision_callbacks_t
 * @brief Callback functions for the decision engine.
 */
typedef struct {
    tk_on_action_completed_cb on_action_completed;
    tk_on_response_ready_cb on_response_ready;
    void* user_data;                        /**< User data passed to all callbacks. */
} tk_decision_callbacks_t;

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Engine Lifecycle Management
//------------------------------------------------------------------------------

/**
 * @brief Creates and initializes a new Decision Engine instance.
 *
 * @param[out] out_engine Pointer to receive the address of the new engine instance.
 * @param[in] config The configuration for the decision engine.
 * @param[in] callbacks Callback functions for action and response handling.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL or config is invalid.
 * @return TK_ERROR_OUT_OF_MEMORY on memory allocation failure.
 */
TK_NODISCARD tk_error_code_t tk_decision_engine_create(
    tk_decision_engine_t** out_engine,
    const tk_decision_config_t* config,
    tk_decision_callbacks_t callbacks
);

/**
 * @brief Destroys a Decision Engine instance.
 *
 * @param[in,out] engine Pointer to the engine instance to be destroyed.
 */
void tk_decision_engine_destroy(tk_decision_engine_t** engine);

//------------------------------------------------------------------------------
// LLM Response Processing
//------------------------------------------------------------------------------

/**
 * @brief Processes a raw text response from the LLM.
 *
 * This function parses the LLM's output text to identify requested actions,
 * response content, and execution parameters. It handles various response
 * formats including structured commands and natural language.
 *
 * @param[in] engine The decision engine instance.
 * @param[in] llm_output_text The raw text response from the LLM.
 * @param[in] context_summary Current context for action validation.
 * @param[out] out_parsed_response Pointer to receive the parsed response.
 *                                 Must be freed with tk_decision_engine_free_response.
 *
 * @return TK_SUCCESS on successful parsing.
 * @return TK_ERROR_INVALID_ARGUMENT if required pointers are NULL.
 * @return TK_ERROR_PARSE_FAILED if the LLM response cannot be parsed.
 * @return TK_ERROR_OUT_OF_MEMORY if response allocation fails.
 */
TK_NODISCARD tk_error_code_t tk_decision_engine_process_llm_response(
    tk_decision_engine_t* engine,
    const char* llm_output_text,
    const tk_context_summary_t* context_summary,
    tk_llm_response_t** out_parsed_response
);

/**
 * @brief Executes a parsed LLM response by queuing its actions.
 *
 * @param[in] engine The decision engine instance.
 * @param[in] parsed_response The parsed response containing actions to execute.
 *
 * @return TK_SUCCESS on successful action queuing.
 * @return TK_ERROR_INVALID_ARGUMENT if required pointers are NULL.
 * @return TK_ERROR_INVALID_STATE if the engine is not ready to accept actions.
 */
TK_NODISCARD tk_error_code_t tk_decision_engine_execute_response(
    tk_decision_engine_t* engine,
    const tk_llm_response_t* parsed_response
);

//------------------------------------------------------------------------------
// Action Management
//------------------------------------------------------------------------------

/**
 * @brief Manually queues a single action for execution.
 *
 * @param[in] engine The decision engine instance.
 * @param[in] action_params The parameters for the action to execute.
 * @param[out] out_action_id Pointer to receive the unique action ID.
 *
 * @return TK_SUCCESS on successful queuing.
 * @return TK_ERROR_INVALID_ARGUMENT if parameters are invalid.
 * @return TK_ERROR_QUEUE_FULL if the action queue is full.
 */
TK_NODISCARD tk_error_code_t tk_decision_engine_queue_action(
    tk_decision_engine_t* engine,
    const tk_action_params_t* action_params,
    uint64_t* out_action_id
);

/**
 * @brief Cancels a queued or executing action.
 *
 * @param[in] engine The decision engine instance.
 * @param[in] action_id The ID of the action to cancel.
 *
 * @return TK_SUCCESS if the action was cancelled.
 * @return TK_ERROR_NOT_FOUND if the action ID is not found.
 * @return TK_ERROR_INVALID_STATE if the action cannot be cancelled.
 */
TK_NODISCARD tk_error_code_t tk_decision_engine_cancel_action(
    tk_decision_engine_t* engine,
    uint64_t action_id
);

/**
 * @brief Processes the action queue and executes pending actions.
 *
 * This function should be called regularly from the main Cortex loop to
 * ensure actions are executed in a timely manner.
 *
 * @param[in] engine The decision engine instance.
 * @param[in] current_time_ns Current timestamp for timeout handling.
 *
 * @return TK_SUCCESS on successful processing.
 */
TK_NODISCARD tk_error_code_t tk_decision_engine_process_actions(
    tk_decision_engine_t* engine,
    uint64_t current_time_ns,
    const tk_context_summary_t* context,
    void* audio_ctx,
    void* nav_ctx,
    void* reasoner_ctx
);

//------------------------------------------------------------------------------
// State Queries
//------------------------------------------------------------------------------

/**
 * @brief Gets the current status of a specific action.
 *
 * @param[in] engine The decision engine instance.
 * @param[in] action_id The ID of the action to query.
 * @param[out] out_action Pointer to receive the action details.
 *
 * @return TK_SUCCESS if the action was found.
 * @return TK_ERROR_NOT_FOUND if the action ID is not found.
 */
TK_NODISCARD tk_error_code_t tk_decision_engine_get_action_status(
    tk_decision_engine_t* engine,
    uint64_t action_id,
    tk_action_t* out_action
);

/**
 * @brief Gets statistics about the current action queue.
 *
 * @param[in] engine The decision engine instance.
 * @param[out] out_pending_count Number of actions pending execution.
 * @param[out] out_executing_count Number of actions currently executing.
 * @param[out] out_completed_count Number of actions completed since last reset.
 * @param[out] out_failed_count Number of actions that failed execution.
 *
 * @return TK_SUCCESS on success.
 */
TK_NODISCARD tk_error_code_t tk_decision_engine_get_queue_stats(
    tk_decision_engine_t* engine,
    size_t* out_pending_count,
    size_t* out_executing_count,
    size_t* out_completed_count,
    size_t* out_failed_count
);

//------------------------------------------------------------------------------
// Emergency and Safety Controls
//------------------------------------------------------------------------------

/**
 * @brief Immediately cancels all actions and clears the queue.
 *
 * This is an emergency function that stops all current activities.
 * Should be used in safety-critical situations.
 *
 * @param[in] engine The decision engine instance.
 *
 * @return TK_SUCCESS on successful emergency stop.
 */
TK_NODISCARD tk_error_code_t tk_decision_engine_emergency_stop(
    tk_decision_engine_t* engine
);

/**
 * @brief Forces an immediate high-priority response to be spoken.
 *
 * This bypasses the normal action queue for emergency communications.
 *
 * @param[in] engine The decision engine instance.
 * @param[in] emergency_text The text to speak immediately.
 * @param[in] repeat_count Number of times to repeat (0 = once).
 *
 * @return TK_SUCCESS on successful emergency response.
 */
TK_NODISCARD tk_error_code_t tk_decision_engine_emergency_response(
    tk_decision_engine_t* engine,
    const char* emergency_text,
    uint32_t repeat_count
);

//------------------------------------------------------------------------------
// Memory Management
//------------------------------------------------------------------------------

/**
 * @brief Frees the memory allocated for a parsed LLM response.
 *
 * @param[in,out] response Pointer to the response structure to be freed.
 */
void tk_decision_engine_free_response(tk_llm_response_t** response);

/**
 * @brief Frees the memory allocated for action parameters.
 *
 * @param[in,out] params Pointer to the action parameters to be freed.
 */
void tk_decision_engine_free_action_params(tk_action_params_t* params);


//------------------------------------------------------------------------------
// High-Level Cortex Commands
//------------------------------------------------------------------------------

/**
 * @brief Runs a full cycle to describe the current environment.
 *
 * This function orchestrates a complete sequence:
 * 1. It calls into the Rust 'reasoning' module to generate a prompt based on the world model.
 * 2. It sends this prompt to the LLM.
 * 3. It takes the LLM's textual response and sends it to the TTS engine to be spoken.
 *
 * This is a high-level command that represents a primary function of the Cortex.
 *
 * @param engine The decision engine instance, used to access other subsystems.
 * @param cortex_context A pointer to the main tk_cortex_t object to access runners.
 * @return TK_SUCCESS on successful execution of the full cycle.
 */
TK_NODISCARD tk_error_code_t tk_decision_engine_describe_environment(
    tk_decision_engine_t* engine,
    void* llm_runner_ctx,    // Opaque pointer to tk_llm_runner_t
    void* audio_pipeline_ctx // Opaque pointer to tk_audio_pipeline_t
);


#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_CORTEX_TK_DECISION_ENGINE_H