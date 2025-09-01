/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_contextual_reasoner.h
*
* This header file defines the public API for the Contextual Reasoning Engine,
* a critical component of the TrackieLLM Cortex. This module is responsible for
* maintaining and processing the dynamic context that feeds into the Large
* Language Model (LLM).
*
* The Contextual Reasoner acts as a sophisticated memory and state management
* system. It aggregates information from all sensory inputs (vision, audio,
* navigation) and maintains a structured, temporal understanding of the user's
* environment and interaction history.
*
* Key responsibilities include:
*   1. Context Aggregation: Combining multi-modal inputs into coherent context.
*   2. Memory Management: Maintaining short-term and working memory of recent
*      events, conversations, and environmental changes.
*   3. Relevance Filtering: Determining which context is most relevant for
*      the current situation to optimize LLM token usage.
*   4. Temporal Reasoning: Understanding the sequence and timing of events.
*
* The reasoner maintains multiple context streams:
*   - Environmental Context (objects, hazards, navigation state)
*   - Conversational Context (recent dialogue, user intent)
*   - Temporal Context (sequence of events, state transitions)
*   - User Context (preferences, accessibility needs, current activity)
*
* SPDX-License-Identifier:
*/

#ifndef TRACKIELLM_CORTEX_TK_CONTEXTUAL_REASONER_H
#define TRACKIELLM_CORTEX_TK_CONTEXTUAL_REASONER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "vision/tk_vision_pipeline.h"      // For vision results
#include "navigation/tk_path_planner.h"     // For navigation data
#include "navigation/tk_free_space_detector.h" // For space analysis
#include "navigation/tk_obstacle_avoider.h" // For obstacle tracking

// Forward-declare the primary reasoner object as an opaque type.
typedef struct tk_contextual_reasoner_s tk_contextual_reasoner_t;

/**
 * @struct tk_context_config_t
 * @brief Configuration for initializing the Contextual Reasoning Engine.
 */
typedef struct {
    size_t max_context_history_items;      /**< Maximum number of context items to maintain in memory. */
    size_t max_conversation_history_turns; /**< Maximum number of conversation turns to remember. */
    float  context_relevance_threshold;    /**< Minimum relevance score for context inclusion (0.0-1.0). */
    float  memory_decay_rate;             /**< Rate at which old context items lose relevance over time. */
    uint32_t context_update_interval_ms;   /**< Minimum time between context updates (performance control). */
} tk_context_config_t;

/**
 * @enum tk_context_priority_e
 * @brief Priority levels for different types of context information.
 */
typedef enum {
    TK_CONTEXT_PRIORITY_CRITICAL = 0,   /**< Immediate safety hazards, urgent user requests. */
    TK_CONTEXT_PRIORITY_HIGH = 1,       /**< Navigation guidance, direct user interaction. */
    TK_CONTEXT_PRIORITY_MEDIUM = 2,     /**< Environmental awareness, object descriptions. */
    TK_CONTEXT_PRIORITY_LOW = 3,        /**< Background information, historical data. */
    TK_CONTEXT_PRIORITY_COUNT
} tk_context_priority_e;

/**
 * @enum tk_context_type_e
 * @brief Types of contextual information maintained by the reasoner.
 */
typedef enum {
    TK_CONTEXT_TYPE_ENVIRONMENTAL,  /**< Current environment state (objects, hazards). */
    TK_CONTEXT_TYPE_NAVIGATIONAL,   /**< Path planning and movement guidance. */
    TK_CONTEXT_TYPE_CONVERSATIONAL, /**< Dialogue history and user intent. */
    TK_CONTEXT_TYPE_TEMPORAL,       /**< Time-sequenced events and state changes. */
    TK_CONTEXT_TYPE_USER_STATE,     /**< User preferences, activity, and context. */
    TK_CONTEXT_TYPE_SYSTEM_STATE    /**< Internal system status and performance. */
} tk_context_type_e;

/**
 * @struct tk_context_item_t
 * @brief Represents a single piece of contextual information.
 */
typedef struct {
    uint64_t            timestamp_ns;   /**< When this context was created/updated. */
    tk_context_type_e   type;           /**< The type of context this represents. */
    tk_context_priority_e priority;     /**< The priority level of this context. */
    float               relevance_score;/**< Current relevance score (0.0-1.0). */
    char*               description;    /**< Human-readable description of the context. */
    void*               data;           /**< Type-specific context data. */
    size_t              data_size;      /**< Size of the context data. */
} tk_context_item_t;

/**
 * @struct tk_conversation_turn_t
 * @brief Represents a single turn in the conversation history.
 */
typedef struct {
    uint64_t    timestamp_ns;       /**< When this turn occurred. */
    bool        is_user_input;      /**< True if this was user input, false if system response. */
    char*       content;            /**< The actual text content of the turn. */
    float       confidence;         /**< Confidence in transcription/generation (if applicable). */
} tk_conversation_turn_t;

/**
 * @struct tk_context_summary_t
 * @brief A structured summary of the current context for LLM consumption.
 */
typedef struct {
    // Environmental awareness
    size_t              visible_object_count;
    const tk_vision_object_t* visible_objects;
    
    // Navigation state
    bool                has_clear_path;
    float               clear_path_direction_deg;
    float               clear_path_distance_m;
    size_t              hazard_count;
    const tk_navigation_hazard_t* hazards;
    
    // Recent conversation
    size_t              conversation_turn_count;
    const tk_conversation_turn_t* recent_conversation;
    
    // Temporal context
    char*               recent_events_summary;  /**< Brief summary of recent events. */
    
    // System state
    bool                is_navigation_active;
    bool                is_listening_for_commands;
    float               system_confidence;      /**< Overall system confidence in current state. */
} tk_context_summary_t;

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Reasoner Lifecycle Management
//------------------------------------------------------------------------------

/**
 * @brief Creates and initializes a new Contextual Reasoning Engine instance.
 *
 * @param[out] out_reasoner Pointer to receive the address of the new reasoner instance.
 * @param[in] config The configuration for the reasoning engine.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL or config is invalid.
 * @return TK_ERROR_OUT_OF_MEMORY on memory allocation failure.
 */
TK_NODISCARD tk_error_code_t tk_contextual_reasoner_create(
    tk_contextual_reasoner_t** out_reasoner,
    const tk_context_config_t* config
);

/**
 * @brief Destroys a Contextual Reasoning Engine instance.
 *
 * @param[in,out] reasoner Pointer to the reasoner instance to be destroyed.
 */
void tk_contextual_reasoner_destroy(tk_contextual_reasoner_t** reasoner);

//------------------------------------------------------------------------------
// Context Input and Updates
//------------------------------------------------------------------------------

/**
 * @brief Updates the environmental context with new vision analysis results.
 *
 * @param[in] reasoner The contextual reasoner instance.
 * @param[in] vision_result The latest vision analysis results.
 *
 * @return TK_SUCCESS on successful update.
 * @return TK_ERROR_INVALID_ARGUMENT if required pointers are NULL.
 */
TK_NODISCARD tk_error_code_t tk_contextual_reasoner_update_vision_context(
    tk_contextual_reasoner_t* reasoner,
    const tk_vision_result_t* vision_result
);

/**
 * @brief Updates the navigational context with new path planning data.
 *
 * @param[in] reasoner The contextual reasoner instance.
 * @param[in] traversability_map The current traversability analysis.
 * @param[in] free_space_analysis The current free space analysis.
 * @param[in] obstacles The current obstacle tracking data.
 * @param[in] obstacle_count Number of tracked obstacles.
 *
 * @return TK_SUCCESS on successful update.
 */
TK_NODISCARD tk_error_code_t tk_contextual_reasoner_update_navigation_context(
    tk_contextual_reasoner_t* reasoner,
    const tk_traversability_map_t* traversability_map,
    const tk_free_space_analysis_t* free_space_analysis,
    const tk_obstacle_t* obstacles,
    size_t obstacle_count
);

/**
 * @brief Adds a new conversational turn to the context history.
 *
 * @param[in] reasoner The contextual reasoner instance.
 * @param[in] is_user_input True if this is user input, false if system response.
 * @param[in] content The text content of the conversation turn.
 * @param[in] confidence Confidence score for the transcription/generation.
 *
 * @return TK_SUCCESS on successful addition.
 * @return TK_ERROR_INVALID_ARGUMENT if required pointers are NULL.
 * @return TK_ERROR_OUT_OF_MEMORY if unable to allocate space for the turn.
 */
TK_NODISCARD tk_error_code_t tk_contextual_reasoner_add_conversation_turn(
    tk_contextual_reasoner_t* reasoner,
    bool is_user_input,
    const char* content,
    float confidence
);

/**
 * @brief Adds a custom context item to the reasoner's memory.
 *
 * @param[in] reasoner The contextual reasoner instance.
 * @param[in] type The type of context being added.
 * @param[in] priority The priority level of this context.
 * @param[in] description Human-readable description.
 * @param[in] data Optional type-specific data (can be NULL).
 * @param[in] data_size Size of the data (0 if data is NULL).
 *
 * @return TK_SUCCESS on successful addition.
 * @return TK_ERROR_OUT_OF_MEMORY if unable to allocate space for the context.
 */
TK_NODISCARD tk_error_code_t tk_contextual_reasoner_add_context_item(
    tk_contextual_reasoner_t* reasoner,
    tk_context_type_e type,
    tk_context_priority_e priority,
    const char* description,
    const void* data,
    size_t data_size
);

//------------------------------------------------------------------------------
// Context Processing and Queries
//------------------------------------------------------------------------------

/**
 * @brief Performs a full context update and relevance recalculation.
 *
 * This function should be called periodically to update relevance scores,
 * apply memory decay, and prune irrelevant context items.
 *
 * @param[in] reasoner The contextual reasoner instance.
 * @param[in] current_time_ns Current timestamp for relevance calculations.
 *
 * @return TK_SUCCESS on successful processing.
 */
TK_NODISCARD tk_error_code_t tk_contextual_reasoner_process_context(
    tk_contextual_reasoner_t* reasoner,
    uint64_t current_time_ns
);

/**
 * @brief Generates a structured context summary for LLM consumption.
 *
 * This function aggregates and prioritizes the current context to create
 * a coherent summary that can be efficiently processed by the LLM.
 *
 * @param[in] reasoner The contextual reasoner instance.
 * @param[out] out_summary Pointer to a summary structure that will be filled.
 *                         The data is owned by the reasoner and valid until
 *                         the next context update.
 *
 * @return TK_SUCCESS on successful summary generation.
 * @return TK_ERROR_INVALID_STATE if no context is available.
 */
TK_NODISCARD tk_error_code_t tk_contextual_reasoner_get_context_summary(
    tk_contextual_reasoner_t* reasoner,
    tk_context_summary_t* out_summary
);

/**
 * @brief Generates a formatted context string for direct LLM input.
 *
 * @param[in] reasoner The contextual reasoner instance.
 * @param[out] out_context_string Pointer to receive a dynamically allocated
 *                                context string. Must be freed by the caller.
 * @param[in] max_token_budget Maximum number of tokens to use for context
 *                             (approximate, used for length control).
 *
 * @return TK_SUCCESS on successful string generation.
 * @return TK_ERROR_OUT_OF_MEMORY if string allocation fails.
 */
TK_NODISCARD tk_error_code_t tk_contextual_reasoner_generate_context_string(
    tk_contextual_reasoner_t* reasoner,
    char** out_context_string,
    size_t max_token_budget
);

//------------------------------------------------------------------------------
// Context Memory Management
//------------------------------------------------------------------------------

/**
 * @brief Clears all context history and resets the reasoner state.
 *
 * @param[in] reasoner The contextual reasoner instance.
 *
 * @return TK_SUCCESS on successful reset.
 */
TK_NODISCARD tk_error_code_t tk_contextual_reasoner_clear_context(
    tk_contextual_reasoner_t* reasoner
);

/**
 * @brief Gets the current memory usage statistics of the reasoner.
 *
 * @param[in] reasoner The contextual reasoner instance.
 * @param[out] out_total_items Total number of context items in memory.
 * @param[out] out_total_memory_bytes Total memory used by context data.
 * @param[out] out_conversation_turns Number of conversation turns in memory.
 *
 * @return TK_SUCCESS on success.
 */
TK_NODISCARD tk_error_code_t tk_contextual_reasoner_get_memory_stats(
    tk_contextual_reasoner_t* reasoner,
    size_t* out_total_items,
    size_t* out_total_memory_bytes,
    size_t* out_conversation_turns
);

#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_CORTEX_TK_CONTEXTUAL_REASONER_H