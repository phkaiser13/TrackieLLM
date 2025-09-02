/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_contextual_reasoner.c
*
* This file contains the implementation of the Contextual Reasoning Engine.
* This component maintains and processes the dynamic context that feeds into
* the Large Language Model, acting as the system's memory and situational
* awareness center.
*
* The reasoner implements sophisticated algorithms for:
*   - Context relevance scoring and temporal decay
*   - Multi-modal data fusion and correlation
*   - Memory management and efficient context retrieval
*   - Structured summarization for LLM consumption
*
* SPDX-License-Identifier:
*/

#include "cortex/tk_contextual_reasoner.h"
#include "utils/tk_logging.h"
#include "utils/tk_error_handling.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

//------------------------------------------------------------------------------
// Internal Data Structures
//------------------------------------------------------------------------------

/**
 * @struct context_memory_t
 * @brief Internal structure for managing context items in memory.
 */
typedef struct context_memory_s {
    tk_context_item_t* items;           /**< Array of context items. */
    size_t item_count;                  /**< Current number of items. */
    size_t item_capacity;               /**< Maximum number of items. */
    size_t next_item_index;             /**< Circular buffer index for new items. */
    pthread_mutex_t mutex;              /**< Thread safety for context access. */
} context_memory_t;

/**
 * @struct conversation_memory_t
 * @brief Internal structure for managing conversation history.
 */
typedef struct conversation_memory_s {
    tk_conversation_turn_t* turns;      /**< Array of conversation turns. */
    size_t turn_count;                  /**< Current number of turns. */
    size_t turn_capacity;               /**< Maximum number of turns. */
    size_t next_turn_index;             /**< Circular buffer index for new turns. */
    pthread_mutex_t mutex;              /**< Thread safety for conversation access. */
} conversation_memory_t;

/**
 * @struct tk_contextual_reasoner_s
 * @brief The main contextual reasoner instance.
 */
struct tk_contextual_reasoner_s {
    tk_context_config_t config;         /**< Configuration parameters. */
    
    context_memory_t context_memory;    /**< General context items storage. */
    conversation_memory_t conversation_memory; /**< Conversation history storage. */
    
    // Current environmental state
    struct {
        tk_vision_object_t* visible_objects;
        size_t visible_object_count;
        size_t visible_object_capacity;
        uint64_t last_vision_update_ns;
    } environmental_state;
    
    // Current navigation state
    struct {
        bool has_clear_path;
        float clear_path_direction_deg;
        float clear_path_distance_m;
        tk_navigation_hazard_t* hazards;
        size_t hazard_count;
        size_t hazard_capacity;
        uint64_t last_navigation_update_ns;
    } navigation_state;
    
    // System state
    struct {
        uint64_t last_context_process_ns;
        uint64_t total_memory_bytes;
        bool is_listening_for_commands;
        float system_confidence;
    } system_state;
    
    pthread_mutex_t state_mutex;        /**< Mutex for state updates. */
};

//------------------------------------------------------------------------------
// Internal Function Declarations
//------------------------------------------------------------------------------

static tk_error_code_t init_context_memory(context_memory_t* memory, size_t capacity);
static void cleanup_context_memory(context_memory_t* memory);
static tk_error_code_t init_conversation_memory(conversation_memory_t* memory, size_t capacity);
static void cleanup_conversation_memory(conversation_memory_t* memory);

static tk_error_code_t add_context_item_internal(tk_contextual_reasoner_t* reasoner, const tk_context_item_t* item);
static void update_context_relevance_scores(tk_contextual_reasoner_t* reasoner, uint64_t current_time_ns);
static void prune_irrelevant_context(tk_contextual_reasoner_t* reasoner);
static float calculate_relevance_score(const tk_context_item_t* item, uint64_t current_time_ns, float decay_rate);

static char* generate_environmental_description(tk_contextual_reasoner_t* reasoner);
static char* generate_navigation_description(tk_contextual_reasoner_t* reasoner);
static char* generate_conversation_summary(tk_contextual_reasoner_t* reasoner, size_t max_turns);

static uint64_t get_current_time_ns(void);

//------------------------------------------------------------------------------
// Public API Implementation
//------------------------------------------------------------------------------

tk_error_code_t tk_contextual_reasoner_create(tk_contextual_reasoner_t** out_reasoner, const tk_context_config_t* config) {
    if (!out_reasoner || !config) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    tk_log_info("Creating Contextual Reasoning Engine");
    
    tk_contextual_reasoner_t* reasoner = calloc(1, sizeof(tk_contextual_reasoner_t));
    if (!reasoner) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Copy configuration
    reasoner->config = *config;
    
    // Initialize memory structures
    tk_error_code_t result = init_context_memory(&reasoner->context_memory, config->max_context_history_items);
    if (result != TK_SUCCESS) {
        free(reasoner);
        return result;
    }
    
    result = init_conversation_memory(&reasoner->conversation_memory, config->max_conversation_history_turns);
    if (result != TK_SUCCESS) {
        cleanup_context_memory(&reasoner->context_memory);
        free(reasoner);
        return result;
    }
    
    // Initialize environmental state
    reasoner->environmental_state.visible_object_capacity = 50;
    reasoner->environmental_state.visible_objects = calloc(reasoner->environmental_state.visible_object_capacity, sizeof(tk_vision_object_t));
    if (!reasoner->environmental_state.visible_objects) {
        cleanup_conversation_memory(&reasoner->conversation_memory);
        cleanup_context_memory(&reasoner->context_memory);
        free(reasoner);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Initialize navigation state
    reasoner->navigation_state.hazard_capacity = 20;
    reasoner->navigation_state.hazards = calloc(reasoner->navigation_state.hazard_capacity, sizeof(tk_navigation_hazard_t));
    if (!reasoner->navigation_state.hazards) {
        free(reasoner->environmental_state.visible_objects);
        cleanup_conversation_memory(&reasoner->conversation_memory);
        cleanup_context_memory(&reasoner->context_memory);
        free(reasoner);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Initialize mutexes
    if (pthread_mutex_init(&reasoner->state_mutex, NULL) != 0) {
        free(reasoner->navigation_state.hazards);
        free(reasoner->environmental_state.visible_objects);
        cleanup_conversation_memory(&reasoner->conversation_memory);
        cleanup_context_memory(&reasoner->context_memory);
        free(reasoner);
        return TK_ERROR_SYSTEM_ERROR;
    }
    
    // Initialize system state
    reasoner->system_state.system_confidence = 0.8f;
    reasoner->system_state.is_listening_for_commands = false;
    
    *out_reasoner = reasoner;
    
    tk_log_info("Contextual Reasoning Engine created successfully");
    return TK_SUCCESS;
}

void tk_contextual_reasoner_destroy(tk_contextual_reasoner_t** reasoner) {
    if (!reasoner || !*reasoner) {
        return;
    }
    
    tk_log_info("Destroying Contextual Reasoning Engine");
    
    tk_contextual_reasoner_t* r = *reasoner;
    
    // Cleanup memory structures
    cleanup_conversation_memory(&r->conversation_memory);
    cleanup_context_memory(&r->context_memory);
    
    // Free state arrays
    free(r->environmental_state.visible_objects);
    free(r->navigation_state.hazards);
    
    // Destroy mutex
    pthread_mutex_destroy(&r->state_mutex);
    
    free(r);
    *reasoner = NULL;
    
    tk_log_info("Contextual Reasoning Engine destroyed");
}

tk_error_code_t tk_contextual_reasoner_update_vision_context(tk_contextual_reasoner_t* reasoner, const tk_vision_result_t* vision_result) {
    if (!reasoner || !vision_result) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    pthread_mutex_lock(&reasoner->state_mutex);
    
    // Update visible objects
    size_t objects_to_copy = (vision_result->object_count < reasoner->environmental_state.visible_object_capacity) 
                           ? vision_result->object_count 
                           : reasoner->environmental_state.visible_object_capacity;
    
    if (objects_to_copy > 0) {
        memcpy(reasoner->environmental_state.visible_objects, 
               vision_result->objects, 
               objects_to_copy * sizeof(tk_vision_object_t));
    }
    
    reasoner->environmental_state.visible_object_count = objects_to_copy;
    reasoner->environmental_state.last_vision_update_ns = get_current_time_ns();
    
    pthread_mutex_unlock(&reasoner->state_mutex);
    
    // Create context items for significant detections
    for (size_t i = 0; i < objects_to_copy; i++) {
        const tk_vision_object_t* obj = &vision_result->objects[i];
        
        // Only create context for high-confidence detections
        if (obj->confidence > 0.7f) {
            char description[256];
            snprintf(description, sizeof(description), 
                "Detected %s at %.1fm distance with %.0f%% confidence",
                obj->label, obj->distance_meters, obj->confidence * 100.0f);
            
            tk_context_item_t context_item = {
                .timestamp_ns = get_current_time_ns(),
                .type = TK_CONTEXT_TYPE_ENVIRONMENTAL,
                .priority = (obj->distance_meters < 2.0f) ? TK_CONTEXT_PRIORITY_HIGH : TK_CONTEXT_PRIORITY_MEDIUM,
                .relevance_score = obj->confidence,
                .description = NULL, // Will be allocated in add_context_item_internal
                .data = NULL,
                .data_size = 0
            };
            
            // Allocate and copy description
            context_item.description = strdup(description);
            if (context_item.description) {
                add_context_item_internal(reasoner, &context_item);
            }
        }
    }
    
    return TK_SUCCESS;
}

tk_error_code_t tk_contextual_reasoner_update_navigation_context(
    tk_contextual_reasoner_t* reasoner,
    const tk_traversability_map_t* traversability_map,
    const tk_free_space_analysis_t* free_space_analysis,
    const tk_obstacle_t* obstacles,
    size_t obstacle_count) {
    
    if (!reasoner || !traversability_map || !free_space_analysis) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    pthread_mutex_lock(&reasoner->state_mutex);
    
    // Update navigation state
    reasoner->navigation_state.has_clear_path = free_space_analysis->is_any_path_clear;
    reasoner->navigation_state.clear_path_direction_deg = free_space_analysis->clearest_path_angle_deg;
    reasoner->navigation_state.clear_path_distance_m = free_space_analysis->clearest_path_distance_m;
    reasoner->navigation_state.last_navigation_update_ns = get_current_time_ns();
    
    // Reset hazard count (will be updated below)
    reasoner->navigation_state.hazard_count = 0;
    
    pthread_mutex_unlock(&reasoner->state_mutex);
    
    // Create context items for navigation situation
    char nav_description[512];
    if (free_space_analysis->is_any_path_clear) {
        snprintf(nav_description, sizeof(nav_description),
            "Clear path available at %.0f degrees, distance %.1fm",
            free_space_analysis->clearest_path_angle_deg,
            free_space_analysis->clearest_path_distance_m);
        
        tk_contextual_reasoner_add_context_item(reasoner,
            TK_CONTEXT_TYPE_NAVIGATIONAL,
            TK_CONTEXT_PRIORITY_HIGH,
            nav_description,
            NULL, 0);
    } else {
        tk_contextual_reasoner_add_context_item(reasoner,
            TK_CONTEXT_TYPE_NAVIGATIONAL,
            TK_CONTEXT_PRIORITY_CRITICAL,
            "No clear navigation path detected",
            NULL, 0);
    }
    
    // Process obstacles
    if (obstacles && obstacle_count > 0) {
        for (size_t i = 0; i < obstacle_count && i < 5; i++) { // Limit to 5 most relevant obstacles
            const tk_obstacle_t* obstacle = &obstacles[i];
            
            char obstacle_desc[256];
            snprintf(obstacle_desc, sizeof(obstacle_desc),
                "Obstacle detected at (%.1f, %.1f)m, size %.1fx%.1fm",
                obstacle->position_m.x, obstacle->position_m.y,
                obstacle->dimensions_m.x, obstacle->dimensions_m.y);
            
            tk_context_priority_e priority = TK_CONTEXT_PRIORITY_MEDIUM;
            
            // Higher priority for close obstacles
            float distance = sqrtf(obstacle->position_m.x * obstacle->position_m.x + 
                                 obstacle->position_m.y * obstacle->position_m.y);
            if (distance < 1.5f) {
                priority = TK_CONTEXT_PRIORITY_HIGH;
            }
            
            tk_contextual_reasoner_add_context_item(reasoner,
                TK_CONTEXT_TYPE_NAVIGATIONAL,
                priority,
                obstacle_desc,
                NULL, 0);
        }
    }
    
    return TK_SUCCESS;
}

tk_error_code_t tk_contextual_reasoner_add_conversation_turn(
    tk_contextual_reasoner_t* reasoner,
    bool is_user_input,
    const char* content,
    float confidence) {
    
    if (!reasoner || !content) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    pthread_mutex_lock(&reasoner->conversation_memory.mutex);
    
    conversation_memory_t* mem = &reasoner->conversation_memory;
    
    // Get next slot (circular buffer)
    size_t index = mem->next_turn_index;
    tk_conversation_turn_t* turn = &mem->turns[index];
    
    // Free existing content if overwriting
    if (turn->content) {
        free(turn->content);
        turn->content = NULL;
    }
    
    // Set up new turn
    turn->timestamp_ns = get_current_time_ns();
    turn->is_user_input = is_user_input;
    turn->content = strdup(content);