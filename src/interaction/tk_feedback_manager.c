/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_feedback_manager.c
 *
 * This source file implements the User Feedback Manager module. It provides
 * functionality to prioritize and manage feedback messages to ensure critical
 * safety alerts are delivered promptly while preventing auditory spam.
 *
 * The implementation uses a priority queue system with configurable verbosity
 * levels and message suppression to optimize user experience.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "interaction/tk_feedback_manager.h"
#include "utils/tk_logging.h"
#include "utils/tk_error_handling.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

// Maximum number of messages in the queue
#define MAX_MESSAGE_QUEUE_SIZE 32

// Internal structures for message queue
typedef struct {
    char*                    text;
    tk_feedback_priority_e   priority;
    tk_feedback_verbosity_e  verbosity_level;
    uint32_t                 suppression_key;
    uint32_t                 suppression_cooldown_ms;
    bool                     is_interrupt;
    float                    time_until_cooldown_reset; // in seconds
    uint64_t                 tts_request_id;
} tk_internal_message_t;

// Suppression key tracking structure
typedef struct {
    uint32_t key;
    float    cooldown_time_remaining; // in seconds
} tk_suppression_entry_t;

struct tk_feedback_manager_s {
    tk_tts_request_func_t tts_request_callback;
    void*                 user_data;
    
    tk_feedback_verbosity_e current_verbosity;
    
    // Message queue
    tk_internal_message_t message_queue[MAX_MESSAGE_QUEUE_SIZE];
    size_t                queue_size;
    size_t                queue_head;
    size_t                queue_tail;
    
    // Currently playing message
    bool                  is_tts_busy;
    uint64_t              current_tts_request_id;
    
    // Suppression tracking
    tk_suppression_entry_t suppression_table[MAX_MESSAGE_QUEUE_SIZE];
    size_t                 suppression_count;
};

//------------------------------------------------------------------------------
// Private helper functions
//------------------------------------------------------------------------------

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
 * @brief Initializes an internal message from a request
 */
static tk_error_code_t init_internal_message(
    tk_internal_message_t* msg,
    const tk_feedback_request_t* request
) {
    if (!msg || !request) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    msg->text = duplicate_string(request->text);
    if (!msg->text && request->text) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    msg->priority = request->priority;
    msg->verbosity_level = request->verbosity_level;
    msg->suppression_key = request->suppression_key;
    msg->suppression_cooldown_ms = request->suppression_cooldown_ms;
    msg->is_interrupt = request->is_interrupt;
    msg->time_until_cooldown_reset = request->suppression_cooldown_ms / 1000.0f; // Convert to seconds
    msg->tts_request_id = 0;
    
    return TK_SUCCESS;
}

/**
 * @brief Frees resources associated with an internal message
 */
static void free_internal_message(tk_internal_message_t* msg) {
    if (!msg) return;
    
    if (msg->text) {
        free(msg->text);
        msg->text = NULL;
    }
}

/**
 * @brief Checks if a suppression key is currently active
 */
static bool is_suppression_key_active(
    tk_feedback_manager_t* manager,
    uint32_t key
) {
    if (!manager || key == 0) return false;
    
    for (size_t i = 0; i < manager->suppression_count; i++) {
        if (manager->suppression_table[i].key == key &&
            manager->suppression_table[i].cooldown_time_remaining > 0.0f) {
            return true;
        }
    }
    
    return false;
}

/**
 * @brief Adds a suppression key to the tracking table
 */
static void add_suppression_key(
    tk_feedback_manager_t* manager,
    uint32_t key,
    float cooldown_time_seconds
) {
    if (!manager || key == 0 || cooldown_time_seconds <= 0.0f) return;
    
    // Check if key already exists
    for (size_t i = 0; i < manager->suppression_count; i++) {
        if (manager->suppression_table[i].key == key) {
            manager->suppression_table[i].cooldown_time_remaining = cooldown_time_seconds;
            return;
        }
    }
    
    // Add new key if there's space
    if (manager->suppression_count < MAX_MESSAGE_QUEUE_SIZE) {
        manager->suppression_table[manager->suppression_count].key = key;
        manager->suppression_table[manager->suppression_count].cooldown_time_remaining = cooldown_time_seconds;
        manager->suppression_count++;
    }
}

/**
 * @brief Updates suppression cooldowns
 */
static void update_suppression_cooldowns(
    tk_feedback_manager_t* manager,
    float delta_time_s
) {
    if (!manager) return;
    
    for (size_t i = 0; i < manager->suppression_count; ) {
        if (manager->suppression_table[i].cooldown_time_remaining > 0.0f) {
            manager->suppression_table[i].cooldown_time_remaining -= delta_time_s;
            
            // Remove expired entries
            if (manager->suppression_table[i].cooldown_time_remaining <= 0.0f) {
                // Move last entry to current position
                manager->suppression_table[i] = manager->suppression_table[manager->suppression_count - 1];
                manager->suppression_count--;
            } else {
                i++; // Only increment if we didn't remove an entry
            }
        } else {
            i++;
        }
    }
}

/**
 * @brief Compares two messages for priority sorting
 */
static bool should_prioritize_message(
    const tk_internal_message_t* msg1,
    const tk_internal_message_t* msg2
) {
    if (!msg1 || !msg2) return false;
    
    // Higher priority messages come first
    if (msg1->priority > msg2->priority) return true;
    if (msg1->priority < msg2->priority) return false;
    
    // For same priority, earlier messages come first (FIFO)
    return false; // We don't track submission time in this simplified version
}

/**
 * @brief Finds the highest priority message that meets verbosity requirements
 */
static tk_internal_message_t* find_highest_priority_message(
    tk_feedback_manager_t* manager
) {
    if (!manager || manager->queue_size == 0) return NULL;
    
    tk_internal_message_t* best_msg = NULL;
    
    // Search through the queue
    size_t index = manager->queue_head;
    for (size_t i = 0; i < manager->queue_size; i++) {
        tk_internal_message_t* msg = &manager->message_queue[index];
        
        // Check if message meets verbosity requirements
        if (msg->verbosity_level <= manager->current_verbosity) {
            // Check if this message should be prioritized
            if (!best_msg || should_prioritize_message(msg, best_msg)) {
                best_msg = msg;
            }
        }
        
        index = (index + 1) % MAX_MESSAGE_QUEUE_SIZE;
    }
    
    return best_msg;
}

/**
 * @brief Removes a message from the queue
 */
static void remove_message_from_queue(
    tk_feedback_manager_t* manager,
    tk_internal_message_t* msg_to_remove
) {
    if (!manager || !msg_to_remove || manager->queue_size == 0) return;
    
    size_t index = manager->queue_head;
    for (size_t i = 0; i < manager->queue_size; i++) {
        if (&manager->message_queue[index] == msg_to_remove) {
            // Free the message
            free_internal_message(msg_to_remove);
            
            // Shift remaining messages
            size_t move_index = index;
            while (move_index != manager->queue_tail) {
                size_t next_index = (move_index + 1) % MAX_MESSAGE_QUEUE_SIZE;
                manager->message_queue[move_index] = manager->message_queue[next_index];
                move_index = next_index;
            }
            
            // Update queue tail
            manager->queue_tail = (manager->queue_tail - 1 + MAX_MESSAGE_QUEUE_SIZE) % MAX_MESSAGE_QUEUE_SIZE;
            manager->queue_size--;
            return;
        }
        
        index = (index + 1) % MAX_MESSAGE_QUEUE_SIZE;
    }
}

/**
 * @brief Clears all messages from the queue
 */
static void clear_message_queue(tk_feedback_manager_t* manager) {
    if (!manager) return;
    
    size_t index = manager->queue_head;
    for (size_t i = 0; i < manager->queue_size; i++) {
        free_internal_message(&manager->message_queue[index]);
        index = (index + 1) % MAX_MESSAGE_QUEUE_SIZE;
    }
    
    manager->queue_size = 0;
    manager->queue_head = 0;
    manager->queue_tail = 0;
}

//------------------------------------------------------------------------------
// Public API Implementation
//------------------------------------------------------------------------------

tk_error_code_t tk_feedback_manager_create(
    tk_feedback_manager_t** out_manager,
    const tk_feedback_manager_config_t* config
) {
    if (!out_manager || !config || !config->tts_request_callback) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_manager = NULL;
    
    // Allocate manager structure
    tk_feedback_manager_t* manager = calloc(1, sizeof(tk_feedback_manager_t));
    if (!manager) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Initialize fields
    manager->tts_request_callback = config->tts_request_callback;
    manager->user_data = config->user_data;
    manager->current_verbosity = TK_VERBOSITY_LEVEL_NORMAL;
    manager->queue_size = 0;
    manager->queue_head = 0;
    manager->queue_tail = 0;
    manager->is_tts_busy = false;
    manager->current_tts_request_id = 0;
    manager->suppression_count = 0;
    
    *out_manager = manager;
    return TK_SUCCESS;
}

void tk_feedback_manager_destroy(tk_feedback_manager_t** manager) {
    if (!manager || !*manager) return;
    
    tk_feedback_manager_t* m = *manager;
    
    // Clear any remaining messages
    clear_message_queue(m);
    
    // Free manager itself
    free(m);
    *manager = NULL;
}

tk_error_code_t tk_feedback_manager_enqueue(
    tk_feedback_manager_t* manager,
    const tk_feedback_request_t* request
) {
    if (!manager || !request) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Check if suppression key is active
    if (request->suppression_key != 0 && 
        is_suppression_key_active(manager, request->suppression_key)) {
        // Message suppressed, silently drop it
        return TK_SUCCESS;
    }
    
    // Check if queue is full
    if (manager->queue_size >= MAX_MESSAGE_QUEUE_SIZE) {
        TK_LOG_WARN("Feedback manager queue is full, dropping message");
        return TK_SUCCESS; // Not an error, just drop the message
    }
    
    // Add suppression key to tracking table
    if (request->suppression_key != 0 && request->suppression_cooldown_ms > 0) {
        add_suppression_key(
            manager,
            request->suppression_key,
            request->suppression_cooldown_ms / 1000.0f
        );
    }
    
    // Create internal message
    tk_internal_message_t* msg = &manager->message_queue[manager->queue_tail];
    tk_error_code_t result = init_internal_message(msg, request);
    
    if (result != TK_SUCCESS) {
        return result;
    }
    
    // Update queue pointers
    manager->queue_tail = (manager->queue_tail + 1) % MAX_MESSAGE_QUEUE_SIZE;
    manager->queue_size++;
    
    // Handle interrupt messages
    if (request->is_interrupt) {
        // In a real implementation, we might want to clear lower priority messages
        // For now, we'll just let the update function handle prioritization
    }
    
    return TK_SUCCESS;
}

tk_error_code_t tk_feedback_manager_update(
    tk_feedback_manager_t* manager,
    float delta_time_s
) {
    if (!manager) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Update suppression cooldowns
    update_suppression_cooldowns(manager, delta_time_s);
    
    // If TTS is busy, we can't do anything
    if (manager->is_tts_busy) {
        return TK_SUCCESS;
    }
    
    // Find the highest priority message that meets verbosity requirements
    tk_internal_message_t* best_msg = find_highest_priority_message(manager);
    
    // If we found a message to play
    if (best_msg) {
        // Request TTS synthesis
        if (best_msg->text) {
            uint64_t request_id = manager->tts_request_callback(
                best_msg->text,
                manager->user_data
            );
            
            // Mark as busy and store request ID
            manager->is_tts_busy = true;
            manager->current_tts_request_id = request_id;
            best_msg->tts_request_id = request_id;
        }
        
        // Remove the message from the queue
        remove_message_from_queue(manager, best_msg);
    }
    
    return TK_SUCCESS;
}

void tk_feedback_manager_notify_tts_finished(
    tk_feedback_manager_t* manager,
    uint64_t request_id
) {
    if (!manager) return;
    
    // Check if this is the current TTS request
    if (manager->is_tts_busy && manager->current_tts_request_id == request_id) {
        manager->is_tts_busy = false;
        manager->current_tts_request_id = 0;
    }
}

void tk_feedback_manager_set_verbosity(
    tk_feedback_manager_t* manager,
    tk_feedback_verbosity_e level
) {
    if (!manager) return;
    
    manager->current_verbosity = level;
}

void tk_feedback_manager_clear_and_interrupt(tk_feedback_manager_t* manager) {
    if (!manager) return;
    
    // Clear the message queue
    clear_message_queue(manager);
    
    // Reset suppression tracking
    manager->suppression_count = 0;
    
    // If TTS is busy, we can't actually interrupt it here
    // The audio pipeline would need to handle that separately
    // We just mark ourselves as not busy so we can process new messages
    manager->is_tts_busy = false;
    manager->current_tts_request_id = 0;
}
