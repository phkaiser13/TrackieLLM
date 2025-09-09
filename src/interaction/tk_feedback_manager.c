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
    tk_feedback_type_e       type;
    tk_feedback_priority_e   priority;
    tk_feedback_verbosity_e  verbosity_level;
    uint32_t                 suppression_key;
    uint32_t                 suppression_cooldown_ms;
    bool                     is_interrupt;
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
    
    // Message queue - using a simple array for now, not a true priority queue
    tk_internal_message_t message_queue[MAX_MESSAGE_QUEUE_SIZE];
    size_t                queue_size;
    
    // Currently playing message
    bool                  is_tts_busy;
    uint64_t              current_tts_request_id;
    tk_feedback_priority_e current_tts_priority;
    
    // Suppression tracking
    tk_suppression_entry_t suppression_table[MAX_MESSAGE_QUEUE_SIZE];
    size_t                 suppression_count;
};

//------------------------------------------------------------------------------
// Private helper functions
//------------------------------------------------------------------------------

static char* duplicate_string(const char* src) {
    if (!src) return NULL;
    size_t len = strlen(src) + 1;
    char* dup = malloc(len);
    if (dup) {
        memcpy(dup, src, len);
    }
    return dup;
}

static void free_internal_message(tk_internal_message_t* msg) {
    if (msg) {
        free(msg->text);
        msg->text = NULL;
    }
}

static void update_suppression_cooldowns(tk_feedback_manager_t* manager, float delta_time_s) {
    for (size_t i = 0; i < manager->suppression_count; ) {
        manager->suppression_table[i].cooldown_time_remaining -= delta_time_s;
        if (manager->suppression_table[i].cooldown_time_remaining <= 0.0f) {
            manager->suppression_table[i] = manager->suppression_table[manager->suppression_count - 1];
            manager->suppression_count--;
        } else {
            i++;
        }
    }
}

static bool is_suppression_key_active(tk_feedback_manager_t* manager, uint32_t key) {
    if (key == 0) return false;
    for (size_t i = 0; i < manager->suppression_count; i++) {
        if (manager->suppression_table[i].key == key) {
            return true;
        }
    }
    return false;
}

static void add_suppression_key(tk_feedback_manager_t* manager, uint32_t key, float cooldown_seconds) {
    if (key == 0 || manager->suppression_count >= MAX_MESSAGE_QUEUE_SIZE) return;
    for (size_t i = 0; i < manager->suppression_count; i++) {
        if (manager->suppression_table[i].key == key) {
            manager->suppression_table[i].cooldown_time_remaining = cooldown_seconds;
            return;
        }
    }
    manager->suppression_table[manager->suppression_count].key = key;
    manager->suppression_table[manager->suppression_count].cooldown_time_remaining = cooldown_seconds;
    manager->suppression_count++;
}

static void clear_message_queue(tk_feedback_manager_t* manager) {
    for (size_t i = 0; i < manager->queue_size; i++) {
        free_internal_message(&manager->message_queue[i]);
    }
    manager->queue_size = 0;
}

//------------------------------------------------------------------------------
// Public API Implementation
//------------------------------------------------------------------------------

tk_error_code_t tk_feedback_manager_create(tk_feedback_manager_t** out_manager, const tk_feedback_manager_config_t* config) {
    if (!out_manager || !config || !config->tts_request_callback) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    tk_feedback_manager_t* manager = calloc(1, sizeof(tk_feedback_manager_t));
    if (!manager) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    manager->tts_request_callback = config->tts_request_callback;
    manager->user_data = config->user_data;
    manager->current_verbosity = TK_VERBOSITY_LEVEL_NORMAL;
    *out_manager = manager;
    return TK_SUCCESS;
}

void tk_feedback_manager_destroy(tk_feedback_manager_t** manager) {
    if (manager && *manager) {
        clear_message_queue(*manager);
        free(*manager);
        *manager = NULL;
    }
}

tk_error_code_t tk_feedback_manager_enqueue(tk_feedback_manager_t* manager, const tk_feedback_request_t* request) {
    if (!manager || !request) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    if (is_suppression_key_active(manager, request->suppression_key)) {
        return TK_SUCCESS; // Suppressed
    }

    // --- Interrupt Logic ---
    if (request->is_interrupt) {
        // If TTS is busy, check if we can interrupt it
        if (manager->is_tts_busy && request->priority > manager->current_tts_priority) {
            LOG_INFO("Interrupting current TTS for higher priority message.");
            // In a real system, we'd call a tts_interrupt_callback.
            // For now, we just reset the busy flag.
            manager->is_tts_busy = false;
            manager->current_tts_request_id = 0;
            manager->current_tts_priority = 0;
        }

        // Remove all lower-priority messages from the queue
        size_t write_idx = 0;
        for (size_t read_idx = 0; read_idx < manager->queue_size; read_idx++) {
            if (manager->message_queue[read_idx].priority >= request->priority) {
                if (write_idx != read_idx) {
                    manager->message_queue[write_idx] = manager->message_queue[read_idx];
                }
                write_idx++;
            } else {
                free_internal_message(&manager->message_queue[read_idx]);
            }
        }
        manager->queue_size = write_idx;
    }

    if (manager->queue_size >= MAX_MESSAGE_QUEUE_SIZE) {
        LOG_WARN("Feedback manager queue full, dropping message.");
        return TK_SUCCESS;
    }

    // Add message to queue
    tk_internal_message_t* msg = &manager->message_queue[manager->queue_size];
    msg->text = duplicate_string(request->text);
    if (request->text && !msg->text) return TK_ERROR_OUT_OF_MEMORY;
    
    msg->type = request->type;
    msg->priority = request->priority;
    msg->verbosity_level = request->verbosity_level;
    msg->suppression_key = request->suppression_key;
    msg->suppression_cooldown_ms = request->suppression_cooldown_ms;
    msg->is_interrupt = request->is_interrupt;

    manager->queue_size++;

    if (request->suppression_key != 0) {
        add_suppression_key(manager, request->suppression_key, request->suppression_cooldown_ms / 1000.0f);
    }

    return TK_SUCCESS;
}

tk_error_code_t tk_feedback_manager_update(tk_feedback_manager_t* manager, float delta_time_s) {
    if (!manager) return TK_ERROR_INVALID_ARGUMENT;

    update_suppression_cooldowns(manager, delta_time_s);

    if (manager->is_tts_busy || manager->queue_size == 0) {
        return TK_SUCCESS;
    }

    // Find the highest priority message that meets verbosity requirements
    int best_msg_idx = -1;
    for (int i = 0; i < manager->queue_size; i++) {
        if (manager->message_queue[i].verbosity_level <= manager->current_verbosity) {
            if (best_msg_idx == -1 || manager->message_queue[i].priority > manager->message_queue[best_msg_idx].priority) {
                best_msg_idx = i;
            }
        }
    }

    if (best_msg_idx != -1) {
        tk_internal_message_t* msg_to_play = &manager->message_queue[best_msg_idx];

        // --- Dispatch Logic ---
        bool audio_requested = (msg_to_play->type == TK_FEEDBACK_TYPE_AUDIO || msg_to_play->type == TK_FEEDBACK_TYPE_AUDIO_HAPTIC) && msg_to_play->text;
        bool haptic_requested = (msg_to_play->type == TK_FEEDBACK_TYPE_HAPTIC || msg_to_play->type == TK_FEEDBACK_TYPE_AUDIO_HAPTIC);

        if (haptic_requested) {
            LOG_INFO("HAPTIC_EVENT: Triggering haptic feedback.");
            // In a real system: tk_haptic_controller_play(pattern);
        }

        if (audio_requested) {
            uint64_t request_id = manager->tts_request_callback(msg_to_play->text, manager->user_data);
            manager->is_tts_busy = true;
            manager->current_tts_request_id = request_id;
            manager->current_tts_priority = msg_to_play->priority;
        }

        // Remove message from queue by swapping with the last element
        free_internal_message(msg_to_play);
        manager->queue_size--;
        if (best_msg_idx != manager->queue_size) {
             manager->message_queue[best_msg_idx] = manager->message_queue[manager->queue_size];
        }
    }

    return TK_SUCCESS;
}

void tk_feedback_manager_notify_tts_finished(tk_feedback_manager_t* manager, uint64_t request_id) {
    if (manager && manager->is_tts_busy && manager->current_tts_request_id == request_id) {
        manager->is_tts_busy = false;
        manager->current_tts_request_id = 0;
        manager->current_tts_priority = 0;
    }
}

void tk_feedback_manager_set_verbosity(tk_feedback_manager_t* manager, tk_feedback_verbosity_e level) {
    if (manager) {
        manager->current_verbosity = level;
    }
}

void tk_feedback_manager_clear_and_interrupt(tk_feedback_manager_t* manager) {
    if (!manager) return;
    
    clear_message_queue(manager);
    manager->suppression_count = 0;
    
    if (manager->is_tts_busy) {
        // In a real system, we'd call a tts_interrupt_callback.
        manager->is_tts_busy = false;
        manager->current_tts_request_id = 0;
        manager->current_tts_priority = 0;
    }
}
