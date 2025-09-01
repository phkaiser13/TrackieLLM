/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_feedback_manager.h
*
* This header file defines the public API for the User Feedback Manager. This is
* not a simple Text-to-Speech (TTS) wrapper; it is a mission-critical
* communication arbitration system. Its primary responsibility is to manage the
* flow of information to the user, ensuring that critical safety alerts are
* prioritized over all other forms of communication.
*
* The architecture is built around a prioritized message queue and a set of
* configurable rules for verbosity and message suppression. This prevents
* "auditory spam" and ensures the user receives the right information at the
* right time. The Cortex submits feedback "requests" rather than raw text,
* allowing the manager to make intelligent decisions about what to say, when to
* say it, and what to interrupt.
*
* This module acts as the gatekeeper to the TTS engine.
*
* SPDX-License-Identifier:
*/

#ifndef TRACKIELLM_INTERACTION_TK_FEEDBACK_MANAGER_H
#define TRACKIELLM_INTERACTION_TK_FEEDBACK_MANAGER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"

// Forward-declare the primary manager object as an opaque type.
typedef struct tk_feedback_manager_s tk_feedback_manager_t;

/**
 * @enum tk_feedback_priority_e
 * @brief Defines the priority of a feedback message. Higher values interrupt lower values.
 */
typedef enum {
    TK_PRIORITY_LOW_AMBIENT = 10,   /**< Non-critical ambient information (e.g., "a car passed by"). */
    TK_PRIORITY_NORMAL_RESPONSE = 20, /**< A direct response to a user's question. */
    TK_PRIORITY_HIGH_OBSTACLE = 30,   /**< A warning about a static obstacle in the user's path. */
    TK_PRIORITY_CRITICAL_HAZARD = 40  /**< A critical, immediate danger (e.g., a step down, an approaching object). */
} tk_feedback_priority_e;

/**
 * @enum tk_feedback_verbosity_e
 * @brief Defines verbosity levels to categorize and filter messages.
 */
typedef enum {
    TK_VERBOSITY_LEVEL_QUIET,   /**< Only allows CRITICAL_HAZARD priority messages. */
    TK_VERBOSITY_LEVEL_NORMAL,  /**< Allows hazards and normal responses. */
    TK_VERBOSITY_LEVEL_VERBOSE  /**< Allows all messages, including ambient information. */
} tk_feedback_verbosity_e;

/**
 * @typedef tk_tts_request_func_t
 * @brief A function pointer that the Feedback Manager will call to synthesize speech.
 *
 * This decouples the manager from the audio pipeline, following the dependency
 * inversion principle. The Cortex provides this function during initialization.
 * @param text_to_speak The text to be synthesized.
 * @param user_data The opaque pointer provided during manager creation.
 * @return A unique ID for this speech request, to be used in callbacks.
 */
typedef uint64_t (*tk_tts_request_func_t)(const char* text_to_speak, void* user_data);

/**
 * @struct tk_feedback_manager_config_t
 * @brief Configuration for initializing the Feedback Manager.
 */
typedef struct {
    tk_tts_request_func_t tts_request_callback; /**< The function to call to request TTS synthesis. */
    void*                 user_data;            /**< Opaque pointer passed to the TTS callback. */
} tk_feedback_manager_config_t;

/**
 * @struct tk_feedback_request_t
 * @brief A rich structure describing a single feedback request.
 */
typedef struct {
    const char*              text;          /**< The text content of the message. */
    tk_feedback_priority_e   priority;      /**< The priority of the message. */
    tk_feedback_verbosity_e  verbosity_level; /**< The minimum verbosity level at which this message should be played. */
    
    /**
     * @brief A key used for suppressing duplicate messages. Messages with the same
     * non-zero key submitted within the cooldown period will be dropped.
     */
    uint32_t                 suppression_key;
    uint32_t                 suppression_cooldown_ms; /**< Cooldown in milliseconds for the suppression key. */
    
    /**
     * @brief If true, this message can interrupt and clear the queue of any
     * messages with a lower priority. If false, it will be queued normally.
     */
    bool                     is_interrupt;
} tk_feedback_request_t;


#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Manager Lifecycle Management
//------------------------------------------------------------------------------

/**
 * @brief Creates and initializes a new Feedback Manager instance.
 *
 * @param[out] out_manager Pointer to receive the address of the new manager instance.
 * @param[in] config The configuration, including the essential TTS callback.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL or config is invalid.
 * @return TK_ERROR_OUT_OF_MEMORY on memory allocation failure.
 */
TK_NODISCARD tk_error_code_t tk_feedback_manager_create(tk_feedback_manager_t** out_manager, const tk_feedback_manager_config_t* config);

/**
 * @brief Destroys a Feedback Manager instance.
 *
 * @param[in,out] manager Pointer to the manager instance to be destroyed.
 */
void tk_feedback_manager_destroy(tk_feedback_manager_t** manager);

//------------------------------------------------------------------------------
// Core Processing and Control
//------------------------------------------------------------------------------

/**
 * @brief Submits a new feedback request to the manager.
 *
 * The manager will process this request according to its priority, the current
 * verbosity level, and the suppression rules.
 *
 * @param[in] manager The feedback manager instance.
 * @param[in] request A pointer to the request structure. The manager may copy
 *                    the contents, so the caller can reclaim the structure
 *                    immediately after the call.
 *
 * @return TK_SUCCESS if the message was accepted (either queued or dropped due
 *         to suppression).
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL.
 */
TK_NODISCARD tk_error_code_t tk_feedback_manager_enqueue(tk_feedback_manager_t* manager, const tk_feedback_request_t* request);

/**
 * @brief Updates the manager's internal state.
 *
 * This function should be called periodically by the Cortex. It processes the
 * message queue, handles cooldowns for suppression, and sends the highest-priority
 * valid message to the TTS engine if it's not already busy.
 *
 * @param[in] manager The feedback manager instance.
 * @param[in] delta_time_s Time elapsed in seconds since the last update.
 *
 * @return TK_SUCCESS on success.
 */
TK_NODISCARD tk_error_code_t tk_feedback_manager_update(tk_feedback_manager_t* manager, float delta_time_s);

/**
 * @brief Notifies the manager that a TTS synthesis/playback has finished.
 *
 * The audio pipeline must call this function when it finishes playing a message,
 * allowing the manager to process the next item in the queue.
 *
 * @param[in] manager The feedback manager instance.
 * @param[in] request_id The unique ID that was returned by the TTS callback.
 */
void tk_feedback_manager_notify_tts_finished(tk_feedback_manager_t* manager, uint64_t request_id);

//------------------------------------------------------------------------------
// Configuration
//------------------------------------------------------------------------------

/**
 * @brief Sets the user-defined verbosity level at runtime.
 *
 * @param[in] manager The feedback manager instance.
 * @param[in] level The new verbosity level.
 */
void tk_feedback_manager_set_verbosity(tk_feedback_manager_t* manager, tk_feedback_verbosity_e level);

/**
 * @brief Immediately stops any currently playing message and clears the entire queue.
 *
 * This is a hard interrupt, useful for when the user gives a new command.
 *
 * @param[in] manager The feedback manager instance.
 */
void tk_feedback_manager_clear_and_interrupt(tk_feedback_manager_t* manager);


#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_INTERACTION_TK_FEEDBACK_MANAGER_H