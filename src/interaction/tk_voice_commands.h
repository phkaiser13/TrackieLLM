/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_voice_commands.h
*
* This header file defines the public API for the Voice Command Parser. This
* module is a critical component of the interaction system, responsible for
* interpreting the user's intent from transcribed text.
*
* The architecture is designed for maximum flexibility and localization. Instead
* of hardcoding commands, the parser is initialized with a data blob (e.g.,
* compiled from a JSON or MessagePack file) that defines the entire command
* grammar, including wake words, command phrases, and parameter "slots".
*
* This approach decouples the command logic from the recognition logic, allowing
* for easy updates, personalization, and addition of new languages without
* recompiling the core application. The parser's output is a structured command
* object, not a raw string, which simplifies the decision-making logic in the Cortex.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TRACKIELLM_INTERACTION_TK_VOICE_COMMANDS_H
#define TRACKIELLM_INTERACTION_TK_VOICE_COMMANDS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"

// Forward-declare the primary parser object as an opaque type.
typedef struct tk_command_parser_s tk_command_parser_t;

/**
 * @struct tk_command_slot_t
 * @brief Represents a single parameter (slot) extracted from a user command.
 *
 * For a command like "find my keys", the slot would be:
 *   key = "object_name"
 *   value = "keys"
 */
typedef struct {
    const char* key;    /**< The name of the slot (e.g., "object_name"). */
    char*       value;  /**< The extracted value. Owned by the parsed command object. */
} tk_command_slot_t;

/**
 * @struct tk_parsed_command_t
 * @brief Represents a fully parsed and understood command, ready for execution.
 *
 * This structure and its contents are allocated by the parser and must be freed
 * by the caller using `tk_command_parser_free_command`.
 */
typedef struct {
    uint32_t            command_id;     /**< A unique, numeric ID for the command, defined in the grammar. */
    const char*         command_name;   /**< The canonical name of the command (e.g., "locate_object"). */
    tk_command_slot_t*  slots;          /**< A dynamically allocated array of extracted slots. */
    size_t              slot_count;     /**< The number of slots in the array. */
} tk_parsed_command_t;


#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Parser Lifecycle Management
//------------------------------------------------------------------------------

/**
 * @brief Creates and initializes a new Command Parser instance.
 *
 * This function parses a pre-compiled grammar definition from a memory buffer.
 *
 * @param[out] out_parser Pointer to receive the address of the new parser instance.
 * @param[in] grammar_data A pointer to the memory buffer containing the command
 *                         grammar (e.g., in MessagePack format).
 * @param[in] grammar_size The size of the grammar data buffer in bytes.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL or grammar is invalid.
 * @return TK_ERROR_OUT_OF_MEMORY on memory allocation failure.
 * @return TK_ERROR_CONFIG_PARSE_FAILED if the grammar data is corrupt or malformed.
 */
TK_NODISCARD tk_error_code_t tk_command_parser_create(
    tk_command_parser_t** out_parser,
    const uint8_t* grammar_data,
    size_t grammar_size
);

/**
 * @brief Destroys a Command Parser instance.
 *
 * @param[in,out] parser Pointer to the parser instance to be destroyed.
 */
void tk_command_parser_destroy(tk_command_parser_t** parser);

//------------------------------------------------------------------------------
// Command Parsing and Analysis
//------------------------------------------------------------------------------

/**
 * @brief Scans a text buffer specifically for a defined wake word.
 *
 * This function is optimized to only look for wake word phrases (e.g., "Hey Trackie").
 * It should be used when the system is in a standby/listening state.
 *
 * @param[in] parser The command parser instance.
 * @param[in] text The transcribed text to analyze.
 * @param[out] out_is_wake_word_present Pointer to a boolean that will be set to true
 *                                    if a wake word is detected.
 *
 * @return TK_SUCCESS on successful analysis.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL.
 */
TK_NODISCARD tk_error_code_t tk_command_parser_find_wake_word(
    tk_command_parser_t* parser,
    const char* text,
    bool* out_is_wake_word_present
);

/**
 * @brief Parses a text buffer to identify and extract a specific command and its parameters.
 *
 * This is the main parsing function, used after a wake word has been detected.
 * It attempts to match the input text against all command phrases defined in the
 * grammar and extracts any defined slots.
 *
 * @param[in] parser The command parser instance.
 * @param[in] text The transcribed text to analyze (e.g., "where are my keys").
 * @param[out] out_command Pointer to receive the newly allocated parsed command
 *                         structure. If no command is matched, this will be NULL.
 *                         The caller assumes ownership and must free it with
 *                         `tk_command_parser_free_command`.
 *
 * @return TK_SUCCESS on successful analysis (even if no command was matched).
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL.
 * @return TK_ERROR_OUT_OF_MEMORY if allocation for the command result fails.
 */
TK_NODISCARD tk_error_code_t tk_command_parser_parse_command(
    tk_command_parser_t* parser,
    const char* text,
    tk_parsed_command_t** out_command
);

//------------------------------------------------------------------------------
// Result Data Management
//------------------------------------------------------------------------------

/**
 * @brief Frees the memory allocated for a parsed command object.
 *
 * @param[in,out] command Pointer to the command object to be freed.
 */
void tk_command_parser_free_command(tk_parsed_command_t** command);


#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_INTERACTION_TK_VOICE_COMMANDS_H