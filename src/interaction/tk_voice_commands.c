/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_voice_commands.c
 *
 * This source file implements the Voice Command Parser module. It provides
 * functionality to parse user commands based on a pre-compiled grammar definition,
 * enabling flexible and localized command interpretation.
 *
 * The parser supports wake word detection and full command parsing with parameter
 * extraction. It uses a data-driven approach where the command structure is defined
 * externally and loaded at runtime.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "interaction/tk_voice_commands.h"
#include "utils/tk_logging.h"
#include "utils/tk_error_handling.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>

// Maximum number of command slots we support per command
#define MAX_SLOTS_PER_COMMAND 16

// Internal structures for grammar representation
typedef struct {
    uint32_t id;
    char* name;
    char** phrases;      // Array of possible phrases for this command
    size_t phrase_count;
    char** slot_names;   // Names of slots that can be filled
    size_t slot_count;
} tk_internal_command_t;

typedef struct {
    char* phrase;
    size_t phrase_len;
} tk_wake_word_phrase_t;

struct tk_command_parser_s {
    tk_internal_command_t* commands;
    size_t command_count;
    
    tk_wake_word_phrase_t* wake_words;
    size_t wake_word_count;
    
    // Grammar metadata
    uint32_t version;
    char* language_code;
};

//------------------------------------------------------------------------------
// Private helper functions
//------------------------------------------------------------------------------

/**
 * @brief Converts a string to lowercase in-place
 */
static void to_lowercase(char* str) {
    if (!str) return;
    
    for (int i = 0; str[i]; i++) {
        str[i] = tolower(str[i]);
    }
}

/**
 * @brief Duplicates a string with memory allocation
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
 * @brief Frees memory allocated for an internal command structure
 */
static void free_internal_command(tk_internal_command_t* cmd) {
    if (!cmd) return;
    
    if (cmd->name) {
        free(cmd->name);
        cmd->name = NULL;
    }
    
    if (cmd->phrases) {
        for (size_t i = 0; i < cmd->phrase_count; i++) {
            if (cmd->phrases[i]) {
                free(cmd->phrases[i]);
            }
        }
        free(cmd->phrases);
        cmd->phrases = NULL;
    }
    
    if (cmd->slot_names) {
        for (size_t i = 0; i < cmd->slot_count; i++) {
            if (cmd->slot_names[i]) {
                free(cmd->slot_names[i]);
            }
        }
        free(cmd->slot_names);
        cmd->slot_names = NULL;
    }
}

/**
 * @brief Frees memory allocated for wake word phrases
 */
static void free_wake_words(tk_wake_word_phrase_t* wake_words, size_t count) {
    if (!wake_words) return;
    
    for (size_t i = 0; i < count; i++) {
        if (wake_words[i].phrase) {
            free(wake_words[i].phrase);
        }
    }
    free(wake_words);
}

/**
 * @brief Simple pattern matching with wildcards (*)
 * This is a basic implementation for demonstration purposes
 */
static bool match_pattern(const char* text, const char* pattern) {
    if (!text || !pattern) return false;
    
    const char* star = NULL;
    const char* ss = text;
    
    while (*text) {
        if (*pattern == '*') {
            star = pattern++;
            ss = text;
        } else if (*pattern == *text || *pattern == '?') {
            pattern++;
            text++;
        } else if (star) {
            pattern = star + 1;
            text = ++ss;
        } else {
            return false;
        }
    }
    
    while (*pattern == '*') pattern++;
    return !*pattern;
}

/**
 * @brief Finds the best matching command and extracts slots
 * This is a simplified implementation - in production, you might use NLP techniques
 */
static bool find_best_command_match(
    tk_command_parser_t* parser,
    const char* text,
    tk_parsed_command_t* out_command
) {
    if (!parser || !text || !out_command) return false;
    
    char* lower_text = duplicate_string(text);
    if (!lower_text) return false;
    
    to_lowercase(lower_text);
    
    // Try to match each command
    for (size_t i = 0; i < parser->command_count; i++) {
        tk_internal_command_t* cmd = &parser->commands[i];
        
        // Check each phrase for this command
        for (size_t j = 0; j < cmd->phrase_count; j++) {
            if (match_pattern(lower_text, cmd->phrases[j])) {
                // Found a match
                out_command->command_id = cmd->id;
                out_command->command_name = cmd->name;
                
                // For simplicity, we'll just copy the first slot name if any exist
                // In a real implementation, you'd extract actual values from the text
                if (cmd->slot_count > 0) {
                    out_command->slots = calloc(cmd->slot_count, sizeof(tk_command_slot_t));
                    if (!out_command->slots) {
                        free(lower_text);
                        return false;
                    }
                    
                    out_command->slot_count = cmd->slot_count;
                    
                    for (size_t k = 0; k < cmd->slot_count; k++) {
                        out_command->slots[k].key = cmd->slot_names[k];
                        // In a real implementation, extract value from text
                        out_command->slots[k].value = duplicate_string("extracted_value");
                    }
                } else {
                    out_command->slots = NULL;
                    out_command->slot_count = 0;
                }
                
                free(lower_text);
                return true;
            }
        }
    }
    
    free(lower_text);
    return false;
}

//------------------------------------------------------------------------------
// Public API Implementation
//------------------------------------------------------------------------------

tk_error_code_t tk_command_parser_create(
    tk_command_parser_t** out_parser,
    const uint8_t* grammar_data,
    size_t grammar_size
) {
    if (!out_parser || !grammar_data || grammar_size == 0) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_parser = NULL;
    
    // Allocate parser structure
    tk_command_parser_t* parser = calloc(1, sizeof(tk_command_parser_t));
    if (!parser) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // In a real implementation, we would parse the grammar_data here
    // For this example, we'll create a simple hardcoded grammar
    
    // Set version and language
    parser->version = 1;
    parser->language_code = duplicate_string("pt-BR");
    if (!parser->language_code) {
        free(parser);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Create some sample commands
    parser->command_count = 3;
    parser->commands = calloc(parser->command_count, sizeof(tk_internal_command_t));
    if (!parser->commands) {
        free(parser->language_code);
        free(parser);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Command 1: "locate_object"
    parser->commands[0].id = 1001;
    parser->commands[0].name = duplicate_string("locate_object");
    parser->commands[0].phrase_count = 2;
    parser->commands[0].phrases = calloc(parser->commands[0].phrase_count, sizeof(char*));
    parser->commands[0].phrases[0] = duplicate_string("onde * *");
    parser->commands[0].phrases[1] = duplicate_string("encontre * *");
    parser->commands[0].slot_count = 1;
    parser->commands[0].slot_names = calloc(parser->commands[0].slot_count, sizeof(char*));
    parser->commands[0].slot_names[0] = duplicate_string("object_name");
    
    // Command 2: "navigate_to"
    parser->commands[1].id = 1002;
    parser->commands[1].name = duplicate_string("navigate_to");
    parser->commands[1].phrase_count = 2;
    parser->commands[1].phrases = calloc(parser->commands[1].phrase_count, sizeof(char*));
    parser->commands[1].phrases[0] = duplicate_string("vá para *");
    parser->commands[1].phrases[1] = duplicate_string("me leve até *");
    parser->commands[1].slot_count = 1;
    parser->commands[1].slot_names = calloc(parser->commands[1].slot_count, sizeof(char*));
    parser->commands[1].slot_names[0] = duplicate_string("destination");
    
    // Command 3: "read_text"
    parser->commands[2].id = 1003;
    parser->commands[2].name = duplicate_string("read_text");
    parser->commands[2].phrase_count = 2;
    parser->commands[2].phrases = calloc(parser->commands[2].phrase_count, sizeof(char*));
    parser->commands[2].phrases[0] = duplicate_string("leia *");
    parser->commands[2].phrases[1] = duplicate_string("o que *");
    parser->commands[2].slot_count = 1;
    parser->commands[2].slot_names = calloc(parser->commands[2].slot_count, sizeof(char*));
    parser->commands[2].slot_names[0] = duplicate_string("text_content");
    
    // Create wake words
    parser->wake_word_count = 2;
    parser->wake_words = calloc(parser->wake_word_count, sizeof(tk_wake_word_phrase_t));
    if (!parser->wake_words) {
        // Cleanup already allocated commands
        for (size_t i = 0; i < parser->command_count; i++) {
            free_internal_command(&parser->commands[i]);
        }
        free(parser->commands);
        free(parser->language_code);
        free(parser);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    parser->wake_words[0].phrase = duplicate_string("hey trackie");
    parser->wake_words[0].phrase_len = strlen(parser->wake_words[0].phrase);
    parser->wake_words[1].phrase = duplicate_string("oi trackie");
    parser->wake_words[1].phrase_len = strlen(parser->wake_words[1].phrase);
    
    *out_parser = parser;
    return TK_SUCCESS;
}

void tk_command_parser_destroy(tk_command_parser_t** parser) {
    if (!parser || !*parser) return;
    
    tk_command_parser_t* p = *parser;
    
    // Free commands
    if (p->commands) {
        for (size_t i = 0; i < p->command_count; i++) {
            free_internal_command(&p->commands[i]);
        }
        free(p->commands);
    }
    
    // Free wake words
    free_wake_words(p->wake_words, p->wake_word_count);
    
    // Free language code
    if (p->language_code) {
        free(p->language_code);
    }
    
    // Free parser itself
    free(p);
    *parser = NULL;
}

tk_error_code_t tk_command_parser_find_wake_word(
    tk_command_parser_t* parser,
    const char* text,
    bool* out_is_wake_word_present
) {
    if (!parser || !text || !out_is_wake_word_present) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_is_wake_word_present = false;
    
    // Convert input text to lowercase for comparison
    char* lower_text = duplicate_string(text);
    if (!lower_text) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    to_lowercase(lower_text);
    
    // Check for each wake word
    for (size_t i = 0; i < parser->wake_word_count; i++) {
        if (strstr(lower_text, parser->wake_words[i].phrase)) {
            *out_is_wake_word_present = true;
            break;
        }
    }
    
    free(lower_text);
    return TK_SUCCESS;
}

tk_error_code_t tk_command_parser_parse_command(
    tk_command_parser_t* parser,
    const char* text,
    tk_parsed_command_t** out_command
) {
    if (!parser || !text || !out_command) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_command = NULL;
    
    // Allocate result structure
    tk_parsed_command_t* command = calloc(1, sizeof(tk_parsed_command_t));
    if (!command) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Try to find a matching command
    if (find_best_command_match(parser, text, command)) {
        *out_command = command;
        return TK_SUCCESS;
    } else {
        // No command matched, but that's not an error
        free(command);
        *out_command = NULL;
        return TK_SUCCESS;
    }
}

void tk_command_parser_free_command(tk_parsed_command_t** command) {
    if (!command || !*command) return;
    
    tk_parsed_command_t* cmd = *command;
    
    if (cmd->slots) {
        for (size_t i = 0; i < cmd->slot_count; i++) {
            if (cmd->slots[i].value) {
                free(cmd->slots[i].value);
            }
        }
        free(cmd->slots);
    }
    
    free(cmd);
    *command = NULL;
}
