/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_voice_commands.c
 *
 * This source file implements the C-side wrapper for the Voice Command Parser.
 * It acts as a thin Foreign Function Interface (FFI) layer, delegating all
 * parsing logic and state management to the Rust implementation in the
_ * `command_parser.rs` module.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "interaction/tk_voice_commands.h"
#include "utils/tk_error_handling.h"

// Opaque handle to the Rust-managed command parser object.
typedef void* CommandParserHandle;

// FFI declarations for the functions implemented in Rust.
// These would typically be in a generated header file.
extern CommandParserHandle rust_command_parser_create(const uint8_t* grammar_data, size_t grammar_size, int* out_error_code);
extern void rust_command_parser_destroy(CommandParserHandle handle);
extern tk_parsed_command_t* rust_command_parser_parse_command(CommandParserHandle handle, const char* text);
extern void rust_command_parser_free_command(tk_parsed_command_t* command);


//------------------------------------------------------------------------------
// Public API Implementation
//------------------------------------------------------------------------------

tk_error_code_t tk_command_parser_create(
    tk_command_parser_t** out_parser,
    const uint8_t* grammar_data,
    size_t grammar_size
) {
    if (!out_parser) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    int error_code = 0;
    *out_parser = (tk_command_parser_t*)rust_command_parser_create(grammar_data, grammar_size, &error_code);

    if (error_code != 0) {
        return (tk_error_code_t)error_code;
    }
    
    return *out_parser ? TK_SUCCESS : TK_ERROR_OUT_OF_MEMORY;
}

void tk_command_parser_destroy(tk_command_parser_t** parser) {
    if (parser && *parser) {
        rust_command_parser_destroy((CommandParserHandle)*parser);
        *parser = NULL;
    }
}

// NOTE: The tk_command_parser_find_wake_word function is omitted for simplicity,
// as the core logic will be handled by the more general parse_command.

tk_error_code_t tk_command_parser_parse_command(
    tk_command_parser_t* parser,
    const char* text,
    tk_parsed_command_t** out_command
) {
    if (!parser || !text || !out_command) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    *out_command = rust_command_parser_parse_command((CommandParserHandle)parser, text);
    
    // The Rust function returns a NULL pointer if no command is matched.
    // This is not an error condition.
    return TK_SUCCESS;
}

void tk_command_parser_free_command(tk_parsed_command_t** command) {
    if (command && *command) {
        rust_command_parser_free_command(*command);
        *command = NULL;
    }
}
