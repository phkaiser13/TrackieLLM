/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/interaction/command_parser.rs
 *
 * This file provides the primary Rust implementation for the Voice Command Parser.
 * It is called by the C FFI wrappers in `tk_voice_commands.c`.
 *
 * This module is responsible for the actual parsing logic. For this initial
 * implementation, it uses a simple keyword-matching approach. A more robust
 * future version could use a proper parsing library or NLP model.
 *
 * It also handles the complex memory management required for creating C-compatible
 * structs and transferring their ownership across the FFI boundary.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

// Assuming the tk_parsed_command_t struct definition is available from a bindings file.
// For now, we'll redefine it here to match the C header.
#[repr(C)]
pub struct tk_command_slot_t {
    pub key: *const c_char,
    pub value: *mut c_char,
}

#[repr(C)]
pub struct tk_parsed_command_t {
    pub command_id: u32,
    pub command_name: *const c_char,
    pub slots: *mut tk_command_slot_t,
    pub slot_count: usize,
}

/// The main Rust struct for the command parser.
/// The `tk_command_parser_t` in C is just an opaque pointer to this struct.
pub struct CommandParser {
    commands: HashMap<&'static str, &'static str>, // keyword -> command_name
}

impl CommandParser {
    /// Creates a new command parser with a hardcoded set of rules.
    fn new() -> Self {
        let mut commands = HashMap::new();
        commands.insert("status", "get_system_status");
        commands.insert("bateria", "get_system_status");
        commands.insert("ajuda", "get_help");
        commands.insert("parar", "stop_action");
        Self { commands }
    }

    /// Parses the input text and returns a C-compatible command struct.
    fn parse(&self, text: &str) -> *mut tk_parsed_command_t {
        let lower_text = text.to_lowercase();
        for (keyword, command_name) in &self.commands {
            if lower_text.contains(keyword) {
                // Found a match. Allocate a C-compatible struct.
                let command = Box::new(tk_parsed_command_t {
                    command_id: 0, // ID can be refined later
                    command_name: CString::new(*command_name).unwrap().into_raw(),
                    slots: ptr::null_mut(),
                    slot_count: 0,
                });
                return Box::into_raw(command);
            }
        }
        ptr::null_mut() // No match found
    }
}

// --- FFI Bridge Implementation ---

/// Creates a `CommandParser` instance and returns an opaque handle to it.
/// The `grammar_data` is ignored in this simple implementation.
#[no_mangle]
pub extern "C" fn rust_command_parser_create(
    _grammar_data: *const u8,
    _grammar_size: usize,
    out_error_code: *mut i32,
) -> *mut CommandParser {
    unsafe { *out_error_code = 0 };
    let parser = Box::new(CommandParser::new());
    Box::into_raw(parser)
}

/// Destroys the `CommandParser` instance.
#[no_mangle]
pub unsafe extern "C" fn rust_command_parser_destroy(handle: *mut CommandParser) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle);
    }
}

/// Parses the text and returns a pointer to a `tk_parsed_command_t`.
/// The caller (in C) takes ownership of the returned pointer.
#[no_mangle]
pub unsafe extern "C" fn rust_command_parser_parse_command(
    handle: *mut CommandParser,
    text: *const c_char,
) -> *mut tk_parsed_command_t {
    if handle.is_null() || text.is_null() {
        return ptr::null_mut();
    }
    let parser = &*handle;
    let text_str = CStr::from_ptr(text).to_string_lossy();
    parser.parse(&text_str)
}

/// Frees the memory for a `tk_parsed_command_t` that was allocated by Rust.
#[no_mangle]
pub unsafe extern "C" fn rust_command_parser_free_command(command: *mut tk_parsed_command_t) {
    if !command.is_null() {
        // Re-box the raw pointers to allow Rust's memory manager to handle them.
        let cmd_box = Box::from_raw(command);
        let _ = CString::from_raw(cmd_box.command_name as *mut i8);

        // In a version with slots, we would free them here too.
        // for i in 0..cmd_box.slot_count {
        //     let slot = &mut *(cmd_box.slots.add(i));
        //     let _ = CString::from_raw(slot.key as *mut i8);
        //     let _ = CString::from_raw(slot.value);
        // }
        // let _ = Box::from_raw(cmd_box.slots);
    }
}
