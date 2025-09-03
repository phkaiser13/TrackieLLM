/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/interaction/command_parser.rs
 *
 * This file provides a safe Rust wrapper for the data-driven Voice Command
 * Parser defined in `tk_voice_commands.h`.
 *
 * The `CommandParser` struct encapsulates the `unsafe` FFI calls required to
 * interact with the C-based parser. It manages the lifecycle of the
 * `tk_command_parser_t` handle using the RAII pattern.
 *
 * A key responsibility of this module is to safely manage the memory of the
 * result object (`tk_parsed_command_t`). The `ParsedCommand` struct takes
 * ownership of the C-allocated result and ensures `tk_command_parser_free_command`
 * is called when it is dropped, preventing memory leaks.
 *
 * Dependencies:
 *   - crate::ffi: For the raw C function bindings.
 *   - crate::InteractionError: For shared error handling.
 *   - thiserror: For ergonomic error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use super::{ffi, InteractionError};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::ptr::null_mut;
use thiserror::Error;

/// Represents errors specific to the command parsing process.
#[derive(Debug, Error)]
pub enum CommandParserError {
    /// The underlying C-level context is not initialized.
    #[error("Command parser context is not initialized.")]
    NotInitialized,

    /// The provided grammar data was invalid or could not be parsed.
    #[error("Failed to parse command grammar: {0}")]
    GrammarParseFailed(String),

    /// An FFI call to the command parser C library failed.
    #[error("Command parser FFI call failed: {0}")]
    Ffi(String),
}

/// A safe, RAII wrapper for a C-allocated `tk_parsed_command_t`.
///
/// This struct takes ownership of the pointer and guarantees that the
/// corresponding C `free` function is called when it goes out of scope.
pub struct ParsedCommand {
    /// The handle to the underlying C object.
    command_handle: *mut ffi::tk_parsed_command_t,
    /// The parsed command ID.
    pub id: u32,
    /// The canonical name of the command.
    pub name: String,
    /// The extracted parameters (slots) for the command.
    pub slots: HashMap<String, String>,
}

impl ParsedCommand {
    /// Creates a new `ParsedCommand` from a raw pointer returned by the FFI.
    ///
    /// # Safety
    /// The caller must ensure that `ptr` is a valid, non-null pointer to a
    /// `tk_parsed_command_t` allocated by `tk_command_parser_parse_command`.
    /// This function takes ownership of the pointer.
    unsafe fn from_raw(ptr: *mut ffi::tk_parsed_command_t) -> Self {
        let c_command = &*ptr;
        let name = CStr::from_ptr(c_command.command_name)
            .to_string_lossy()
            .into_owned();
        
        let mut slots = HashMap::new();
        if !c_command.slots.is_null() {
            let c_slots = std::slice::from_raw_parts(c_command.slots, c_command.slot_count);
            for c_slot in c_slots {
                let key = CStr::from_ptr(c_slot.key).to_string_lossy().into_owned();
                let value = CStr::from_ptr(c_slot.value).to_string_lossy().into_owned();
                slots.insert(key, value);
            }
        }

        Self {
            command_handle: ptr,
            id: c_command.command_id,
            name,
            slots,
        }
    }
}

impl Drop for ParsedCommand {
    /// Ensures the C-allocated command object is always freed.
    fn drop(&mut self) {
        if !self.command_handle.is_null() {
            unsafe { ffi::tk_command_parser_free_command(&mut self.command_handle) };
        }
    }
}

/// A safe, high-level interface to the Voice Command Parser.
pub struct CommandParser {
    /// The handle to the underlying `tk_command_parser_t` C object.
    parser_handle: *mut ffi::tk_command_parser_t,
}

impl CommandParser {
    /// Creates a new `CommandParser` from a compiled grammar blob.
    ///
    /// # Arguments
    /// * `grammar_data` - A byte slice containing the command grammar data.
    pub fn new(grammar_data: &[u8]) -> Result<Self, InteractionError> {
        let mut parser_handle = null_mut();
        let code = unsafe {
            ffi::tk_command_parser_create(
                &mut parser_handle,
                grammar_data.as_ptr(),
                grammar_data.len(),
            )
        };

        if code != ffi::TK_SUCCESS {
            return Err(
                CommandParserError::GrammarParseFailed("C API returned an error".to_string())
                    .into(),
            );
        }

        Ok(Self { parser_handle })
    }

    /// Parses transcribed text to find a matching command.
    ///
    /// # Arguments
    /// * `text` - The text to be parsed.
    ///
    /// # Returns
    /// * `Ok(Some(ParsedCommand))` if a command was successfully matched.
    /// * `Ok(None)` if the text did not match any known commands.
    /// * `Err(InteractionError)` if a parsing or system error occurred.
    pub fn parse_command(&self, text: &str) -> Result<Option<ParsedCommand>, InteractionError> {
        let c_text = CString::new(text)?;
        let mut command_handle = null_mut();

        let code = unsafe {
            ffi::tk_command_parser_parse_command(
                self.parser_handle,
                c_text.as_ptr(),
                &mut command_handle,
            )
        };

        if code != ffi::TK_SUCCESS {
            return Err(InteractionError::Ffi(format!(
                "tk_command_parser_parse_command failed with code {}",
                code
            )));
        }

        if command_handle.is_null() {
            // This is not an error; it simply means no command was matched.
            Ok(None)
        } else {
            // Safely take ownership of the returned C struct.
            let command = unsafe { ParsedCommand::from_raw(command_handle) };
            Ok(Some(command))
        }
    }
}

impl Drop for CommandParser {
    /// Ensures the C-level command parser is always destroyed.
    fn drop(&mut self) {
        if !self.parser_handle.is_null() {
            unsafe { ffi::tk_command_parser_destroy(&mut self.parser_handle) };
        }
    }
}
