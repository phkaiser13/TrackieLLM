/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/cortex/reasoning.rs
 *
 * This file implements a safe Rust wrapper for the Contextual Reasoning Engine
 * defined in `tk_contextual_reasoner.h`. It provides a high-level API for
 * managing the application's short-term memory and situational awareness.
 *
 * The `ContextualReasoner` struct is the primary interface. It encapsulates
 * the `unsafe` FFI calls required to interact with the C-based reasoner,
 * handling resource management (via RAII) and error translation automatically.
 *
 * This module is responsible for converting rich Rust types into the C-style
 * structs required by the FFI and vice-versa, ensuring that the rest of the
 * application can operate with type-safe, idiomatic Rust.
 *
 * Dependencies:
 *   - crate::ffi: For the raw C function bindings.
 *   - crate::CortexError: For shared error handling.
 *   - log: For structured logging.
 *   - thiserror: For ergonomic error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use super::{ffi, CortexError};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr::null_mut;
use thiserror::Error;
use libc;

/// Represents errors specific to the contextual reasoner.
#[derive(Debug, Error)]
pub enum ReasoningError {
    /// The underlying C-level context is not initialized.
    #[error("Reasoner context is not initialized.")]
    NotInitialized,

    /// An FFI call failed, with a message from the C side.
    #[error("Reasoner FFI call failed: {0}")]
    Ffi(String),

    /// A string passed to an FFI function contained an unexpected null byte.
    #[error("FFI string conversion failed: {0}")]
    NulError(#[from] std::ffi::NulError),

    /// A string returned from an FFI function was not valid UTF-8.
    #[error("FFI string decoding failed: {0}")]
    Utf8Error(#[from] std::str::Utf8Error),
}

/// A low-level RAII wrapper for the `tk_contextual_reasoner_t` handle.
/// In a real implementation, this would be managed by the `Cortex` struct.
struct ReasonerContext {
    ptr: *mut ffi::tk_contextual_reasoner_t,
}

impl Drop for ReasonerContext {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // The Cortex owns the reasoner, so it's responsible for destroying it.
            // This Drop impl is just for correctness in case this struct is ever
            // used in a standalone way.
        }
    }
}

/// A safe, high-level interface to the Contextual Reasoning Engine.
pub struct ContextualReasoner {
    // This is not the owner of the pointer. The C-level `tk_cortex_t` is.
    // We just hold a raw pointer to it.
    ptr: *mut ffi::tk_contextual_reasoner_t,
}

impl ContextualReasoner {
    /// Creates a new `ContextualReasoner` wrapper.
    /// This does not create the reasoner, it only wraps an existing one.
    pub fn new(ptr: *mut ffi::tk_contextual_reasoner_t) -> Self {
        Self { ptr }
    }

    /// Adds a new turn of conversation to the context.
    pub fn add_conversation_turn(
        &mut self,
        is_user_input: bool,
        content: &str,
        confidence: f32,
    ) -> Result<(), ReasoningError> {
        if self.ptr.is_null() {
            return Err(ReasoningError::NotInitialized);
        }

        let c_content = CString::new(content)?;

        let status = unsafe {
            ffi::tk_contextual_reasoner_add_conversation_turn(
                self.ptr,
                is_user_input,
                c_content.as_ptr(),
                confidence,
            )
        };

        if status != ffi::tk_error_code_t_TK_SUCCESS {
            let error_msg = ffi::get_last_error_message();
            Err(ReasoningError::Ffi(error_msg))
        } else {
            Ok(())
        }
    }

    /// Generates a textual summary of the current context for the LLM.
    pub fn generate_context_string(
        &self,
        max_token_budget: usize,
    ) -> Result<String, ReasoningError> {
        if self.ptr.is_null() {
            return Err(ReasoningError::NotInitialized);
        }

        let mut c_string_ptr: *mut c_char = null_mut();

        let status = unsafe {
            ffi::tk_contextual_reasoner_generate_context_string(
                self.ptr,
                &mut c_string_ptr,
                max_token_budget,
            )
        };

        if status != ffi::tk_error_code_t_TK_SUCCESS {
            return Err(ReasoningError::Ffi(ffi::get_last_error_message()));
        }

        if c_string_ptr.is_null() {
            // C function succeeded but returned a null pointer, which is unexpected.
            // Treat this as an empty string.
            return Ok(String::new());
        }

        // Unsafe block to convert C string to Rust String and then free it.
        let result = unsafe {
            let rust_string = CStr::from_ptr(c_string_ptr).to_str()?.to_owned();
            // We must free the string that was allocated by the C side.
            // Assuming it was allocated with malloc/calloc/strdup.
            libc::free(c_string_ptr as *mut libc::c_void);
            Ok(rust_string)
        };

        result
    }
}

impl Default for ContextualReasoner {
    fn default() -> Self {
        Self::new()
    }
}
