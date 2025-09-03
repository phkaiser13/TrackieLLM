/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/cortex/src/ffi.rs
 *
 * This file contains the foreign function interface (FFI) bindings and safe
 * wrappers for the C-level functions used by the Rust components of the Cortex.
 * It centralizes all `unsafe` FFI declarations and provides safe, idiomatic
 * Rust functions to the rest of the crate.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::CStr;
use std::os::raw::c_char;

// Re-export the C-level types from the main ffi_bridge crate for consistency.
// In a larger project, these might be in a dedicated `ffi-types` crate.
pub use trackiellm_ffi::{
    tk_contextual_reasoner_t,
    tk_error_code_t,
    tk_error_code_t_TK_SUCCESS
};

// FFI declarations for C functions that we need to call from this module.
#[link(name = "trackiellm_core")]
extern "C" {
    // From tk_contextual_reasoner.c
    pub fn tk_contextual_reasoner_add_conversation_turn(
        reasoner: *mut tk_contextual_reasoner_t,
        is_user_input: bool,
        content: *const c_char,
        confidence: f32,
    ) -> tk_error_code_t;

    pub fn tk_contextual_reasoner_generate_context_string(
        reasoner: *mut tk_contextual_reasoner_t,
        out_context_string: *mut *mut c_char,
        max_token_budget: usize,
    ) -> tk_error_code_t;

    // From the FFI bridge, which forwards to the C implementation.
    fn tk_get_last_error() -> *const c_char;
}

/// Safely gets the last error message from the FFI layer.
pub fn get_last_error_message() -> String {
    unsafe {
        let error_ptr = tk_get_last_error();
        if error_ptr.is_null() {
            return "Unknown FFI error".to_string();
        }
        CStr::from_ptr(error_ptr)
            .to_string_lossy()
            .into_owned()
    }
}
