/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/utils/error_utils.rs
 *
 * This file implements the primary error handling infrastructure for the Rust
 * side of the application. It defines a comprehensive `TrackieError` enum that
 * serves as a unified error type for all crates.
 *
 * A key feature is the direct mapping from the C-level `tk_error_code_t`.
 * This allows for seamless and safe propagation of errors across the FFI
 * boundary. The `From` trait is implemented to automatically convert a C error
 * code into a rich, structured Rust error, preserving the original context.
 *
 * The `Display` implementation for `TrackieError` uses the `tk_error_to_string`
 * FFI function to provide descriptive, human-readable error messages that are
 * consistent with the C side of the codebase.
 *
 * Dependencies:
 *   - crate::ffi: For the raw C type and function bindings.
 *   - thiserror: For ergonomic error enum definitions.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use super::ffi;
use std::ffi::CStr;
use thiserror::Error;

/// The C-level error code enum, mirrored in Rust for type safety.
///
/// This enum should be kept in sync with `tk_error_handling.h`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum tk_error_code_t {
    TK_SUCCESS = 0,
    TK_ERROR_UNKNOWN = 1000,
    TK_ERROR_INVALID_ARGUMENT = 1001,
    TK_ERROR_INVALID_STATE = 1002,
    TK_ERROR_NOT_IMPLEMENTED = 1003,
    TK_ERROR_BUFFER_TOO_SMALL = 1004,
    TK_ERROR_TIMEOUT = 1005,
    TK_ERROR_PERMISSION_DENIED = 1006,
    TK_ERROR_NOT_INITIALIZED = 1007,
    TK_ERROR_OUT_OF_MEMORY = 2000,
    TK_ERROR_FILE_NOT_FOUND = 3001,
    TK_ERROR_FILE_READ = 3002,
    TK_ERROR_FILE_WRITE = 3003,
    TK_ERROR_MODEL_LOAD_FAILED = 4000,
    TK_ERROR_INFERENCE_FAILED = 4002,
    // Add other error codes as needed...
}

/// The primary, unified error type for the entire TrackieLLM Rust application.
///
/// This enum consolidates errors from all possible sources, including FFI calls,
/// I/O operations, and module-specific failures.
#[derive(Debug, Error)]
pub enum TrackieError {
    /// An error that originated from the C side of the codebase.
    #[error("Core C/C++ Error: {message} (Code: {code:?})")]
    CoreError {
        /// The specific C error code.
        code: tk_error_code_t,
        /// The descriptive message for the error code.
        message: String,
    },

    /// An error from one of the utility modules.
    #[error("Utility Error: {0}")]
    Utils(#[from] crate::UtilsError),

    // In a real application, you would add variants for every other crate.
    // Example:
    // #[error("AI Model Error: {0}")]
    // AiModels(#[from] ai_models::AiModelsError),
    //
    // #[error("Vision Error: {0}")]
    // Vision(#[from] vision::VisionError),
}

impl From<tk_error_code_t> for TrackieError {
    /// Converts a raw C error code into a structured `TrackieError`.
    ///
    /// This function calls the `tk_error_to_string` FFI function to get the
    /// descriptive message associated with the code.
    fn from(code: tk_error_code_t) -> Self {
        if code == tk_error_code_t::TK_SUCCESS {
            // This should ideally not happen, as success codes shouldn't be
            // converted into errors. But as a safeguard:
            return TrackieError::CoreError {
                code,
                message: "Success code treated as error".to_string(),
            };
        }

        let message = unsafe {
            // Call the FFI function to get the error string.
            let c_str_ptr = ffi::tk_error_to_string(code);
            // Assume the C function never returns null and the string is valid UTF-8.
            // A more robust implementation would handle potential null pointers.
            CStr::from_ptr(c_str_ptr).to_string_lossy().into_owned()
        };

        TrackieError::CoreError { code, message }
    }
}
