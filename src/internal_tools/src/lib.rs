/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/internal_tools/lib.rs
 *
 * This file is the main library entry point for the 'internal_tools' crate.
 * This crate provides safe and idiomatic Rust abstractions over the C-based
 * utility functions for file system management and configuration parsing.
 *
 * The primary purpose of this crate is to create a secure and robust bridge
 * to the C helper libraries, ensuring that all interactions are memory-safe
 * and that errors are handled gracefully within the Rust ecosystem. It wraps
 * unsafe FFI calls in higher-level constructs that leverage Rust's powerful
 * type system and RAII patterns.
 *
 * The main components are:
 * - `fs_utils`: A safe wrapper around `tk_file_manager.h`, providing a robust
 *   `Path` object and filesystem functions.
 * - `config_loader`: A safe wrapper around `tk_config_parser.h` for loading
 *   and querying key-value configuration files.
 *
 * All FFI declarations are kept in a private `ffi` module within this file
 * to enforce the use of the safe abstractions provided by the public modules.
 *
 * Dependencies:
 *   - log: For structured logging.
 *   - thiserror: For ergonomic error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// === TOC ===
// 1. Crate-level Documentation & Attributes
// 2. FFI Bindings Module (private)
// 3. Public Module Declarations
// 4. Public Prelude
// 5. Core Public Types (Error)
// =============

#![deny(unsafe_code)]
#![deny(missing_docs)]
#![deny(warnings)]

//! # TrackieLLM Internal Tools Crate
//!
//! Provides safe Rust wrappers for internal C-based utilities like the
//! configuration parser and the filesystem manager.
//!
//! ## Design
//!
//! This crate follows a strict "safe over unsafe" design. All `unsafe` FFI
//! calls are encapsulated within higher-level functions that return `Result`
//! types and manage resource lifetimes automatically using the `Drop` trait.
//! This prevents common C errors like memory leaks, use-after-free, and null
//! pointer dereferences from propagating into the Rust part of the application.

// --- FFI Bindings Module ---
// Contains the raw `extern "C"` declarations that mirror the C header files.
// This module is private to the crate.
mod ffi {
    #![allow(non_camel_case_types, non_snake_case, dead_code)]

    // Common types from tk_error_handling.h
    pub type tk_error_code_t = i32;
    pub const TK_SUCCESS: tk_error_code_t = 0;
    pub const TK_ERROR_INVALID_ARGUMENT: tk_error_code_t = -1;
    pub const TK_ERROR_OUT_OF_MEMORY: tk_error_code_t = -2;
    pub const TK_ERROR_FILE_NOT_FOUND: tk_error_code_t = -3;
    pub const TK_ERROR_FILE_READ: tk_error_code_t = -4;
    pub const TK_ERROR_FILE_WRITE: tk_error_code_t = -5;
    pub const TK_ERROR_PERMISSION_DENIED: tk_error_code_t = -6;

    // --- From tk_config_parser.h ---
    pub enum tk_config_s {}
    pub type tk_config_t = tk_config_s;

    extern "C" {
        pub fn tk_config_create(out_config: *mut *mut tk_config_t) -> tk_error_code_t;
        pub fn tk_config_destroy(config: *mut *mut tk_config_t);
        pub fn tk_config_load_from_file(config: *mut tk_config_t, filepath: *const std::os::raw::c_char) -> tk_error_code_t;
        pub fn tk_config_get_string(config: *const tk_config_t, key: *const std::os::raw::c_char, default_value: *const std::os::raw::c_char) -> *const std::os::raw::c_char;
        pub fn tk_config_get_int(config: *const tk_config_t, key: *const std::os::raw::c_char, default_value: i64) -> i64;
        pub fn tk_config_get_double(config: *const tk_config_t, key: *const std::os::raw::c_char, default_value: f64) -> f64;
        pub fn tk_config_get_bool(config: *const tk_config_t, key: *const std::os::raw::c_char, default_value: bool) -> bool;
    }

    // --- From tk_file_manager.h ---
    pub enum tk_path_s {}
    pub type tk_path_t = tk_path_s;

    #[repr(C)]
    pub enum tk_base_path_e {
        TK_BASE_PATH_APP_CONFIG,
        TK_BASE_PATH_CACHE,
        TK_BASE_PATH_EXECUTABLE_DIR,
        TK_BASE_PATH_WORKING_DIR,
    }

    extern "C" {
        pub fn tk_path_create_from_string(out_path: *mut *mut tk_path_t, path_str: *const std::os::raw::c_char) -> tk_error_code_t;
        pub fn tk_path_create_from_base(out_path: *mut *mut tk_path_t, base: tk_base_path_e) -> tk_error_code_t;
        pub fn tk_path_destroy(path: *mut *mut tk_path_t);
        pub fn tk_path_join(path: *mut tk_path_t, segment: *const std::os::raw::c_char) -> tk_error_code_t;
        pub fn tk_path_get_string(path: *const tk_path_t) -> *const std::os::raw::c_char;
        pub fn tk_fs_exists(path: *const tk_path_t, exists: *mut bool) -> tk_error_code_t;
        pub fn tk_fs_is_file(path: *const tk_path_t, is_file: *mut bool) -> tk_error_code_t;
        pub fn tk_fs_is_directory(path: *const tk_path_t, is_directory: *mut bool) -> tk_error_code_t;
        pub fn tk_dir_create_recursive(path: *const tk_path_t) -> tk_error_code_t;
        pub fn tk_file_read_all_bytes(path: *const tk_path_t, out_buffer: *mut *mut u8, out_size: *mut usize, max_size: usize) -> tk_error_code_t;
        pub fn tk_file_write_buffer(path: *const tk_path_t, buffer: *const u8, size: usize) -> tk_error_code_t;
    }
}


// --- Public Module Declarations ---

/// Safe wrappers for configuration file parsing (`tk_config_parser.h`).
pub mod config_loader;

/// Safe wrappers for filesystem operations (`tk_file_manager.h`).
pub mod fs_utils;


// --- Public Prelude ---

pub mod prelude {
    //! A "prelude" for convenient imports of this crate's main types.
    pub use super::{
        config_loader::ConfigLoader,
        fs_utils::{Path, BasePath},
        InternalToolsError,
    };
}


// --- Core Public Types ---

use thiserror::Error;

/// The primary error type for all operations within the internal_tools crate.
#[derive(Debug, Error)]
pub enum InternalToolsError {
    /// An error originating from the configuration loader.
    #[error("Configuration Error: {0}")]
    Config(#[from] config_loader::ConfigError),

    /// An error originating from the filesystem utilities.
    #[error("Filesystem Error: {0}")]
    FileSystem(#[from] fs_utils::FsError),

    /// An FFI call resulted in an invalid argument error.
    #[error("FFI call failed due to an invalid argument in function '{context}': {message}")]
    FfiInvalidArgument {
        context: String,
        message: String,
    },

    /// An FFI call failed due to an out-of-memory error.
    #[error("FFI call failed due to out of memory in function '{context}'")]
    FfiOutOfMemory { context: String },
    
    /// A C-string conversion failed, likely due to interior null bytes.
    #[error("Invalid C-style string: {0}")]
    InvalidCString(#[from] std::ffi::NulError),

    /// A string conversion from C failed, likely due to invalid UTF-8.
    #[error("Invalid UTF-8 sequence in string from FFI")]
    InvalidUtf8,
}

/// A helper to convert C error codes into a Rust `Result`.
fn ffi_result_from_code(code: ffi::tk_error_code_t, context: &str) -> Result<(), InternalToolsError> {
    match code {
        ffi::TK_SUCCESS => Ok(()),
        ffi::TK_ERROR_INVALID_ARGUMENT => Err(InternalToolsError::FfiInvalidArgument {
            context: context.to_string(),
            message: "A null pointer or invalid value was passed.".to_string(),
        }),
        ffi::TK_ERROR_OUT_OF_MEMORY => Err(InternalToolsError::FfiOutOfMemory {
            context: context.to_string(),
        }),
        // Other specific errors will be handled in the respective modules.
        _ => Err(InternalToolsError::FfiInvalidArgument {
            context: context.to_string(),
            message: format!("An unknown FFI error occurred (code: {})", code),
        }),
    }
}
