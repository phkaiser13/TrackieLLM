/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/utils/lib.rs
 *
 * This file is the main library entry point for the 'utils' crate. This
 * crate provides safe Rust wrappers for common, low-level C utilities used
 * throughout the TrackieLLM application, including error handling and logging.
 *
 * The primary purpose is to create a single, reliable bridge to the C helper
 * libraries, ensuring that their functionality can be safely consumed from
 * other Rust modules.
 *
 * The main components are:
 * - `error_utils`: Provides a comprehensive Rust error enum that maps to the
 *   C-level `tk_error_code_t` and functions for error translation.
 * - `debug_helpers`: Exposes the C-level logging macros in a safe, idiomatic
 *   Rust manner, allowing Rust code to log through the same backend as the C code.
 *
 * Dependencies:
 *   - log: For integrating with the Rust logging ecosystem.
 *   - thiserror: For ergonomic error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// === TOC ===
// 1. Crate-level Documentation & Attributes
// 2. FFI Bindings Module (private)
// 3. Public Module Declarations
// 4. Core Public Data Structures & Types
// 5. Public Prelude
// =============

#![deny(unsafe_code)]
#![deny(missing_docs)]
#![deny(warnings)]

//! # TrackieLLM Utilities Crate
//!
//! Provides safe Rust wrappers for core C-based utilities.
//!
//! ## Features
//!
//! - **Error Handling**: A centralized `TrackieError` enum that maps directly
//!   to the C library's `tk_error_code_t`, allowing for seamless error
//!   propagation across the FFI boundary.
//! - **Logging**: Safe functions that wrap the C-level `TK_LOG_*` macros,
//!   enabling Rust modules to log to the same destination as the C code.

// --- FFI Bindings Module ---
mod ffi {
    #![allow(non_camel_case_types, non_snake_case, dead_code)]
    pub use super::error_utils::tk_error_code_t;

    #[repr(C)]
    pub enum tk_log_level_t {
        TK_LOG_LEVEL_TRACE = 0,
        TK_LOG_LEVEL_DEBUG,
        TK_LOG_LEVEL_INFO,
        TK_LOG_LEVEL_WARN,
        TK_LOG_LEVEL_ERROR,
        TK_LOG_LEVEL_FATAL,
    }

    #[repr(C)]
    pub struct tk_log_config_t {
        pub level: tk_log_level_t,
        pub filename: *const std::os::raw::c_char,
        pub log_to_console: bool,
        pub use_utc_time: bool,
        pub quiet_mode: bool,
    }

    extern "C" {
        // From tk_error_handling.h
        pub fn tk_error_to_string(code: tk_error_code_t) -> *const std::os::raw::c_char;

        // From tk_logging.h
        pub fn tk_log_init(config: *const tk_log_config_t) -> tk_error_code_t;
        pub fn tk_log_shutdown();
        pub fn tk_log_set_level(level: tk_log_level_t);
        pub fn tk_log_message(
            level: tk_log_level_t,
            file: *const std::os::raw::c_char,
            line: std::os::raw::c_int,
            func: *const std::os::raw::c_char,
            fmt: *const std::os::raw::c_char,
            ...
        );
    }
}


// --- Public Module Declarations ---

/// Provides a comprehensive error type and conversion utilities.
pub mod error_utils;

/// Provides safe wrappers for the C-level logging macros.
pub mod debug_helpers;


// --- Core Public Data Structures & Types ---

use thiserror::Error;

/// A top-level error type for this crate.
#[derive(Debug, Error)]
pub enum UtilsError {
    /// An error occurred during FFI interaction.
    #[error("Utility FFI call failed: {0}")]
    Ffi(String),

    /// A C-string conversion failed.
    #[error("Invalid C-style string: {0}")]
    InvalidCString(#[from] std::ffi::NulError),
}


// --- Public Prelude ---

pub mod prelude {
    //! A "prelude" for convenient imports of this crate's main types.
    pub use super::{
        debug_helpers::{tk_log_debug, tk_log_error, tk_log_info, tk_log_warn},
        error_utils::{tk_error_code_t, TrackieError},
        init_logging, LogConfig, LogLevel, UtilsError,
    };
}


// --- Public Functions ---

use crate::error_utils::TrackieError;

/// Log level enum for the Rust-side configuration.
#[derive(Debug, Clone, Copy)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Fatal,
}

impl From<LogLevel> for ffi::tk_log_level_t {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Trace => ffi::tk_log_level_t::TK_LOG_LEVEL_TRACE,
            LogLevel::Debug => ffi::tk_log_level_t::TK_LOG_LEVEL_DEBUG,
            LogLevel::Info => ffi::tk_log_level_t::TK_LOG_LEVEL_INFO,
            LogLevel::Warn => ffi::tk_log_level_t::TK_LOG_LEVEL_WARN,
            LogLevel::Error => ffi::tk_log_level_t::TK_LOG_LEVEL_ERROR,
            LogLevel::Fatal => ffi::tk_log_level_t::TK_LOG_LEVEL_FATAL,
        }
    }
}

/// A safe Rust wrapper for the C `tk_log_config_t`.
#[derive(Debug, Clone, Default)]
pub struct LogConfig<'a> {
    pub level: LogLevel,
    pub filename: Option<&'a str>,
    pub log_to_console: bool,
    pub use_utc_time: bool,
    pub quiet_mode: bool,
}

/// Initializes the global C-level logger with a safe configuration.
///
/// This function must be called only once at application startup.
pub fn init_logging(config: &LogConfig) -> Result<(), TrackieError> {
    let c_filename = match config.filename.map(std::ffi::CString::new) {
        Some(Ok(s)) => s,
        Some(Err(e)) => return Err(UtilsError::InvalidCString(e).into()),
        None => std::ffi::CString::new("").unwrap(), // Should not fail
    };

    let c_config = ffi::tk_log_config_t {
        level: config.level.into(),
        filename: if config.filename.is_some() { c_filename.as_ptr() } else { std::ptr::null() },
        log_to_console: config.log_to_console,
        use_utc_time: config.use_utc_time,
        quiet_mode: config.quiet_mode,
    };

    let code = unsafe { ffi::tk_log_init(&c_config) };
    if code == ffi::TK_SUCCESS {
        Ok(())
    } else {
        Err(TrackieError::from(code))
    }
}
