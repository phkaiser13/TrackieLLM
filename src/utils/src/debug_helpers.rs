/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/utils/debug_helpers.rs
 *
 * This file provides safe, idiomatic Rust wrappers for the C-level logging
 * subsystem defined in `tk_logging.h`. It allows Rust code to seamlessly
 * log messages through the same centralized logging backend used by the C code.
 *
 * The primary goal is to abstract away the `unsafe` FFI calls and the manual
 * C-string conversions, offering a set of simple functions (`tk_log_info`,
 * `tk_log_error`, etc.) that feel natural to a Rust developer.
 *
 * The implementation uses macros to capture the file and line number where the
 * log was called, mirroring the behavior of the C `TK_LOG_*` macros and
 * providing rich debugging context.
 *
 * Dependencies:
 *   - crate::ffi: For the raw C function bindings.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use super::ffi;
use std::ffi::CString;

/// The internal, unsafe core logging function.
///
/// This function is not intended to be called directly. Use the public logging
/// macros instead.
#[doc(hidden)]
pub fn _log(
    level: ffi::tk_log_level_t,
    file: &str,
    line: u32,
    func: &str,
    args: std::fmt::Arguments,
) {
    // Format the message in Rust.
    let msg = std::fmt::format(args);

    // Convert all metadata to C-compatible strings.
    let c_msg = CString::new(msg).unwrap_or_else(|_| CString::new("Log message contained null bytes").unwrap());
    let c_file = CString::new(file).unwrap_or_else(|_| CString::new("?").unwrap());
    let c_func = CString::new(func).unwrap_or_else(|_| CString::new("?").unwrap());

    // Unsafe call to the C logging function.
    unsafe {
        ffi::tk_log_message(
            level,
            c_file.as_ptr(),
            line as std::os::raw::c_int,
            c_func.as_ptr(),
            c_msg.as_ptr(),
        );
    }
}

/// Logs a message at the TRACE level.
#[macro_export]
macro_rules! tk_log_trace {
    ($($arg:tt)*) => ($crate::debug_helpers::_log($crate::ffi::tk_log_level_t::TK_LOG_LEVEL_TRACE, file!(), line!(), module_path!(), format_args!($($arg)*)))
}

/// Logs a message at the DEBUG level.
#[macro_export]
macro_rules! tk_log_debug {
    ($($arg:tt)*) => ($crate::debug_helpers::_log($crate::ffi::tk_log_level_t::TK_LOG_LEVEL_DEBUG, file!(), line!(), module_path!(), format_args!($($arg)*)))
}

/// Logs a message at the INFO level.
#[macro_export]
macro_rules! tk_log_info {
    ($($arg:tt)*) => ($crate::debug_helpers::_log($crate::ffi::tk_log_level_t::TK_LOG_LEVEL_INFO, file!(), line!(), module_path!(), format_args!($($arg)*)))
}

/// Logs a message at the WARN level.
#[macro_export]
macro_rules! tk_log_warn {
    ($($arg:tt)*) => ($crate::debug_helpers::_log($crate::ffi::tk_log_level_t::TK_LOG_LEVEL_WARN, file!(), line!(), module_path!(), format_args!($($arg)*)))
}

/// Logs a message at the ERROR level.
#[macro_export]
macro_rules! tk_log_error {
    ($($arg:tt)*) => ($crate::debug_helpers::_log($crate::ffi::tk_log_level_t::TK_LOG_LEVEL_ERROR, file!(), line!(), module_path!(), format_args!($($arg)*)))
}

/// Logs a message at the FATAL level.
#[macro_export]
macro_rules! tk_log_fatal {
    ($($arg:tt)*) => ($crate::debug_helpers::_log($crate::ffi::tk_log_level_t::TK_LOG_LEVEL_FATAL, file!(), line!(), module_path!(), format_args!($($arg)*)))
}
