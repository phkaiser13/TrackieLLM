/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: lib.rs
 *
 * This file is the main library entry point and FFI boundary for the `logging_ext`
 * crate. It exposes capabilities for structured JSON formatting and security audit
 * logging to the C/C++ core. The design allows the C side to leverage Rust's
 * powerful formatting and data structuring capabilities while integrating with an
 * existing C-based logging infrastructure if needed.
 *
 * Dependencies:
 *  - `lazy_static`: For thread-safe initialization of the global manager.
 *  - `log`: For creating `log::Record` instances for the formatter.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Declarations and Imports ---
pub mod audit_helpers;
pub mod event_formatter;

use audit_helpers::{AuditEvent, AuditLogger};
use event_formatter::JsonFormatter;
use lazy_static::lazy_static;
use log::{Level, Record};
use serde::Serialize;
use std::ffi::{c_char, CStr, CString};
use std::panic::{self, AssertUnwindSafe};
use std::sync::Mutex;

// --- Global State Management ---

/// A manager struct to hold instances of our logging services.
struct LoggingManager {
    formatter: JsonFormatter,
    audit_logger: AuditLogger,
}

impl Default for LoggingManager {
    fn default() -> Self {
        Self {
            formatter: JsonFormatter::new(),
            // Default target for audit logs. Can be reconfigured via FFI.
            audit_logger: AuditLogger::new("audit_trail"),
        }
    }
}

lazy_static! {
    static ref MANAGER: Mutex<LoggingManager> = Mutex::new(LoggingManager::default());
}

// --- FFI Helper Functions ---

fn catch_panic<F, R>(f: F) -> R where F: FnOnce() -> R, R: Default {
    panic::catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| {
        eprintln!("Error: A panic occurred within the Rust FFI boundary.");
        R::default()
    })
}

fn serialize_to_c_string<T: Serialize>(data: &T) -> *mut c_char {
    match serde_json::to_string(data) {
        Ok(json_str) => CString::new(json_str).unwrap().into_raw(),
        Err(e) => {
            eprintln!("Error: Failed to serialize response to JSON: {}", e);
            std::ptr::null_mut()
        }
    }
}

// --- FFI Public Interface ---

/// Reconfigures the target for the audit logger.
///
/// # Arguments
/// - `target_c`: A C-string with the new target for audit logs (e.g., "secure_audit").
///
/// # Safety
/// The caller must provide a valid, null-terminated C-string.
#[no_mangle]
pub extern "C" fn logging_ext_setup_audit_logger(target_c: *const c_char) {
    catch_panic(|| {
        let target = unsafe { CStr::from_ptr(target_c).to_str().unwrap_or("audit_trail") };
        let mut manager = MANAGER.lock().unwrap();
        manager.audit_logger = AuditLogger::new(target);
        0 // Return 0 for void-equivalent in catch_panic
    });
}

/// Logs a structured audit event provided as a JSON string.
/// The JSON can be a partial `AuditEvent`; the Rust side will add the timestamp and event_id.
///
/// # Arguments
/// - `event_json_c`: A C-string containing the JSON for the audit event.
///
/// # Returns
/// - `0` on success.
/// - `-1` on failure (e.g., JSON parsing error).
///
/// # Safety
/// The caller must provide a valid, null-terminated C-string with well-formed JSON.
#[no_mangle]
pub extern "C" fn logging_ext_log_audit_event(event_json_c: *const c_char) -> i32 {
    catch_panic(|| {
        let event_json = unsafe { CStr::from_ptr(event_json_c).to_str().unwrap() };

        // Deserialize into a temporary struct that captures the C-side data.
        #[derive(serde::Deserialize)]
        struct PartialEvent {
            actor: audit_helpers::Actor,
            action: String,
            object: audit_helpers::Object,
            outcome: audit_helpers::Outcome,
            details: Option<serde_json::Value>,
        }

        let partial: PartialEvent = match serde_json::from_str(event_json) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Failed to deserialize partial audit event: {}", e);
                return -1;
            }
        };

        // Create a full AuditEvent, allowing Rust to control timestamp and ID.
        let full_event = AuditEvent {
            event_id: format!("{}-{}", chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0), rand::random::<u32>()),
            timestamp: chrono::Utc::now(),
            actor: partial.actor,
            action: partial.action,
            object: partial.object,
            outcome: partial.outcome,
            details: partial.details,
        };

        let manager = MANAGER.lock().unwrap();
        manager.audit_logger.log(&full_event);
        0
    })
}

/// Formats a simple log message into a structured JSON string.
///
/// # Arguments
/// - `message_c`: The log message, which can include `key=value` pairs.
/// - `level_c`: The log level string (e.g., "INFO", "WARN").
/// - `target_c`: The log target string (e.g., "my_module::my_function").
///
/// # Returns
/// A pointer to a new C-string containing the formatted JSON log. This string must be
/// freed by the caller using `logging_ext_free_string`. Returns null on error.
///
/// # Safety
/// Caller must provide valid C-strings and free the returned string.
#[no_mangle]
pub extern "C" fn logging_ext_format_log_message(
    message_c: *const c_char,
    level_c: *const c_char,
    target_c: *const c_char,
) -> *mut c_char {
    catch_panic(|| {
        let message = unsafe { CStr::from_ptr(message_c).to_str().unwrap() };
        let level_str = unsafe { CStr::from_ptr(level_c).to_str().unwrap() };
        let target = unsafe { CStr::from_ptr(target_c).to_str().unwrap() };

        let level = level_str.parse::<Level>().unwrap_or(Level::Info);

        // Create a log::Record to pass to the formatter.
        let record = Record::builder()
            .args(format_args!("{}", message))
            .level(level)
            .target(target)
            .build();

        let manager = MANAGER.lock().unwrap();
        match manager.formatter.format(&record) {
            Ok(formatted_string) => CString::new(formatted_string).unwrap().into_raw(),
            Err(e) => {
                eprintln!("Error formatting log message: {}", e);
                std::ptr::null_mut()
            }
        }
    })
}

/// Frees a C-string that was allocated by this Rust library.
#[no_mangle]
pub extern "C" fn logging_ext_free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    unsafe {
        drop(CString::from_raw(s));
    }
}

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_format_message_and_free() {
        let message = CString::new("Login successful user_id=42").unwrap();
        let level = CString::new("INFO").unwrap();
        let target = CString::new("auth_service").unwrap();

        let result_ptr = logging_ext_format_log_message(message.as_ptr(), level.as_ptr(), target.as_ptr());
        assert!(!result_ptr.is_null());

        let result_str = unsafe { CStr::from_ptr(result_ptr).to_str().unwrap() };
        let json: serde_json::Value = serde_json::from_str(result_str).unwrap();

        assert_eq!(json["message"], "Login successful");
        assert_eq!(json["user_id"], 42);
        assert_eq!(json["level"], "INFO");

        logging_ext_free_string(result_ptr);
    }

    #[test]
    fn test_ffi_log_audit_event() {
        // This test ensures the FFI function can be called without panicking.
        // Verifying the log output would require a more complex test harness.
        let event_json = r#"{
            "actor": {"id": "c_user", "details": {}},
            "action": "test.action",
            "object": {"id": "c_resource", "type": "test"},
            "outcome": "success"
        }"#;
        let event_c = CString::new(event_json).unwrap();

        let result = logging_ext_log_audit_event(event_c.as_ptr());
        assert_eq!(result, 0);
    }
}
