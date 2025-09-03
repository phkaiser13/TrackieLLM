/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: lib.rs
 *
 * This file is the main library entry point and FFI boundary for the `profiling`
 * crate. It exposes a C-compatible API to control the profiler, define measurement
 * scopes, and retrieve detailed statistical reports. It provides a C-idiomatic
 * start/end scope API that maps to the underlying RAII-based profiler core.
 *
 * Dependencies:
 *  - `lazy_static`: For thread-local scope stacks.
 *  - `serde_json`: For serializing the final report.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Declarations and Imports ---
pub mod metrics_collector;
pub mod profiler_core;

use metrics_collector::MetricsCollector;
use profiler_core::ProfileGuard;
use std::ffi::{c_char, CStr, CString};
use std::panic::{self, AssertUnwindSafe};
use std::time::Instant;

// --- FFI-Specific State for Scope Management ---

// To provide a C-friendly start/end API, we need to manage a stack of
// scopes on a per-thread basis. `thread_local!` is perfect for this.
thread_local! {
    static SCOPE_STACK: std::cell::RefCell<Vec<(&'static str, Instant)>> = std::cell::RefCell::new(Vec::new());
}

// --- FFI Helper Functions ---

fn catch_panic<F, R>(f: F) -> R where F: FnOnce() -> R, R: Default {
    panic::catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| R::default())
}

// --- FFI Public Interface ---

/// Enables the profiler globally.
#[no_mangle]
pub extern "C" fn profiling_enable() {
    catch_panic(|| {
        profiler_core::enable();
        0 // Return value for catch_panic
    });
}

/// Disables the profiler globally.
#[no_mangle]
pub extern "C" fn profiling_disable() {
    catch_panic(|| {
        profiler_core::disable();
        0
    });
}

/// Clears all collected profiling data.
#[no_mangle]
pub extern "C" fn profiling_clear() {
    catch_panic(|| {
        profiler_core::clear();
        0
    });
}

/// Marks the beginning of a named scope for profiling.
/// This pushes the scope name and current time onto a thread-local stack.
///
/// # Safety
/// The `name_c` pointer must be a valid, null-terminated C-string with a
/// 'static lifetime. This is a strong requirement; the string must be a
/// compile-time constant or live for the entire program duration.
#[no_mangle]
pub extern "C" fn profiling_start_scope(name_c: *const c_char) {
    if !profiler_core::is_enabled() {
        return;
    }
    catch_panic(|| {
        let name = unsafe { CStr::from_ptr(name_c).to_str().unwrap() };
        // This is `unsafe` because we are trusting the C caller about the lifetime.
        let static_name: &'static str = unsafe { std::mem::transmute(name) };

        SCOPE_STACK.with(|stack| {
            stack.borrow_mut().push((static_name, Instant::now()));
        });
        0
    });
}

/// Marks the end of the most recently started scope.
/// This pops the scope from the thread-local stack, calculates the duration,
/// and records the measurement.
#[no_mangle]
pub extern "C" fn profiling_end_scope() {
    if !profiler_core::is_enabled() {
        return;
    }
    catch_panic(|| {
        SCOPE_STACK.with(|stack| {
            if let Some((name, start_time)) = stack.borrow_mut().pop() {
                let elapsed_micros = start_time.elapsed().as_micros() as u64;
                // This re-uses the same recording logic as the RAII guard, but we
                // can't use the guard itself due to the C API structure. We have to
                // manually implement the recording logic here.
                // This is a simplified version of the logic in ProfileGuard::drop.
                let state_data = profiler_core::PROFILER_STATE.lock().data;
                let hist = state_data
                    .entry(name)
                    .or_insert_with(|| parking_lot::Mutex::new(hdrhistogram::Histogram::new(3).unwrap()));
                let _ = hist.lock().record(elapsed_micros);
            } else {
                eprintln!("Profiling warning: profiling_end_scope called without a matching start_scope.");
            }
        });
        0
    });
}

/// Generates a full statistical report of all collected profiling data.
///
/// # Returns
/// A C-string containing a JSON representation of the `ProfilingReport`.
/// This string must be freed by the caller using `profiling_free_string`.
/// Returns null on error.
#[no_mangle]
pub extern "C" fn profiling_get_report_json() -> *mut c_char {
    catch_panic(|| {
        let collector = MetricsCollector::new();
        let report = collector.collect_and_generate_report();

        match serde_json::to_string_pretty(&report) {
            Ok(json_str) => CString::new(json_str).unwrap().into_raw(),
            Err(e) => {
                eprintln!("Failed to serialize profiling report to JSON: {}", e);
                std::ptr::null_mut()
            }
        }
    })
}

/// Frees a C-string that was allocated by this Rust library.
#[no_mangle]
pub extern "C" fn profiling_free_string(s: *mut c_char) {
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
    fn test_ffi_scope_markers() {
        profiling_clear();
        profiling_enable();

        // Use CString to manage the lifetime, then get a pointer.
        // The `transmute` in `start_scope` is unsafe, but this is how we test it.
        let scope_name = CString::new("c_test_scope").unwrap();

        profiling_start_scope(scope_name.as_ptr());
        std::thread::sleep(std::time::Duration::from_millis(10));
        profiling_end_scope();

        let report_ptr = profiling_get_report_json();
        let report_str = unsafe { CStr::from_ptr(report_ptr).to_str().unwrap() };

        assert!(report_str.contains("c_test_scope"));

        profiling_free_string(report_ptr);
        profiling_disable();
    }
}
