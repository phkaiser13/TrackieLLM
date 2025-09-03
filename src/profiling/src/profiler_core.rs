/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: profiler_core.rs
 *
 * This file implements the core of a lightweight, high-performance, in-application
 * profiler. It uses an RAII guard (`ProfileGuard`) pattern for easy and accurate
 * measurement of code scopes. The collected data is stored in high-dynamic-range
 * (HDR) histograms, which allows for accurate analysis of latency distributions
 * with low overhead. The entire profiler can be globally enabled or disabled
 * at runtime.
 *
 * Dependencies:
 *  - `hdrhistogram`: For recording latency measurements.
 *  - `parking_lot`: For a high-performance, thread-safe Mutex.
 *  - `lazy_static`: For the global profiler state.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Imports ---
use hdrhistogram::Histogram;
use lazy_static::lazy_static;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

// --- Global Profiler State ---

/// The central data store for all profiling data.
/// It maps a scope name to a histogram of its execution times in microseconds.
pub type ProfilerData = HashMap<&'static str, Mutex<Histogram<u64>>>;

/// The global profiler state, encapsulating the data and an enable/disable flag.
struct ProfilerState {
    is_enabled: AtomicBool,
    data: Mutex<ProfilerData>,
}

lazy_static! {
    static ref PROFILER_STATE: ProfilerState = ProfilerState {
        is_enabled: AtomicBool::new(false), // Disabled by default
        data: Mutex::new(HashMap::new()),
    };
}

// --- Public Control Functions ---

/// Enables the profiler globally.
pub fn enable() {
    PROFILER_STATE.is_enabled.store(true, Ordering::Relaxed);
    println!("Profiler enabled.");
}

/// Disables the profiler globally.
/// When disabled, `ProfileGuard` creation is nearly a zero-cost operation.
pub fn disable() {
    PROFILER_STATE.is_enabled.store(false, Ordering::Relaxed);
    println!("Profiler disabled.");
}

/// Clears all collected profiling data.
pub fn clear() {
    PROFILER_STATE.lock().data.clear();
    println!("Profiler data cleared.");
}

/// Checks if the profiler is currently enabled.
pub fn is_enabled() -> bool {
    PROFILER_STATE.is_enabled.load(Ordering::Relaxed)
}

/// Provides read-only access to the collected profiler data.
///
/// This function clones the current state of the histograms for analysis,
/// allowing the profiler to continue running with minimal contention.
pub fn get_data() -> HashMap<&'static str, Histogram<u64>> {
    let state_data = PROFILER_STATE.lock().data;
    state_data
        .iter()
        .map(|(&name, mutex_hist)| (name, mutex_hist.lock().clone()))
        .collect()
}


// --- RAII Profiling Guard ---

/// An RAII guard for profiling a scope.
/// When created, it records a start time. When it goes out of scope (is dropped),
/// it calculates the elapsed time and records it in the global profiler state.
pub struct ProfileGuard {
    name: &'static str,
    start: Instant,
}

impl ProfileGuard {
    /// Creates a new `ProfileGuard` for the given scope name.
    ///
    /// If the profiler is disabled, this function returns immediately, making
    /// the overhead of disabled profiling very low.
    #[inline]
    pub fn new(name: &'static str) -> Option<Self> {
        if is_enabled() {
            Some(Self {
                name,
                start: Instant::now(),
            })
        } else {
            None
        }
    }
}

impl Drop for ProfileGuard {
    /// Executed when the guard goes out of scope. Records the measurement.
    fn drop(&mut self) {
        let elapsed_micros = self.start.elapsed().as_micros() as u64;

        let state_data = PROFILER_STATE.lock().data;

        // Check if a histogram for this scope already exists.
        if let Some(hist_mutex) = state_data.get(self.name) {
            // If it exists, lock and record.
            if let Err(e) = hist_mutex.lock().record(elapsed_micros) {
                eprintln!("Error recording value in histogram for '{}': {}", self.name, e);
            }
        } else {
            // If it doesn't exist, we need to re-lock for writing to create it.
            // This is a simple approach; a more complex one might use `dashmap`
            // or other concurrent hash maps to avoid the re-lock.
            drop(state_data); // Drop the read guard
            let mut state_data_write = PROFILER_STATE.lock().data;
            let hist = state_data_write
                .entry(self.name)
                .or_insert_with(|| Mutex::new(Histogram::new(3).unwrap()));

            if let Err(e) = hist.lock().record(elapsed_micros) {
                 eprintln!("Error recording value in new histogram for '{}': {}", self.name, e);
            }
        }
    }
}

/// A macro to simplify the creation of `ProfileGuard`.
///
/// # Example
///
/// ```rust,ignore
/// fn my_function() {
///     profile_scope!("my_function");
///     // ... code to be measured ...
/// }
/// ```
#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {
        // The `let _` is important. It ensures the guard is not dropped immediately.
        let _profile_guard = $crate::profiler_core::ProfileGuard::new($name);
    };
}


// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_profiler_enable_disable_clear() {
        assert!(!is_enabled());
        enable();
        assert!(is_enabled());
        disable();
        assert!(!is_enabled());

        enable();
        {
            let _guard = ProfileGuard::new("test_scope");
            thread::sleep(std::time::Duration::from_millis(5));
        }

        let data = get_data();
        assert_eq!(data.len(), 1);
        assert!(data.contains_key("test_scope"));

        clear();
        let data_after_clear = get_data();
        assert!(data_after_clear.is_empty());

        disable();
    }

    #[test]
    fn test_profile_guard_records_data() {
        clear();
        enable();

        {
            let _guard = ProfileGuard::new("timed_block");
            thread::sleep(std::time::Duration::from_micros(150));
        }

        let data = get_data();
        let hist = data.get("timed_block").unwrap();

        assert_eq!(hist.len(), 1);
        // The value should be around 150 microseconds.
        assert!(hist.value_at_quantile(0.5) >= 150);
        assert!(hist.value_at_quantile(0.5) < 250); // Give some leeway

        disable();
    }

    #[test]
    fn test_profiler_is_noop_when_disabled() {
        clear();
        disable();

        {
            let _guard = ProfileGuard::new("disabled_scope");
        }

        let data = get_data();
        assert!(data.is_empty());
    }

    #[test]
    fn test_profile_scope_macro() {
        clear();
        enable();

        fn profiled_function() {
            profile_scope!("profiled_function");
            // Work...
        }
        profiled_function();

        let data = get_data();
        assert!(data.contains_key("profiled_function"));

        disable();
    }
}
