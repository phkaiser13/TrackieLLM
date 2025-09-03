/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/profiling/profiler_core.rs
 *
 * This file implements the core components of the profiling framework. It
 * includes the main `Profiler` service, the `ProfiledThread` RAII guard for
 * instrumenting code blocks, and the `ProfilingReport` for presenting the
 * collected data.
 *
 * The design is centered around the RAII pattern. A developer instruments a
 * scope by creating a `ProfiledThread` guard at the beginning of it. The guard
 * records the start time. When the guard goes out of scope at the end of the
 * block, its `Drop` implementation is automatically called, which records the
 * end time and submits the completed sample to the global `Profiler` instance.
 * This makes instrumentation clean and exception-safe.
 *
 * The `Profiler` itself is a singleton-like object that collects samples from
 * all threads in a thread-safe manner.
 *
 * Dependencies:
 *   - crate::ProfilingError: For shared error handling.
 *   - log: For structured logging.
 *   - thiserror: For ergonomic error handling.
 *   - parking_lot::Mutex: A more performant mutex implementation than the
 *     standard library's, suitable for a high-contention profiler.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Represents a single completed profiling sample for a named scope.
#[derive(Debug, Clone)]
pub struct ProfileSample {
    /// The name of the thread this sample was recorded on.
    pub thread_name: String,
    /// The duration of the profiled scope.
    pub duration: Duration,
}

/// The internal state of the profiler, shared across all threads.
#[derive(Debug, Default)]
struct ProfilerState {
    /// The time the profiling session started.
    start_time: Option<Instant>,
    /// A collection of all recorded samples.
    samples: Vec<ProfileSample>,
}

/// The main profiler service.
///
/// This struct acts as a handle to the global profiler state. It can be cloned
/// and passed between threads.
#[derive(Debug, Clone, Default)]
pub struct Profiler {
    /// The shared state, protected by a high-performance mutex.
    state: Arc<Mutex<ProfilerState>>,
}

/// Represents errors that can occur within the profiler core.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CoreError {
    /// An operation was attempted when the profiler was not active.
    #[error("The profiler is not currently running.")]
    NotRunning,
}

impl Profiler {
    /// Creates a new `Profiler` and starts a profiling session.
    pub fn start() -> Self {
        log::info!("Starting profiling session.");
        let profiler = Self::default();
        profiler.state.lock().start_time = Some(Instant::now());
        profiler
    }

    /// Stops the profiling session and returns a report of the collected data.
    pub fn stop(self) -> ProfilingReport {
        log::info!("Stopping profiling session.");
        let mut state = self.state.lock();
        let samples = std::mem::take(&mut state.samples);
        let duration = state.start_time.map_or(Duration::default(), |s| s.elapsed());
        
        ProfilingReport {
            total_duration: duration,
            samples,
        }
    }

    /// Creates a RAII guard to profile the current thread's scope.
    ///
    /// # Arguments
    /// * `thread_name` - A descriptive name for the current thread or scope.
    pub fn profile_thread(&self, thread_name: &'static str) -> ProfiledThread {
        ProfiledThread {
            thread_name,
            start_time: Instant::now(),
            state: self.state.clone(),
        }
    }
}

/// A RAII guard that records a `ProfileSample` when it is dropped.
///
/// Creating a `ProfiledThread` marks the beginning of a profiled scope.
/// When the guard goes out of scope, its `Drop` implementation records the
/// duration and submits it to the `Profiler`.
pub struct ProfiledThread {
    thread_name: &'static str,
    start_time: Instant,
    state: Arc<Mutex<ProfilerState>>,
}

impl Drop for ProfiledThread {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed();
        let sample = ProfileSample {
            thread_name: self.thread_name.to_string(),
            duration,
        };

        let mut state = self.state.lock();
        // Only record the sample if the profiler is still running.
        if state.start_time.is_some() {
            state.samples.push(sample);
        }
    }
}

/// A report containing the results of a profiling session.
pub struct ProfilingReport {
    /// The total duration of the profiling session.
    pub total_duration: Duration,
    /// All the individual samples collected during the session.
    pub samples: Vec<ProfileSample>,
}

impl ProfilingReport {
    /// Generates a human-readable summary of the profiling report.
    pub fn generate_summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str(&format!(
            "Profiling Report (Total Duration: {:.2?})\n",
            self.total_duration
        ));
        summary.push_str("========================================\n");

        if self.samples.is_empty() {
            summary.push_str("No samples were recorded.\n");
            return summary;
        }

        // Aggregate samples by thread name.
        let mut aggregated: HashMap<String, (Duration, u32)> = HashMap::new();
        for sample in &self.samples {
            let entry = aggregated.entry(sample.thread_name.clone()).or_default();
            entry.0 += sample.duration;
            entry.1 += 1;
        }

        let mut sorted_threads: Vec<_> = aggregated.into_iter().collect();
        // Sort by total time spent, descending.
        sorted_threads.sort_by(|a, b| b.1.0.cmp(&a.1.0));
        
        summary.push_str(&format!(
            "{:<20} | {:<15} | {:<10} | {:<15}\n",
            "Thread Name", "Total Time", "Samples", "Avg. Time/Sample"
        ));
        summary.push_str(&"-".repeat(70));
        summary.push('\n');

        for (name, (total_time, count)) in sorted_threads {
            let avg_time = if count > 0 {
                total_time / count
            } else {
                Duration::default()
            };
            summary.push_str(&format!(
                "{:<20} | {:<15.2?} | {:<10} | {:<15.2?}\n",
                name, total_time, count, avg_time
            ));
        }

        summary
    }
}
