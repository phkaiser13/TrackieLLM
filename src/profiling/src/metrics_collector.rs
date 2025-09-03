/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/profiling/metrics_collector.rs
 *
 * This file implements a specialized metrics collector for use during
 * profiling sessions. Unlike the general-purpose system monitor, this
 * collector is designed to sample fine-grained performance data at a high
 * frequency while the profiler is active.
 *
 * In a real-world implementation with deep OS integration, this collector
 * would interface with hardware performance counters (via `perf_event` on
 * Linux, `ETW` on Windows, etc.) to gather metrics such as:
 * - CPU cycles
 * - Instructions retired
 * - Cache misses (L1, L2, L3)
 * - Branch mispredictions
 * - Memory allocations and frees
 *
 * Since we cannot access these directly in this environment, this file provides
 * a mock implementation that establishes the architecture. It simulates the
 * collection of memory allocation metrics to demonstrate how such data would
 * be gathered and integrated into the final `ProfilingReport`.
 *
 * Dependencies:
 *   - log: For structured logging.
 *   - parking_lot::Mutex: For thread-safe data aggregation.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Holds aggregated performance metrics collected during a profiling session.
#[derive(Debug, Clone, Default)]
pub struct ProfiledMetrics {
    /// The total number of simulated memory allocations.
    pub total_allocations: u64,
    /// The total number of simulated memory deallocations.
    pub total_deallocations: u64,
    /// The net number of allocations (total_allocations - total_deallocations),
    /// which can help detect memory leaks.
    pub net_allocations: i64,
}

/// The core metrics collector service.
///
/// This service runs in a background thread, periodically sampling performance
/// data and aggregating it into a `ProfiledMetrics` struct.
pub struct MetricsCollector {
    /// The handle to the background collector thread.
    handle: Option<thread::JoinHandle<()>>,
    /// A flag to signal the collector thread to stop.
    stop_signal: Arc<AtomicBool>,
    /// The shared data structure where collected metrics are aggregated.
    metrics: Arc<Mutex<ProfiledMetrics>>,
}

impl MetricsCollector {
    /// Creates a new `MetricsCollector` and starts the background sampling thread.
    ///
    /// # Arguments
    /// * `sampling_interval` - The interval at which to sample performance metrics.
    pub fn start(sampling_interval: Duration) -> Self {
        let stop_signal = Arc::new(AtomicBool::new(false));
        let metrics = Arc::new(Mutex::new(ProfiledMetrics::default()));

        let thread_stop_signal = stop_signal.clone();
        let thread_metrics = metrics.clone();

        let handle = thread::spawn(move || {
            log::debug!("Profiling metrics collector thread started.");
            while !thread_stop_signal.load(Ordering::Relaxed) {
                // In a real implementation, this is where we would sample
                // hardware performance counters.
                Self::collect_sample(&thread_metrics);
                thread::sleep(sampling_interval);
            }
            log::debug!("Profiling metrics collector thread stopped.");
        });

        Self {
            handle: Some(handle),
            stop_signal,
            metrics,
        }
    }

    /// Stops the collector and returns the aggregated metrics.
    pub fn stop(mut self) -> ProfiledMetrics {
        self.stop_signal.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            // Wait for the collector thread to finish its last cycle.
            let _ = handle.join();
        }

        // Take ownership of the metrics from the Arc.
        // If the Arc has other strong references, it will clone the data.
        Arc::try_unwrap(self.metrics)
            .map(|mutex| mutex.into_inner())
            .unwrap_or_else(|arc| arc.lock().clone())
    }

    /// Simulates the collection of a single sample of performance data.
    fn collect_sample(metrics: &Arc<Mutex<ProfiledMetrics>>) {
        // --- Mock Implementation ---
        // This simulates the collection of memory allocation metrics.
        // A real implementation would hook into the global allocator or use
        // OS-level tools to get this data.

        let mut metrics_guard = metrics.lock();

        // Simulate a random number of allocations and deallocations per sample.
        let allocations_this_tick = (rand::random::<f32>() * 100.0) as u64;
        let deallocations_this_tick = (rand::random::<f32>() * 95.0) as u64; // Slightly fewer deallocations

        metrics_guard.total_allocations += allocations_this_tick;
        metrics_guard.total_deallocations += deallocations_this_tick;
        metrics_guard.net_allocations = metrics_guard.total_allocations as i64
            - metrics_guard.total_deallocations as i64;
    }
}

impl Drop for MetricsCollector {
    /// Ensures the background thread is stopped if the collector is dropped.
    fn drop(&mut self) {
        if self.handle.is_some() {
            self.stop_signal.store(true, Ordering::Relaxed);
            if let Some(handle) = self.handle.take() {
                // We don't want to panic in drop, so we just log if the join fails.
                if handle.join().is_err() {
                    log::error!("Profiling metrics collector thread panicked.");
                }
            }
        }
    }
}
