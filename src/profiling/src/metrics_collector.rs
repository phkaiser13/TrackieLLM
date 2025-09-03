/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: metrics_collector.rs
 *
 * This file implements the analysis and reporting component of the profiler. It
 * is responsible for taking the raw histogram data collected by the `profiler_core`
 * and transforming it into a structured, human-readable report containing meaningful
 * statistical analysis for each profiled scope, including key latency percentiles.
 *
 * Dependencies:
 *  - `hdrhistogram`: For reading statistics from the histogram objects.
 *  - `serde`: For making the final report serializable.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Imports ---
use crate::profiler_core;
use hdrhistogram::Histogram;
use serde::Serialize;

// --- Report Data Structures ---

/// Represents a single percentile measurement.
#[derive(Serialize, Debug, Clone)]
pub struct Percentile {
    /// The quantile (e.g., 0.5 for p50, 0.99 for p99).
    pub quantile: f64,
    /// The value at this percentile, in microseconds.
    pub value_us: u64,
}

/// A comprehensive statistical report for a single profiled scope.
#[derive(Serialize, Debug, Clone)]
pub struct ScopeReport {
    pub scope_name: &'static str,
    /// The total number of times this scope was measured.
    pub count: u64,
    /// The average execution time in microseconds.
    pub mean_us: f64,
    /// The standard deviation of execution time in microseconds.
    pub std_dev_us: f64,
    /// The minimum observed execution time in microseconds.
    pub min_us: u64,
    /// The maximum observed execution time in microseconds.
    pub max_us: u64,
    /// A list of key percentile measurements.
    pub percentiles: Vec<Percentile>,
}

/// The top-level report containing an analysis for all profiled scopes.
#[derive(Serialize, Debug, Clone)]
pub struct ProfilingReport {
    pub report_timestamp_utc: String,
    pub scopes: Vec<ScopeReport>,
}

// --- Metrics Collector Service ---

/// A service for collecting raw profiler data and generating structured reports.
#[derive(Default)]
pub struct MetricsCollector;

impl MetricsCollector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Collects the current profiler data and generates a full statistical report.
    ///
    /// This function fetches the raw histograms, iterates through them, and calculates
    /// key statistics for each one, compiling the results into a `ProfilingReport`.
    pub fn collect_and_generate_report(&self) -> ProfilingReport {
        let raw_data = profiler_core::get_data();
        let mut scopes = Vec::with_capacity(raw_data.len());

        for (name, hist) in raw_data {
            if hist.is_empty() {
                continue;
            }
            scopes.push(self.analyze_histogram(name, &hist));
        }

        // Sort the report by scope name for consistent output.
        scopes.sort_by(|a, b| a.scope_name.cmp(&b.scope_name));

        ProfilingReport {
            report_timestamp_utc: chrono::Utc::now().to_rfc3339(),
            scopes,
        }
    }

    /// Analyzes a single histogram and produces a `ScopeReport`.
    fn analyze_histogram(&self, scope_name: &'static str, hist: &Histogram<u64>) -> ScopeReport {
        let percentiles_to_check = [50.0, 90.0, 95.0, 99.0, 99.9];
        let mut percentiles = Vec::with_capacity(percentiles_to_check.len());

        for p in percentiles_to_check {
            percentiles.push(Percentile {
                quantile: p / 100.0,
                value_us: hist.value_at_percentile(p),
            });
        }

        ScopeReport {
            scope_name,
            count: hist.len(),
            mean_us: hist.mean(),
            std_dev_us: hist.stdev(),
            min_us: hist.min(),
            max_us: hist.max(),
            percentiles,
        }
    }
}


// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiler_core::{self, ProfileGuard};

    #[test]
    fn test_report_generation_from_profiler_data() {
        // 1. Setup: Clear any previous data and enable the profiler.
        profiler_core::clear();
        profiler_core::enable();

        // 2. Action: Run some profiled code.
        for _ in 0..100 {
            let _guard = ProfileGuard::new("fast_operation");
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
        for _ in 0..10 {
            let _guard = ProfileGuard::new("slow_operation");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        // 3. Collect and generate the report.
        let collector = MetricsCollector::new();
        let report = collector.collect_and_generate_report();

        // 4. Assert: Check the report's integrity.
        assert_eq!(report.scopes.len(), 2);

        let fast_op_report = report.scopes.iter().find(|s| s.scope_name == "fast_operation").unwrap();
        assert_eq!(fast_op_report.count, 100);
        assert!(fast_op_report.mean_us > 90.0 && fast_op_report.mean_us < 200.0);
        assert_eq!(fast_op_report.percentiles.len(), 5);
        assert_eq!(fast_op_report.percentiles[0].quantile, 0.5); // p50

        let slow_op_report = report.scopes.iter().find(|s| s.scope_name == "slow_operation").unwrap();
        assert_eq!(slow_op_report.count, 10);
        assert!(slow_op_report.mean_us > 4900.0 && slow_op_report.mean_us < 6000.0); // Around 5000 us

        // 5. Teardown
        profiler_core::disable();
        profiler_core::clear();
    }

    #[test]
    fn test_empty_report_when_no_data() {
        profiler_core::clear();
        profiler_core::enable();

        let collector = MetricsCollector::new();
        let report = collector.collect_and_generate_report();

        assert!(report.scopes.is_empty());

        profiler_core::disable();
    }
}
