/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: metrics_comparator.rs
 *
 * This file provides a robust framework for the statistical comparison of performance
 * metrics from different experiments. It is designed to go beyond simple mean
 * comparisons by employing statistical tests (like Welch's t-test) to determine
 * if observed differences are statistically significant. This is essential for
 * making data-driven decisions in A/B testing scenarios.
 *
 * Dependencies:
 *  - `statrs`: For core statistical functions (mean, std_dev, t-distribution).
 *  - `serde`: For deserializing experiment results.
 *  - `thiserror`: For structured error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Imports ---
use serde::Deserialize;
use statrs::distribution::{StudentsT, ContinuousCDF};
use statrs::statistics::{Data, Mean, OrderStatistics, std_dev};
use std::collections::HashMap;

// --- Custom Error and Result Types ---

/// Represents errors that can occur during metric comparison.
#[derive(Debug, thiserror::Error)]
pub enum ComparatorError {
    #[error("Metric '{metric_name}' is present in one experiment but missing in the other.")]
    MismatchedMetrics { metric_name: String },
    #[error("Not enough data for metric '{metric_name}' to perform a statistical test. Required: at least 2 data points per set.")]
    InsufficientData { metric_name: String },
    #[error("Failed to deserialize experiment data: {0}")]
    DeserializationError(#[from] serde_json::Error),
}

type ComparatorResult<T> = Result<T, ComparatorError>;

// --- Core Data Structures ---

/// A set of observations for a single named metric.
#[derive(Deserialize, Debug, Clone)]
pub struct MetricSet {
    pub name: String,
    pub values: Vec<f64>,
}

/// Represents the complete results from a single experiment run,
/// containing multiple metric sets.
#[derive(Deserialize, Debug, Clone)]
pub struct ExperimentResults {
    #[serde(flatten)]
    pub metrics: HashMap<String, Vec<f64>>,
}

/// A detailed report comparing a single metric between two experiments.
#[derive(Debug, Clone)]
pub struct ComparisonReport {
    pub metric_name: String,
    // Baseline (A) stats
    pub baseline_mean: f64,
    pub baseline_std_dev: f64,
    pub baseline_n: usize,
    // Candidate (B) stats
    pub candidate_mean: f64,
    pub candidate_std_dev: f64,
    pub candidate_n: usize,
    // Comparison stats
    pub percentage_change: f64,
    pub t_statistic: f64,
    pub p_value: f64,
    pub degrees_of_freedom: f64,
    pub is_significant: bool,
}

// --- Metrics Comparison Service ---

/// A service for performing statistical comparisons between experiment results.
#[derive(Default)]
pub struct MetricsComparator;

impl MetricsComparator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compares two sets of experiment results and generates a report for each common metric.
    ///
    /// # Arguments
    /// * `baseline` - The results of the control/baseline experiment (A).
    /// * `candidate` - The results of the new/candidate experiment (B).
    /// * `alpha` - The significance level (e.g., 0.05) for the statistical test.
    ///
    /// # Returns
    /// A map of `ComparisonReport`s, keyed by metric name.
    pub fn compare(
        &self,
        baseline: &ExperimentResults,
        candidate: &ExperimentResults,
        alpha: f64,
    ) -> ComparatorResult<HashMap<String, ComparisonReport>> {
        let mut reports = HashMap::new();
        let baseline_keys: std::collections::HashSet<_> = baseline.metrics.keys().collect();
        let candidate_keys: std::collections::HashSet<_> = candidate.metrics.keys().collect();

        // Find common metrics to compare
        let common_keys = baseline_keys.intersection(&candidate_keys);

        for key in common_keys {
            let metric_name = (*key).clone();
            let baseline_values = baseline.metrics.get(&metric_name).unwrap();
            let candidate_values = candidate.metrics.get(&metric_name).unwrap();

            // 1. Check for sufficient data.
            if baseline_values.len() < 2 || candidate_values.len() < 2 {
                return Err(ComparatorError::InsufficientData { metric_name });
            }

            // 2. Calculate descriptive statistics.
            let baseline_data = Data::new(baseline_values.as_slice());
            let candidate_data = Data::new(candidate_values.as_slice());

            let baseline_mean = baseline_data.mean().unwrap();
            let candidate_mean = candidate_data.mean().unwrap();

            let baseline_std_dev = std_dev(baseline_values.as_slice()).unwrap();
            let candidate_std_dev = std_dev(candidate_values.as_slice()).unwrap();

            let baseline_n = baseline_values.len() as f64;
            let candidate_n = candidate_values.len() as f64;

            // 3. Perform Welch's t-test (for independent samples with possibly unequal variances).
            let baseline_var = baseline_std_dev.powi(2);
            let candidate_var = candidate_std_dev.powi(2);

            let t_statistic = (baseline_mean - candidate_mean) /
                              (baseline_var / baseline_n + candidate_var / candidate_n).sqrt();

            // Satterthwaite approximation for degrees of freedom.
            let df_num = (baseline_var / baseline_n + candidate_var / candidate_n).powi(2);
            let df_den = (baseline_var / baseline_n).powi(2) / (baseline_n - 1.0) +
                         (candidate_var / candidate_n).powi(2) / (candidate_n - 1.0);
            let degrees_of_freedom = df_num / df_den;

            // 4. Calculate the p-value from the t-distribution.
            let t_dist = StudentsT::new(0.0, 1.0, degrees_of_freedom).unwrap();
            // Two-tailed test: P(T <= -|t|) + P(T >= |t|) = 2 * P(T <= -|t|)
            let p_value = 2.0 * t_dist.cdf(-t_statistic.abs());

            // 5. Build the report.
            let report = ComparisonReport {
                metric_name: metric_name.clone(),
                baseline_mean,
                baseline_std_dev,
                baseline_n: baseline_values.len(),
                candidate_mean,
                candidate_std_dev,
                candidate_n: candidate_values.len(),
                percentage_change: (candidate_mean - baseline_mean) / baseline_mean * 100.0,
                t_statistic,
                p_value,
                degrees_of_freedom,
                is_significant: p_value < alpha,
            };
            reports.insert(metric_name, report);
        }

        Ok(reports)
    }
}

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn get_test_data() -> (ExperimentResults, ExperimentResults) {
        let baseline_results = ExperimentResults {
            metrics: HashMap::from([
                ("latency_ms".to_string(), vec![100.0, 105.0, 110.0, 98.0, 102.0]),
                ("accuracy".to_string(), vec![0.91, 0.92, 0.915, 0.925]),
            ]),
        };
        let candidate_results = ExperimentResults {
            metrics: HashMap::from([
                // Significantly lower latency
                ("latency_ms".to_string(), vec![80.0, 85.0, 82.0, 78.0, 81.0]),
                // Similar accuracy
                ("accuracy".to_string(), vec![0.912, 0.921, 0.918, 0.923]),
            ]),
        };
        (baseline_results, candidate_results)
    }

    #[test]
    fn test_comparison_report_generation() {
        let comparator = MetricsComparator::new();
        let (baseline, candidate) = get_test_data();
        let reports = comparator.compare(&baseline, &candidate, 0.05).unwrap();

        // --- Check Latency Report ---
        let latency_report = reports.get("latency_ms").unwrap();
        assert_eq!(latency_report.metric_name, "latency_ms");
        assert_relative_eq!(latency_report.baseline_mean, 103.0, epsilon = 1e-9);
        assert_relative_eq!(latency_report.candidate_mean, 81.2, epsilon = 1e-9);
        assert_relative_eq!(latency_report.percentage_change, -21.165, epsilon = 1e-3);
        assert!(latency_report.is_significant, "Latency change should be significant");

        // --- Check Accuracy Report ---
        let accuracy_report = reports.get("accuracy").unwrap();
        assert!(!accuracy_report.is_significant, "Accuracy change should not be significant");
    }

    #[test]
    fn test_insufficient_data_error() {
        let comparator = MetricsComparator::new();
        let baseline = ExperimentResults {
            metrics: HashMap::from([("latency".to_string(), vec![100.0])]), // Only one data point
        };
        let candidate = ExperimentResults {
            metrics: HashMap::from([("latency".to_string(), vec![80.0, 85.0])]),
        };

        let result = comparator.compare(&baseline, &candidate, 0.05);
        assert!(matches!(result, Err(ComparatorError::InsufficientData { .. })));
    }

    #[test]
    fn test_deserialize_from_json() {
        let json_data = r#"{
            "latency": [10.1, 10.2, 10.3],
            "cpu_usage": [45.5, 46.1, 44.9]
        }"#;
        let results: Result<ExperimentResults, _> = serde_json::from_str(json_data);
        assert!(results.is_ok());
        let exp_results = results.unwrap();
        assert_eq!(exp_results.metrics.get("latency").unwrap().len(), 3);
    }
}
