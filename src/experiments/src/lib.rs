/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: lib.rs
 *
 * This file is the main library entry point and FFI boundary for the `experiments`
 * crate. It exposes functions to the C/C++ core for performing statistical
 * comparison of experiment metrics and detailed analysis of model performance.
 * This FFI layer handles data serialization (JSON) and memory management for
 * strings passed across the C boundary.
 *
 * Dependencies:
 *  - `lazy_static`: For thread-safe initialization of the global manager.
 *  - `serde_json`: For serializing reports and deserializing inputs.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Declarations and Imports ---
pub mod metrics_comparator;
pub mod model_analysis;

use lazy_static::lazy_static;
use metrics_comparator::{ExperimentResults, MetricsComparator};
use model_analysis::{ModelAnalyzer, ModelPrediction, GroundTruth};
use serde::Serialize;
use std::ffi::{c_char, CStr, CString};
use std::os::raw::c_void;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Mutex;

// --- Global State Management ---

/// A manager struct to hold instances of our analysis and comparison services.
struct ExperimentsManager {
    comparator: MetricsComparator,
    analyzer: ModelAnalyzer,
}

lazy_static! {
    static ref EXPERIMENTS_MANAGER: Mutex<ExperimentsManager> = Mutex::new(ExperimentsManager {
        comparator: MetricsComparator::new(),
        analyzer: ModelAnalyzer::new(),
    });
}

// --- FFI Helper Functions ---

/// Helper to run a closure and catch any panics, returning a null pointer if a panic occurs.
/// This is crucial for preventing panics from unwinding across the FFI boundary, which is UB.
fn catch_panic<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
    R: Default,
{
    panic::catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| {
        eprintln!("Error: A panic occurred within the Rust FFI boundary.");
        R::default()
    })
}

/// Serializes a Rust struct into a C-compatible, null-terminated string.
/// The caller in the C code is responsible for freeing this string using `experiments_free_string`.
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

/// Compares metrics from two experiments provided as JSON strings.
///
/// # Arguments
/// - `baseline_json`: A C-string with the JSON for the baseline experiment results.
/// - `candidate_json`: A C-string with the JSON for the candidate experiment results.
/// - `alpha`: The significance level for the statistical test (e.g., 0.05).
///
/// # Returns
/// A pointer to a new C-string containing the JSON report. This string must be freed
/// by the caller using `experiments_free_string`. Returns a null pointer on error.
///
/// # Safety
/// The caller must provide valid, null-terminated C-strings and must free the returned string.
#[no_mangle]
pub extern "C" fn experiments_compare_metrics(
    baseline_json: *const c_char,
    candidate_json: *const c_char,
    alpha: f64,
) -> *mut c_char {
    catch_panic(|| {
        let baseline_str = unsafe { CStr::from_ptr(baseline_json).to_str().unwrap() };
        let candidate_str = unsafe { CStr::from_ptr(candidate_json).to_str().unwrap() };

        let baseline: ExperimentResults = match serde_json::from_str(baseline_str) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Error deserializing baseline JSON: {}", e);
                return std::ptr::null_mut();
            }
        };
        let candidate: ExperimentResults = match serde_json::from_str(candidate_str) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Error deserializing candidate JSON: {}", e);
                return std::ptr::null_mut();
            }
        };

        let manager = EXPERIMENTS_MANAGER.lock().unwrap();
        match manager.comparator.compare(&baseline, &candidate, alpha) {
            Ok(report) => serialize_to_c_string(&report),
            Err(e) => {
                eprintln!("Error comparing metrics: {}", e);
                std::ptr::null_mut()
            }
        }
    })
}

/// Analyzes a set of classification predictions against ground truth data.
///
/// # Arguments
/// - `predictions_json`: JSON string of `ModelPrediction::Classification` objects.
/// - `truth_json`: JSON string of `GroundTruth::Classification` objects.
///
/// # Returns
/// A C-string with the JSON `EvaluationReport`. Must be freed by the caller.
///
/// # Safety
/// Caller must provide valid JSON C-strings and free the returned string.
#[no_mangle]
pub extern "C" fn experiments_analyze_classification(
    predictions_json: *const c_char,
    truth_json: *const c_char,
) -> *mut c_char {
    catch_panic(|| {
        let predictions_str = unsafe { CStr::from_ptr(predictions_json).to_str().unwrap() };
        let truth_str = unsafe { CStr::from_ptr(truth_json).to_str().unwrap() };

        let predictions: Vec<ModelPrediction> = serde_json::from_str(predictions_str).unwrap();
        let ground_truth: Vec<GroundTruth> = serde_json::from_str(truth_str).unwrap();

        let manager = EXPERIMENTS_MANAGER.lock().unwrap();
        match manager.analyzer.analyze_classification(&predictions, &ground_truth) {
            Ok(report) => serialize_to_c_string(&report),
            Err(e) => {
                eprintln!("Error analyzing classification: {}", e);
                std::ptr::null_mut()
            }
        }
    })
}

/// Frees a C-string that was allocated by this Rust library.
/// This function is essential for preventing memory leaks on the C side.
///
/// # Safety
/// The pointer `s` must be a non-null pointer that was previously returned by a
/// function from this library. Calling it with any other pointer will lead to UB.
#[no_mangle]
pub extern "C" fn experiments_free_string(s: *mut c_char) {
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
    use serde_json::Value;

    #[test]
    fn test_ffi_compare_metrics_and_free() {
        let baseline_json = r#"{ "latency": [100.0, 105.0], "accuracy": [0.9] }"#;
        let candidate_json = r#"{ "latency": [80.0, 85.0], "accuracy": [0.91] }"#;

        let baseline_c = CString::new(baseline_json).unwrap();
        let candidate_c = CString::new(candidate_json).unwrap();

        let result_ptr = experiments_compare_metrics(baseline_c.as_ptr(), candidate_c.as_ptr(), 0.05);
        assert!(!result_ptr.is_null());

        let result_str = unsafe { CStr::from_ptr(result_ptr).to_str().unwrap() };
        let result_json: Value = serde_json::from_str(result_str).unwrap();

        assert!(result_json.get("latency").is_some());
        assert!(result_json.get("latency").unwrap().get("is_significant").is_some());

        // Crucially, test that the string can be freed without error.
        experiments_free_string(result_ptr);
    }

    #[test]
    fn test_ffi_analyze_classification_and_free() {
        let predictions_json = r#"[{"class": "cat", "confidence": 0.9}, {"class": "dog"}]"#;
        let truth_json = r#"[{"class": "cat"}, {"class": "cat"}]"#;

        let predictions_c = CString::new(predictions_json).unwrap();
        let truth_c = CString::new(truth_json).unwrap();

        let result_ptr = experiments_analyze_classification(predictions_c.as_ptr(), truth_c.as_ptr());
        assert!(!result_ptr.is_null());

        let result_str = unsafe { CStr::from_ptr(result_ptr).to_str().unwrap() };
        let report_json: Value = serde_json::from_str(result_str).unwrap();

        assert_eq!(report_json.get("num_samples").unwrap().as_u64(), Some(2));
        assert_eq!(report_json.get("metrics").unwrap().get("accuracy").unwrap().as_f64(), Some(0.5));

        experiments_free_string(result_ptr);
    }

    #[test]
    fn test_free_null_string() {
        // This test ensures that calling free on a null pointer doesn't crash.
        experiments_free_string(std::ptr::null_mut());
    }
}
