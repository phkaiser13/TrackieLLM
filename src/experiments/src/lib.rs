/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/experiments/lib.rs
 *
 * This file is the main library entry point for the 'experiments' crate.
 * This crate provides a framework for running, comparing, and analyzing the
 * performance of various AI models within the TrackieLLM ecosystem. It is
 * designed to facilitate A/B testing, regression testing, and benchmarking
 * of different model architectures, weights, or configurations.
 *
 * The core components are:
 * - `model_analysis`: Contains tools for in-depth analysis of a single model's
 *   performance on a given dataset, producing detailed metrics.
 * - `metrics_comparator`: Provides utilities to compare the performance metrics
 *   of two or more models, highlighting differences in accuracy, speed, and
 *   resource usage.
 *
 * The crate is orchestrated by the `ExperimentService`, which provides a high-level
 * API to define and execute experiments.
 *
 * Dependencies:
 *   - log: For structured logging during experiments.
 *   - thiserror: For ergonomic error handling.
 *   - serde: For serializing and deserializing experiment configurations and results.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// === TOC ===
// 1. Crate-level Documentation & Attributes
// 2. Public Module Declarations
// 3. Public Prelude
// 4. Core Public Types (Config, Error, Results)
// 5. Main Service Interface (ExperimentService)
// =============

#![deny(unsafe_code)]
#![deny(missing_docs)]
#![deny(warnings)]

//! # TrackieLLM Experiments Crate
//!
//! A framework for conducting performance analysis and comparative
//! benchmarking of AI models.
//!
//! ## Core Concepts
//!
//! - **Experiment**: A defined test run that typically involves one or more
//!   models, a dataset, and a set of metrics to be collected.
//! - **Analysis**: The process of running a single model against a dataset
//!   to generate a detailed performance report.
//! - **Comparison**: The process of comparing the performance reports of two
//!   or more models to identify key differences.
//!
//! ## Example Workflow
//!
//! ```rust,ignore
//! use experiments::{ExperimentConfig, ExperimentService, ModelSpec};
//!
//! fn run_ab_test() -> Result<(), Box<dyn std::error::Error>> {
//!     let model_a = ModelSpec {
//!         name: "Model-A-v1.0".to_string(),
//!         path: "/path/to/model_a.onnx".to_string(),
//!     };
//!
//!     let model_b = ModelSpec {
//!         name: "Model-B-v1.2".to_string(),
//!         path: "/path/to/model_b.onnx".to_string(),
//!     };
//!
//!     let config = ExperimentConfig {
//!         name: "A/B Test: Model A vs. Model B".to_string(),
//!         dataset_path: "/path/to/test_dataset.json".to_string(),
//!         models_to_test: vec![model_a, model_b],
//!     };
//!
//!     let mut service = ExperimentService::new(config);
//!     let results = service.run()?;
//!
//!     println!("Experiment Complete: {}", results.summary);
//!     Ok(())
//! }
//! ```

// --- Public Module Declarations ---

/// Provides tools for analyzing the performance of a single AI model.
pub mod model_analysis;

/// Provides tools for comparing metrics between different models.
pub mod metrics_comparator;


// --- Public Prelude ---

pub mod prelude {
    //! A "prelude" for convenient imports of the experiments crate's main types.
    pub use super::{
        ExperimentConfig, ExperimentError, ExperimentResult, ExperimentService, ModelSpec,
        model_analysis::ModelPerformanceReport,
        metrics_comparator::ComparisonReport,
    };
}


// --- Core Public Types ---

use thiserror::Error;
use crate::model_analysis::ModelPerformanceReport;

/// Specifies a model to be used in an experiment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    /// A unique name for the model (e.g., "yolov8n-v1.2").
    pub name: String,
    /// The file path to the model's weights or configuration.
    pub path: String,
}

/// Defines the configuration for a single experiment run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExperimentConfig {
    /// A descriptive name for the experiment.
    pub name: String,
    /// The path to the dataset to be used for testing.
    pub dataset_path: String,
    /// A list of models to be included in the experiment.
    pub models_to_test: Vec<ModelSpec>,
}

/// Represents the final results of a completed experiment.
#[derive(Debug, Clone)]
pub struct ExperimentResult {
    /// The name of the experiment.
    pub experiment_name: String,
    /// A collection of performance reports, one for each model tested.
    pub individual_reports: Vec<ModelPerformanceReport>,
    /// A high-level summary of the experiment's findings.
    pub summary: String,
}

/// Represents all possible errors that can occur within the experiments crate.
#[derive(Debug, Error)]
pub enum ExperimentError {
    /// An error occurred during the analysis of a model.
    #[error("Model analysis failed for '{model_name}': {source}")]
    AnalysisFailed {
        /// The name of the model that failed analysis.
        model_name: String,
        /// The underlying error.
        #[source]
        source: model_analysis::AnalysisError,
    },

    /// An error occurred during the comparison of metrics.
    #[error("Metric comparison failed: {0}")]
    ComparisonFailed(#[from] metrics_comparator::ComparatorError),

    /// The experiment configuration was invalid.
    #[error("Invalid experiment configuration: {0}")]
    InvalidConfig(String),

    /// No models were specified to be tested.
    #[error("No models were provided in the experiment configuration.")]
    NoModelsToTest,
}


// --- Main Service Interface ---

/// A service for defining and executing AI model experiments.
pub struct ExperimentService {
    config: ExperimentConfig,
}

impl ExperimentService {
    /// Creates a new `ExperimentService` with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The `ExperimentConfig` defining the experiment to be run.
    pub fn new(config: ExperimentConfig) -> Self {
        Self { config }
    }

    /// Runs the configured experiment.
    ///
    /// This method iterates through each model specified in the configuration,
    /// runs a performance analysis on it, and then generates a comparative
    /// summary of the results.
    ///
    /// # Returns
    ///
    /// An `ExperimentResult` containing the detailed reports and summary
    /// upon successful completion of the experiment.
    ///
    /// # Errors
    ///
    /// Returns an `ExperimentError` if any part of the experiment fails.
    pub fn run(&mut self) -> Result<ExperimentResult, ExperimentError> {
        log::info!("Starting experiment: '{}'", self.config.name);

        if self.config.models_to_test.is_empty() {
            return Err(ExperimentError::NoModelsToTest);
        }

        let mut individual_reports = Vec::new();

        // 1. Run analysis for each model.
        for model_spec in &self.config.models_to_test {
            log::info!("Analyzing model: '{}' from '{}'", model_spec.name, model_spec.path);
            let report =
                model_analysis::analyze_model(model_spec, &self.config.dataset_path).map_err(
                    |e| ExperimentError::AnalysisFailed {
                        model_name: model_spec.name.clone(),
                        source: e,
                    },
                )?;
            individual_reports.push(report);
        }

        // 2. Generate a comparison and summary.
        let summary = if individual_reports.len() > 1 {
            log::info!("Comparing performance of {} models.", individual_reports.len());
            let comparison_report = metrics_comparator::compare_reports(&individual_reports)?;
            comparison_report.summary
        } else {
            "Experiment complete. Only one model was analyzed.".to_string()
        };

        log::info!("Experiment '{}' finished successfully.", self.config.name);

        Ok(ExperimentResult {
            experiment_name: self.config.name.clone(),
            individual_reports,
            summary,
        })
    }
}
