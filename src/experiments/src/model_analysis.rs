/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: model_analysis.rs
 *
 * This file provides tools for the analysis of machine learning model outputs.
 * It is designed to evaluate a model's performance against a labeled dataset,
 * calculating common metrics for tasks like classification and regression. The
 * framework is structured to be extensible for different types of analysis.
 *
 * Dependencies:
 *  - `serde`: For deserializing datasets and model outputs.
 *  - `statrs`: For calculating descriptive statistics on model outputs.
 *  - `thiserror`: For structured error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Imports ---
use serde::Deserialize;
use statrs::statistics::{Data, Mean, Median, OrderStatistics, std_dev};
use std::collections::HashMap;

// --- Custom Error and Result Types ---

/// Represents errors that can occur during model analysis.
#[derive(Debug, thiserror::Error)]
pub enum AnalysisError {
    #[error("Input data and ground truth datasets have different lengths. Inputs: {input_len}, Ground Truth: {truth_len}.")]
    MismatchedDataLength {
        input_len: usize,
        truth_len: usize,
    },
    #[error("Required data for analysis is missing: {0}")]
    MissingData(String),
    #[error("Invalid data format for analysis: {0}")]
    InvalidData(String),
}

type AnalysisResult<T> = Result<T, AnalysisError>;

// --- Core Data Structures ---

/// Represents the output of a model for a single prediction.
/// This is a flexible enum to handle different types of model outputs.
#[derive(Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)] // Allows deserializing into the first matching variant
pub enum ModelPrediction {
    /// For classification tasks, contains the predicted class label.
    Classification { class: String, confidence: Option<f64> },
    /// For regression tasks, contains the predicted numerical value.
    Regression { value: f64 },
    /// For generation tasks, contains the generated text.
    Generation { text: String },
}

/// Represents the ground truth (correct answer) for a single data point.
#[derive(Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum GroundTruth {
    Classification { class: String },
    Regression { value: f64 },
    Generation { references: Vec<String> },
}

/// A comprehensive report summarizing the evaluation of a model.
#[derive(Debug, Clone)]
pub struct EvaluationReport {
    pub num_samples: usize,
    pub metrics: HashMap<String, f64>,
    pub notes: Vec<String>,
}

// --- Model Analysis Service ---

/// A service for performing offline analysis of model performance.
#[derive(Default)]
pub struct ModelAnalyzer;

impl ModelAnalyzer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Evaluates a set of classification predictions against ground truth labels.
    ///
    /// # Arguments
    /// * `predictions` - A slice of `ModelPrediction::Classification`.
    /// * `ground_truth` - A slice of `GroundTruth::Classification`.
    ///
    /// # Returns
    /// An `EvaluationReport` containing metrics like accuracy and statistics on confidence scores.
    pub fn analyze_classification(
        &self,
        predictions: &[ModelPrediction],
        ground_truth: &[GroundTruth],
    ) -> AnalysisResult<EvaluationReport> {
        if predictions.len() != ground_truth.len() {
            return Err(AnalysisError::MismatchedDataLength {
                input_len: predictions.len(),
                truth_len: ground_truth.len(),
            });
        }

        let mut correct_predictions = 0;
        let mut confidence_scores = Vec::new();

        for (pred, truth) in predictions.iter().zip(ground_truth.iter()) {
            if let (
                ModelPrediction::Classification { class: pred_class, confidence },
                GroundTruth::Classification { class: true_class },
            ) = (pred, truth) {
                if pred_class == true_class {
                    correct_predictions += 1;
                }
                if let Some(score) = confidence {
                    confidence_scores.push(*score);
                }
            } else {
                return Err(AnalysisError::InvalidData(
                    "Mismatched prediction/truth types; expected Classification for all items.".to_string(),
                ));
            }
        }

        let num_samples = predictions.len();
        let accuracy = if num_samples > 0 {
            correct_predictions as f64 / num_samples as f64
        } else { 0.0 };

        let mut metrics = HashMap::from([
            ("accuracy".to_string(), accuracy)
        ]);

        // Add statistics for confidence scores if available
        if !confidence_scores.is_empty() {
            let data = Data::new(confidence_scores.as_slice());
            metrics.insert("confidence_mean".to_string(), data.mean().unwrap_or(0.0));
            metrics.insert("confidence_median".to_string(), data.median());
            metrics.insert("confidence_std_dev".to_string(), std_dev(confidence_scores.as_slice()).unwrap_or(0.0));
        }

        Ok(EvaluationReport {
            num_samples,
            metrics,
            notes: vec![],
        })
    }

    /// Evaluates a set of regression predictions against ground truth values.
    pub fn analyze_regression(
        &self,
        predictions: &[ModelPrediction],
        ground_truth: &[GroundTruth],
    ) -> AnalysisResult<EvaluationReport> {
        if predictions.len() != ground_truth.len() {
            return Err(AnalysisError::MismatchedDataLength {
                input_len: predictions.len(),
                truth_len: ground_truth.len(),
            });
        }

        let mut squared_errors = Vec::new();
        let mut truth_values = Vec::new();

        for (pred, truth) in predictions.iter().zip(ground_truth.iter()) {
            if let (
                ModelPrediction::Regression { value: pred_val },
                GroundTruth::Regression { value: true_val },
            ) = (pred, truth) {
                squared_errors.push((true_val - pred_val).powi(2));
                truth_values.push(*true_val);
            } else {
                 return Err(AnalysisError::InvalidData(
                    "Mismatched prediction/truth types; expected Regression for all items.".to_string(),
                ));
            }
        }

        let num_samples = predictions.len();
        let mse = Data::new(squared_errors.as_slice()).mean().unwrap_or(0.0);

        let mut metrics = HashMap::from([
            ("mean_squared_error".to_string(), mse),
            ("root_mean_squared_error".to_string(), mse.sqrt()),
        ]);

        Ok(EvaluationReport {
            num_samples,
            metrics,
            notes: vec![],
        })
    }

    // Note: A real implementation of text analysis (e.g., BLEU, ROUGE) would be
    // much more complex and likely require dedicated crates. This is a placeholder
    // for the structure of such an analysis.
    pub fn analyze_generation(
        &self,
        predictions: &[ModelPrediction],
        ground_truth: &[GroundTruth],
    ) -> AnalysisResult<EvaluationReport> {
        // Placeholder for a more complex text analysis.
        let mut notes = vec![
            "Text generation analysis is highly complex.".to_string(),
            "Metrics like BLEU or ROUGE would be calculated here.".to_string(),
        ];

        let mut metrics = HashMap::new();
        // Simulate a dummy score.
        metrics.insert("simulated_bleu_score".to_string(), 0.75);

        Ok(EvaluationReport {
            num_samples: predictions.len(),
            metrics,
            notes,
        })
    }
}

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classification_analysis() {
        let analyzer = ModelAnalyzer::new();
        let predictions = vec![
            ModelPrediction::Classification { class: "cat".to_string(), confidence: Some(0.9) },
            ModelPrediction::Classification { class: "dog".to_string(), confidence: Some(0.8) },
            ModelPrediction::Classification { class: "cat".to_string(), confidence: Some(0.95) },
            ModelPrediction::Classification { class: "cat".to_string(), confidence: Some(0.7) },
        ];
        let ground_truth = vec![
            GroundTruth::Classification { class: "cat".to_string() },
            GroundTruth::Classification { class: "fish".to_string() }, // Incorrect
            GroundTruth::Classification { class: "cat".to_string() },
            GroundTruth::Classification { class: "cat".to_string() },
        ];

        let report = analyzer.analyze_classification(&predictions, &ground_truth).unwrap();

        assert_eq!(report.num_samples, 4);
        assert_eq!(*report.metrics.get("accuracy").unwrap(), 0.75);
        assert_eq!(*report.metrics.get("confidence_mean").unwrap(), 0.8375);
    }

    #[test]
    fn test_regression_analysis() {
        let analyzer = ModelAnalyzer::new();
        let predictions = vec![
            ModelPrediction::Regression { value: 10.5 },
            ModelPrediction::Regression { value: 22.0 },
            ModelPrediction::Regression { value: 31.0 },
        ];
        let ground_truth = vec![
            GroundTruth::Regression { value: 10.0 },
            GroundTruth::Regression { value: 20.0 },
            GroundTruth::Regression { value: 30.0 },
        ];

        let report = analyzer.analyze_regression(&predictions, &ground_truth).unwrap();
        // MSE = ((0.5^2) + (2.0^2) + (1.0^2)) / 3 = (0.25 + 4.0 + 1.0) / 3 = 5.25 / 3 = 1.75
        assert_eq!(*report.metrics.get("mean_squared_error").unwrap(), 1.75);
        assert_eq!(*report.metrics.get("root_mean_squared_error").unwrap(), 1.75_f64.sqrt());
    }

    #[test]
    fn test_mismatched_length_error() {
        let analyzer = ModelAnalyzer::new();
        let predictions = vec![ModelPrediction::Regression { value: 1.0 }];
        let ground_truth = vec![];

        let result = analyzer.analyze_regression(&predictions, &ground_truth);
        assert!(matches!(result, Err(AnalysisError::MismatchedDataLength { .. })));
    }
}
