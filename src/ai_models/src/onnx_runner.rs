/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: onnx_runner.rs
 *
 * This file implements a runner for ONNX (Open Neural Network Exchange) models.
 * It provides a high-level interface over the `ort` crate (Rust bindings for
 * ONNX Runtime) to load models, configure execution providers (like CUDA or
 * TensorRT), and run inference sessions. The design encapsulates the complexity
 * of session management and tensor manipulation.
 *
 * Dependencies:
 *  - `ort`: For the core ONNX Runtime functionality.
 *  - `ndarray`: For creating and manipulating input/output tensors.
 *  - `thiserror`: For structured error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Imports ---
use ort::{environment::Environment, ExecutionProvider, GraphOptimizationLevel, Session, SessionBuilder, Value};
use std::path::Path;
use std::sync::Arc;

// --- Custom Error and Result Types ---

/// Represents errors that can occur during ONNX model loading or inference.
#[derive(Debug, thiserror::Error)]
pub enum OnnxError {
    #[error("Failed to build ONNX Runtime environment: {0}")]
    EnvironmentError(ort::OrtError),
    #[error("Failed to build ONNX session: {0}")]
    SessionCreationError(ort::OrtError),
    #[error("Failed to run inference: {0}")]
    InferenceError(ort::OrtError),
    #[error("I/O error while reading model file: {0}")]
    IoError(#[from] std::io::Error),
    #[error("The number of inputs provided ({provided}) does not match the number required by the model ({expected}).")]
    InputCountMismatch { provided: usize, expected: usize },
}

pub type OnnxResult<T> = Result<T, OnnxError>;

// --- ONNX Runner Service ---

/// A wrapper around an ONNX Runtime session that manages its lifecycle and execution.
pub struct OnnxSession {
    session: Session,
    // Store input and output names for convenience and validation.
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
}

impl OnnxSession {
    /// Creates a new `OnnxSession` by loading a model from a file.
    ///
    /// # Arguments
    /// * `env` - A shared reference to the ONNX Runtime environment.
    /// * `model_path` - The path to the `.onnx` model file.
    /// * `use_gpu` - If true, attempts to configure a GPU execution provider (e.g., CUDA).
    ///
    /// # Returns
    /// A new `OnnxSession` instance.
    pub fn new(env: Arc<Environment>, model_path: &Path, use_gpu: bool) -> OnnxResult<Self> {
        let mut session_builder = SessionBuilder::new(&env).map_err(OnnxError::SessionCreationError)?;

        if use_gpu {
            // Attempt to configure the most powerful GPU provider available.
            // The `ort` crate will find the best one (e.g., CUDA, TensorRT).
            if let Ok(provider) = ExecutionProvider::default_gpu() {
                println!("Using GPU execution provider: {:?}", provider.get_provider());
                session_builder = session_builder.with_execution_providers([provider]).map_err(OnnxError::SessionCreationError)?;
            } else {
                eprintln!("Warning: GPU execution was requested, but no compatible provider was found. Falling back to CPU.");
            }
        }

        // Set graph optimization level for performance.
        session_builder = session_builder.with_optimization_level(GraphOptimizationLevel::Level3).map_err(OnnxError::SessionCreationError)?;

        // Load the model into the session.
        let session = session_builder
            .with_model_from_file(model_path)
            .map_err(OnnxError::SessionCreationError)?;

        let input_names = session.inputs.iter().map(|i| i.name.clone()).collect();
        let output_names = session.outputs.iter().map(|o| o.name.clone()).collect();

        Ok(OnnxSession {
            session,
            input_names,
            output_names,
        })
    }

    /// Runs an inference pass using the loaded model.
    ///
    /// # Arguments
    /// * `inputs` - A slice of `ort::Value` tensors. The order and number of tensors
    ///   must match the model's requirements.
    ///
    /// # Returns
    /// A `Vec<ort::Value>` containing the output tensors from the model.
    pub fn run<'a, 'b>(&'a self, inputs: Vec<Value<'b>>) -> OnnxResult<Vec<Value<'a>>> {
        if inputs.len() != self.input_names.len() {
            return Err(OnnxError::InputCountMismatch {
                provided: inputs.len(),
                expected: self.input_names.len(),
            });
        }

        self.session.run(inputs).map_err(OnnxError::InferenceError)
    }
}

/// Manages the ONNX Runtime environment and session creation.
pub struct OnnxRunner {
    environment: Arc<Environment>,
}

impl OnnxRunner {
    /// Creates a new `OnnxRunner` and initializes the global ONNX Runtime environment.
    pub fn new() -> OnnxResult<Self> {
        let environment = Environment::builder()
            .with_name("TrackieLLM_ONNX_Runner")
            .build()
            .map_err(OnnxError::EnvironmentError)?;

        Ok(Self {
            environment: Arc::new(environment),
        })
    }

    /// Loads a model and creates a new `OnnxSession`.
    pub fn load_session(&self, model_path: &Path, use_gpu: bool) -> OnnxResult<OnnxSession> {
        OnnxSession::new(self.environment.clone(), model_path, use_gpu)
    }
}

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, Ix2};
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    // Helper to create a dummy ONNX model file for testing.
    // In a real scenario, you'd use a pre-existing, simple model.
    // This function creates a file with a minimal (but invalid) ONNX structure
    // just to test the file loading path. A real test would require a valid model.
    fn create_dummy_onnx_file(path: &Path) {
        // A minimal ONNX model representing a simple graph.
        // This is a placeholder; a real test needs a valid ONNX file.
        // For now, we assume `ort`'s file loading will fail gracefully.
        // This byte sequence is not a valid model, but allows testing I/O.
        let onnx_placeholder_bytes: &[u8] = b"\x08\x07\x12\x07\x4f\x4e\x4e\x58\x2d\x4d\x4c";
        let mut file = File::create(path).unwrap();
        file.write_all(onnx_placeholder_bytes).unwrap();
    }

    // Note: To truly test the runner, you need a valid ONNX model file.
    // For example, a simple model that adds two tensors. Without such a file,
    // we can only test the setup and error handling paths.
    #[test]
    fn test_runner_creation() {
        let runner = OnnxRunner::new();
        assert!(runner.is_ok());
    }

    #[test]
    fn test_session_loading_error_handling() {
        let dir = tempdir().unwrap();
        let dummy_model_path = dir.path().join("dummy.onnx");
        create_dummy_onnx_file(&dummy_model_path);

        let runner = OnnxRunner::new().unwrap();
        // This is expected to fail because the byte sequence is not a valid model.
        // We are testing that the error is correctly propagated.
        let session_result = runner.load_session(&dummy_model_path, false);

        assert!(session_result.is_err());
        assert!(matches!(session_result.unwrap_err(), OnnxError::SessionCreationError(_)));
    }

    // This is a conceptual test. It would require a real ONNX model file
    // that performs a known operation (e.g., matrix multiplication).
    #[test]
    #[ignore] // Ignored because it requires a real ONNX model file.
    fn conceptual_test_run_inference() {
        // 1. Assume `model.onnx` exists and takes a [1, 2] tensor and returns a [1, 2] tensor.
        let model_path = Path::new("path/to/your/model.onnx");

        let runner = OnnxRunner::new().unwrap();
        let session = runner.load_session(model_path, false).unwrap();

        // 2. Create an input tensor.
        let input_array = Array::from_shape_vec((1, 2), vec![1.0f32, 2.0f32]).unwrap();
        let input_tensor = vec![Value::from_array(input_array).unwrap()];

        // 3. Run inference.
        let outputs = session.run(input_tensor).unwrap();

        // 4. Check the output.
        let output_tensor = outputs[0].try_extract::<f32>().unwrap();
        let output_array = output_tensor.view();

        // Let's say the model doubles the input.
        let expected_array = Array::from_shape_vec((1, 2), vec![2.0f32, 4.0f32]).unwrap();
        assert_eq!(output_array, expected_array.view());
    }
}
