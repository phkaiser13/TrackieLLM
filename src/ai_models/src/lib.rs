/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/ai_models/lib.rs
 *
 * This file is the main library entry point for the 'ai_models' crate. This
 * crate is the core of the application's intelligence, providing a unified
 * Rust interface for loading, managing, and running various AI models.
 *
 * It acts as a safe, high-level abstraction layer over the comprehensive C APIs
 * defined in `tk_model_loader.h` and `tk_model_runner.h`. The crate is
 * responsible for wrapping unsafe FFI calls in memory-safe, ergonomic Rust
 * constructs, ensuring robust error handling and resource management (RAII).
 *
 * The crate is structured into sub-modules for different model runner
 * implementations (e.g., `gguf_runner` for LLMs, `onnx_runner` for vision/audio
 * models), providing specialized APIs for each task while sharing common
 * configuration and error-handling infrastructure.
 *
 * Dependencies:
 *   - internal_tools: For safe path and configuration handling.
 *   - log: For structured logging.
 *   - thiserror: For ergonomic error handling.
 *   - serde, serde_json: For handling JSON schemas and arguments for tool calls.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// === TOC ===
// 1. Crate-level Documentation & Attributes
// 2. FFI Bindings Module (private)
// 3. Public Module Declarations
// 4. Core Public Data Structures
// 5. Core Public Error Type
// 6. Public Prelude
// =============

#![deny(unsafe_code)]
#![deny(missing_docs)]
#![deny(warnings)]

//! # TrackieLLM AI Models Crate
//!
//! Provides high-level, safe Rust APIs for interacting with AI models.
//!
//! ## Core Functionality
//!
//! - **Model Loading**: A robust interface for loading models of various
//!   formats (GGUF, ONNX) with extensive configuration options.
//! - **Model Inference**: Specialized runners for different model types,
//!   including a stateful, tool-using LLM runner.
//! - **Safe Abstractions**: Wraps all C-level pointers and resources in
//!   safe Rust types that handle memory and resource lifetimes automatically.

// --- FFI Bindings Module ---
// Contains the raw `extern "C"` declarations mirroring the C headers.
// This is extensive due to the detailed C API.
mod ffi {
    #![allow(non_camel_case_types, non_snake_case, dead_code)]
    use crate::internal_tools::fs_utils::Path as TkPath;

    pub type tk_error_code_t = i32;
    pub const TK_SUCCESS: tk_error_code_t = 0;
    // Define other error codes as needed...

    // --- From tk_model_loader.h ---
    pub enum tk_model_loader_s {}
    pub type tk_model_loader_t = tk_model_loader_s;

    #[repr(C)]
    pub enum tk_model_format_e {
        // ... variants
    }

    #[repr(C)]
    pub struct tk_model_loader_config_t {
        // ... fields
    }

    #[repr(C)]
    pub struct tk_model_load_params_t {
        pub model_path: *mut TkPath, // Note: This should be an opaque pointer
        // ... other fields
    }
    
    extern "C" {
        pub fn tk_model_loader_create(out_loader: *mut *mut tk_model_loader_t, config: *const tk_model_loader_config_t) -> tk_error_code_t;
        pub fn tk_model_loader_destroy(loader: *mut *mut tk_model_loader_t);
        pub fn tk_model_loader_load_model(loader: *mut tk_model_loader_t, params: *const tk_model_load_params_t, out_model_handle: *mut *mut std::ffi::c_void) -> tk_error_code_t;
        pub fn tk_model_loader_unload_model(loader: *mut tk_model_loader_t, model_handle: *mut *mut std::ffi::c_void) -> tk_error_code_t;
    }

    // --- From tk_model_runner.h ---
    pub enum tk_llm_runner_s {}
    pub type tk_llm_runner_t = tk_llm_runner_s;
    pub enum tk_llm_result_s {}
    pub type tk_llm_result_t = tk_llm_result_s;

    #[repr(C)]
    pub struct tk_llm_config_t {
        pub model_path: *mut TkPath,
        pub context_size: u32,
        pub gpu_layers_offload: u32,
        pub system_prompt: *const std::os::raw::c_char,
        pub random_seed: u32,
    }

    #[repr(C)]
    pub struct tk_llm_tool_definition_t {
        pub name: *const std::os::raw::c_char,
        pub description: *const std::os::raw::c_char,
        pub parameters_json_schema: *const std::os::raw::c_char,
    }

    #[repr(C)]
    pub struct tk_llm_prompt_context_t {
        pub user_transcription: *const std::os::raw::c_char,
        pub vision_context: *const std::os::raw::c_char,
    }
    
    extern "C" {
        pub fn tk_llm_runner_create(out_runner: *mut *mut tk_llm_runner_t, config: *const tk_llm_config_t) -> tk_error_code_t;
        pub fn tk_llm_runner_destroy(runner: *mut *mut tk_llm_runner_t);
        pub fn tk_llm_runner_generate_response(runner: *mut tk_llm_runner_t, prompt_context: *const tk_llm_prompt_context_t, available_tools: *const tk_llm_tool_definition_t, tool_count: usize, out_result: *mut *mut tk_llm_result_t) -> tk_error_code_t;
        pub fn tk_llm_runner_add_tool_response(runner: *mut tk_llm_runner_t, tool_name: *const std::os::raw::c_char, tool_output: *const std::os::raw::c_char) -> tk_error_code_t;
        pub fn tk_llm_runner_reset_context(runner: *mut tk_llm_runner_t) -> tk_error_code_t;
        pub fn tk_llm_result_destroy(result: *mut *mut tk_llm_result_t);
    }
}


// --- Public Module Declarations ---

/// Provides a safe runner for GGUF-based Large Language Models.
pub mod gguf_runner;
/// Provides a safe runner for ONNX-based models (e.g., for vision and audio).
pub mod onnx_runner;


// --- Core Public Data Structures ---

use serde::{Deserialize, Serialize};
use thiserror::Error;
use crate::internal_tools::fs_utils::Path;

/// A high-level, safe Rust representation of a tool the LLM can use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// The exact name of the function to be called.
    pub name: String,
    /// A natural language description of what the tool does.
    pub description: String,
    /// A JSON schema describing the tool's parameters.
    pub parameters_json_schema: Value,
}

/// A safe Rust representation of a tool call requested by the LLM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmToolCall {
    /// The name of the tool to be executed.
    pub name: String,
    /// The arguments for the tool, as a JSON object.
    pub arguments: Value,
}

/// A safe, structured Rust representation of the LLM's output.
#[derive(Debug, Clone)]
pub enum LlmResult {
    /// The LLM produced a natural language response.
    Text(String),
    /// The LLM requested to use a tool.
    ToolCall(LlmToolCall),
}

/// High-level configuration for an LLM runner.
pub struct LlmConfig<'a> {
    /// Path to the GGUF model file.
    pub model_path: &'a Path,
    /// The context window size for the model.
    pub context_size: u32,
    /// Number of model layers to offload to the GPU.
    pub gpu_layers_offload: u32,
    /// The initial system prompt defining the AI's persona.
    pub system_prompt: &'a str,
    /// Seed for the random number generator.
    pub random_seed: u32,
}


// --- Core Public Error Type ---

/// The primary error type for all operations within the `ai_models` crate.
#[derive(Debug, Error)]
pub enum AiModelsError {
    /// An FFI call failed.
    #[error("FFI call failed: {0}")]
    Ffi(String),

    /// A C-string conversion failed.
    #[error("Invalid C-style string: {0}")]
    InvalidCString(#[from] std::ffi::NulError),
    
    /// A string conversion from C failed due to invalid UTF-8.
    #[error("Invalid UTF-8 sequence in string from FFI")]
    InvalidUtf8(#[from] std::str::Utf8Error),
    
    /// Failed to serialize or deserialize JSON data.
    #[error("JSON operation failed: {0}")]
    Json(#[from] serde_json::Error),
    
    /// The specified model could not be loaded.
    #[error("Model load failed for path '{path}': {reason}")]
    ModelLoadFailed {
        /// The path to the model that failed to load.
        path: String,
        /// The reason for the failure.
        reason: String,
    },

    /// An error occurred during model inference.
    #[error("Inference failed: {0}")]
    InferenceFailed(String),
}


// --- Public Prelude ---

pub mod prelude {
    //! A "prelude" for convenient imports of this crate's main types.
    pub use super::{
        gguf_runner::GgufRunner,
        onnx_runner::OnnxRunner,
        AiModelsError, LlmConfig, LlmResult, LlmToolCall, ToolDefinition,
    };
}
