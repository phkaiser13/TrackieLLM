/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/ai_models/gguf_runner.rs
 *
 * This file implements a safe Rust wrapper for the GGUF-based LLM runner
 * defined in `tk_model_runner.h`. It provides a high-level, idiomatic Rust
 * interface for stateful, tool-using conversational AI.
 *
 * The primary component is the `GgufRunner`, which encapsulates all the
 * complexity of interacting with the C FFI, including:
 * - Resource management of the `tk_llm_runner_t` handle via the RAII pattern.
 * - Safe conversion between Rust data structures (e.g., `LlmConfig`, `LlmResult`)
 *   and their C-style FFI counterparts.
 * - Robust error handling that translates C error codes into a rich Rust enum.
 *
 * This module allows the rest of the application to interact with the LLM
 * in a completely safe and predictable way, without directly touching any
 * `unsafe` code.
 *
 * Dependencies:
 *   - crate::ffi: For the raw C function bindings.
 *   - crate::{AiModelsError, LlmConfig, LlmResult, ...}: For shared types.
 *   - log: For structured logging.
 *   - serde_json: For parsing tool call arguments.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use super::{
    ffi, AiModelsError, LlmConfig, LlmResult, LlmToolCall, ToolDefinition,
};
use std::ffi::{CStr, CString};
use std::ptr::null_mut;

/// A low-level RAII wrapper for the `tk_llm_runner_t` handle.
///
/// This struct ensures that the C-level context is created and destroyed
/// correctly. It is a private implementation detail of the `GgufRunner`.
struct LlmContext {
    ptr: *mut ffi::tk_llm_runner_t,
}

impl LlmContext {
    /// Creates a new `LlmContext` by wrapping the C FFI `tk_llm_runner_create`.
    fn new(config: &LlmConfig) -> Result<Self, AiModelsError> {
        // Convert the safe Rust config into a C-compatible struct.
        let c_system_prompt = CString::new(config.system_prompt)?;
        let c_config = ffi::tk_llm_config_t {
            model_path: config.model_path.as_ptr() as *mut _, // In a real scenario, Path would have an `as_mut_ptr` or be handled differently.
            context_size: config.context_size,
            gpu_layers_offload: config.gpu_layers_offload,
            system_prompt: c_system_prompt.as_ptr(),
            random_seed: config.random_seed,
        };

        let mut ptr = null_mut();
        let code = unsafe { ffi::tk_llm_runner_create(&mut ptr, &c_config) };

        if code != ffi::TK_SUCCESS {
            return Err(AiModelsError::ModelLoadFailed {
                path: config.model_path.to_string(),
                reason: format!("FFI call to tk_llm_runner_create failed with code {}", code),
            });
        }
        Ok(Self { ptr })
    }
}

impl Drop for LlmContext {
    /// Ensures the C context is always destroyed when the `LlmContext` goes out of scope.
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::tk_llm_runner_destroy(&mut self.ptr) };
        }
    }
}

/// A safe, high-level runner for GGUF-based Large Language Models.
pub struct GgufRunner {
    context: LlmContext,
}

impl GgufRunner {
    /// Creates a new `GgufRunner` and loads the specified model.
    pub fn new(config: &LlmConfig) -> Result<Self, AiModelsError> {
        let context = LlmContext::new(config)?;
        Ok(Self { context })
    }

    /// Generates a response from the LLM based on the provided context and tools.
    ///
    /// # Arguments
    /// * `user_transcription` - The text from the user.
    /// * `vision_context` - A textual description of the visual scene.
    /// * `available_tools` - A slice of tools the LLM can use.
    ///
    /// # Returns
    /// An `LlmResult` which can be either a text response or a tool call.
    pub fn generate_response(
        &mut self,
        user_transcription: &str,
        vision_context: Option<&str>,
        available_tools: &[ToolDefinition],
    ) -> Result<LlmResult, AiModelsError> {
        // 1. Convert Rust types to C-compatible types.
        let c_user_transcription = CString::new(user_transcription)?;
        let c_vision_context = vision_context.map(CString::new).transpose()?.map(|s| s.as_ptr()).unwrap_or(null_mut());

        let c_tools: Vec<_> = available_tools.iter().map(|t| {
            // These CStrings must live long enough for the FFI call.
            let name = CString::new(t.name.as_str()).unwrap();
            let desc = CString::new(t.description.as_str()).unwrap();
            let schema = CString::new(t.parameters_json_schema.to_string()).unwrap();
            (name, desc, schema)
        }).collect();

        let c_tool_defs: Vec<_> = c_tools.iter().map(|(name, desc, schema)| {
            ffi::tk_llm_tool_definition_t {
                name: name.as_ptr(),
                description: desc.as_ptr(),
                parameters_json_schema: schema.as_ptr(),
            }
        }).collect();

        let prompt_context = ffi::tk_llm_prompt_context_t {
            user_transcription: c_user_transcription.as_ptr(),
            vision_context: c_vision_context,
        };

        // 2. Make the FFI call.
        let mut result_ptr = null_mut();
        let code = unsafe {
            ffi::tk_llm_runner_generate_response(
                self.context.ptr,
                &prompt_context,
                c_tool_defs.as_ptr(),
                c_tool_defs.len(),
                &mut result_ptr,
            )
        };

        if code != ffi::TK_SUCCESS {
            return Err(AiModelsError::InferenceFailed(format!(
                "FFI call to tk_llm_runner_generate_response failed with code {}", code
            )));
        }

        // 3. Convert the C result back to a safe Rust type.
        let result = unsafe { self.parse_c_result(result_ptr) };
        
        // 4. Free the C result object.
        unsafe { ffi::tk_llm_result_destroy(&mut result_ptr) };

        result
    }

    /// Parses the raw C result pointer into a safe Rust `LlmResult`.
    /// This function is unsafe because it dereferences raw pointers.
    unsafe fn parse_c_result(&self, result_ptr: *mut ffi::tk_llm_result_t) -> Result<LlmResult, AiModelsError> {
        let c_result = &*result_ptr;
        match c_result.type_ {
            ffi::tk_llm_result_type_e::TK_LLM_RESULT_TYPE_TEXT_RESPONSE => {
                let text = CStr::from_ptr(c_result.data.text_response).to_str()?.to_owned();
                Ok(LlmResult::Text(text))
            }
            ffi::tk_llm_result_type_e::TK_LLM_RESULT_TYPE_TOOL_CALL => {
                let tool_call = &c_result.data.tool_call;
                let name = CStr::from_ptr(tool_call.name).to_str()?.to_owned();
                let args_str = CStr::from_ptr(tool_call.arguments_json).to_str()?;
                let arguments = serde_json::from_str(args_str)?;

                Ok(LlmResult::ToolCall(LlmToolCall { name, arguments }))
            }
            _ => Err(AiModelsError::InferenceFailed("Unknown result type from FFI".to_string())),
        }
    }
}
