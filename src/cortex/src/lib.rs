/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/cortex/lib.rs
 *
 * This file is the main library entry point for the 'cortex' crate, the
 * central nervous system of the TrackieLLM application. The Cortex is
 * responsible for orchestrating the entire perception-reasoning-action loop.
 *
 * It integrates all other modules to create a cohesive system:
 * - It receives sensory data from the `vision`, `audio`, and `sensors` modules.
 * - It uses the `ai_models` crate to run inference.
 * - It manages the application's state and memory via the `reasoning` and
 *   `memory_manager` modules.
 * - It translates the LLM's decisions into actions, such as generating speech
 *   via the `audio` module's TTS or providing guidance.
 *
 * This crate provides a high-level, safe Rust abstraction over the detailed C
 * APIs defined in `tk_cortex_main.h`, `tk_contextual_reasoner.h`, and
 * `tk_decision_engine.h`.
 *
 * Dependencies:
 *   - All other core crates (`vision`, `audio`, `ai_models`, etc.).
 *   - log: For structured logging of the main application loop.
 *   - thiserror: For ergonomic error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// === TOC ===
// 1. Crate-level Documentation & Attributes
// 2. FFI Bindings Module (private)
// 3. Public Module Declarations
// 4. Core Public Data Structures & Types
// 5. Public Prelude
// =============

#![deny(unsafe_code)]
#![deny(missing_docs)]
#![deny(warnings)]

//! # TrackieLLM Cortex Crate
//!
//! The central reasoning and orchestration engine of the application.
//!
//! ## Architecture
//!
//! The `Cortex` struct is the primary entry point. It is a stateful service
//! that is initialized once at application startup. It runs a main processing
//_ loop in a dedicated thread, which periodically:
//! 1. Gathers the latest context from the `reasoning` engine.
//! 2. If triggered (e.g., by user voice command), runs the LLM via the
//!    `ai_models` crate.
//! 3. Processes the LLM's response, which could be text or a tool call.
//! 4. Executes actions (e.g., speaks text, sends navigation commands).
//!
//! Communication with the Cortex from other threads (e.g., for injecting
//! sensor data) is handled via a thread-safe command pattern.

// --- FFI Bindings Module ---
// This module contains the raw FFI declarations for the entire Cortex C API.
mod ffi {
    #![allow(non_camel_case_types, non_snake_case, dead_code)]
    // This would be extremely large, containing all definitions from the
    // three cortex-related C headers. For brevity, only a few key items
    // are included here as placeholders for the real generated bindings.
    pub type tk_error_code_t = i32;
    pub const TK_SUCCESS: tk_error_code_t = 0;

    pub enum tk_cortex_s {}
    pub type tk_cortex_t = tk_cortex_s;

    pub enum tk_contextual_reasoner_s {}
    pub type tk_contextual_reasoner_t = tk_contextual_reasoner_s;
    
    pub enum tk_decision_engine_s {}
    pub type tk_decision_engine_t = tk_decision_engine_s;

    #[repr(C)]
    pub struct tk_cortex_config_t {
        // ... fields
    }
    
    #[repr(C)]
    pub struct tk_cortex_callbacks_t {
        // ... fields
    }

    extern "C" {
        pub fn tk_cortex_create(
            out_cortex: *mut *mut tk_cortex_t,
            config: *const tk_cortex_config_t,
            callbacks: tk_cortex_callbacks_t,
        ) -> tk_error_code_t;
        pub fn tk_cortex_destroy(cortex: *mut *mut tk_cortex_t);
        pub fn tk_cortex_run(cortex: *mut tk_cortex_t) -> tk_error_code_t;
        pub fn tk_cortex_stop(cortex: *mut tk_cortex_t) -> tk_error_code_t;
    }
}


// --- Public Module Declarations ---

/// Manages the application's contextual understanding and short-term memory.
/// Wraps `tk_contextual_reasoner.h`.
pub mod reasoning;

/// A Rust-native system for managing long-term memory and context summarization.
pub mod memory_manager;


// --- Core Public Types ---

use thiserror::Error;

/// The primary error type for all operations within the Cortex.
#[derive(Debug, Error)]
pub enum CortexError {
    /// The Cortex has not been initialized.
    #[error("Cortex is not initialized.")]
    NotInitialized,

    /// The Cortex is already running and cannot be started again.
    #[error("Cortex is already running.")]
    AlreadyRunning,

    /// An FFI call to the underlying C library failed.
    #[error("Cortex FFI call failed: {0}")]
    Ffi(String),

    /// A required model or configuration file could not be loaded.
    #[error("Failed to load a required model: {0}")]
    ModelLoadFailed(String),
}


// --- Public Prelude ---

pub mod prelude {
    //! A "prelude" for convenient imports of this crate's main types.
    pub use super::{reasoning::ContextualReasoner, Cortex, CortexError};
}


// --- Main Service Interface ---

/// The main Cortex service.
///
/// This struct is the primary public interface to the application's core logic.
pub struct Cortex {
    /// A handle to the underlying `tk_cortex_t` C object.
    /// This is a private field, managed internally via RAII.
    cortex_handle: *mut ffi::tk_cortex_t,
    // In a real implementation, this would also hold handles to the main loop
    // thread and channels for communication.
}

// Note: A full implementation would be much more extensive, including:
// - A `CortexBuilder` to construct the `tk_cortex_config_t`.
// - A `CortexHandle` for thread-safe communication with the running Cortex.
// - Implementations of the callback functions to be passed to the C API.
// - Safe wrappers for `tk_cortex_inject_audio_frame`, etc.

impl Drop for Cortex {
    /// Ensures the C-level Cortex object is always destroyed.
    fn drop(&mut self) {
        if !self.cortex_handle.is_null() {
            unsafe { ffi::tk_cortex_destroy(&mut self.cortex_handle) };
        }
    }
}
