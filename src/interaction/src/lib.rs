/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/interaction/lib.rs
 *
 * This file is the main library entry point for the 'interaction' crate. This
 * crate is responsible for the user-facing interaction logic, including parsing
 * voice commands and managing spoken feedback. It acts as the bridge between
 * the Cortex's decisions and the user.
 *
 * It provides safe Rust abstractions over the C-based interaction components:
 * - `tk_voice_commands.h`: A data-driven command parser.
 * - `tk_feedback_manager.h`: A priority-based queue for spoken feedback.
 *
 * The primary interface is the `InteractionService`, which composes the
 * `CommandParser` and `FeedbackManager` to provide a unified service to the Cortex.
 *
 * Dependencies:
 *   - log: For structured logging.
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

//! # TrackieLLM Interaction Crate
//!
//! Manages voice command parsing and spoken feedback arbitration.
//!
//! ## Architecture
//!
//! This crate provides two main services:
//!
//! - **`CommandParser`**: Takes transcribed text from the ASR engine and, using
//!   a pre-compiled grammar, attempts to match it to a known command and
//!   extract its parameters (slots).
//! - **`FeedbackManager`**: Receives feedback requests from various parts of
//!   the system, prioritizes them based on urgency (e.g., a critical hazard
//!   warning vs. a simple conversational reply), and sends the highest-priority
//!   message to the TTS engine for synthesis.

// --- FFI Bindings Module ---
mod ffi {
    #![allow(non_camel_case_types, non_snake_case, dead_code)]
    pub type tk_error_code_t = i32;
    pub const TK_SUCCESS: tk_error_code_t = 0;

    // Forward declarations from the C headers
    pub enum tk_command_parser_s {}
    pub type tk_command_parser_t = tk_command_parser_s;
    pub enum tk_feedback_manager_s {}
    pub type tk_feedback_manager_t = tk_feedback_manager_s;
    pub enum tk_parsed_command_s {}
    pub type tk_parsed_command_t = tk_parsed_command_s;

    // A subset of the FFI functions for demonstration
    extern "C" {
        // From tk_voice_commands.h
        pub fn tk_command_parser_create(
            out_parser: *mut *mut tk_command_parser_t,
            grammar_data: *const u8,
            grammar_size: usize,
        ) -> tk_error_code_t;
        pub fn tk_command_parser_destroy(parser: *mut *mut tk_command_parser_t);
        pub fn tk_command_parser_parse_command(
            parser: *mut tk_command_parser_t,
            text: *const std::os::raw::c_char,
            out_command: *mut *mut tk_parsed_command_t,
        ) -> tk_error_code_t;
        pub fn tk_command_parser_free_command(command: *mut *mut tk_parsed_command_t);

        // From tk_feedback_manager.h
        pub fn tk_feedback_manager_create(
            out_manager: *mut *mut tk_feedback_manager_t,
            config: *const std::ffi::c_void, // Placeholder
        ) -> tk_error_code_t;
        pub fn tk_feedback_manager_destroy(manager: *mut *mut tk_feedback_manager_t);
        pub fn tk_feedback_manager_enqueue(
            manager: *mut tk_feedback_manager_t,
            request: *const std::ffi::c_void, // Placeholder
        ) -> tk_error_code_t;
    }
}

// --- Public Module Declarations ---

/// Provides a safe wrapper for the voice command parser.
pub mod command_parser;
/// Provides a safe wrapper for the feedback arbitration logic.
pub mod feedback_logic;


// --- Core Public Data Structures & Types ---

use thiserror::Error;

/// The primary error type for all operations within the interaction crate.
#[derive(Debug, Error)]
pub enum InteractionError {
    /// An error occurred in the command parsing system.
    #[error("Command parser error: {0}")]
    CommandParser(#[from] command_parser::CommandParserError),

    /// An error occurred in the feedback management system.
    #[error("Feedback manager error: {0}")]
    Feedback(#[from] feedback_logic::FeedbackError),
    
    /// An FFI call failed.
    #[error("Interaction FFI call failed: {0}")]
    Ffi(String),

    /// A C-string conversion failed.
    #[error("Invalid C-style string: {0}")]
    InvalidCString(#[from] std::ffi::NulError),
}


// --- Public Prelude ---

pub mod prelude {
    //! A "prelude" for convenient imports of this crate's main types.
    pub use super::{
        command_parser::{CommandParser, ParsedCommand},
        feedback_logic::{FeedbackManager, FeedbackPriority, FeedbackRequest},
        InteractionError,
    };
}
