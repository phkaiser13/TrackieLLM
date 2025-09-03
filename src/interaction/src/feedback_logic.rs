/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/interaction/feedback_logic.rs
 *
 * This file provides a safe Rust wrapper for the Feedback Manager defined in
 * `tk_feedback_manager.h`. The `FeedbackManager` is a sophisticated system
 * for arbitrating and prioritizing spoken feedback to the user, ensuring that
 * critical alerts are heard over less important information.
 *
 * This module encapsulates the `unsafe` FFI calls to the C-level manager,
 * managing the lifecycle of the `tk_feedback_manager_t` handle and providing
 * a safe, idiomatic API for submitting feedback requests.
 *
 * Dependencies:
 *   - crate::ffi: For the raw C function bindings.
 *   - crate::InteractionError: For shared error handling.
 *   - thiserror: For ergonomic error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use super::{ffi, InteractionError};
use std::ffi::CString;
use std::ptr::null_mut;
use thiserror::Error;

/// Represents errors specific to the feedback management process.
#[derive(Debug, Error)]
pub enum FeedbackError {
    /// The underlying C-level context is not initialized.
    #[error("Feedback manager context is not initialized.")]
    NotInitialized,

    /// An FFI call to the feedback manager C library failed.
    #[error("Feedback manager FFI call failed: {0}")]
    Ffi(String),
}

/// Defines the priority of a feedback message. Higher values are more important.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u32)]
pub enum FeedbackPriority {
    LowAmbient = 10,
    NormalResponse = 20,
    HighObstacle = 30,
    CriticalHazard = 40,
}

/// Defines verbosity levels to categorize and filter messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum FeedbackVerbosity {
    Quiet,
    Normal,
    Verbose,
}

/// A safe Rust representation of a feedback request.
#[derive(Debug, Clone)]
pub struct FeedbackRequest<'a> {
    pub text: &'a str,
    pub priority: FeedbackPriority,
    pub verbosity_level: FeedbackVerbosity,
    pub suppression_key: u32,
    pub suppression_cooldown_ms: u32,
    pub is_interrupt: bool,
}

/// A safe, high-level interface to the Feedback Manager.
pub struct FeedbackManager {
    /// The handle to the underlying `tk_feedback_manager_t` C object.
    manager_handle: *mut ffi::tk_feedback_manager_t,
}

impl FeedbackManager {
    /// Creates a new `FeedbackManager`.
    ///
    /// This is a simplified constructor. A real implementation would take a
    /// configuration struct with callbacks and call `tk_feedback_manager_create`.
    pub fn new() -> Result<Self, InteractionError> {
        // Placeholder for creating the feedback manager.
        Ok(Self {
            manager_handle: null_mut(),
        })
    }

    /// Submits a new feedback request to the manager's queue.
    ///
    /// # Arguments
    /// * `request` - The `FeedbackRequest` to be processed.
    pub fn enqueue(&self, request: &FeedbackRequest) -> Result<(), InteractionError> {
        // Mock Implementation:
        // 1. Convert the Rust `FeedbackRequest` into a C `tk_feedback_request_t`.
        //    This involves creating a CString for the text.
        // 2. Make the unsafe FFI call to `tk_feedback_manager_enqueue`.
        // 3. Check the return code.

        let c_text = CString::new(request.text)?;
        
        // This is a placeholder as the C struct is not fully defined in the mock FFI
        let c_request = ffi::tk_parsed_command_s {
            // ... fields would be mapped here ...
        };

        log::debug!("Enqueueing feedback request: '{}' with priority {:?}", request.text, request.priority);

        let code = unsafe {
            // This is a conceptual call
            // ffi::tk_feedback_manager_enqueue(self.manager_handle, &c_request)
            ffi::TK_SUCCESS // Assume success for the mock
        };

        if code != ffi::TK_SUCCESS {
            return Err(InteractionError::Ffi(format!(
                "tk_feedback_manager_enqueue failed with code {}",
                code
            )));
        }

        Ok(())
    }
}

impl Drop for FeedbackManager {
    /// Ensures the C-level feedback manager is always destroyed.
    fn drop(&mut self) {
        if !self.manager_handle.is_null() {
            unsafe { ffi::tk_feedback_manager_destroy(&mut self.manager_handle) };
        }
    }
}

impl Default for FeedbackManager {
    fn default() -> Self {
        Self::new().expect("Failed to create default FeedbackManager")
    }
}
