/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/logging_ext/lib.rs
 *
 * This file is the main library entry point for the 'logging_ext' crate.
 * This crate provides extensions to standard logging functionalities, with a
 * focus on structured, machine-readable log formats (like JSON) and helpers
 * for creating detailed audit trails.
 *
 * The goal is to provide a robust logging framework that not only helps with
 * debugging but also satisfies security and compliance requirements for
 * auditing user and system actions.
 *
 * The main components are:
 * - `event_formatter`: Provides logic for formatting log records into
 *   structured formats (e.g., JSON). This can be plugged into logging
 *   implementations like `fern` or `env_logger`.
 * - `audit_helpers`: Offers high-level functions to easily generate
 *   standardized audit log messages for critical events like authentication,
 *   data access, or configuration changes.
 *
 * Since the corresponding C header files are empty, this implementation is
 * fully Rust-native and is intended to be the primary provider of this
*  functionality.
 *
 * Dependencies:
 *   - log: The logging facade this crate builds upon.
 *   - serde, serde_json: For serializing log data into JSON.
 *   - thiserror: For ergonomic error handling.
 *   - chrono: For precise timestamps.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// === TOC ===
// 1. Crate-level Documentation & Attributes
// 2. Public Module Declarations
// 3. Public Prelude
// 4. Core Public Types (Error)
// =============

#![deny(unsafe_code)]
#![deny(missing_docs)]
#![deny(warnings)]

//! # TrackieLLM Logging Extensions Crate
//!
//! Provides structured logging formatters and helpers for creating audit trails.
//!
//! ## Features
//!
//! - **JSON Logging**: A flexible JSON formatter that can be integrated with
//!   popular logging backends to produce machine-readable logs.
//! - **Audit Helpers**: A set of functions to easily log security-critical
//!   events in a standardized format.
//!
//! ## Usage
//!
//! This crate is designed to be used alongside a logging implementation.
//! The `JsonLogFormatter` can be used to configure the logger at startup.
//!
//! ```rust,ignore
//! use logging_ext::event_formatter::JsonLogFormatter;
//! use log::info;
//!
//! fn setup_logger() -> Result<(), fern::InitError> {
//!     fern::Dispatch::new()
//!         .format(JsonLogFormatter::format)
//!         .level(log::LevelFilter::Debug)
//!         .chain(std::io::stdout())
//!         .apply()?;
//!     Ok(())
//! }
//!
//! fn main() {
//!     setup_logger().expect("Failed to set up logger");
//!     info!(target: "app_startup", "Application starting up");
//!     logging_ext::audit_helpers::log_authentication_success("user@example.com");
//! }
//! ```

// --- Public Module Declarations ---

/// Provides logic for formatting log records into structured data.
pub mod event_formatter;

/// Provides helper functions for generating standardized audit log events.
pub mod audit_helpers;


// --- Public Prelude ---

pub mod prelude {
    //! A "prelude" for convenient imports of this crate's main types.
    pub use super::{
        audit_helpers::{log_authentication_success, log_config_change, AuditEvent},
        event_formatter::JsonLogFormatter,
        LoggingExtError,
    };
}


// --- Core Public Types ---

use thiserror::Error;

/// The primary error type for all operations within the logging_ext crate.
#[derive(Debug, Error)]
pub enum LoggingExtError {
    /// An error occurred during the formatting of a log event.
    #[error("Log formatting failed: {0}")]
    Format(#[from] std::fmt::Error),

    /// An error occurred during JSON serialization.
    #[error("JSON serialization failed: {0}")]
    Json(#[from] serde_json::Error),
}
