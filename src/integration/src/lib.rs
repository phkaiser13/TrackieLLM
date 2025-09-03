/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/integration/lib.rs
 *
 * This file is the main library entry point for the 'integration' crate.
 * This crate is responsible for managing all integrations with external
 * systems, third-party services, and dynamically loaded plugins. It serves
 * as the bridge between the core application logic and the outside world.
 *
 * The core components are:
 * - `plugin_manager`: A service for discovering, loading, and managing the
 *   lifecycle of dynamic plugins (e.g., shared libraries).
 * - `bridge`: Provides a generic interface for communication between the core
 *   application and the loaded plugins or external services.
 *
 * Since the corresponding C header file is empty, this is a Rust-native
 * implementation that defines a flexible architecture for extensibility.
 *
 * Dependencies:
 *   - libloading: For dynamically loading shared libraries (.so, .dll, .dylib).
 *   - log: For structured logging.
 *   - thiserror: For ergonomic error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// === TOC ===
// 1. Crate-level Documentation & Attributes
// 2. Public Module Declarations
// 3. Public Prelude
// 4. Core Public Types (Error, Trait definitions)
// =============

#![deny(unsafe_code)]
#![deny(missing_docs)]
#![deny(warnings)]

//! # TrackieLLM Integration Crate
//!
//! Provides a framework for extending the application with plugins and
//! third-party integrations.
//!
//! ## Architecture
//!
//! The heart of this crate is the `Plugin` trait, which defines a common
//! interface that all dynamically loaded plugins must implement. The
//! `PluginManager` is responsible for loading shared libraries from a
//! designated directory, finding symbols that implement the `Plugin` trait,
//! and managing their lifecycle (initialization, execution, shutdown).
//!
//! The `bridge` module provides the communication channel between the core
//! application and the plugins, allowing for safe, structured data exchange.

// --- Public Module Declarations ---

/// Provides a generic interface for communication with external components.
pub mod bridge;

/// Manages the discovery, loading, and lifecycle of dynamic plugins.
pub mod plugin_manager;


// --- Public Prelude ---

pub mod prelude {
    //! A "prelude" for convenient imports of this crate's main types.
    pub use super::{
        bridge::IntegrationBridge,
        plugin_manager::{Plugin, PluginManager},
        IntegrationError,
    };
}


// --- Core Public Types ---

use crate::bridge::BridgeError;
use crate::plugin_manager::PluginError;
use thiserror::Error;

/// The primary error type for all operations within the integration crate.
#[derive(Debug, Error)]
pub enum IntegrationError {
    /// An error occurred in the plugin management system.
    #[error("Plugin error: {0}")]
    Plugin(#[from] PluginError),

    /// An error occurred in the integration bridge.
    #[error("Bridge error: {0}")]
    Bridge(#[from] BridgeError),

    /// The requested integration or plugin is not found.
    #[error("Integration '{0}' not found.")]
    NotFound(String),
}
