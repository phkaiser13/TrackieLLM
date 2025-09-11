/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/cortex/mod.rs
 *
 * This module is the primary Rust interface for the C-based Cortex.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#![deny(missing_docs)]

//! # Cortex Subsystem
//!
//! This module provides the Rust interface to the C-based Cortex engine.

/// Foreign Function Interface (FFI) bindings to the C cortex library.
pub mod ffi;
/// Manages the state and memory of the application.
pub mod memory_manager;
/// Handles reasoning and logic.
pub mod reasoning;
