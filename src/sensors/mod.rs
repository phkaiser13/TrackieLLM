/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/sensors/mod.rs
 *
 * This module is the primary Rust interface for the sensor subsystems. It
 * provides the necessary FFI bindings to the C-based sensor fusion library
 * and defines safe Rust types to interact with it.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#![deny(missing_docs)]

//! # Sensor Subsystem
//!
//! This module provides the Rust interface to the sensor fusion engine.

/// Foreign Function Interface (FFI) bindings to the C sensor library.
pub mod ffi;
