/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/profiling/lib.rs
 *
 * This file is the main library entry point for the 'profiling' crate.
 * This crate provides tools for performance profiling and analysis, helping
 * developers identify and diagnose bottlenecks within the application.
 *
 * The profiling system is designed to be lightweight and can be enabled or
 * disabled at runtime to minimize performance overhead in production builds.
 *
 * The core components are:
 * - `profiler_core`: Contains the main logic for starting and stopping profiling
 *   sessions and for capturing profiling data (e.g., call stacks, timings).
 * - `metrics_collector`: A specialized collector (distinct from the one in the
 *   `monitoring` crate) that gathers fine-grained performance metrics during
 *   a profiling session.
 *
 * Since the corresponding C header files are empty, this is a Rust-native
 * implementation. It could be extended in the future to integrate with
 * external profiling tools like `perf`, `DTrace`, or `Tracy`.
 *
 * Dependencies:
 *   - log: For structured logging.
 *   - thiserror: For ergonomic error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// === TOC ===
// 1. Crate-level Documentation & Attributes
// 2. Public Module Declarations
// 3. Public Prelude
// 4. Core Public Types (Error, Profiler)
// =============

#![deny(unsafe_code)]
#![deny(missing_docs)]
#![deny(warnings)]

//! # TrackieLLM Profiling Crate
//!
//! A lightweight, embeddable profiler for performance analysis.
//!
//! ## Overview
//!
//! This crate provides a simple yet powerful API for profiling sections of
//! code. The `Profiler` service can be used to start a profiling session,
//! which will collect data in the background. When the session is stopped,
//! it produces a report that can be used to understand the performance
//! characteristics of the application.
//!
//! ## Example
//!
//! ```rust,ignore
//! use profiling::{Profiler, ProfiledThread};
//! use std::thread;
//! use std::time::Duration;
//!
//! fn main() {
//!     let profiler = Profiler::start();
//!
//!     // Profile some work on the main thread
//!     {
//!         let _main_guard = profiler.profile_thread("main_thread");
//!         thread::sleep(Duration::from_millis(100));
//!     }
//!
//!     // Profile work on a background thread
//!     let handle = thread::spawn(move || {
//!         let _worker_guard = profiler.profile_thread("worker_thread");
//!         thread::sleep(Duration::from_millis(200));
//!     });
//!     handle.join().unwrap();
//!
//!     let report = profiler.stop();
//!     println!("Profiling Report:\n{}", report.generate_summary());
//! }
//! ```

// --- Public Module Declarations ---

/// Contains the core logic for capturing and managing profiling data.
pub mod profiler_core;

/// Collects fine-grained metrics during a profiling session.
pub mod metrics_collector;


// --- Public Prelude ---

pub mod prelude {
    //! A "prelude" for convenient imports of this crate's main types.
    pub use super::{
        profiler_core::{ProfiledThread, Profiler, ProfilingReport},
        ProfilingError,
    };
}


// --- Core Public Types ---

use thiserror::Error;

/// The primary error type for all operations within the profiling crate.
#[derive(Debug, Error)]
pub enum ProfilingError {
    /// The profiler is already running and cannot be started again.
    #[error("Profiler is already running.")]
    AlreadyRunning,

    /// The profiler is not currently running.
    #[error("Profiler is not running.")]
    NotRunning,

    /// An error occurred within the core profiling engine.
    #[error("Profiler core error: {0}")]
    Core(#[from] profiler_core::CoreError),
}
