/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/async_tasks/lib.rs
 *
 * This file is the main library entry point for the 'async_tasks' crate.
 * This crate provides the core infrastructure for managing asynchronous
 * operations, background jobs, and concurrent tasks within the TrackieLLM
 * application.
 *
 * It is built upon a flexible, high-performance asynchronous runtime and
 * provides two key components:
 *
 * - `async_executor`: A thin abstraction over the chosen async runtime (e.g.,
 *   `tokio`). It is responsible for spawning tasks and running the event loop.
 * - `task_manager`: A high-level service for submitting, tracking, and managing
 *   the lifecycle of background jobs. It provides features like task
 *   cancellation, status tracking, and result retrieval.
 *
 * Since the corresponding C header files are empty, this is a Rust-native
 * implementation that leverages the powerful concurrency features of the
 * Rust ecosystem.
 *
 * Dependencies:
 *   - tokio: For the asynchronous runtime, channels, and synchronization primitives.
 *   - futures: For core asynchronous traits and utilities.
 *   - log: For structured logging.
 *   - thiserror: For ergonomic error handling.
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

//! # TrackieLLM Asynchronous Task Management Crate
//!
//! Provides a robust framework for running and managing concurrent tasks.
//!
//! ## Architecture
//!
//! The crate is designed around a central `TaskManager` that operates on top
//! of an `AsyncExecutor`. The executor is responsible for the low-level details
//! of running futures on a thread pool, while the manager provides a higher-level
//! API for application logic to interact with background jobs.
//!
//! ### Example Usage
//!
//! ```rust,ignore
//! use async_tasks::{TaskManager, Task, TaskError};
//! use std::time::Duration;
//!
//! // A simple async task that simulates some work.
//! async fn my_background_task(n: u32) -> Result<String, TaskError> {
//!     println!("Task starting: waiting for {} seconds.", n);
//!     tokio::time::sleep(Duration::from_secs(n as u64)).await;
//!     if n == 3 {
//!         return Err(TaskError::Failed("Simulated failure".to_string()));
//!     }
//!     Ok(format!("Task finished after {} seconds", n))
//! }
//!
//! #[tokio::main]
//! async fn main() {
//!     let task_manager = TaskManager::new();
//!
//!     // Submit a task and get a handle to it.
//!     let handle = task_manager.submit(Task::new(my_background_task(5)));
//!
//!     // ... do other work ...
//!
//!     // Await the result of the task.
//!     match handle.await {
//!         Ok(Ok(result_string)) => println!("Task succeeded: {}", result_string),
//!         Ok(Err(task_err)) => println!("Task failed with internal error: {}", task_err),
//!         Err(join_err) => println!("Failed to join task: {}", join_err),
//!     }
//! }
//! ```

// --- Public Module Declarations ---

/// Provides the core asynchronous executor and task spawning capabilities.
pub mod async_executor;

/// Provides a high-level service for managing background tasks.
pub mod task_manager;


// --- Public Prelude ---

pub mod prelude {
    //! A "prelude" for convenient imports of this crate's main types.
    pub use super::{
        async_executor::AsyncExecutor,
        task_manager::{Task, TaskError, TaskManager},
        AsyncTasksError,
    };
}


// --- Core Public Types ---

use thiserror::Error;

/// The primary error type for the `async_tasks` crate.
///
/// This enum consolidates errors from the task manager and the underlying
/// async runtime.
#[derive(Debug, Error)]
pub enum AsyncTasksError {
    /// The async runtime failed to initialize.
    #[error("Async runtime failed to start: {0}")]
    RuntimeStartup(String),

    /// A submitted task handle was lost or could not be joined.
    #[error("Failed to join task handle: {0}")]
    TaskJoin(#[from] tokio::task::JoinError),

    /// The task manager's communication channel is broken.
    #[error("Task manager channel is broken.")]
    ChannelBroken,
}
