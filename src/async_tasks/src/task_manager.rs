/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/async_tasks/task_manager.rs
 *
 * This file implements a high-level `TaskManager` service. Its purpose is to
 * provide a simple and unified API for submitting background jobs without
 * needing to directly interact with the low-level details of the async
 * runtime or task spawning.
 *
 * The manager acts as a "fire-and-forget" service for tasks where the result
 * can be awaited via a `JoinHandle`. This is a common pattern for offloading
 * work from a primary thread (e.g., a UI thread or a request handler) to keep
 * it responsive.
 *
 * This implementation is built on top of the `AsyncExecutor` and showcases how
 * different components of the async framework can be composed to build a
 * higher-level service.
 *
 * Dependencies:
 *   - crate::async_executor::AsyncExecutor: The underlying executor.
 *   - futures: For the `Future` trait.
 *   - tokio: For `task::JoinHandle`.
 *   - thiserror: For ergonomic error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use crate::async_executor::AsyncExecutor;
use futures::future::BoxFuture;
use std::future::Future;
use thiserror::Error;
use tokio::task::JoinHandle;

/// A type-erased, pinnable, and sendable future.
/// This is the definition of a "task" that can be run by our executor.
pub typeTaskFuture<T> = BoxFuture<'static, T>;

/// Represents an error that occurs within the logic of a running task.
/// This is distinct from errors related to the task's execution (like panics).
#[derive(Debug, Error)]
#[error("Task failed: {0}")]
pub struct TaskError(pub String);

/// A concrete task that can be submitted to the `TaskManager`.
///
/// It wraps a `BoxFuture` to provide type erasure, allowing the manager to
/// handle futures with different underlying types, as long as they resolve
/// to the same `Output`.
pub struct Task<T: Send + 'static> {
    future: TaskFuture<T>,
}

impl<T: Send + 'static> Task<T> {
    /// Creates a new `Task` from any future that is `Send` and has a `'static`
    /// lifetime.
    ///
    /// # Arguments
    ///
    /// * `future` - The future to be wrapped.
    pub fn new<F>(future: F) -> Self
    where
        F: Future<Output = T> + Send + 'static,
    {
        Self {
            future: Box::pin(future),
        }
    }
}

/// A high-level service for submitting and managing background tasks.
pub struct TaskManager {
    /// The underlying executor that will run the tasks.
    /// An `Arc` is used to allow the manager to be shared across threads if needed.
    executor: std::sync::Arc<AsyncExecutor>,
}

impl TaskManager {
    /// Creates a new `TaskManager`.
    ///
    /// It initializes a new `AsyncExecutor` which will be used for all tasks
    /// submitted to this manager.
    pub fn new() -> Self {
        Self {
            executor: std::sync::Arc::new(AsyncExecutor::new()),
        }
    }

    /// Submits a task to the executor to be run in the background.
    ///
    /// The task is immediately spawned on the async runtime's thread pool.
    ///
    /// # Arguments
    ///
    /// * `task` - The `Task` to be executed.
    ///
    /// # Returns
    ///
    /// A `JoinHandle` which can be used by the caller to await the task's
    /// completion and retrieve its output.
    pub fn submit<T>(&self, task: Task<T>) -> JoinHandle<T>
    where
        T: Send + 'static,
    {
        log::debug!("Submitting new task to the executor.");
        self.executor.spawn(task.future)
    }

    /// A convenience method to spawn a future directly without wrapping it in a `Task`.
    pub fn spawn<F, T>(&self, future: F) -> JoinHandle<T>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        log::debug!("Spawning new future on the executor.");
        self.executor.spawn(future)
    }
}

impl Default for TaskManager {
    /// Creates a new `TaskManager` with a default `AsyncExecutor`.
    fn default() -> Self {
        Self::new()
    }
}
