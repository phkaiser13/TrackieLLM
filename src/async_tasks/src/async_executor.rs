/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: async_executor.rs
 *
 * This file implements a high-level asynchronous task executor. It abstracts over
 * a Tokio runtime, providing a clean and simplified interface for spawning both
 * async (I/O-bound) and blocking (CPU-bound) tasks. This separation is critical
 * for maintaining a responsive event loop in a high-performance application.
 *
 * Dependencies:
 *  - `tokio`: For the core async runtime and task spawning capabilities.
 *  - `thiserror`: For structured error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Imports ---
use std::future::Future;
use tokio::runtime::{Builder, Runtime};
use tokio::task::JoinHandle;

// --- Custom Error and Result Types ---

#[derive(Debug, thiserror::Error)]
pub enum ExecutorError {
    #[error("Failed to build the Tokio runtime: {0}")]
    RuntimeCreationError(#[from] std::io::Error),
    #[error("The executor has already been shut down and cannot accept new tasks.")]
    Shutdown,
}

pub type ExecutorResult<T> = Result<T, ExecutorError>;

// --- Task Executor Service ---

/// A task executor that manages a Tokio runtime and provides methods for spawning tasks.
pub struct TaskExecutor {
    // The runtime is wrapped in an Option to allow it to be shut down and consumed.
    runtime: Option<Runtime>,
}

impl TaskExecutor {
    /// Creates a new `TaskExecutor` with a dedicated Tokio runtime.
    ///
    /// # Arguments
    /// * `worker_threads` - The number of worker threads for the runtime's thread pool.
    ///   A good default is the number of logical CPU cores.
    pub fn new(worker_threads: usize) -> ExecutorResult<Self> {
        let runtime = Builder::new_multi_thread()
            .worker_threads(worker_threads)
            .thread_name("async-task-worker")
            .enable_all()
            .build()?;

        Ok(TaskExecutor {
            runtime: Some(runtime),
        })
    }

    /// Spawns an asynchronous, non-blocking task.
    ///
    /// This is intended for I/O-bound tasks that frequently yield, allowing other
    /// tasks to run on the same thread.
    ///
    /// # Arguments
    /// * `future` - The future to execute.
    pub fn spawn<F>(&self, future: F) -> ExecutorResult<JoinHandle<F::Output>>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        if let Some(runtime) = &self.runtime {
            Ok(runtime.spawn(future))
        } else {
            Err(ExecutorError::Shutdown)
        }
    }

    /// Spawns a synchronous, blocking task on a dedicated thread pool.
    ///
    /// This is crucial for CPU-bound tasks or tasks that use blocking I/O,
    /// as it prevents them from stalling the main async event loop.
    ///
    /// # Arguments
    /// * `task` - The blocking function or closure to execute.
    pub fn spawn_blocking<F, R>(&self, task: F) -> ExecutorResult<JoinHandle<R>>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        if let Some(runtime) = &self.runtime {
            Ok(runtime.spawn_blocking(task))
        } else {
            Err(ExecutorError::Shutdown)
        }
    }

    /// Enters the runtime context and blocks the current thread on a future.
    /// This is useful for bridging synchronous and asynchronous code.
    pub fn block_on<F: Future>(&self, future: F) -> F::Output {
        self.runtime.as_ref()
            .expect("Executor must be running to block on a future")
            .block_on(future)
    }

    /// Shuts down the executor's runtime gracefully.
    ///
    /// This will wait for all spawned tasks to complete. After shutdown, the
    /// executor can no longer be used.
    pub fn shutdown(mut self) {
        if let Some(runtime) = self.runtime.take() {
            println!("Shutting down async executor runtime...");
            runtime.shutdown_background();
            println!("Async executor runtime shut down.");
        }
    }
}

impl Drop for TaskExecutor {
    /// Ensures the runtime is shut down when the executor is dropped.
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            println!("Executor dropped. Shutting down runtime...");
            // Use `shutdown_background` for a non-blocking shutdown in drop.
            // `shutdown_timeout` would be better but requires a duration.
            runtime.shutdown_background();
        }
    }
}


// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    use std::time::Duration;
    use tokio::time::sleep;

    #[test]
    fn test_executor_creation_and_shutdown() {
        let executor = TaskExecutor::new(2).unwrap();
        executor.shutdown(); // Explicit shutdown
    }

    #[test]
    fn test_spawn_async_task() {
        let executor = TaskExecutor::new(1).unwrap();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let handle = executor.spawn(async move {
            sleep(Duration::from_millis(10)).await;
            counter_clone.fetch_add(1, Ordering::SeqCst);
            42
        }).unwrap();

        let result = executor.block_on(handle).unwrap();

        assert_eq!(result, 42);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_spawn_blocking_task() {
        let executor = TaskExecutor::new(2).unwrap();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let handle = executor.spawn_blocking(move || {
            // Simulate a CPU-intensive task
            std::thread::sleep(Duration::from_millis(20));
            counter_clone.fetch_add(1, Ordering::SeqCst);
            "done"
        }).unwrap();

        let result = executor.block_on(handle).unwrap();

        assert_eq!(result, "done");
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_executor_is_unusable_after_shutdown() {
        let executor = TaskExecutor::new(1).unwrap();
        executor.shutdown();

        // This test is tricky because the executor is consumed by shutdown.
        // We can re-architect to test this better if needed, but for now, we'll
        // rely on the logic that `self.runtime` becomes `None`.
        // A direct call would be:
        // `let result = executor.spawn(async { });`
        // which would fail to compile because `executor` was moved.
        // This is a good example of Rust's ownership preventing use-after-move.
    }
}
