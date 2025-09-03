/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/async_tasks/async_executor.rs
 *
 * This file implements the `AsyncExecutor`, which is a thin wrapper around the
 * chosen asynchronous runtime. The primary purpose of this module is to
 * decouple the rest of the application from the specific choice of runtime
 * (e.g., `tokio`, `async-std`).
 *
 * By creating this abstraction, we can easily swap out the underlying async
 * runtime in the future without having to refactor large parts of the codebase.
 * All task spawning and management should go through this executor or a
 * higher-level service like the `TaskManager`.
 *
 * This implementation uses the `tokio` multi-threaded runtime, which is a
 * high-performance, work-stealing runtime suitable for a wide range of
 * asynchronous workloads.
 *
 * Dependencies:
 *   - tokio: For the `Runtime` and `JoinHandle`.
 *   - futures: For the `Future` trait.
 *   - log: For structured logging.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use futures::Future;
use tokio::runtime::{Builder, Runtime};
use tokio::task::JoinHandle;

/// A wrapper around an asynchronous runtime.
///
/// This struct holds an instance of the `tokio` runtime and provides a simple
/// interface for spawning tasks onto it.
pub struct AsyncExecutor {
    /// The underlying `tokio` runtime.
    /// The `Option` is used to allow us to `take` ownership of the runtime
    /// during a graceful shutdown.
    runtime: Option<Runtime>,
}

impl AsyncExecutor {
    /// Creates a new `AsyncExecutor` and starts its underlying runtime.
    ///
    /// This will build a new `tokio` multi-threaded runtime with a configurable
    /// number of worker threads.
    ///
    /// # Panics
    ///
    /// Panics if the `tokio` runtime fails to be created.
    pub fn new() -> Self {
        log::info!("Initializing new multi-threaded async executor.");
        let runtime = Builder::new_multi_thread()
            .worker_threads(num_cpus::get_physical()) // Use physical cores for less context switching
            .thread_name("async-worker")
            .enable_all() // Enable both I/O and time drivers
            .build()
            .expect("Failed to build tokio runtime.");

        Self {
            runtime: Some(runtime),
        }
    }

    /// Spawns a new asynchronous task to be run on this executor.
    ///
    /// The task will be executed on the runtime's thread pool.
    ///
    /// # Arguments
    ///
    /// * `future` - The future to be executed. It must be `Send` and have a
    ///   `'static` lifetime to be safely moved between threads.
    ///
    /// # Returns
    ///
    /// A `JoinHandle` that can be used to await the result of the future.
    pub fn spawn<F, T>(&self, future: F) -> JoinHandle<T>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        self.runtime
            .as_ref()
            .expect("AsyncExecutor runtime is not initialized.")
            .spawn(future)
    }

    /// Shuts down the executor's runtime gracefully.
    ///
    /// This will wait for all spawned tasks to complete. It is automatically
    /// called when the `AsyncExecutor` is dropped.
    pub fn shutdown(mut self) {
        log::info!("Shutting down async executor...");
        if let Some(runtime) = self.runtime.take() {
            // `shutdown_background` will wait for all spawned tasks to complete.
            runtime.shutdown_background();
            log::info!("Async executor has been shut down.");
        }
    }
}

impl Default for AsyncExecutor {
    /// Creates a new `AsyncExecutor` with a default configuration.
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AsyncExecutor {
    /// Ensures that the runtime is shut down when the executor goes out of scope.
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            log::warn!("AsyncExecutor dropped without explicit shutdown. Shutting down now.");
            // `shutdown_background` is used here because a blocking shutdown
            // can cause issues in `drop`. The OS will still clean up, but this
            // gives tasks a chance to finish.
            runtime.shutdown_background();
        }
    }
}
