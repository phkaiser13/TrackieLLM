/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: task_manager.rs
 *
 * This file implements a high-level, stateful task manager. It builds upon the
 * `TaskExecutor` to provide a system for submitting, tracking, querying, and
 * cancelling asynchronous tasks. Each task is assigned a unique ID, allowing for
 * fine-grained control and observability over the application's concurrent operations.
 *
 * Dependencies:
 *  - `tokio`, `futures`: For task handles and cancellation via `Abortable`.
 *  - `uuid`: For generating unique, non-sequential task identifiers.
 *  - `thiserror`: For structured error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Imports ---
use crate::async_executor::{TaskExecutor, ExecutorResult};
use futures::future::{AbortHandle, Abortable};
use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};
use tokio::task::JoinHandle;
use uuid::Uuid;

// --- Custom Types and Enums ---

pub type TaskId = Uuid;

/// Represents the current status of a managed task.
#[derive(Debug, Clone)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed(String),
    Cancelled,
}

/// A wrapper around a task's handles and state.
struct ManagedTask {
    abort_handle: AbortHandle,
    // The JoinHandle is stored within an Arc<Mutex<Option<...>>> to allow it
    // to be `take`n and awaited once, while still being accessible for status checks.
    join_handle: Arc<Mutex<Option<JoinHandle<Result<(), String>>>>>,
    status: Arc<Mutex<TaskStatus>>,
}

// --- Task Management Service ---

/// A service for managing the lifecycle of asynchronous tasks.
pub struct TaskManager {
    executor: Arc<TaskExecutor>,
    // The main collection of all tasks known to the manager.
    tasks: Arc<Mutex<HashMap<TaskId, ManagedTask>>>,
}

impl TaskManager {
    /// Creates a new `TaskManager`.
    ///
    /// # Arguments
    /// * `executor` - A shared `TaskExecutor` that will be used to spawn all tasks.
    pub fn new(executor: Arc<TaskExecutor>) -> Self {
        TaskManager {
            executor,
            tasks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Submits an asynchronous, non-blocking task to be managed.
    pub fn submit<F>(&self, future: F) -> ExecutorResult<TaskId>
    where
        F: Future<Output = Result<(), String>> + Send + 'static,
    {
        let task_id = Uuid::new_v4();
        let (abort_handle, abort_registration) = AbortHandle::new_pair();

        let status = Arc::new(Mutex::new(TaskStatus::Pending));
        let status_clone = Arc::clone(&status);

        let managed_future = Abortable::new(async move {
            *status_clone.lock().unwrap() = TaskStatus::Running;
            let result = future.await;
            // Update status based on the future's result
            *status_clone.lock().unwrap() = match result {
                Ok(()) => TaskStatus::Completed,
                Err(e) => TaskStatus::Failed(e),
            };
            result
        }, abort_registration);

        let join_handle = self.executor.spawn(managed_future)?;

        // The outer JoinHandle gives us a `Result<Result<(), Aborted>, JoinError>`.
        // We need to handle this to properly update the status upon cancellation.
        let status_clone2 = Arc::clone(&status);
        let final_join_handle = self.executor.spawn(async move {
            match join_handle.await {
                Ok(Ok(inner_result)) => inner_result, // Task completed or failed normally
                Ok(Err(_)) => { // Aborted
                    *status_clone2.lock().unwrap() = TaskStatus::Cancelled;
                    Err("Task was cancelled".to_string())
                }
                Err(e) => { // JoinError (e.g., runtime shutdown, panic)
                    let err_msg = format!("Task panicked or runtime was shut down: {}", e);
                    *status_clone2.lock().unwrap() = TaskStatus::Failed(err_msg.clone());
                    Err(err_msg)
                }
            }
        })?;

        let task = ManagedTask {
            abort_handle,
            join_handle: Arc::new(Mutex::new(Some(final_join_handle))),
            status,
        };

        self.tasks.lock().unwrap().insert(task_id, task);
        Ok(task_id)
    }

    /// Retrieves the current status of a task.
    pub fn get_status(&self, id: &TaskId) -> Option<TaskStatus> {
        self.tasks
            .lock()
            .unwrap()
            .get(id)
            .map(|task| task.status.lock().unwrap().clone())
    }

    /// Cancels a running task.
    ///
    /// If the task has already completed, failed, or been cancelled, this has no effect.
    ///
    /// # Returns
    /// `true` if the cancellation signal was sent, `false` if the task was not found.
    pub fn cancel_task(&self, id: &TaskId) -> bool {
        if let Some(task) = self.tasks.lock().unwrap().get(id) {
            // AbortHandle.abort() can be called multiple times safely.
            task.abort_handle.abort();
            true
        } else {
            false
        }
    }

    /// Waits for a task to complete and returns its result.
    /// This consumes the task's `JoinHandle` and removes it from the manager.
    pub async fn await_result(&self, id: &TaskId) -> Option<Result<(), String>> {
        let task = self.tasks.lock().unwrap().remove(id);
        if let Some(task) = task {
            if let Some(handle) = task.join_handle.lock().unwrap().take() {
                return handle.await.ok();
            }
        }
        None
    }
}

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::sleep;

    fn setup_manager() -> TaskManager {
        let executor = Arc::new(TaskExecutor::new(2).unwrap());
        TaskManager::new(executor)
    }

    #[tokio::test]
    async fn test_submit_and_get_status() {
        let manager = setup_manager();
        let task_id = manager.submit(async {
            sleep(Duration::from_millis(50)).await;
            Ok(())
        }).unwrap();

        // Status should be Pending or Running immediately after submission
        let status = manager.get_status(&task_id).unwrap();
        assert!(matches!(status, TaskStatus::Pending) || matches!(status, TaskStatus::Running));

        sleep(Duration::from_millis(100)).await;

        // Status should be Completed after the task finishes
        let final_status = manager.get_status(&task_id).unwrap();
        assert!(matches!(final_status, TaskStatus::Completed));
    }

    #[tokio::test]
    async fn test_task_cancellation() {
        let manager = setup_manager();
        let task_id = manager.submit(async {
            sleep(Duration::from_secs(5)).await; // A long-running task
            Ok(())
        }).unwrap();

        let cancelled = manager.cancel_task(&task_id);
        assert!(cancelled);

        // Give a moment for the cancellation to propagate
        sleep(Duration::from_millis(10)).await;

        let status = manager.get_status(&task_id).unwrap();
        assert!(matches!(status, TaskStatus::Cancelled));
    }

    #[tokio::test]
    async fn test_task_failure_status() {
        let manager = setup_manager();
        let task_id = manager.submit(async {
            Err("Something went wrong!".to_string())
        }).unwrap();

        sleep(Duration::from_millis(50)).await;

        let status = manager.get_status(&task_id).unwrap();
        assert!(matches!(status, TaskStatus::Failed(_)));
        if let TaskStatus::Failed(e) = status {
            assert_eq!(e, "Something went wrong!");
        }
    }

    #[tokio::test]
    async fn test_await_result() {
        let manager = setup_manager();
        let task_id = manager.submit(async { Ok(()) }).unwrap();

        let result = manager.await_result(&task_id).await;
        assert!(result.is_some());
        assert!(result.unwrap().is_ok());

        // Task should be removed from the manager after awaiting
        assert!(manager.get_status(&task_id).is_none());
    }
}
