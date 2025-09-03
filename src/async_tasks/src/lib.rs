/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: lib.rs
 *
 * This file is the main library entry point and FFI boundary for the `async_tasks`
 * crate. It exposes a powerful asynchronous task execution framework to a C/C++
 * environment. The most critical feature is the ability to accept a C function
 * pointer and execute it on a managed Rust thread pool, allowing the C side to
 * offload blocking work safely and efficiently.
 *
 * Dependencies:
 *  - `lazy_static`: For the global, thread-safe task manager.
 *  - `serde_json`: For serializing status objects for the C side.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Declarations and Imports ---
pub mod async_executor;
pub mod task_manager;

use async_executor::TaskExecutor;
use lazy_static::lazy_static;
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Arc, Mutex};
use task_manager::{TaskId, TaskManager};
use uuid::Uuid;

// --- Global State Management ---

lazy_static! {
    // The global TaskManager, wrapped for thread-safe access.
    // It's an Option so it can be initialized by an explicit FFI call.
    static ref TASK_MANAGER: Mutex<Option<Arc<TaskManager>>> = Mutex::new(None);
}

// --- FFI Helper and Safety Wrapper ---

/// A wrapper to make a raw pointer `Send`.
/// This is unsafe and relies on the C caller to guarantee that the provided
/// `user_data` pointer is safe to be sent and used across threads.
struct SendablePointer(*mut c_void);
unsafe impl Send for SendablePointer {}

/// Helper to run a closure and catch any panics.
fn catch_panic<F, R>(f: F) -> R where F: FnOnce() -> R, R: Default {
    panic::catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| {
        eprintln!("Error: A panic occurred within the Rust FFI boundary.");
        R::default()
    })
}

// --- FFI Public Interface ---

/// Initializes the global task manager and its underlying thread pool.
/// This function must be called once before any other functions in this module.
///
/// # Arguments
/// - `num_threads`: The number of worker threads to create. If `0`, it will use
///   the number of logical CPU cores.
///
/// # Returns
/// `0` on success, `-1` if the manager is already initialized.
#[no_mangle]
pub extern "C" fn async_tasks_init(num_threads: usize) -> i32 {
    catch_panic(|| {
        let mut manager_guard = TASK_MANAGER.lock().unwrap();
        if manager_guard.is_some() {
            eprintln!("Task manager is already initialized.");
            return -1;
        }

        let threads = if num_threads == 0 {
            num_cpus::get()
        } else {
            num_threads
        };

        let executor = Arc::new(TaskExecutor::new(threads).expect("Failed to create executor"));
        let manager = Arc::new(TaskManager::new(executor));
        *manager_guard = Some(manager);

        println!("Async task manager initialized with {} threads.", threads);
        0
    })
}

/// Submits a C function to be executed asynchronously on the Rust thread pool.
///
/// # Arguments
/// - `func_ptr`: A pointer to a C function with the signature `extern "C" fn(*mut c_void)`.
/// - `user_data`: A raw pointer to data that will be passed to the C function.
///
/// # Returns
/// A C-string containing the unique `TaskId` for the submitted task. This string
/// must be freed by the caller using `async_tasks_free_string`. Returns null on error.
///
/// # Safety
/// - The caller must ensure `func_ptr` is a valid, non-null function pointer.
/// - The caller is responsible for the thread safety of the data pointed to by `user_data`.
/// - The lifetime of `user_data` must exceed the execution time of the task.
#[no_mangle]
pub extern "C" fn async_tasks_submit_c_task(
    func_ptr: extern "C" fn(*mut c_void),
    user_data: *mut c_void,
) -> *mut c_char {
    catch_panic(|| {
        let manager_guard = TASK_MANAGER.lock().unwrap();
        let manager = match manager_guard.as_ref() {
            Some(m) => m.clone(),
            None => {
                eprintln!("Task manager not initialized. Call async_tasks_init first.");
                return std::ptr::null_mut();
            }
        };

        let ptr_wrapper = SendablePointer(user_data);

        // We use `submit` with a blocking task wrapper because the C function is opaque
        // and likely blocking.
        let task_id_result = manager.submit(async move {
            // This closure is `Send` because `SendablePointer` is `Send`.
            tokio::task::spawn_blocking(move || {
                // Execute the C function pointer.
                func_ptr(ptr_wrapper.0);
            }).await.map_err(|e| e.to_string())?;
            Ok(())
        });

        match task_id_result {
            Ok(task_id) => {
                let id_str = task_id.to_string();
                CString::new(id_str).unwrap().into_raw()
            }
            Err(e) => {
                eprintln!("Failed to submit task: {}", e);
                std::ptr::null_mut()
            }
        }
    })
}

/// Retrieves the current status of a task by its ID.
///
/// # Returns
/// A C-string with the status (e.g., "running", "completed"). Must be freed.
#[no_mangle]
pub extern "C" fn async_tasks_get_status(task_id_c: *const c_char) -> *mut c_char {
    catch_panic(|| {
        let task_id_str = unsafe { CStr::from_ptr(task_id_c).to_str().unwrap() };
        let task_id = Uuid::parse_str(task_id_str).unwrap();

        let manager_guard = TASK_MANAGER.lock().unwrap();
        if let Some(manager) = manager_guard.as_ref() {
            let status = manager.get_status(&task_id);
            let status_str = format!("{:?}", status.unwrap_or(task_manager::TaskStatus::Pending)); // Default to Pending if not found yet
            CString::new(status_str).unwrap().into_raw()
        } else {
            CString::new("Uninitialized").unwrap().into_raw()
        }
    })
}

/// Signals a task to be cancelled.
#[no_mangle]
pub extern "C" fn async_tasks_cancel_task(task_id_c: *const c_char) -> bool {
    catch_panic(|| {
        let task_id_str = unsafe { CStr::from_ptr(task_id_c).to_str().unwrap() };
        let task_id = Uuid::parse_str(task_id_str).unwrap();

        let manager_guard = TASK_MANAGER.lock().unwrap();
        if let Some(manager) = manager_guard.as_ref() {
            manager.cancel_task(&task_id)
        } else {
            false
        }
    })
}

/// Frees a C-string that was allocated by this Rust library.
#[no_mangle]
pub extern "C" fn async_tasks_free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    unsafe {
        drop(CString::from_raw(s));
    }
}

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    // A C-style function to be used in tests.
    extern "C" fn test_c_function(data: *mut c_void) {
        // Cast the void pointer back to its real type and modify it.
        let finished_flag = unsafe { &*(data as *const AtomicBool) };
        std::thread::sleep(std::time::Duration::from_millis(50));
        finished_flag.store(true, Ordering::SeqCst);
    }

    #[test]
    fn test_ffi_task_submission_and_status() {
        async_tasks_init(1);
        let finished = Arc::new(AtomicBool::new(false));
        // Get a raw pointer to the Arc's data.
        let data_ptr = Arc::into_raw(finished.clone()) as *mut c_void;

        let task_id_ptr = async_tasks_submit_c_task(test_c_function, data_ptr);
        assert!(!task_id_ptr.is_null());

        let status_ptr = async_tasks_get_status(task_id_ptr);
        let status_str = unsafe { CStr::from_ptr(status_ptr).to_str().unwrap() };
        assert!(status_str == "Running" || status_str == "Pending");
        async_tasks_free_string(status_ptr);

        // Wait for the task to finish
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert_eq!(finished.load(Ordering::SeqCst), true);

        let final_status_ptr = async_tasks_get_status(task_id_ptr);
        let final_status_str = unsafe { CStr::from_ptr(final_status_ptr).to_str().unwrap() };
        assert_eq!(final_status_str, "Completed");

        // Clean up
        async_tasks_free_string(task_id_ptr);
        async_tasks_free_string(final_status_ptr);
        // "Reconstitute" the Arc to drop it properly and avoid memory leaks.
        unsafe { drop(Arc::from_raw(data_ptr as *const AtomicBool)); }
    }
}
