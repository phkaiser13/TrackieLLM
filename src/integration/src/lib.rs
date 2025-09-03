/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: lib.rs
 *
 * This file is the main library entry point and FFI boundary for the `integration`
 * crate. It exposes the `PluginManager`'s functionality to the C/C++ core, allowing
 * it to load, unload, and list dynamic plugins in a safe, controlled manner.
 *
 * Dependencies:
 *  - `lazy_static`: For the global, thread-safe plugin manager instance.
 *  - `serde_json`: For serializing the list of plugins for the C side.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Declarations and Imports ---
pub mod bridge;
pub mod plugin_manager;

use bridge::BridgeContext;
use lazy_static::lazy_static;
use plugin_manager::PluginManager;
use std::ffi::{c_char, CStr, CString};
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;
use std::sync::Mutex;

// --- Global State Management ---

lazy_static! {
    // The global PluginManager, wrapped in a Mutex for thread-safe access from FFI.
    static ref PLUGIN_MANAGER: Mutex<PluginManager> = Mutex::new(PluginManager::new());
}

// --- FFI Helper Functions ---

/// Helper to run a closure and catch any panics.
fn catch_panic<F, R>(f: F) -> R where F: FnOnce() -> R, R: Default {
    panic::catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| {
        eprintln!("Error: A panic occurred within the Rust FFI boundary.");
        R::default()
    })
}

// --- FFI Public Interface ---

/// Loads a plugin from a specified shared library path.
///
/// # Arguments
/// - `path_c`: A C-string with the path to the shared library file (.so, .dll, .dylib).
///
/// # Returns
/// - `0` on success.
/// - `-1` on failure (e.g., file not found, symbol missing, plugin init failed).
///
/// # Safety
/// The caller must provide a valid, null-terminated C-string path.
#[no_mangle]
pub extern "C" fn integration_load_plugin(path_c: *const c_char) -> i32 {
    catch_panic(|| {
        let path_str = unsafe { CStr::from_ptr(path_c).to_str().unwrap() };
        let library_path = Path::new(path_str);

        // For now, we create a default context. In a real app, this might be
        // constructed with more meaningful data from the C side.
        let context = BridgeContext {
            host_version: env!("CARGO_PKG_VERSION").to_string(),
        };

        let mut manager = PLUGIN_MANAGER.lock().unwrap();
        match manager.load(library_path, context) {
            Ok(_) => 0,
            Err(e) => {
                eprintln!("Failed to load plugin from '{}': {}", path_str, e);
                -1
            }
        }
    })
}

/// Unloads a plugin by its registered name.
///
/// # Arguments
/// - `name_c`: A C-string with the name of the plugin to unload.
///
/// # Returns
/// - `0` on success.
/// - `-1` if no plugin with that name was found.
///
/// # Safety
/// The caller must provide a valid, null-terminated C-string.
#[no_mangle]
pub extern "C" fn integration_unload_plugin(name_c: *const c_char) -> i32 {
    catch_panic(|| {
        let name = unsafe { CStr::from_ptr(name_c).to_str().unwrap() };
        let mut manager = PLUGIN_MANAGER.lock().unwrap();
        match manager.unload(name) {
            Ok(_) => 0,
            Err(e) => {
                eprintln!("Failed to unload plugin '{}': {}", name, e);
                -1
            }
        }
    })
}

/// Retrieves a list of all currently loaded plugin names.
///
/// # Returns
/// A C-string containing a JSON array of plugin names. This string must be freed
/// by the caller using `integration_free_string`. Returns null on error.
#[no_mangle]
pub extern "C" fn integration_list_plugins() -> *mut c_char {
    catch_panic(|| {
        let manager = PLUGIN_MANAGER.lock().unwrap();
        let plugin_names = manager.list_plugins();

        match serde_json::to_string(&plugin_names) {
            Ok(json_str) => CString::new(json_str).unwrap().into_raw(),
            Err(e) => {
                eprintln!("Failed to serialize plugin list to JSON: {}", e);
                std::ptr::null_mut()
            }
        }
    })
}


/// Frees a C-string that was allocated by this Rust library.
///
/// # Safety
/// Must be called with a non-null pointer previously returned by this library.
#[no_mangle]
pub extern "C" fn integration_free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    unsafe {
        drop(CString::from_raw(s));
    }
}

// --- Unit Tests ---
// FFI tests are conceptual as they require a C test harness or a compiled plugin.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_list_plugins_initially_empty() {
        let result_ptr = integration_list_plugins();
        assert!(!result_ptr.is_null());

        let result_str = unsafe { CStr::from_ptr(result_ptr).to_str().unwrap() };
        assert_eq!(result_str, "[]");

        integration_free_string(result_ptr);
    }

    // This test is conceptual and would fail without a real, compiled plugin
    // available at a known path.
    #[test]
    #[ignore]
    fn conceptual_test_ffi_load_unload_and_list() {
        // Assume "dummy_plugin.so" is available
        let plugin_path = CString::new("path/to/dummy_plugin.so").unwrap();

        // Load
        let res_load = integration_load_plugin(plugin_path.as_ptr());
        assert_eq!(res_load, 0);

        // List
        let list_ptr_1 = integration_list_plugins();
        let list_str_1 = unsafe { CStr::from_ptr(list_ptr_1).to_str().unwrap() };
        assert_eq!(list_str_1, r#"["DummyPlugin"]"#);
        integration_free_string(list_ptr_1);

        // Unload
        let plugin_name = CString::new("DummyPlugin").unwrap();
        let res_unload = integration_unload_plugin(plugin_name.as_ptr());
        assert_eq!(res_unload, 0);

        // List again
        let list_ptr_2 = integration_list_plugins();
        let list_str_2 = unsafe { CStr::from_ptr(list_ptr_2).to_str().unwrap() };
        assert_eq!(list_str_2, "[]");
        integration_free_string(list_ptr_2);
    }
}
