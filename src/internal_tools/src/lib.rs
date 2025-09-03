/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: lib.rs
 *
 * This file is the main library entry point and FFI boundary for the `internal_tools`
 * crate. It exposes utility functions for configuration loading and file system
 * operations to the C/C++ core. The FFI layer uses JSON for data interchange and
 * provides a memory management function for strings allocated by Rust.
 *
 * Dependencies:
 *  - `lazy_static`: For thread-safe initialization of the global manager.
 *  - `serde_json`: For serializing results to JSON strings for the C side.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Declarations and Imports ---
pub mod config_loader;
pub mod fs_utils;

use config_loader::{AppConfig, ConfigLoader, Configurable};
use fs_utils::{FileSystemUtils, FindOptions};
use lazy_static::lazy_static;
use serde::Serialize;
use std::ffi::{c_char, CStr, CString};
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;
use std::sync::Mutex;

// --- Global State Management ---

/// A manager struct to hold instances of our utility services.
struct InternalToolsManager {
    config_loader: ConfigLoader,
    fs_utils: FileSystemUtils,
}

lazy_static! {
    static ref MANAGER: Mutex<InternalToolsManager> = Mutex::new(InternalToolsManager {
        config_loader: ConfigLoader::new(),
        fs_utils: FileSystemUtils::new(),
    });
}

// --- FFI Helper Functions ---

/// Helper to run a closure and catch any panics.
fn catch_panic<F>(f: F) -> *mut c_char
where
    F: FnOnce() -> *mut c_char,
{
    panic::catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| {
        eprintln!("Error: A panic occurred within the Rust FFI boundary.");
        std::ptr::null_mut()
    })
}

/// Serializes a Rust struct into a C-compatible, null-terminated string.
fn serialize_to_c_string<T: Serialize>(data: &T) -> *mut c_char {
    match serde_json::to_string(data) {
        Ok(json_str) => CString::new(json_str).unwrap().into_raw(),
        Err(e) => {
            eprintln!("Error: Failed to serialize response to JSON: {}", e);
            std::ptr::null_mut()
        }
    }
}

// --- FFI Public Interface ---

/// Loads an application configuration from a specified file path.
///
/// # Arguments
/// - `config_path_c`: A C-string with the path to the configuration file (e.g., "config.toml").
///
/// # Returns
/// A pointer to a new C-string containing the loaded configuration as JSON.
/// This string must be freed by the caller using `internal_tools_free_string`.
/// Returns a null pointer on error.
///
/// # Safety
/// The caller must provide a valid, null-terminated C-string and must free the returned string.
#[no_mangle]
pub extern "C" fn internal_tools_load_config(config_path_c: *const c_char) -> *mut c_char {
    catch_panic(|| {
        let path_str = unsafe { CStr::from_ptr(config_path_c).to_str().unwrap() };
        let path = Path::new(path_str);

        let manager = MANAGER.lock().unwrap();
        match manager.config_loader.load_from_file::<AppConfig>(path) {
            Ok(config) => serialize_to_c_string(&config),
            Err(e) => {
                eprintln!("Error loading configuration: {}", e);
                std::ptr::null_mut()
            }
        }
    })
}

/// Finds files recursively in a directory based on a single glob pattern.
///
/// # Arguments
/// - `root_c`: A C-string with the path to the root directory for the search.
/// - `pattern_c`: A C-string with the glob pattern (e.g., "*.log").
///
/// # Returns
/// A pointer to a new C-string containing a JSON array of the found file paths.
/// This string must be freed by the caller using `internal_tools_free_string`.
/// Returns a null pointer on error.
///
/// # Safety
/// The caller must provide valid C-strings and free the returned string.
#[no_mangle]
pub extern "C" fn internal_tools_find_files(
    root_c: *const c_char,
    pattern_c: *const c_char,
) -> *mut c_char {
    catch_panic(|| {
        let root_str = unsafe { CStr::from_ptr(root_c).to_str().unwrap() };
        let pattern_str = unsafe { CStr::from_ptr(pattern_c).to_str().unwrap() };

        let root = Path::new(root_str);
        let patterns = [pattern_str];
        let options = FindOptions::default(); // Use default options for simplicity in FFI

        let manager = MANAGER.lock().unwrap();
        match manager.fs_utils.find_files(root, &patterns, &options) {
            Ok(files) => {
                // Convert PathBufs to strings for serialization
                let file_strings: Vec<_> = files.iter().map(|p| p.to_string_lossy()).collect();
                serialize_to_c_string(&file_strings)
            },
            Err(e) => {
                eprintln!("Error finding files: {}", e);
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
pub extern "C" fn internal_tools_free_string(s: *mut c_char) {
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
    use assert_fs::prelude::*;
    use assert_fs::TempDir;

    #[test]
    fn test_ffi_load_config_and_free() {
        let dir = TempDir::new().unwrap();
        let config_file = dir.child("settings.json");
        config_file.write_str(r#"{ "log_level": "info" }"#).unwrap();

        let path_c = CString::new(config_file.path().to_str().unwrap()).unwrap();
        let result_ptr = internal_tools_load_config(path_c.as_ptr());
        assert!(!result_ptr.is_null());

        let result_str = unsafe { CStr::from_ptr(result_ptr).to_str().unwrap() };
        assert!(result_str.contains(r#""log_level":"info""#));

        internal_tools_free_string(result_ptr);
    }

    #[test]
    fn test_ffi_find_files_and_free() {
        let dir = TempDir::new().unwrap();
        dir.child("data.csv").touch().unwrap();
        dir.child("image.jpg").touch().unwrap();
        dir.child("archive.zip").touch().unwrap();

        let root_c = CString::new(dir.path().to_str().unwrap()).unwrap();
        let pattern_c = CString::new("*.{csv,zip}").unwrap(); // Note: this is a Brace expression, not two patterns

        // The glob crate doesn't support brace expansion like a shell.
        // We'll test for a single pattern.
        let pattern_c_single = CString::new("*.csv").unwrap();
        let result_ptr = internal_tools_find_files(root_c.as_ptr(), pattern_c_single.as_ptr());
        assert!(!result_ptr.is_null());

        let result_str = unsafe { CStr::from_ptr(result_ptr).to_str().unwrap() };
        let result_json: Vec<String> = serde_json::from_str(result_str).unwrap();

        assert_eq!(result_json.len(), 1);
        assert!(result_json[0].ends_with("data.csv"));

        internal_tools_free_string(result_ptr);
    }
}
