/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/internal_tools/config_loader.rs
 *
 * This file provides a safe Rust wrapper for the C-based configuration
 * parser defined in `tk_config_parser.h`. It abstracts away the unsafe
 * details of FFI calls, C string handling, and manual resource management,
 * presenting a clean, ergonomic, and memory-safe API.
 *
 * The main entry point is the `ConfigLoader` struct, which uses the RAII
 * pattern via an internal `ConfigContext` to ensure that the underlying C
 * resource (`tk_config_t`) is always properly cleaned up.
 *
 * Dependencies:
 *   - crate::ffi: For the raw C function bindings.
 *   - crate::{InternalToolsError, ffi_result_from_code}: For error handling.
 *   - log: For logging operations.
 *   - thiserror: For ergonomic error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use super::{ffi, InternalToolsError};
use std::ffi::{CStr, CString};
use std::ptr::null_mut;
use thiserror::Error;

/// Represents errors specific to the configuration loading process.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// The specified configuration file was not found.
    #[error("Configuration file not found at path: {0}")]
    FileNotFound(String),

    /// There was an error reading the configuration file.
    #[error("Failed to read configuration file: {0}")]
    FileRead(String),
}

/// A low-level RAII wrapper for the `tk_config_t` handle.
///
/// This struct ensures that the C-level context is created and destroyed
/// correctly. When a `ConfigContext` is dropped, it automatically calls
/// `tk_config_destroy`. This is a private implementation detail of the module.
struct ConfigContext {
    ptr: *mut ffi::tk_config_t,
}

impl ConfigContext {
    /// Creates a new `ConfigContext` by calling the C FFI.
    fn new() -> Result<Self, InternalToolsError> {
        let mut ptr = null_mut();
        // Unsafe block to call the C constructor.
        let code = unsafe { ffi::tk_config_create(&mut ptr) };
        super::ffi_result_from_code(code, "tk_config_create")?;

        if ptr.is_null() {
            return Err(InternalToolsError::FfiOutOfMemory {
                context: "tk_config_create returned a null pointer".to_string(),
            });
        }
        Ok(Self { ptr })
    }

    /// Provides access to the raw pointer for FFI calls.
    fn as_ptr(&self) -> *mut ffi::tk_config_t {
        self.ptr
    }
}

/// The `Drop` implementation ensures the C context is always cleaned up.
impl Drop for ConfigContext {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // Unsafe block to call the C destructor.
            unsafe { ffi::tk_config_destroy(&mut self.ptr) };
        }
    }
}

/// A safe, high-level wrapper for loading and accessing key-value configurations.
pub struct ConfigLoader {
    context: ConfigContext,
}

impl ConfigLoader {
    /// Creates a new, empty configuration loader.
    pub fn new() -> Result<Self, InternalToolsError> {
        let context = ConfigContext::new()?;
        Ok(Self { context })
    }

    /// Loads configuration settings from a specified file path.
    ///
    /// This method populates the configuration object by parsing the given file.
    ///
    /// # Arguments
    /// * `filepath` - The path to the configuration file.
    pub fn load_from_file(&mut self, filepath: &str) -> Result<(), InternalToolsError> {
        let c_filepath = CString::new(filepath)?;
        let code = unsafe {
            ffi::tk_config_load_from_file(self.context.as_ptr(), c_filepath.as_ptr())
        };

        // Translate specific C error codes to our richer Rust error types.
        match code {
            ffi::TK_SUCCESS => Ok(()),
            ffi::TK_ERROR_FILE_NOT_FOUND => {
                Err(ConfigError::FileNotFound(filepath.to_string()).into())
            }
            ffi::TK_ERROR_FILE_READ => Err(ConfigError::FileRead(filepath.to_string()).into()),
            _ => super::ffi_result_from_code(code, "tk_config_load_from_file"),
        }
    }

    /// Retrieves a string value for a given key.
    ///
    /// # Arguments
    /// * `key` - The configuration key.
    /// * `default_value` - The value to return if the key is not found.
    pub fn get_string(&self, key: &str, default_value: &str) -> Result<String, InternalToolsError> {
        let c_key = CString::new(key)?;
        let c_default = CString::new(default_value)?;

        let result_ptr = unsafe {
            ffi::tk_config_get_string(self.context.as_ptr(), c_key.as_ptr(), c_default.as_ptr())
        };

        // Convert the raw C string pointer back to a safe Rust String.
        let c_str = unsafe { CStr::from_ptr(result_ptr) };
        c_str
            .to_str()
            .map(|s| s.to_owned())
            .map_err(|_| InternalToolsError::InvalidUtf8)
    }

    /// Retrieves an integer value for a given key.
    ///
    /// # Arguments
    /// * `key` - The configuration key.
    /// * `default_value` - The value to return if the key is not found or parsing fails.
    pub fn get_int(&self, key: &str, default_value: i64) -> Result<i64, InternalToolsError> {
        let c_key = CString::new(key)?;
        let value =
            unsafe { ffi::tk_config_get_int(self.context.as_ptr(), c_key.as_ptr(), default_value) };
        Ok(value)
    }

    /// Retrieves a floating-point value for a given key.
    ///
    /// # Arguments
    /// * `key` - The configuration key.
    /// * `default_value` - The value to return if the key is not found or parsing fails.
    pub fn get_double(&self, key: &str, default_value: f64) -> Result<f64, InternalToolsError> {
        let c_key = CString::new(key)?;
        let value = unsafe {
            ffi::tk_config_get_double(self.context.as_ptr(), c_key.as_ptr(), default_value)
        };
        Ok(value)
    }

    /// Retrieves a boolean value for a given key.
    ///
    /// The C library interprets "true", "yes", "on", "1" as true.
    ///
    /// # Arguments
    /// * `key` - The configuration key.
    /// * `default_value` - The value to return if the key is not found.
    pub fn get_bool(&self, key: &str, default_value: bool) -> Result<bool, InternalToolsError> {
        let c_key = CString::new(key)?;
        let value =
            unsafe { ffi::tk_config_get_bool(self.context.as_ptr(), c_key.as_ptr(), default_value) };
        Ok(value)
    }
}
