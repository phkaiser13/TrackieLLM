/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/internal_tools/fs_utils.rs
 *
 * This file provides a safe Rust wrapper for the C-based filesystem
 * abstraction layer defined in `tk_file_manager.h`. Its primary goal is to
 * offer a secure and ergonomic API for all filesystem operations, abstracting
 * away the risks and complexities of raw path manipulation and FFI calls.
 *
 * The central abstraction is the `Path` struct, a robust RAII wrapper around
 * the opaque `tk_path_t` C handle. This ensures that path resources are
 * managed automatically and safely. By forcing all filesystem operations to
 * use this `Path` type instead of raw `String`s, we gain centralized control
 * over path normalization and validation, which is a critical security feature.
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
use std::fmt;
use std::ptr::null_mut;
use thiserror::Error;

/// Represents errors specific to filesystem operations.
#[derive(Debug, Error)]
pub enum FsError {
    /// The requested file or directory was not found.
    #[error("Path not found: {0}")]
    NotFound(String),

    /// Permission was denied for the requested operation.
    #[error("Permission denied for path: {0}")]
    PermissionDenied(String),

    /// An error occurred while reading from a file.
    #[error("Failed to read from file: {0}")]
    FileRead(String),

    /// An error occurred while writing to a file.
    #[error("Failed to write to file: {0}")]
    FileWrite(String),
}

/// An enumeration of well-known base directories, mirroring `tk_base_path_e`.
///
/// This provides a platform-independent way to reference standard system
/// locations like the application config or cache directories.
#[derive(Debug, Clone, Copy)]
pub enum BasePath {
    /// The primary, persistent data directory for the application.
    AppConfig,
    /// A directory for storing cached data that can be regenerated.
    Cache,
    /// The directory where the main executable is located.
    ExecutableDir,
    /// The current working directory from which the application was launched.
    WorkingDir,
}

impl From<BasePath> for ffi::tk_base_path_e {
    fn from(val: BasePath) -> Self {
        match val {
            BasePath::AppConfig => ffi::tk_base_path_e::TK_BASE_PATH_APP_CONFIG,
            BasePath::Cache => ffi::tk_base_path_e::TK_BASE_PATH_CACHE,
            BasePath::ExecutableDir => ffi::tk_base_path_e::TK_BASE_PATH_EXECUTABLE_DIR,
            BasePath::WorkingDir => ffi::tk_base_path_e::TK_BASE_PATH_WORKING_DIR,
        }
    }
}

/// A safe, RAII-compliant wrapper for a `tk_path_t` handle.
///
/// This struct is the primary way to interact with the filesystem in a safe
/// manner. It ensures that the underlying C resources are always freed.
pub struct Path {
    ptr: *mut ffi::tk_path_t,
}

impl Path {
    /// Creates a new `Path` from a string slice.
    pub fn from_str(path_str: &str) -> Result<Self, InternalToolsError> {
        let mut ptr = null_mut();
        let c_path_str = CString::new(path_str)?;
        let code = unsafe { ffi::tk_path_create_from_string(&mut ptr, c_path_str.as_ptr()) };
        super::ffi_result_from_code(code, "tk_path_create_from_string")?;
        Ok(Self { ptr })
    }

    /// Creates a new `Path` from a well-known base directory.
    pub fn from_base(base: BasePath) -> Result<Self, InternalToolsError> {
        let mut ptr = null_mut();
        let code = unsafe { ffi::tk_path_create_from_base(&mut ptr, base.into()) };
        super::ffi_result_from_code(code, "tk_path_create_from_base")?;
        Ok(Self { ptr })
    }

    /// Appends a segment to the path in-place.
    pub fn join(&mut self, segment: &str) -> Result<(), InternalToolsError> {
        let c_segment = CString::new(segment)?;
        let code = unsafe { ffi::tk_path_join(self.ptr, c_segment.as_ptr()) };
        super::ffi_result_from_code(code, "tk_path_join")
    }

    /// Provides access to the raw pointer for FFI calls.
    fn as_ptr(&self) -> *const ffi::tk_path_t {
        self.ptr
    }
}

impl fmt::Display for Path {
    /// Provides a string representation of the path for display purposes.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let result_ptr = unsafe { ffi::tk_path_get_string(self.ptr) };
        if result_ptr.is_null() {
            return write!(f, "[Invalid Path]");
        }
        let c_str = unsafe { CStr::from_ptr(result_ptr) };
        write!(f, "{}", c_str.to_string_lossy())
    }
}

impl Drop for Path {
    /// Ensures the underlying C `tk_path_t` object is always destroyed.
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::tk_path_destroy(&mut self.ptr) };
        }
    }
}

// --- Filesystem Functions ---

/// Checks if a file or directory exists at the given path.
pub fn exists(path: &Path) -> Result<bool, InternalToolsError> {
    let mut exists_bool = false;
    let code = unsafe { ffi::tk_fs_exists(path.as_ptr(), &mut exists_bool) };
    super::ffi_result_from_code(code, "tk_fs_exists")?;
    Ok(exists_bool)
}

/// Checks if the path points to a regular file.
pub fn is_file(path: &Path) -> Result<bool, InternalToolsError> {
    let mut is_file_bool = false;
    let code = unsafe { ffi::tk_fs_is_file(path.as_ptr(), &mut is_file_bool) };
    super::ffi_result_from_code(code, "tk_fs_is_file")?;
    Ok(is_file_bool)
}

/// Checks if the path points to a directory.
pub fn is_directory(path: &Path) -> Result<bool, InternalToolsError> {
    let mut is_dir_bool = false;
    let code = unsafe { ffi::tk_fs_is_directory(path.as_ptr(), &mut is_dir_bool) };
    super::ffi_result_from_code(code, "tk_fs_is_directory")?;
    Ok(is_dir_bool)
}

/// Creates a directory and all its parent directories if they do not exist.
pub fn create_dir_recursive(path: &Path) -> Result<(), InternalToolsError> {
    let code = unsafe { ffi::tk_dir_create_recursive(path.as_ptr()) };
    super::ffi_result_from_code(code, "tk_dir_create_recursive")
}

/// Reads the entire content of a file into a byte vector.
pub fn read_all_bytes(path: &Path, max_size: usize) -> Result<Vec<u8>, InternalToolsError> {
    let mut buffer_ptr = null_mut();
    let mut buffer_size = 0;
    let code = unsafe {
        ffi::tk_file_read_all_bytes(path.as_ptr(), &mut buffer_ptr, &mut buffer_size, max_size)
    };

    match code {
        ffi::TK_SUCCESS => {
            // Unsafe block to take ownership of the buffer allocated by C.
            // We trust the C API to have returned a valid pointer and size.
            // The buffer is then immediately wrapped in a Vec, which will manage
            // its memory from now on (including freeing it when the Vec is dropped).
            let data = unsafe { Vec::from_raw_parts(buffer_ptr, buffer_size, buffer_size) };
            Ok(data)
        },
        ffi::TK_ERROR_FILE_NOT_FOUND => Err(FsError::NotFound(path.to_string()).into()),
        ffi::TK_ERROR_FILE_READ => Err(FsError::FileRead(path.to_string()).into()),
        _ => super::ffi_result_from_code(code, "tk_file_read_all_bytes"),
    }
}

/// Writes the content of a byte slice to a file, overwriting it if it exists.
pub fn write_buffer(path: &Path, data: &[u8]) -> Result<(), InternalToolsError> {
    let code = unsafe { ffi::tk_file_write_buffer(path.as_ptr(), data.as_ptr(), data.len()) };
    match code {
        ffi::TK_SUCCESS => Ok(()),
        ffi::TK_ERROR_PERMISSION_DENIED => Err(FsError::PermissionDenied(path.to_string()).into()),
        ffi::TK_ERROR_FILE_WRITE => Err(FsError::FileWrite(path.to_string()).into()),
        _ => super::ffi_result_from_code(code, "tk_file_write_buffer"),
    }
}
