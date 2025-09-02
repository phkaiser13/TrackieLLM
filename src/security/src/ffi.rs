/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: ffi.rs
 *
 * Foreign Function Interface (FFI) definitions for the TrackieLLM security module.
 * This module defines the types and function signatures for interoperability 
 * between Rust and C components.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use std::ffi::c_void;

/// C-compatible error codes matching tk_error_handling.h
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TkErrorCode {
    Success = 0,
    InvalidArgument = 1,
    OutOfMemory = 2,
    Internal = 3,
    NotFound = 4,
    BufferTooSmall = 5,
    DecryptionFailed = 6,
    InvalidState = 7,
    NetworkError = 8,
    FileError = 9,
    PermissionDenied = 10,
    Timeout = 11,
    AlreadyExists = 12,
    NotImplemented = 13,
}

impl Default for TkErrorCode {
    fn default() -> Self {
        TkErrorCode::Success
    }
}

/// Convert from our internal error type to C error codes
impl From<crate::key_management::KeyManagementError> for TkErrorCode {
    fn from(error: crate::key_management::KeyManagementError) -> Self {
        use crate::key_management::KeyManagementError;
        match error {
            KeyManagementError::InvalidArgument => TkErrorCode::InvalidArgument,
            KeyManagementError::OutOfMemory => TkErrorCode::OutOfMemory,
            KeyManagementError::CryptoError => TkErrorCode::DecryptionFailed,
            KeyManagementError::KeyNotFound => TkErrorCode::NotFound,
            KeyManagementError::KeyDerivationFailed => TkErrorCode::Internal,
            KeyManagementError::DeviceInfoUnavailable => TkErrorCode::Internal,
            KeyManagementError::InternalError => TkErrorCode::Internal,
        }
    }
}

/// Constants for key and buffer sizes
pub const TK_ENCRYPTION_KEY_SIZE: usize = 32;
pub const TK_ENCRYPTION_NONCE_SIZE: usize = 12;
pub const TK_ENCRYPTION_TAG_SIZE: usize = 16;
pub const TK_ENCRYPTION_OVERHEAD: usize = TK_ENCRYPTION_NONCE_SIZE + TK_ENCRYPTION_TAG_SIZE;

/// Opaque handle for encryption contexts from Rust side
pub type TkSecurityContextHandle = *mut c_void;

// External C functions that we need to call
extern "C" {
    /// Logging functions from tk_logging.h
    pub fn tk_log_debug(format: *const i8, ...);
    pub fn tk_log_info(format: *const i8, ...);
    pub fn tk_log_warning(format: *const i8, ...);
    pub fn tk_log_error(format: *const i8, ...);
}

/// Utility macro for safe C string logging from Rust
#[macro_export]
macro_rules! tk_log_debug {
    ($msg:expr) => {
        unsafe {
            let c_str = std::ffi::CString::new($msg).unwrap_or_default();
            crate::ffi::tk_log_debug(b"%s\0".as_ptr() as *const i8, c_str.as_ptr());
        }
    };
}

#[macro_export]
macro_rules! tk_log_info {
    ($msg:expr) => {
        unsafe {
            let c_str = std::ffi::CString::new($msg).unwrap_or_default();
            crate::ffi::tk_log_info(b"%s\0".as_ptr() as *const i8, c_str.as_ptr());
        }
    };
}

#[macro_export]
macro_rules! tk_log_warning {
    ($msg:expr) => {
        unsafe {
            let c_str = std::ffi::CString::new($msg).unwrap_or_default();
            crate::ffi::tk_log_warning(b"%s\0".as_ptr() as *const i8, c_str.as_ptr());
        }
    };
}

#[macro_export]
macro_rules! tk_log_error {
    ($msg:expr) => {
        unsafe {
            let c_str = std::ffi::CString::new($msg).unwrap_or_default();
            crate::ffi::tk_log_error(b"%s\0".as_ptr() as *const i8, c_str.as_ptr());
        }
    };
}

/// FFI-safe structure for passing device information
#[repr(C)]
pub struct TkDeviceInfo {
    pub machine_id: *const i8,
    pub machine_id_len: usize,
    pub hostname: *const i8,
    pub hostname_len: usize,
    pub platform: *const i8,
    pub platform_len: usize,
}

/// FFI-safe structure for key derivation parameters
#[repr(C)]
pub struct TkKeyDerivationParams {
    pub context: *const u8,
    pub context_len: usize,
    pub salt: *const u8,
    pub salt_len: usize,
    pub iterations: u32,
}

// Additional FFI exports for extended functionality

/// Gets device-specific information for key derivation
#[no_mangle]
pub extern "C" fn tk_security_get_device_info(
    info: *mut TkDeviceInfo,
) -> TkErrorCode {
    if info.is_null() {
        return TkErrorCode::InvalidArgument;
    }

    // This is a simplified implementation
    // In a real implementation, you would populate the device info structure
    // with actual device-specific data
    
    tk_log_debug!("tk_security_get_device_info: called");
    
    // For now, return not implemented
    TkErrorCode::NotImplemented
}

/// Derives a key using device-specific information and custom parameters
#[no_mangle]
pub extern "C" fn tk_security_derive_key(
    params: *const TkKeyDerivationParams,
    key_buffer: *mut u8,
    key_buffer_size: usize,
) -> TkErrorCode {
    if params.is_null() || key_buffer.is_null() {
        return TkErrorCode::InvalidArgument;
    }

    if key_buffer_size < TK_ENCRYPTION_KEY_SIZE {
        return TkErrorCode::BufferTooSmall;
    }

    tk_log_debug!("tk_security_derive_key: called");
    
    // This would implement custom key derivation
    // For now, return not implemented
    TkErrorCode::NotImplemented
}

/// Validates that the security subsystem is properly initialized
#[no_mangle]
pub extern "C" fn tk_security_validate_initialization() -> TkErrorCode {
    tk_log_debug!("tk_security_validate_initialization: checking security subsystem");
    
    // Try to get the master key to validate initialization
    match crate::key_management::get_master_key() {
        Ok(_) => {
            tk_log_info!("tk_security_validate_initialization: security subsystem is ready");
            TkErrorCode::Success
        }
        Err(e) => {
            tk_log_error!(&format!("tk_security_validate_initialization: failed - {:?}", e));
            TkErrorCode::from(e)
        }
    }
}

/// Performs a self-test of the encryption functionality
#[no_mangle]
pub extern "C" fn tk_security_self_test() -> TkErrorCode {
    tk_log_info!("tk_security_self_test: starting security subsystem self-test");
    
    // Test 1: Master key derivation
    let master_key = match crate::key_management::get_master_key() {
        Ok(key) => {
            tk_log_debug!("tk_security_self_test: master key derivation - OK");
            key
        }
        Err(e) => {
            tk_log_error!(&format!("tk_security_self_test: master key derivation failed - {:?}", e));
            return TkErrorCode::from(e);
        }
    };

    // Test 2: Encryption context creation
    let mut ctx = match crate::key_management::EncryptionContext::new() {
        Ok(ctx) => {
            tk_log_debug!("tk_security_self_test: encryption context creation - OK");
            ctx
        }
        Err(e) => {
            tk_log_error!(&format!("tk_security_self_test: encryption context creation failed - {:?}", e));
            return TkErrorCode::from(e);
        }
    };

    // Test 3: Key setting
    if let Err(e) = ctx.set_key(&master_key) {
        tk_log_error!(&format!("tk_security_self_test: key setting failed - {:?}", e));
        return TkErrorCode::from(e);
    }
    tk_log_debug!("tk_security_self_test: key setting - OK");

    // Test 4: Encryption/Decryption round-trip
    let test_data = b"TrackieLLM Security Self-Test Data";
    let encrypted = match ctx.encrypt(test_data) {
        Ok(data) => {
            tk_log_debug!("tk_security_self_test: encryption - OK");
            data
        }
        Err(e) => {
            tk_log_error!(&format!("tk_security_self_test: encryption failed - {:?}", e));
            return TkErrorCode::from(e);
        }
    };

    let decrypted = match ctx.decrypt(&encrypted) {
        Ok(data) => {
            tk_log_debug!("tk_security_self_test: decryption - OK");
            data
        }
        Err(e) => {
            tk_log_error!(&format!("tk_security_self_test: decryption failed - {:?}", e));
            return TkErrorCode::from(e);
        }
    };

    // Test 5: Data integrity verification
    if test_data != decrypted.as_slice() {
        tk_log_error!("tk_security_self_test: data integrity check failed");
        return TkErrorCode::Internal;
    }
    tk_log_debug!("tk_security_self_test: data integrity - OK");

    // Test 6: Master context consistency
    let master_ctx = match crate::key_management::create_master_context() {
        Ok(ctx) => {
            tk_log_debug!("tk_security_self_test: master context creation - OK");
            ctx
        }
        Err(e) => {
            tk_log_error!(&format!("tk_security_self_test: master context creation failed - {:?}", e));
            return TkErrorCode::from(e);
        }
    };

    // Test that the master context can decrypt data encrypted with the same master key
    match master_ctx.decrypt(&encrypted) {
        Ok(data) => {
            if data == test_data {
                tk_log_debug!("tk_security_self_test: master context consistency - OK");
            } else {
                tk_log_error!("tk_security_self_test: master context consistency failed");
                return TkErrorCode::Internal;
            }
        }
        Err(e) => {
            tk_log_error!(&format!("tk_security_self_test: master context decryption failed - {:?}", e));
            return TkErrorCode::from(e);
        }
    }

    tk_log_info!("tk_security_self_test: all tests passed successfully");
    TkErrorCode::Success
}

/// Gets the version information of the security module
#[no_mangle]
pub extern "C" fn tk_security_get_version(
    major: *mut u32,
    minor: *mut u32,
    patch: *mut u32,
) -> TkErrorCode {
    if major.is_null() || minor.is_null() || patch.is_null() {
        return TkErrorCode::InvalidArgument;
    }

    unsafe {
        *major = crate::VERSION_MAJOR;
        *minor = crate::VERSION_MINOR;
        *patch = crate::VERSION_PATCH;
    }

    TkErrorCode::Success
}

/// Emergency function to clear all security state
#[no_mangle]
pub extern "C" fn tk_security_emergency_clear() -> TkErrorCode {
    tk_log_warning!("tk_security_emergency_clear: performing emergency security state clear");
    
    // Clear all cached keys
    let result = crate::key_management::tk_security_clear_keys();
    
    if result == TkErrorCode::Success {
        tk_log_info!("tk_security_emergency_clear: security state cleared successfully");
    } else {
        tk_log_error!("tk_security_emergency_clear: failed to clear security state");
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_conversion() {
        use crate::key_management::KeyManagementError;
        
        assert_eq!(TkErrorCode::from(KeyManagementError::InvalidArgument), TkErrorCode::InvalidArgument);
        assert_eq!(TkErrorCode::from(KeyManagementError::OutOfMemory), TkErrorCode::OutOfMemory);
        assert_eq!(TkErrorCode::from(KeyManagementError::KeyNotFound), TkErrorCode::NotFound);
    }

    #[test]
    fn test_ffi_constants() {
        assert_eq!(TK_ENCRYPTION_KEY_SIZE, 32);
        assert_eq!(TK_ENCRYPTION_NONCE_SIZE, 12);
        assert_eq!(TK_ENCRYPTION_TAG_SIZE, 16);
        assert_eq!(TK_ENCRYPTION_OVERHEAD, 28);
    }

    #[test]
    fn test_validation() {
        let result = tk_security_validate_initialization();
        assert_eq!(result, TkErrorCode::Success);
    }

    #[test]
    fn test_self_test() {
        let result = tk_security_self_test();
        assert_eq!(result, TkErrorCode::Success);
    }

    #[test]
    fn test_version() {
        let mut major = 0;
        let mut minor = 0;
        let mut patch = 0;
        
        let result = tk_security_get_version(&mut major, &mut minor, &mut patch);
        
        assert_eq!(result, TkErrorCode::Success);
        assert_eq!(major, crate::VERSION_MAJOR);
        assert_eq!(minor, crate::VERSION_MINOR);
        assert_eq!(patch, crate::VERSION_PATCH);
    }
}