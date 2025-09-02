/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: lib.rs
 *
 * Main library file for the TrackieLLM security module. This module provides
 * secure key management, encryption abstractions, and secure communication
 * channels for the TrackieLLM project.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

pub mod key_management;
pub mod secure_channels;
pub mod ffi;

// Re-exports for convenience
pub use key_management::{
    EncryptionContext,
    KeyManagementError,
    TkErrorCode,
    get_master_key,
    create_master_context,
};

pub use secure_channels::*;
pub use ffi::*;

/// Security module version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VERSION_MAJOR: u32 = 1;
pub const VERSION_MINOR: u32 = 0;
pub const VERSION_PATCH: u32 = 0;

/// Initialize the security module
///
/// This function should be called once at application startup to ensure
/// all security subsystems are properly initialized.
pub fn init() -> Result<(), KeyManagementError> {
    // Initialize key management subsystem
    let _ = key_management::get_master_key()?;
    
    // Future: Initialize other security subsystems here
    
    Ok(())
}

/// Cleanup and secure memory wipe
///
/// This function should be called during application shutdown to ensure
/// all sensitive data is securely wiped from memory.
pub fn cleanup() {
    // Clear all cached keys
    let _ = key_management::tk_security_clear_keys();
    
    // Future: Cleanup other security subsystems here
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_init() {
        let result = init();
        assert!(result.is_ok());
        
        cleanup();
    }
}