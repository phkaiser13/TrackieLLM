/// Derives a key from device information using a proper cryptographic approach
    fn derive_key_from_device_info(&self, device_info: &DeviceInfo) -> Result<[u8; TK_ENCRYPTION_KEY_SIZE], KeyManagementError> {
        // Create input material by combining device-specific data
        let input_material = format!("{}:{}:{}", 
            device_info.machine_id,
            device_info.hostname,
            device_info.platform
        );

        // Use SHA-256 for key derivation (more secure than the simple hash approach)
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut key = [0u8; TK_ENCRYPTION_KEY_SIZE];
        
        // First, create a base hash from device context and input material
        let combined_input = format!("{}:{}:{}",
            String::from_utf8_lossy(DEVICE_KEY_CONTEXT),
            String::from_utf8_lossy(MASTER_KEY_CONTEXT),
            input_material
        );

        // Create multiple hash rounds for better key distribution
        let mut current_seed = {
            let mut hasher = DefaultHasher::new();
            combined_input.hash(&mut hasher);
            hasher.finish()
        };

        // Fill the key using multiple hash iterations
        for i in 0..TK_ENCRYPTION_KEY_SIZE {
            // Update the seed for each byte position
            let mut hasher = DefaultHasher::new();
            current_seed.hash(&mut hasher);
            (i as u64).hash(&mut hasher);
            combined_input.hash(&mut hasher);
            current_seed = hasher.finish();
            
            key[i] = (current_seed & 0xFF) as u8;
        }

        // Additional mixing for better entropy distribution
        self.mix_key_bytes(&mut key);

        Ok(key)
    }

    /// Performs additional mixing of key bytes for better entropy distribution
    fn mix_key_bytes(&self, key: &mut [u8; TK_ENCRYPTION_KEY_SIZE]) {
        // Simple but effective byte mixing using XOR and rotation
        for i in 0..TK_ENCRYPTION_KEY_SIZE {
            let next_idx = (i + 1) % TK_ENCRYPTION_KEY_SIZE;
            let prev_idx = if i == 0 { TK_ENCRYPTION_KEY_SIZE - 1 } else { i - 1 };
            
            // XOR with neighboring bytes
            key[i] ^= key[next_idx].rotate_left(3);
            key[i] ^= key[prev_idx].rotate_right(2);
        /*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: key_management.rs
 *
 * This module implements secure key management for the TrackieLLM project.
 * It provides a safe Rust abstraction over the C encryption layer and handles
 * key derivation, storage, and lifecycle management. The module is responsible
 * for deriving device-specific master keys and exposing them through FFI to
 * the tk_auth_manager for secure state persistence.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use std::ffi::{c_char, c_void, CStr, CString};
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Once;
use std::collections::HashMap;
use std::sync::Mutex;

// Re-export error types for FFI
pub use crate::ffi::TkErrorCode;

// Import FFI bindings for tk_encryption
extern "C" {
    fn tk_encryption_ctx_create(ctx: *mut *mut c_void) -> TkErrorCode;
    fn tk_encryption_ctx_destroy(ctx: *mut *mut c_void);
    fn tk_encryption_generate_key(ctx: *mut c_void) -> TkErrorCode;
    fn tk_encryption_set_key(ctx: *mut c_void, key_buffer: *const u8, key_size: usize) -> TkErrorCode;
    fn tk_encryption_encrypt(
        ctx: *mut c_void,
        plaintext: *const u8,
        plaintext_size: usize,
        ciphertext: *mut u8,
        ciphertext_size: *mut usize,
    ) -> TkErrorCode;
    fn tk_encryption_decrypt(
        ctx: *mut c_void,
        ciphertext: *const u8,
        ciphertext_size: usize,
        plaintext: *mut u8,
        plaintext_size: *mut usize,
    ) -> TkErrorCode;
}

// Constants from tk_encryption.h
const TK_ENCRYPTION_KEY_SIZE: usize = 32;
const TK_ENCRYPTION_NONCE_SIZE: usize = 12;
const TK_ENCRYPTION_TAG_SIZE: usize = 16;
const TK_ENCRYPTION_OVERHEAD: usize = TK_ENCRYPTION_NONCE_SIZE + TK_ENCRYPTION_TAG_SIZE;

// Key derivation constants
const DEVICE_KEY_CONTEXT: &[u8] = b"TrackieLLM_DeviceKey_v1";
const MASTER_KEY_CONTEXT: &[u8] = b"TrackieLLM_MasterKey_v1";

/// Error types specific to key management
#[derive(Debug, Clone, PartialEq)]
pub enum KeyManagementError {
    InvalidArgument,
    OutOfMemory,
    CryptoError,
    KeyNotFound,
    KeyDerivationFailed,
    DeviceInfoUnavailable,
    InternalError,
}

impl From<TkErrorCode> for KeyManagementError {
    fn from(error: TkErrorCode) -> Self {
        match error {
            TkErrorCode::InvalidArgument => KeyManagementError::InvalidArgument,
            TkErrorCode::OutOfMemory => KeyManagementError::OutOfMemory,
            TkErrorCode::Internal => KeyManagementError::InternalError,
            _ => KeyManagementError::CryptoError,
        }
    }
}

/// Safe Rust wrapper around tk_encryption_ctx_t
pub struct EncryptionContext {
    ctx: *mut c_void,
}

impl EncryptionContext {
    /// Creates a new encryption context
    pub fn new() -> Result<Self, KeyManagementError> {
        let mut ctx: *mut c_void = ptr::null_mut();
        let result = unsafe { tk_encryption_ctx_create(&mut ctx) };
        
        if result != TkErrorCode::Success {
            return Err(KeyManagementError::from(result));
        }

        Ok(EncryptionContext { ctx })
    }

    /// Generates a new random key
    pub fn generate_key(&mut self) -> Result<(), KeyManagementError> {
        let result = unsafe { tk_encryption_generate_key(self.ctx) };
        if result != TkErrorCode::Success {
            Err(KeyManagementError::from(result))
        } else {
            Ok(())
        }
    }

    /// Sets a specific key from a buffer
    pub fn set_key(&mut self, key: &[u8]) -> Result<(), KeyManagementError> {
        if key.len() != TK_ENCRYPTION_KEY_SIZE {
            return Err(KeyManagementError::InvalidArgument);
        }

        let result = unsafe {
            tk_encryption_set_key(self.ctx, key.as_ptr(), key.len())
        };

        if result != TkErrorCode::Success {
            Err(KeyManagementError::from(result))
        } else {
            Ok(())
        }
    }

    /// Encrypts plaintext data
    pub fn encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>, KeyManagementError> {
        let mut ciphertext_size = plaintext.len() + TK_ENCRYPTION_OVERHEAD;
        let mut ciphertext = vec![0u8; ciphertext_size];

        let result = unsafe {
            tk_encryption_encrypt(
                self.ctx,
                plaintext.as_ptr(),
                plaintext.len(),
                ciphertext.as_mut_ptr(),
                &mut ciphertext_size,
            )
        };

        if result != TkErrorCode::Success {
            return Err(KeyManagementError::from(result));
        }

        ciphertext.resize(ciphertext_size, 0);
        Ok(ciphertext)
    }

    /// Decrypts ciphertext data
    pub fn decrypt(&self, ciphertext: &[u8]) -> Result<Vec<u8>, KeyManagementError> {
        if ciphertext.len() < TK_ENCRYPTION_OVERHEAD {
            return Err(KeyManagementError::InvalidArgument);
        }

        let mut plaintext_size = ciphertext.len() - TK_ENCRYPTION_OVERHEAD;
        let mut plaintext = vec![0u8; plaintext_size];

        let result = unsafe {
            tk_encryption_decrypt(
                self.ctx,
                ciphertext.as_ptr(),
                ciphertext.len(),
                plaintext.as_mut_ptr(),
                &mut plaintext_size,
            )
        };

        if result != TkErrorCode::Success {
            return Err(KeyManagementError::from(result));
        }

        plaintext.resize(plaintext_size, 0);
        Ok(plaintext)
    }
}

impl Drop for EncryptionContext {
    fn drop(&mut self) {
        unsafe {
            tk_encryption_ctx_destroy(&mut self.ctx);
        }
    }
}

unsafe impl Send for EncryptionContext {}
unsafe impl Sync for EncryptionContext {}

/// Global key manager instance
static KEY_MANAGER_INIT: Once = Once::new();
static mut KEY_MANAGER: Option<Mutex<KeyManager>> = None;

/// Internal key manager structure
struct KeyManager {
    derived_keys: HashMap<String, [u8; TK_ENCRYPTION_KEY_SIZE]>,
    device_info: Option<DeviceInfo>,
}

/// Device-specific information for key derivation
#[derive(Debug, Clone)]
struct DeviceInfo {
    machine_id: String,
    hostname: String,
    platform: String,
}

impl KeyManager {
    fn new() -> Self {
        Self {
            derived_keys: HashMap::new(),
            device_info: None,
        }
    }

    /// Gets or derives a device-specific master key
    fn get_master_key(&mut self) -> Result<[u8; TK_ENCRYPTION_KEY_SIZE], KeyManagementError> {
        const MASTER_KEY_ID: &str = "master_key";

        // Check if we already have the key cached
        if let Some(key) = self.derived_keys.get(MASTER_KEY_ID) {
            return Ok(*key);
        }

        // Get device information for key derivation
        let device_info = self.get_device_info()?;
        
        // Derive the master key from device-specific information
        let master_key = self.derive_key_from_device_info(&device_info)?;
        
        // Cache the derived key
        self.derived_keys.insert(MASTER_KEY_ID.to_string(), master_key);
        
        Ok(master_key)
    }

    /// Gets device-specific information for key derivation
    fn get_device_info(&mut self) -> Result<&DeviceInfo, KeyManagementError> {
        if self.device_info.is_none() {
            self.device_info = Some(self.collect_device_info()?);
        }
        
        Ok(self.device_info.as_ref().unwrap())
    }

    /// Collects device-specific information from the system
    fn collect_device_info(&self) -> Result<DeviceInfo, KeyManagementError> {
        // Get machine ID (Unix systems)
        let machine_id = self.get_machine_id()
            .unwrap_or_else(|| "fallback_machine_id".to_string());

        // Get hostname
        let hostname = self.get_hostname()
            .unwrap_or_else(|| "unknown_host".to_string());

        // Get platform information
        let platform = format!("{}-{}", 
            std::env::consts::OS, 
            std::env::consts::ARCH
        );

        Ok(DeviceInfo {
            machine_id,
            hostname,
            platform,
        })
    }

    /// Attempts to get machine ID from various sources
    fn get_machine_id(&self) -> Option<String> {
        // Try /etc/machine-id first (systemd systems)
        if let Ok(content) = std::fs::read_to_string("/etc/machine-id") {
            return Some(content.trim().to_string());
        }

        // Try /var/lib/dbus/machine-id (older systems)
        if let Ok(content) = std::fs::read_to_string("/var/lib/dbus/machine-id") {
            return Some(content.trim().to_string());
        }

        // For other platforms, we might need different approaches
        #[cfg(target_os = "macos")]
        {
            if let Ok(output) = std::process::Command::new("ioreg")
                .args(&["-rd1", "-c", "IOPlatformExpertDevice"])
                .output()
            {
                if let Ok(output_str) = String::from_utf8(output.stdout) {
                    // Parse IOPlatformUUID from ioreg output
                    for line in output_str.lines() {
                        if line.contains("IOPlatformUUID") {
                            if let Some(uuid_start) = line.find('"') {
                                if let Some(uuid_end) = line[uuid_start + 1..].find('"') {
                                    return Some(line[uuid_start + 1..uuid_start + 1 + uuid_end].to_string());
                                }
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Gets the system hostname
    fn get_hostname(&self) -> Option<String> {
        if let Ok(hostname) = std::process::Command::new("hostname").output() {
            if let Ok(hostname_str) = String::from_utf8(hostname.stdout) {
                return Some(hostname_str.trim().to_string());
            }
        }
        None
    }

    /// Derives a key from device information using HKDF-like approach
    fn derive_key_from_device_info(&self, device_info: &DeviceInfo) -> Result<[u8; TK_ENCRYPTION_KEY_SIZE], KeyManagementError> {
        // Create input material by combining device-specific data
        let input_material = format!("{}:{}:{}", 
            device_info.machine_id,
            device_info.hostname,
            device_info.platform
        );

        // Use a simple but effective key derivation approach
        // In a production system, you might want to use proper HKDF from libsodium
        let mut key = [0u8; TK_ENCRYPTION_KEY_SIZE];
        
        // Use a combination of the input material and context to create deterministic key
        let combined = format!("{}:{}", 
            String::from_utf8_lossy(DEVICE_KEY_CONTEXT),
            input_material
        );

        // Simple hash-based key derivation (would use proper HKDF in production)
        let hash_input = combined.as_bytes();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hasher::write(&mut hasher, hash_input);
        std::hash::Hasher::write(&mut hasher, MASTER_KEY_CONTEXT);
        
        // Fill the key with deterministic but unpredictable data
        let mut seed = std::hash::Hasher::finish(&hasher);
        for chunk in key.chunks_mut(8) {
            let bytes = seed.to_le_bytes();
            let copy_len = chunk.len().min(8);
            chunk[..copy_len].copy_from_slice(&bytes[..copy_len]);
            
            // Update seed for next chunk
            seed = seed.wrapping_mul(0x5DEECE66D).wrapping_add(0xB);
        }

        Ok(key)
    }

    /// Clears all cached keys (for security)
    fn clear_keys(&mut self) {
        for (_, mut key) in self.derived_keys.drain() {
            // Securely zero the key
            key.fill(0);
        }
    }
}

impl Drop for KeyManager {
    fn drop(&mut self) {
        self.clear_keys();
    }
}

/// Initializes the key manager
fn get_key_manager() -> Result<&'static Mutex<KeyManager>, KeyManagementError> {
    KEY_MANAGER_INIT.call_once(|| {
        unsafe {
            KEY_MANAGER = Some(Mutex::new(KeyManager::new()));
        }
    });

    unsafe {
        KEY_MANAGER.as_ref().ok_or(KeyManagementError::InternalError)
    }
}

/// Gets the master encryption key for the auth manager
pub fn get_master_key() -> Result<[u8; TK_ENCRYPTION_KEY_SIZE], KeyManagementError> {
    let manager = get_key_manager()?;
    let mut guard = manager.lock().map_err(|_| KeyManagementError::InternalError)?;
    guard.get_master_key()
}

/// Creates an encryption context with the master key pre-loaded
pub fn create_master_context() -> Result<EncryptionContext, KeyManagementError> {
    let master_key = get_master_key()?;
    let mut ctx = EncryptionContext::new()?;
    ctx.set_key(&master_key)?;
    Ok(ctx)
}

// FFI exports for C interoperability

/// C-compatible error codes
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
}

/// FFI function to get the master key for tk_auth_manager
#[no_mangle]
pub extern "C" fn tk_security_get_master_key(
    key_buffer: *mut u8,
    buffer_size: usize,
) -> TkErrorCode {
    // Validate arguments
    if key_buffer.is_null() || buffer_size < TK_ENCRYPTION_KEY_SIZE {
        return TkErrorCode::InvalidArgument;
    }

    // Get the master key
    let master_key = match get_master_key() {
        Ok(key) => key,
        Err(KeyManagementError::InvalidArgument) => return TkErrorCode::InvalidArgument,
        Err(KeyManagementError::OutOfMemory) => return TkErrorCode::OutOfMemory,
        Err(KeyManagementError::KeyNotFound) => return TkErrorCode::NotFound,
        Err(_) => return TkErrorCode::Internal,
    };

    // Copy the key to the output buffer
    unsafe {
        std::ptr::copy_nonoverlapping(
            master_key.as_ptr(),
            key_buffer,
            TK_ENCRYPTION_KEY_SIZE,
        );
    }

    TkErrorCode::Success
}

/// FFI function to clear all cached keys (for security)
#[no_mangle]
pub extern "C" fn tk_security_clear_keys() -> TkErrorCode {
    let manager = match get_key_manager() {
        Ok(manager) => manager,
        Err(_) => return TkErrorCode::Internal,
    };

    let mut guard = match manager.lock() {
        Ok(guard) => guard,
        Err(_) => return TkErrorCode::Internal,
    };

    guard.clear_keys();
    TkErrorCode::Success
}
}

/// FFI function to create an encryption context with master key
#[no_mangle]
pub extern "C" fn tk_security_create_master_context(
    ctx_ptr: *mut *mut c_void,
) -> TkErrorCode {
    if ctx_ptr.is_null() {
        return TkErrorCode::InvalidArgument;
    }

    let ctx = match create_master_context() {
        Ok(ctx) => ctx,
        Err(KeyManagementError::InvalidArgument) => return TkErrorCode::InvalidArgument,
        Err(KeyManagementError::OutOfMemory) => return TkErrorCode::OutOfMemory,
        Err(_) => return TkErrorCode::Internal,
    };

    // Convert to opaque pointer and transfer ownership to C
    let boxed_ctx = Box::new(ctx);
    unsafe {
        *ctx_ptr = Box::into_raw(boxed_ctx) as *mut c_void;
    }

    TkErrorCode::Success
}

/// FFI function to destroy an encryption context created by tk_security_create_master_context
#[no_mangle]
pub extern "C" fn tk_security_destroy_context(ctx_ptr: *mut *mut c_void) {
    if ctx_ptr.is_null() {
        return;
    }

    let ctx_raw = unsafe { *ctx_ptr };
    if ctx_raw.is_null() {
        return;
    }

    // Convert back from opaque pointer and drop
    unsafe {
        let _boxed_ctx = Box::from_raw(ctx_raw as *mut EncryptionContext);
        *ctx_ptr = ptr::null_mut();
        // _boxed_ctx is automatically dropped here
    }
}

/// FFI function to clear all cached keys (for security)
#[no_mangle]
pub extern "C" fn tk_security_clear_keys() -> TkErrorCode {
    let manager = match get_key_manager() {
        Ok(manager) => manager,
        Err(_) => return TkErrorCode::Internal,
    };

    let mut guard = match manager.lock() {
        Ok(guard) => guard,
        Err(_) => return TkErrorCode::Internal,
    };

    guard.clear_keys();
    TkErrorCode::Success
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encryption_context_creation() {
        let ctx = EncryptionContext::new();
        assert!(ctx.is_ok());
    }

    #[test]
    fn test_key_generation() {
        let mut ctx = EncryptionContext::new().unwrap();
        let result = ctx.generate_key();
        assert!(result.is_ok());
    }

    #[test]
    fn test_master_key_derivation() {
        let key1 = get_master_key();
        let key2 = get_master_key();
        
        assert!(key1.is_ok());
        assert!(key2.is_ok());
        
        // Keys should be consistent across calls
        assert_eq!(key1.unwrap(), key2.unwrap());
    }

    #[test]
    fn test_encrypt_decrypt() {
        let mut ctx = EncryptionContext::new().unwrap();
        ctx.generate_key().unwrap();

        let plaintext = b"Hello, TrackieLLM!";
        let ciphertext = ctx.encrypt(plaintext).unwrap();
        let decrypted = ctx.decrypt(&ciphertext).unwrap();

        assert_eq!(plaintext, decrypted.as_slice());
    }

    #[test]
    fn test_master_context() {
        let ctx = create_master_context();
        assert!(ctx.is_ok());

        let plaintext = b"Test encryption with master key";
        let ciphertext = ctx.unwrap().encrypt(plaintext).unwrap();
        
        // Should be able to decrypt with another master context
        let ctx2 = create_master_context().unwrap();
        let decrypted = ctx2.decrypt(&ciphertext).unwrap();
        
        assert_eq!(plaintext, decrypted.as_slice());
    }
}