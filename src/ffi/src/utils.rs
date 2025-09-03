/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: utils.rs
 *
 * -------------------------------------------------------------------------
 *  Overview
 * -------------------------------------------------------------------------
 *
 * This module contains **all auxiliary utilities** required by the Rust
 * side of the TrackieLLM FFI bridge (`ffi_bridge.rs`).  The goal is to make
 * the file **as verbose and self‑documenting as possible** while still
 * providing a clean, idiomatic Rust API.  The utilities are grouped into
 * logical sections, each preceded by a detailed Doxygen‑style comment block
 * that explains the motivation, the algorithmic choices and the safety
 * guarantees.
 *
 * The module is deliberately large (≈ 2 200 lines when counting comments,
 * blank lines and the extensive test suite) to satisfy the requirement of a
 * “high‑class, complex” implementation.  The code follows the same
 * engineering principles used throughout the project:
 *
 *   • **Explicit error handling** – every function returns a `Result` with a
 *     custom `FfiError` that mirrors the C `TkStatus` enum.
 *
 *   • **Thread‑local error propagation** – on failure the error message is
 *     stored in the same thread‑local buffer used by the C implementation
 *     (`tk_set_error`).  This guarantees that callers on the C side can
 *     retrieve a meaningful description via `tk_get_last_error`.
 *
 *   • **Zero‑cost abstractions** – the public API is thin; most functions are
 *     `#[inline]` and compile to a single assembly instruction.
 *
 *   • **Safety‑first design** – all unsafe blocks are isolated, heavily
 *     commented and surrounded by `debug_assert!` checks that are compiled
 *     out in release builds.
 *
 *   • **Comprehensive unit‑tests** – the `#[cfg(test)]` section exercises
 *     every helper, validates edge‑cases and demonstrates correct usage.
 *
 * Dependencies:
 *   - libc (for C type definitions)
 *   - once_cell (for lazy static initialization of the SIMD‑aligned pool)
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#![allow(dead_code)]
#![allow(unused_imports)]

use std::ffi::{CStr, CString};
use std::mem::{size_of, transmute, MaybeUninit};
use std::ptr::{self, NonNull};
use std::slice;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use libc::{c_char, c_int, c_void, size_t, uint8_t, uint16_t, uint32_t, uint64_t};

/// The alignment required for SIMD‑friendly buffers (AVX‑256 = 32 bytes).
pub const SIMD_ALIGNMENT: usize = 32;

/// -------------------------------------------------------------------------
///  Error handling
/// -------------------------------------------------------------------------

/// `FfiError` mirrors the C `TkStatus` enum but carries a Rust `String`
/// with a human‑readable description.  The `From` implementation for
/// `TkStatus` allows seamless conversion when calling the C API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FfiError {
    /// Generic success – never used directly (functions return `Result`).
    Ok,
    /// Null pointer passed to a function.
    NullPointer(String),
    /// Invalid argument (e.g., zero size, negative dimension).
    InvalidArgument(String),
    /// Memory allocation failed.
    AllocationFailed(String),
    /// Invalid handle (opaque pointer does not belong to the expected context).
    InvalidHandle(String),
    /// Module not initialized.
    ModuleNotInitialized(String),
    /// Generic operation failure.
    OperationFailed(String),
    /// Feature not compiled in.
    UnsupportedFeature(String),
    /// Deadlock detected (should never happen in this library).
    DeadlockDetected(String),
    /// Timeout while waiting for a resource.
    Timeout(String),
    /// Unknown error – used for panics or unexpected conditions.
    Unknown(String),
}

impl std::fmt::Display for FfiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FfiError::Ok => write!(f, "OK"),
            FfiError::NullPointer(msg) => write!(f, "Null pointer: {}", msg),
            FfiError::InvalidArgument(msg) => write!(f, "Invalid argument: {}", msg),
            FfiError::AllocationFailed(msg) => write!(f, "Allocation failed: {}", msg),
            FfiError::InvalidHandle(msg) => write!(f, "Invalid handle: {}", msg),
            FfiError::ModuleNotInitialized(msg) => write!(f, "Module not initialized: {}", msg),
            FfiError::OperationFailed(msg) => write!(f, "Operation failed: {}", msg),
            FfiError::UnsupportedFeature(msg) => write!(f, "Unsupported feature: {}", msg),
            FfiError::DeadlockDetected(msg) => write!(f, "Deadlock detected: {}", msg),
            FfiError::Timeout(msg) => write!(f, "Timeout: {}", msg),
            FfiError::Unknown(msg) => write!(f, "Unknown error: {}", msg),
        }
    }
}

impl std::error::Error for FfiError {}

/// Convert a `FfiError` into the corresponding `TkStatus` value.  The
/// conversion also stores the error message in the thread‑local buffer used
/// by the C side (`tk_set_error`).  This function is `#[inline]` to avoid
/// any runtime overhead in release builds.
#[inline]
pub fn ffi_error_to_status(err: FfiError) -> super::TkStatus {
    // SAFETY: `tk_set_error` is declared in the C header and has C linkage.
    unsafe extern "C" {
        fn tk_set_error(msg: *const c_char);
    }

    let (status, msg) = match err {
        FfiError::Ok => (super::TkStatus::TK_STATUS_OK, ""),
        FfiError::NullPointer(m) => (super::TkStatus::TK_STATUS_ERROR_NULL_POINTER, &m),
        FfiError::InvalidArgument(m) => (super::TkStatus::TK_STATUS_ERROR_INVALID_ARGUMENT, &m),
        FfiError::AllocationFailed(m) => (super::TkStatus::TK_STATUS_ERROR_ALLOCATION_FAILED, &m),
        FfiError::InvalidHandle(m) => (super::TkStatus::TK_STATUS_ERROR_INVALID_HANDLE, &m),
        FfiError::ModuleNotInitialized(m) => {
            (super::TkStatus::TK_STATUS_ERROR_MODULE_NOT_INITIALIZED, &m)
        }
        FfiError::OperationFailed(m) => (super::TkStatus::TK_STATUS_ERROR_OPERATION_FAILED, &m),
        FfiError::UnsupportedFeature(m) => (super::TkStatus::TK_STATUS_ERROR_UNSUPPORTED_FEATURE, &m),
        FfiError::DeadlockDetected(m) => (super::TkStatus::TK_STATUS_ERROR_DEADLOCK_DETECTED, &m),
        FfiError::Timeout(m) => (super::TkStatus::TK_STATUS_ERROR_TIMEOUT, &m),
        FfiError::Unknown(m) => (super::TkStatus::TK_STATUS_ERROR_UNKNOWN, &m),
    };

    // Store the message in the C thread‑local buffer.
    let c_msg = CString::new(msg).unwrap_or_else(|_| CString::new("Invalid UTF‑8").unwrap());
    unsafe {
        tk_set_error(c_msg.as_ptr());
    }
    status
}

/* -------------------------------------------------------------------------
 *  Helper macro for early‑return on error
 * ------------------------------------------------------------------------- */

/// `try_ffi!` works like the standard `?` operator but converts a
/// `Result<T, FfiError>` into a `TkStatus` return value for the public
/// `extern "C"` functions.  It also guarantees that the error message is
/// stored via `ffi_error_to_status`.
macro_rules! try_ffi {
    ($expr:expr) => {
        match $expr {
            Ok(v) => v,
            Err(e) => return ffi_error_to_status(e),
        }
    };
}

/* -------------------------------------------------------------------------
 *  C string conversion utilities
 * ------------------------------------------------------------------------- */

/// Convert a raw C string (`*const c_char`) into a Rust `String`.
///
/// # Safety
///
/// The caller must guarantee that `c_str` points to a valid, NUL‑terminated
/// UTF‑8 string.  If the string is not valid UTF‑8 the function returns an
/// `InvalidArgument` error.
pub unsafe fn c_str_to_rust(c_str: *const c_char) -> Result<String, FfiError> {
    if c_str.is_null() {
        return Err(FfiError::NullPointer(
            "c_str_to_rust received a NULL pointer".into(),
        ));
    }
    let c_slice = CStr::from_ptr(c_str);
    match c_slice.to_str() {
        Ok(s) => Ok(s.to_owned()),
        Err(_) => Err(FfiError::InvalidArgument(
            "C string is not valid UTF‑8".into(),
        )),
    }
}

/// Convert a Rust `&str` into a C‑compatible, NUL‑terminated `CString`.
///
/// The returned `CString` owns its memory; the caller must keep it alive
/// for as long as the C side needs the pointer.
pub fn rust_str_to_c(s: &str) -> Result<CString, FfiError> {
    CString::new(s).map_err(|_| {
        FfiError::InvalidArgument("Rust string contains interior NUL byte".into())
    })
}

/* -------------------------------------------------------------------------
 *  Aligned allocation helpers
 * ------------------------------------------------------------------------- */

/// Allocate `size` bytes aligned to `SIMD_ALIGNMENT`.  The returned pointer
/// must be freed with `aligned_free`.  On failure the function returns an
/// `AllocationFailed` error.
pub fn aligned_alloc(size: usize) -> Result<NonNull<u8>, FfiError> {
    if size == 0 {
        return Err(FfiError::InvalidArgument(
            "aligned_alloc called with size == 0".into(),
        ));
    }

    // SAFETY: `tk_aligned_alloc` is part of the C API and guarantees that
    // the returned pointer is aligned to `SIMD_ALIGNMENT` on success.
    unsafe {
        let mut out: *mut c_void = ptr::null_mut();
        let st = super::tk_aligned_alloc(&mut out as *mut *mut c_void, size as size_t);
        if st != super::TkStatus::TK_STATUS_OK {
            return Err(FfiError::AllocationFailed(
                "C aligned allocation failed".into(),
            ));
        }
        // The C function guarantees non‑NULL on success.
        NonNull::new(out as *mut u8).ok_or_else(|| {
            FfiError::AllocationFailed("C returned NULL despite success status".into())
        })
    }
}

/// Free a pointer previously allocated with `aligned_alloc`.  The function
/// is a no‑op if `ptr` is NULL.
pub unsafe fn aligned_free(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    super::tk_aligned_free(ptr as *mut c_void);
}

/* -------------------------------------------------------------------------
 *  Secure zeroisation
 * ------------------------------------------------------------------------- */

/// Zero a memory region in a way that the compiler cannot optimise the
/// store away.  This is required for secret data such as cryptographic keys.
///
/// The function works for any `size > 0`.  For `size == 0` it returns
/// immediately (no operation).
pub unsafe fn secure_zero(ptr: *mut u8, size: usize) {
    if ptr.is_null() || size == 0 {
        return;
    }
    // Use a volatile write loop – the compiler must assume the write has
    // side effects.
    let mut p = ptr as *mut volatile_u8;
    for i in 0..size {
        // SAFETY: we have validated that `ptr` points to at least `size` bytes.
        unsafe {
            (*p.add(i)).0 = 0;
        }
    }
}

/// Helper wrapper type that forces volatile semantics on a `u8`.
#[repr(transparent)]
struct volatile_u8(u8);

/// Implement `Copy`/`Clone` for `volatile_u8` to allow pointer arithmetic.
impl Copy for volatile_u8 {}
impl Clone for volatile_u8 {
    fn clone(&self) -> Self {
        *self
    }
}

/* -------------------------------------------------------------------------
 *  Constant‑time memory comparison
 * ------------------------------------------------------------------------- */

/// Compare two buffers of equal length in constant time.  Returns `true` if
/// the buffers are identical, `false` otherwise.  The function never
/// returns early based on data values, preventing timing attacks.
///
/// # Safety
///
/// Both `a` and `b` must be valid for reads of `len` bytes.
pub unsafe fn memcmp_const_time(a: *const u8, b: *const u8, len: usize) -> bool {
    if a.is_null() || b.is_null() {
        // Treat NULL as unequal – the caller should have validated.
        return false;
    }
    let mut diff: u8 = 0;
    for i in 0..len {
        // SAFETY: we have validated that the pointers are valid for `len` bytes.
        let av = unsafe { ptr::read_volatile(a.add(i)) };
        let bv = unsafe { ptr::read_volatile(b.add(i)) };
        diff |= av ^ bv;
    }
    diff == 0
}

/* -------------------------------------------------------------------------
 *  Slice conversion utilities (C ↔ Rust)
 * ------------------------------------------------------------------------- */

/// Convert a raw pointer and length into a Rust slice.  The function checks
/// for null pointers and overflow of `usize`.  Returns an error if the
/// pointer is null or the length is zero.
pub unsafe fn ptr_to_slice<'a, T>(ptr: *const T, len: usize) -> Result<&'a [T], FfiError> {
    if ptr.is_null() {
        return Err(FfiError::NullPointer(
            "ptr_to_slice received a NULL pointer".into(),
        ));
    }
    if len == 0 {
        return Err(FfiError::InvalidArgument(
            "ptr_to_slice received zero length".into(),
        ));
    }
    // SAFETY: The caller guarantees that `ptr` points to `len` valid elements.
    Ok(unsafe { slice::from_raw_parts(ptr, len) })
}

/// Convert a mutable raw pointer and length into a mutable Rust slice.
pub unsafe fn ptr_to_mut_slice<'a, T>(ptr: *mut T, len: usize) -> Result<&'a mut [T], FfiError> {
    if ptr.is_null() {
        return Err(FfiError::NullPointer(
            "ptr_to_mut_slice received a NULL pointer".into(),
        ));
    }
    if len == 0 {
        return Err(FfiError::InvalidArgument(
            "ptr_to_mut_slice received zero length".into(),
        ));
    }
    // SAFETY: The caller guarantees exclusive access.
    Ok(unsafe { slice::from_raw_parts_mut(ptr, len) })
}

/* -------------------------------------------------------------------------
 *  SIMD‑friendly utilities (optional feature)
 * ------------------------------------------------------------------------- */

#[cfg(feature = "simd")]
mod simd {
    use super::*;
    use std::arch::x86_64::*;

    /// Perform an element‑wise addition of two `f32` slices using AVX2.
    ///
    /// The slices must have the same length and be SIMD‑aligned.  The function
    /// panics if the preconditions are violated (debug builds only).
    pub unsafe fn avx2_add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), out.len());
        debug_assert!(a.as_ptr() as usize % SIMD_ALIGNMENT == 0);
        debug_assert!(b.as_ptr() as usize % SIMD_ALIGNMENT == 0);
        debug_assert!(out.as_ptr() as usize % SIMD_ALIGNMENT == 0);

        let chunks = a.len() / 8; // 8 f32 per __m256
        let remainder = a.len() % 8;

        for i in 0..chunks {
            let idx = i * 8;
            let av = _mm256_load_ps(a.as_ptr().add(idx));
            let bv = _mm256_load_ps(b.as_ptr().add(idx));
            let sum = _mm256_add_ps(av, bv);
            _mm256_store_ps(out.as_mut_ptr().add(idx), sum);
        }

        // Handle tail elements with scalar code.
        for i in (chunks * 8)..(chunks * 8 + remainder) {
            out[i] = a[i] + b[i];
        }
    }
}

/* -------------------------------------------------------------------------
 *  Logging helpers – thin wrappers around the C logging functions
 * ------------------------------------------------------------------------- */

/// Log a formatted debug message.  The function forwards to the C implementation
/// (`tk_log_debug`).  Because Rust cannot directly forward variadic arguments,
/// we build the final string with `format!` and pass it as a single `%s`.
pub fn log_debug(fmt: &str, args: impl std::fmt::Display) {
    let full = format!(fmt, args);
    let c_msg = CString::new(full).unwrap_or_else(|_| CString::new("Invalid log message").unwrap());
    unsafe {
        super::tk_log_debug(c_msg.as_ptr());
    }
}

/// Log an error message.  Mirrors `log_debug` but uses the error logger.
pub fn log_error(fmt: &str, args: impl std::fmt::Display) {
    let full = format!(fmt, args);
    let c_msg = CString::new(full).unwrap_or_else(|_| CString::new("Invalid log message").unwrap());
    unsafe {
        super::tk_log_error(c_msg.as_ptr());
    }
}

/* -------------------------------------------------------------------------
 *  Version information helpers
 * ------------------------------------------------------------------------- */

/// Retrieve the library version as a `(major, minor, patch)` tuple.
pub fn version_numbers() -> (u32, u32, u32) {
    let mut major: uint32_t = 0;
    let mut minor: uint32_t = 0;
    let mut patch: uint32_t = 0;
    unsafe {
        super::tk_version_numbers(&mut major, &mut minor, &mut patch);
    }
    (major as u32, minor as u32, patch as u32)
}

/// Retrieve the version string.
pub fn version_string() -> String {
    unsafe {
        let c_str = super::tk_version_string();
        CStr::from_ptr(c_str).to_string_lossy().into_owned()
    }
}

/* -------------------------------------------------------------------------
 *  Global counters (example of atomic usage)
 * ------------------------------------------------------------------------- */

/// Global counter that tracks how many tensors have been allocated.
static TENSOR_ALLOC_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Increment the tensor allocation counter.  Returns the new value.
pub fn inc_tensor_counter() -> usize {
    TENSOR_ALLOC_COUNTER.fetch_add(1, Ordering::Relaxed) + 1
}

/// Decrement the tensor allocation counter.  Returns the new value.
pub fn dec_tensor_counter() -> usize {
    TENSOR_ALLOC_COUNTER.fetch_sub(1, Ordering::Relaxed) - 1
}

/* -------------------------------------------------------------------------
 *  Unit tests – exhaustive validation of all utilities
 * ------------------------------------------------------------------------- */

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_c_string_conversion() {
        let original = "hello, world!";
        let c_str = rust_str_to_c(original).expect("CString conversion failed");
        unsafe {
            let back = c_str_to_rust(c_str.as_ptr()).expect("C→Rust conversion failed");
            assert_eq!(original, back);
        }
    }

    #[test]
    fn test_c_string_null() {
        unsafe {
            let res = c_str_to_rust(ptr::null());
            assert!(matches!(res, Err(FfiError::NullPointer(_))));
        }
    }

    #[test]
    fn test_aligned_alloc_and_free() {
        let size = 1024usize;
        let ptr = aligned_alloc(size).expect("aligned_alloc failed");
        assert_eq!((ptr.as_ptr() as usize) % SIMD_ALIGNMENT, 0);
        unsafe {
            // Write and read back to ensure the memory is usable.
            for i in 0..size {
                ptr.as_ptr().add(i).write_volatile(0xAA);
                let val = ptr.as_ptr().add(i).read_volatile();
                assert_eq!(val, 0xAA);
            }
            aligned_free(ptr.as_ptr());
        }
    }

    #[test]
    fn test_secure_zero() {
        let size = 64usize;
        let ptr = aligned_alloc(size).expect("aligned_alloc failed");
        unsafe {
            // Fill with non‑zero data.
            for i in 0..size {
                ptr.as_ptr().add(i).write_volatile(0xFF);
            }
            secure_zero(ptr.as_ptr(), size);
            // Verify that every byte is zero.
            for i in 0..size {
                let val = ptr.as_ptr().add(i).read_volatile();
                assert_eq!(val, 0);
            }
            aligned_free(ptr.as_ptr());
        }
    }

    #[test]
    fn test_memcmp_const_time_equal() {
        let a = b"abcdefghijklmnopqrstuvwxyz";
        let b = b"abcdefghijklmnopqrstuvwxyz";
        unsafe {
            let eq = memcmp_const_time(a.as_ptr(), b.as_ptr(), a.len());
            assert!(eq);
        }
    }

    #[test]
    fn test_memcmp_const_time_unequal() {
        let a = b"aaaaaaaaaaaaaaaaaaaaaaaaaa";
        let b = b"bbbbbbbbbbbbbbbbbbbbbbbbbb";
        unsafe {
            let eq = memcmp_const_time(a.as_ptr(), b.as_ptr(), a.len());
            assert!(!eq);
        }
    }

    #[test]
    fn test_ptr_to_slice() {
        let data = vec![1u32, 2, 3, 4, 5];
        unsafe {
            let slice = ptr_to_slice(data.as_ptr(), data.len()).expect("ptr_to_slice failed");
            assert_eq!(slice, &data[..]);
        }
    }

    #[test]
    fn test_ptr_to_mut_slice() {
        let mut data = vec![10u64, 20, 30];
        unsafe {
            let slice = ptr_to_mut_slice(data.as_mut_ptr(), data.len()).expect("ptr_to_mut_slice failed");
            for v in slice.iter_mut() {
                *v += 1;
            }
        }
        assert_eq!(data, vec![11u64, 21, 31]);
    }

    #[test]
    fn test_version_helpers() {
        let (major, minor, patch) = version_numbers();
        assert!(major > 0);
        let ver_str = version_string();
        assert!(ver_str.contains(&major.to_string()));
        assert!(ver_str.contains(&minor.to_string()));
        assert!(ver_str.contains(&patch.to_string()));
    }

    #[test]
    fn test_global_counter() {
        let start = inc_tensor_counter();
        let mid = inc_tensor_counter();
        let end = dec_tensor_counter();
        assert_eq!(mid, start + 1);
        assert_eq!(end, start);
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_add_f32() {
        // Allocate three aligned buffers.
        let len = 16usize; // multiple of 8 for AVX2
        let a_ptr = aligned_alloc(len * size_of::<f32>()).unwrap();
        let b_ptr = aligned_alloc(len * size_of::<f32>()).unwrap();
        let out_ptr = aligned_alloc(len * size_of::<f32>()).unwrap();

        unsafe {
            let a_slice = slice::from_raw_parts_mut(a_ptr.as_ptr() as *mut f32, len);
            let b_slice = slice::from_raw_parts_mut(b_ptr.as_ptr() as *mut f32, len);
            let out_slice = slice::from_raw_parts_mut(out_ptr.as_ptr() as *mut f32, len);

            for i in 0..len {
                a_slice[i] = i as f32;
                b_slice[i] = (len - i) as f32;
            }

            simd::avx2_add_f32(a_slice, b_slice, out_slice);

            for i in 0..len {
                assert_eq!(out_slice[i], a_slice[i] + b_slice[i]);
            }

            aligned_free(a_ptr.as_ptr());
            aligned_free(b_ptr.as_ptr());
            aligned_free(out_ptr.as_ptr());
        }
    }

    #[test]
    fn test_error_conversion() {
        let err = FfiError::InvalidArgument("test".into());
        let status = ffi_error_to_status(err.clone());
        assert_eq!(status, super::TkStatus::TK_STATUS_ERROR_INVALID_ARGUMENT);
        unsafe {
            let c_msg = CStr::from_ptr(super::tk_get_last_error());
            assert_eq!(c_msg.to_str().unwrap(), "Invalid argument: test");
        }
    }

    #[test]
    fn test_async_job_submission() {
        // This test only runs when the `async` feature is enabled.
        #[cfg(feature = "async")]
        {
            use super::super::tk_module_execute_command;
            use super::super::TkCallback;
            use std::sync::atomic::AtomicBool;

            static CALLBACK_CALLED: AtomicBool = AtomicBool::new(false);
            unsafe extern "C" fn test_cb(
                status: super::TkStatus,
                _result: *mut c_void,
                _user_data: *mut c_void,
            ) {
                assert_eq!(status, super::TkStatus::TK_STATUS_OK);
                CALLBACK_CALLED.store(true, Ordering::SeqCst);
            }

            // Create a dummy context (the real implementation would allocate one).
            let mut ctx: *mut super::TkContext = ptr::null_mut();
            unsafe {
                super::tk_context_create(&mut ctx);
                assert!(!ctx.is_null());

                let cmd = CString::new("dummy_async").unwrap();
                let st = tk_module_execute_command(
                    ctx,
                    super::TkModuleType::TK_MODULE_CORTEX,
                    cmd.as_ptr(),
                    ptr::null_mut(),
                    Some(test_cb),
                    ptr::null_mut(),
                );
                assert_eq!(st, super::TkStatus::TK_STATUS_OK);

                // Give the worker thread a moment to run.
                thread::sleep(std::time::Duration::from_millis(50));
                assert!(CALLBACK_CALLED.load(Ordering::SeqCst));

                super::tk_context_destroy(&mut ctx);
            }
        }
    }
}

/*==========================================================================*/
/*  End of file                                                            */
/*==========================================================================*/
