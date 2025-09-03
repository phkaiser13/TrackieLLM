/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: lib.rs
 *
 * -------------------------------------------------------------------------
 *  Overview
 * -------------------------------------------------------------------------
 *
 * This crate implements the **Rust side** of the TrackieLLM Foreign Function
 * Interface (FFI).  The public API consists of a set of `extern "C"` functions
 * declared in `ffi_bridge.rs`.  The `lib.rs` file glues everything together:
 *
 *   • Re‑exports the symbols from `ffi_bridge.rs` so that they become part of
 *     the crate’s public interface.
 *
 *   • Provides a small amount of crate‑level documentation and version
 *     information.
 *
 *   • Contains a `#[cfg(test)]` integration test suite that builds the whole
 *     library (including the C++ and C layers) and exercises the end‑to‑end
 *     flow.  The tests are deliberately verbose to demonstrate correct usage
 *     of the FFI functions from Rust.
 *
 * The implementation follows the same engineering principles used throughout
 * the project: explicit error handling, panic safety, thread‑local error
 * propagation, and zero‑cost abstractions.
 *
 * Dependencies:
 *   - libc (C type definitions)
 *   - once_cell (lazy static initialization)
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#![allow(dead_code)]
#![allow(unused_imports)]

/*==========================================================================*/
/*  Crate‑level documentation                                               */
/*==========================================================================*/

/// TrackieLLM FFI bridge – Rust implementation.
///
/// The crate builds a **C‑compatible dynamic library** (`cdylib`) and a
/// **static library** (`staticlib`).  The exported symbols are declared in
/// `ffi_bridge.rs`.  Consumers (C, C++, or other languages) link against the
/// generated library and call the functions using the C ABI.
///
/// # Version
///
/// The library follows semantic versioning.  The version string and the three
/// numeric components can be queried at runtime via the C functions
/// `tk_version_string` and `tk_version_numbers`.
///
/// # Features
///
/// * `async` – enables the asynchronous command executor based on a fixed‑size
///   thread‑pool.  When disabled the `tk_module_execute_command` function
///   executes synchronously.
/// * `simd` – enables SIMD‑accelerated helper functions (e.g. AVX2‑based
///   tensor addition).  The feature is optional because not all target
///   architectures support the required intrinsics.
///
/// # Safety
///
/// All exported functions are `extern "C"` and therefore **unsafe** to call
/// from Rust code.  The functions perform thorough pointer validation,
/// translate panics into `TkStatus` error codes and store a human‑readable
/// message in a thread‑local buffer that can be retrieved with
/// `tk_get_last_error`.  This design guarantees that the library never
/// propagates undefined behaviour across the FFI boundary.
///
/// # Usage example (Rust side)
///
/// ```rust
/// use trackie_ffi::ffi_bridge::*;
///
/// // Create a context
/// let mut ctx: *mut TkContext = std::ptr::null_mut();
/// let status = unsafe { tk_context_create(&mut ctx) };
/// assert_eq!(status, TkStatus::TK_STATUS_OK);
///
/// // Create a 2×3 float tensor filled with zeros
/// let shape = [2_i64, 3];
/// let mut tensor: *mut TkTensor = std::ptr::null_mut();
/// let status = unsafe {
///     tk_tensor_create(
///         &mut tensor,
///         TkDataType::TK_DATA_TYPE_FLOAT32,
///         shape.as_ptr(),
///         shape.len() as size_t,
///         std::ptr::null(),
///     )
/// };
/// assert_eq!(status, TkStatus::TK_STATUS_OK);
///
/// // Fill the tensor with a constant value
/// let value: f32 = 1.23;
/// let status = unsafe { tk_tensor_fill(tensor, &value as *const _ as *const std::ffi::c_void) };
/// assert_eq!(status, TkStatus::TK_STATUS_OK);
///
/// // Clean up
/// unsafe {
///     tk_tensor_destroy(&mut tensor);
///     tk_context_destroy(&mut ctx);
/// }
/// ```
pub const LIB_NAME: &str = "trackie_ffi";

/*==========================================================================*/
/*  Re‑export of the public FFI symbols                                    */
/*==========================================================================*/

pub mod ffi_bridge;
pub use ffi_bridge::*;

/*==========================================================================*/
/*  Version helpers (convenient Rust wrappers)                             */
/*==========================================================================*/

/// Returns the library version as a `(major, minor, patch)` tuple.
pub fn version() -> (u32, u32, u32) {
    let (major, minor, patch) = crate::utils::version_numbers();
    (major, minor, patch)
}

/// Returns the version string (e.g. `"1.0.3"`).
pub fn version_str() -> String {
    crate::utils::version_string()
}

/*==========================================================================*/
/*  Integration tests – end‑to‑end validation of the whole FFI stack        */
/*==========================================================================*/

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::utils::{c_str_to_rust, rust_str_to_c, aligned_alloc, aligned_free};
    use std::ffi::CString;
    use std::ptr;
    use std::slice;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Helper that creates a fresh `TkContext` for each test.
    fn new_context() -> *mut TkContext {
        let mut ctx: *mut TkContext = ptr::null_mut();
        let st = unsafe { tk_context_create(&mut ctx) };
        assert_eq!(st, TkStatus::TK_STATUS_OK);
        assert!(!ctx.is_null());
        ctx
    }

    /// Helper that destroys a `TkContext`.
    fn destroy_context(ctx: *mut TkContext) {
        let mut ctx_ptr = ctx;
        let st = unsafe { tk_context_destroy(&mut ctx_ptr) };
        assert_eq!(st, TkStatus::TK_STATUS_OK);
        assert!(ctx_ptr.is_null());
    }

    #[test]
    fn test_version_functions() {
        let (major, minor, patch) = version();
        assert!(major > 0);
        let ver_str = version_str();
        assert!(ver_str.contains(&major.to_string()));
        assert!(ver_str.contains(&minor.to_string()));
        assert!(ver_str.contains(&patch.to_string()));
    }

    #[test]
    fn test_context_lifecycle() {
        let ctx = new_context();
        destroy_context(ctx);
    }

    #[test]
    fn test_tensor_creation_and_fill() {
        let ctx = new_context();

        // Shape: 4 × 2 (row‑major)
        let shape = [4_i64, 2];
        let mut tensor: *mut TkTensor = ptr::null_mut();

        // Create tensor with uninitialized data.
        let st = unsafe {
            tk_tensor_create(
                &mut tensor,
                TkDataType::TK_DATA_TYPE_FLOAT32,
                shape.as_ptr(),
                shape.len() as size_t,
                ptr::null(),
            )
        };
        assert_eq!(st, TkStatus::TK_STATUS_OK);
        assert!(!tensor.is_null());

        // Fill with constant 3.14f.
        let value: f32 = 3.14;
        let st = unsafe {
            tk_tensor_fill(tensor, &value as *const _ as *const std::ffi::c_void)
        };
        assert_eq!(st, TkStatus::TK_STATUS_OK);

        // Retrieve data pointer and verify contents.
        let mut data_ptr: *const std::ffi::c_void = ptr::null();
        let st = unsafe { tk_tensor_get_data(tensor, &mut data_ptr) };
        assert_eq!(st, TkStatus::TK_STATUS_OK);
        assert!(!data_ptr.is_null());

        let num_elements = (shape[0] * shape[1]) as usize;
        let slice = unsafe {
            slice::from_raw_parts(data_ptr as *const f32, num_elements)
        };
        for &v in slice {
            assert!((v - value).abs() < f32::EPSILON);
        }

        // Clean up.
        let st = unsafe { tk_tensor_destroy(&mut tensor) };
        assert_eq!(st, TkStatus::TK_STATUS_OK);
        destroy_context(ctx);
    }

    #[test]
    fn test_tensor_addition() {
        let ctx = new_context();

        let shape = [2_i64, 3];
        let a_vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_vals: Vec<f32> = vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        // Create tensors A and B.
        let mut a: *mut TkTensor = ptr::null_mut();
        let mut b: *mut TkTensor = ptr::null_mut();
        let st_a = unsafe {
            tk_tensor_create(
                &mut a,
                TkDataType::TK_DATA_TYPE_FLOAT32,
                shape.as_ptr(),
                shape.len() as size_t,
                a_vals.as_ptr() as *const std::ffi::c_void,
            )
        };
        let st_b = unsafe {
            tk_tensor_create(
                &mut b,
                TkDataType::TK_DATA_TYPE_FLOAT32,
                shape.as_ptr(),
                shape.len() as size_t,
                b_vals.as_ptr() as *const std::ffi::c_void,
            )
        };
        assert_eq!(st_a, TkStatus::TK_STATUS_OK);
        assert_eq!(st_b, TkStatus::TK_STATUS_OK);
        assert!(!a.is_null() && !b.is_null());

        // Allocate result tensor (same shape, uninitialized).
        let mut result: *mut TkTensor = ptr::null_mut();
        let st_res = unsafe {
            tk_tensor_create(
                &mut result,
                TkDataType::TK_DATA_TYPE_FLOAT32,
                shape.as_ptr(),
                shape.len() as size_t,
                ptr::null(),
            )
        };
        assert_eq!(st_res, TkStatus::TK_STATUS_OK);
        assert!(!result.is_null());

        // Perform addition.
        let st_add = unsafe { tk_tensor_add(a, b, result) };
        assert_eq!(st_add, TkStatus::TK_STATUS_OK);

        // Verify result.
        let mut data_ptr: *const std::ffi::c_void = ptr::null();
        let st = unsafe { tk_tensor_get_data(result, &mut data_ptr) };
        assert_eq!(st, TkStatus::TK_STATUS_OK);
        let slice = unsafe {
            slice::from_raw_parts(data_ptr as *const f32, a_vals.len())
        };
        let expected: Vec<f32> = a_vals.iter().zip(b_vals.iter()).map(|(x, y)| x + y).collect();
        assert_eq!(slice, expected.as_slice());

        // Clean up.
        unsafe {
            tk_tensor_destroy(&mut a);
            tk_tensor_destroy(&mut b);
            tk_tensor_destroy(&mut result);
        }
        destroy_context(ctx);
    }

    #[test]
    fn test_matrix_multiplication() {
        let ctx = new_context();

        // A: 2×3 matrix
        let shape_a = [2_i64, 3];
        let a_vals: Vec<f32> = vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
        ];

        // B: 3×2 matrix
        let shape_b = [3_i64, 2];
        let b_vals: Vec<f32> = vec![
            7.0, 8.0, // col 0
            9.0, 10.0, // col 1
            11.0, 12.0, // col 2
        ];

        // Expected result C = A × B => 2×2 matrix
        // Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // Row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        let expected = vec![58.0, 64.0, 139.0, 154.0];

        // Create tensors.
        let mut a: *mut TkTensor = ptr::null_mut();
        let mut b: *mut TkTensor = ptr::null_mut();
        let st_a = unsafe {
            tk_tensor_create(
                &mut a,
                TkDataType::TK_DATA_TYPE_FLOAT32,
                shape_a.as_ptr(),
                shape_a.len() as size_t,
                a_vals.as_ptr() as *const std::ffi::c_void,
            )
        };
        let st_b = unsafe {
            tk_tensor_create(
                &mut b,
                TkDataType::TK_DATA_TYPE_FLOAT32,
                shape_b.as_ptr(),
                shape_b.len() as size_t,
                b_vals.as_ptr() as *const std::ffi::c_void,
            )
        };
        assert_eq!(st_a, TkStatus::TK_STATUS_OK);
        assert_eq!(st_b, TkStatus::TK_STATUS_OK);

        // Allocate result tensor with shape 2×2.
        let shape_c = [2_i64, 2];
        let mut c: *mut TkTensor = ptr::null_mut();
        let st_c = unsafe {
            tk_tensor_create(
                &mut c,
                TkDataType::TK_DATA_TYPE_FLOAT32,
                shape_c.as_ptr(),
                shape_c.len() as size_t,
                ptr::null(),
            )
        };
        assert_eq!(st_c, TkStatus::TK_STATUS_OK);

        // Perform matrix multiplication.
        let st_mul = unsafe { tk_tensor_matmul(a, b, c) };
        assert_eq!(st_mul, TkStatus::TK_STATUS_OK);

        // Verify result.
        let mut data_ptr: *const std::ffi::c_void = ptr::null();
        let st = unsafe { tk_tensor_get_data(c, &mut data_ptr) };
        assert_eq!(st, TkStatus::TK_STATUS_OK);
        let slice = unsafe {
            slice::from_raw_parts(data_ptr as *const f32, expected.len())
        };
        assert_eq!(slice, expected.as_slice());

        // Clean up.
        unsafe {
            tk_tensor_destroy(&mut a);
            tk_tensor_destroy(&mut b);
            tk_tensor_destroy(&mut c);
        }
        destroy_context(ctx);
    }

    #[test]
    fn test_audio_stream_write_and_read() {
        let ctx = new_context();

        // Create an audio stream: 48 kHz, stereo, 16‑bit, capacity 256 frames.
        let mut stream: *mut TkAudioStream = ptr::null_mut();
        let st = unsafe {
            tk_audio_stream_create(
                &mut stream,
                48_000,
                2,
                TkAudioFormat::TK_AUDIO_FMT_S16LE,
                256,
            )
        };
        assert_eq!(st, TkStatus::TK_STATUS_OK);
        assert!(!stream.is_null());

        // Generate dummy audio data (interleaved L,R samples).
        let frames = 128usize;
        let mut samples: Vec<i16> = Vec::with_capacity(frames * 2);
        for i in 0..frames {
            samples.push(i as i16); // left channel
            samples.push((-i) as i16); // right channel
        }

        // Write to the stream.
        let st_write = unsafe {
            tk_audio_stream_write(
                stream,
                frames as size_t,
                samples.as_ptr() as *const std::ffi::c_void,
            )
        };
        assert_eq!(st_write, TkStatus::TK_STATUS_OK);

        // Read back.
        let mut out_buf: Vec<i16> = vec![0; frames * 2];
        let mut frames_read: size_t = 0;
        let st_read = unsafe {
            tk_audio_stream_read(
                stream,
                frames as size_t,
                out_buf.as_mut_ptr() as *mut std::ffi::c_void,
                &mut frames_read,
            )
        };
        assert_eq!(st_read, TkStatus::TK_STATUS_OK);
        assert_eq!(frames_read as usize, frames);
        assert_eq!(samples, out_buf);

        // Reset and verify empty.
        let st_reset = unsafe { tk_audio_stream_reset(stream) };
        assert_eq!(st_reset, TkStatus::TK_STATUS_OK);
        let mut zero_read: size_t = 0;
        let st_read2 = unsafe {
            tk_audio_stream_read(
                stream,
                frames as size_t,
                out_buf.as_mut_ptr() as *mut std::ffi::c_void,
                &mut zero_read,
            )
        };
        assert_eq!(st_read2, TkStatus::TK_STATUS_OK);
        assert_eq!(zero_read, 0);

        // Clean up.
        let st_destroy = unsafe { tk_audio_stream_destroy(&mut stream) };
        assert_eq!(st_destroy, TkStatus::TK_STATUS_OK);
        destroy_context(ctx);
    }

    #[test]
    fn test_vision_frame_creation_and_access() {
        let ctx = new_context();

        // Create a 320×240 RGB24 frame.
        let mut frame: *mut TkVisionFrame = ptr::null_mut();
        let st = unsafe {
            tk_vision_frame_create(
                &mut frame,
                320,
                240,
                TkVisionFormat::TK_VISION_FMT_RGB24,
            )
        };
        assert_eq!(st, TkStatus::TK_STATUS_OK);
        assert!(!frame.is_null());

        // Retrieve metadata.
        let mut width: uint32_t = 0;
        let mut height: uint32_t = 0;
        let mut fmt: TkVisionFormat = TkVisionFormat::TK_VISION_FMT_RGB24;
        let st_info = unsafe {
            tk_vision_frame_get_info(
                frame,
                &mut width,
                &mut height,
                &mut fmt,
            )
        };
        assert_eq!(st_info, TkStatus::TK_STATUS_OK);
        assert_eq!(width, 320);
        assert_eq!(height, 240);
        assert_eq!(fmt, TkVisionFormat::TK_VISION_FMT_RGB24);

        // Get mutable data pointer and fill with a solid colour (e.g., green).
        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let st_mut = unsafe { tk_vision_frame_get_mutable_data(frame, &mut data_ptr) };
        assert_eq!(st_mut, TkStatus::TK_STATUS_OK);
        assert!(!data_ptr.is_null());

        // Fill the buffer: each pixel = (R=0, G=255, B=0)
        let stride = 320 * 3; // RGB24 => 3 bytes per pixel
        let total_bytes = stride * 240;
        unsafe {
            let buf = std::slice::from_raw_parts_mut(data_ptr as *mut u8, total_bytes);
            for chunk in buf.chunks_exact_mut(3) {
                chunk[0] = 0;   // R
                chunk[1] = 255; // G
                chunk[2] = 0;   // B
            }
        }

        // Verify a few pixels via read‑only accessor.
        let mut const_ptr: *const std::ffi::c_void = ptr::null();
        let st_const = unsafe { tk_vision_frame_get_data(frame, &mut const_ptr) };
        assert_eq!(st_const, TkStatus::TK_STATUS_OK);
        assert!(!const_ptr.is_null());
        unsafe {
            let buf = std::slice::from_raw_parts(const_ptr as *const u8, total_bytes);
            for chunk in buf.chunks_exact(3).take(10) {
                assert_eq!(chunk[0], 0);
                assert_eq!(chunk[1], 255);
                assert_eq!(chunk[2], 0);
            }
        }

        // Clean up.
        let st_destroy = unsafe { tk_vision_frame_destroy(&mut frame) };
        assert_eq!(st_destroy, TkStatus::TK_STATUS_OK);
        destroy_context(ctx);
    }

    #[test]
    fn test_aligned_allocation_and_secure_zero() {
        // Allocate 4096 bytes aligned to SIMD_ALIGNMENT.
        let size = 4096usize;
        let ptr = aligned_alloc(size).expect("aligned_alloc failed");
        assert_eq!((ptr.as_ptr() as usize) % utils::SIMD_ALIGNMENT, 0);

        // Fill with a pattern.
        unsafe {
            for i in 0..size {
                ptr.as_ptr().add(i).write_volatile(0xAB);
            }
        }

        // Securely zero the memory.
        unsafe {
            utils::secure_zero(ptr.as_ptr(), size);
        }

        // Verify zeroed.
        unsafe {
            for i in 0..size {
                let val = ptr.as_ptr().add(i).read_volatile();
                assert_eq!(val, 0);
            }
        }

        unsafe {
            aligned_free(ptr.as_ptr());
        }
    }

    #[test]
    fn test_error_propagation() {
        // Pass a NULL pointer to trigger an error.
        let st = unsafe { tk_context_create(ptr::null_mut()) };
        assert_eq!(st, TkStatus::TK_STATUS_ERROR_NULL_POINTER);
        // Retrieve the error message.
        let err_c = unsafe { tk_get_last_error() };
        let err_str = unsafe { CStr::from_ptr(err_c) }
            .to_string_lossy()
            .into_owned();
        assert!(err_str.contains("NULL"));
    }

    #[test]
    fn test_async_command_execution() {
        // This test only runs when the `async` feature is enabled.
        #[cfg(feature = "async")]
        {
            use std::sync::atomic::AtomicBool;
            use std::sync::Arc;

            static CALLBACK_CALLED: AtomicBool = AtomicBool::new(false);
            unsafe extern "C" fn async_cb(
                status: TkStatus,
                _result: *mut c_void,
                _user_data: *mut c_void,
            ) {
                assert_eq!(status, TkStatus::TK_STATUS_OK);
                CALLBACK_CALLED.store(true, Ordering::SeqCst);
            }

            let ctx = new_context();
            let cmd = CString::new("async_test").unwrap();
            let st = unsafe {
                tk_module_execute_command(
                    ctx,
                    TkModuleType::TK_MODULE_CORTEX,
                    cmd.as_ptr(),
                    ptr::null_mut(),
                    Some(async_cb),
                    ptr::null_mut(),
                )
            };
            assert_eq!(st, TkStatus::TK_STATUS_OK);

            // Give the async worker a moment to run.
            std::thread::sleep(std::time::Duration::from_millis(100));
            assert!(CALLBACK_CALLED.load(Ordering::SeqCst));

            destroy_context(ctx);
        }
    }

    #[test]
    fn test_c_string_helpers() {
        // Verify conversion utilities.
        let original = "rust ↔ C interop";
        let c_str = rust_str_to_c(original).expect("CString conversion failed");
        let back = unsafe { c_str_to_rust(c_str.as_ptr()) }.expect("C→Rust conversion failed");
        assert_eq!(original, back);
    }
}

/*==========================================================================*/
/*  End of file                                                            */
/*==========================================================================*/
