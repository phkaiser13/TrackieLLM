/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/sensors/ffi_bridge.rs
 *
 * This file serves as the Foreign Function Interface (FFI) bridge from C to
 * the Rust-side sensor processing logic. It exposes `extern "C"` functions
 * that can be called from the C implementation of the sensor fusion engine
 * (`tk_sensors_fusion.c`).
 *
 * The primary purpose is to allow the C code to leverage the safe, high-performance
 * Rust implementations of filters (like the Complementary Filter) without needing
 * to reimplement that logic in C.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use crate::sensor_filters::{ComplementaryFilter, Filter};
use nalgebra::{UnitQuaternion, Vector3};
use std::ffi::c_void;

/// A handle to a Rust-managed filter object.
/// This is an opaque pointer from the perspective of C.
pub type FilterHandle = *mut c_void;

/// Creates an instance of the ComplementaryFilter and returns an opaque handle to it.
///
/// # Safety
/// The caller is responsible for eventually calling `rust_destroy_filter` on the
/// returned handle to avoid memory leaks.
#[no_mangle]
pub unsafe extern "C" fn rust_create_complementary_filter(alpha: f32) -> FilterHandle {
    let filter = Box::new(ComplementaryFilter::new(alpha));
    Box::into_raw(filter) as FilterHandle
}

/// Destroys a filter instance that was created by `rust_create_complementary_filter`.
///
/// # Safety
/// The handle must be a valid pointer returned by `rust_create_complementary_filter`.
/// Calling this function with an invalid handle leads to undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn rust_destroy_filter(handle: FilterHandle) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle as *mut ComplementaryFilter);
    }
}

/// Processes IMU data using the filter associated with the given handle.
///
/// # Safety
/// The handle must be a valid pointer to a filter. The pointers for gyro, accel,
/// and out_quaternion must be valid and non-null.
#[no_mangle]
pub unsafe extern "C" fn rust_filter_update(
    handle: FilterHandle,
    dt: f32,
    gyro_x: f32,
    gyro_y: f32,
    gyro_z: f32,
    acc_x: f32,
    acc_y: f32,
    acc_z: f32,
    out_quat_w: *mut f32,
    out_quat_x: *mut f32,
    out_quat_y: *mut f32,
    out_quat_z: *mut f32,
) {
    let filter = &mut *(handle as *mut ComplementaryFilter);
    let gyro = Vector3::new(gyro_x, gyro_y, gyro_z);
    let acc = Vector3::new(acc_x, acc_y, acc_z);

    filter.update(acc, gyro, dt);
    let orientation: &UnitQuaternion<f32> = filter.output();

    *out_quat_w = orientation.w;
    *out_quat_x = orientation.i;
    *out_quat_y = orientation.j;
    *out_quat_z = orientation.k;
}
