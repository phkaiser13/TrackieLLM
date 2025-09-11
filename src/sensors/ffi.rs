/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/sensors/ffi.rs
 *
 * This file contains the Foreign Function Interface (FFI) bindings for the
 * C-based sensor fusion library (`tk_sensors_fusion`). It defines the Rust
 * equivalents of the C structs and declares the function signatures that can
 * be called from Rust.
 *
 * The structs use `#[repr(C)]` to ensure their memory layout is compatible
 * with the C counterparts.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#![allow(non_camel_case_types, non_snake_case)]
#![deny(missing_docs)]

//! FFI bindings for `tk_sensors_fusion.h`.

use std::os::raw::{c_char, c_int, c_void};

// Opaque type for the sensor fusion engine handle.
/// Opaque handle to the C sensor fusion engine.
pub enum tk_sensor_fusion_s {}
/// A type alias for the pointer to the opaque engine struct.
pub type tk_sensor_fusion_t = tk_sensor_fusion_s;

// Mirror of the C error code enum/type.
/// Represents an error code from the C API.
pub type tk_error_code_t = c_int;

/// @brief Configuration for initializing the Sensor Fusion engine.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tk_sensor_fusion_config_t {
    /// The target frequency at which sensor data is processed.
    pub update_rate_hz: f32,
    /// Beta parameter for Madgwick or similar filters.
    pub gyro_trust_factor: f32,
}

/// @brief High-level classification of the user's current motion.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum tk_motion_state_e {
    /// The state cannot be determined yet.
    TK_MOTION_STATE_UNKNOWN,
    /// The user is still.
    TK_MOTION_STATE_STATIONARY,
    /// The user is walking at a steady pace.
    TK_MOTION_STATE_WALKING,
    /// The user is running.
    TK_MOTION_STATE_RUNNING,
    /// A potential fall event has been detected (freefall).
    TK_MOTION_STATE_FALLING,
}

/// @brief Represents orientation in 3D space.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tk_quaternion_t {
    /// W component of the quaternion.
    pub w: f32,
    /// X component of the quaternion.
    pub x: f32,
    /// Y component of the quaternion.
    pub y: f32,
    /// Z component of the quaternion.
    pub z: f32,
}

/// @brief The fused, high-level output of the sensor fusion engine.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tk_world_state_t {
    /// Timestamp of the last update.
    pub last_update_timestamp_ns: u64,
    /// The absolute orientation of the device in space.
    pub orientation: tk_quaternion_t,
    /// The user's classified motion state.
    pub motion_state: tk_motion_state_e,
    /// The current state from the Voice Activity Detector.
    pub is_speech_detected: bool,
}


extern "C" {
    // --- Engine Lifecycle Management ---

    /// @brief Creates and initializes a new Sensor Fusion engine instance.
    pub fn tk_sensor_fusion_create(
        out_engine: *mut *mut tk_sensor_fusion_t,
        config: *const tk_sensor_fusion_config_t,
    ) -> tk_error_code_t;

    /// @brief Destroys a Sensor Fusion engine instance.
    pub fn tk_sensor_fusion_destroy(engine: *mut *mut tk_sensor_fusion_t);

    // --- State Retrieval ---

    /// @brief Retrieves the latest, most up-to-date world state.
    pub fn tk_sensor_fusion_get_world_state(
        engine: *const tk_sensor_fusion_t,
        out_state: *mut tk_world_state_t,
    ) -> tk_error_code_t;

    // --- Data Injection & Update (for future use if needed from Rust) ---

    // We don't need these for the sensor_worker, but include them for completeness.
    // pub fn tk_sensor_fusion_update(engine: *mut tk_sensor_fusion_t, delta_time_s: f32) -> tk_error_code_t;
}
