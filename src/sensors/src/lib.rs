/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/sensors/lib.rs
 *
 * This file is the main library entry point for the 'sensors' crate. This
 * crate is responsible for processing and fusing data from various low-level
 * sensors (like an IMU) to produce a high-level, coherent understanding of
 * the device's physical state in the world.
 *
 * It provides a safe Rust abstraction layer over the C-based sensor fusion
 * engine defined in `tk_sensors_fusion.h`. The primary goal is to provide the
 * Cortex with a simple, high-level "world state" (containing orientation,
 * motion state, etc.) instead of noisy, high-frequency raw sensor data.
 *
 * The main components are:
 * - `sensor_fusion`: A safe wrapper around the main sensor fusion engine.
 * - `sensor_filters`: Rust-native implementations of common sensor filtering
 *   algorithms (e.g., Kalman, Madgwick) that could be used by the fusion
 *   engine or for other purposes.
 *
 * Dependencies:
 *   - log: For structured logging.
 *   - thiserror: For ergonomic error handling.
 *   - nalgebra: A popular Rust library for linear algebra, useful for handling
 *     quaternions and vectors.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// === TOC ===
// 1. Crate-level Documentation & Attributes
// 2. FFI Bindings Module (private)
// 3. Public Module Declarations
// 4. Core Public Data Structures & Types
// 5. Public Prelude
// =============

#![deny(unsafe_code)]
#![deny(missing_docs)]
#![deny(warnings)]

//! # TrackieLLM Sensors Crate
//!
//! Provides sensor data processing and fusion services.
//!
//! ## Architecture
//!
//! The `SensorFusionService` is the central component of this crate. It is
//! designed to be run by the Cortex. Other parts of the system, such as a
//! dedicated hardware interface layer, are responsible for capturing raw sensor
//! data and injecting it into the fusion service. The service then runs its
//! internal filtering and fusion algorithms to continuously update its
//! estimate of the device's state, which can be queried at any time.

// --- FFI Bindings Module ---
mod ffi {
    #![allow(non_camel_case_types, non_snake_case, dead_code)]

    use std::ffi::c_void;

    pub type tk_error_code_t = i32;
    pub const TK_SUCCESS: tk_error_code_t = 0;

    // Opaque type for the fusion engine handle
    pub enum tk_sensor_fusion_s {}
    pub type tk_sensor_fusion_t = tk_sensor_fusion_s;

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct tk_sensor_fusion_config_t {
        pub update_rate_hz: f32,
        pub gyro_trust_factor: f32,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct tk_imu_data_t {
        pub timestamp_ns: u64,
        pub acc_x: f32,
        pub acc_y: f32,
        pub acc_z: f32,
        pub gyro_x: f32,
        pub gyro_y: f32,
        pub gyro_z: f32,
        pub has_mag_data: bool,
        pub mag_x: f32,
        pub mag_y: f32,
        pub mag_z: f32,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum tk_motion_state_e {
        TK_MOTION_STATE_UNKNOWN,
        TK_MOTION_STATE_STATIONARY,
        TK_MOTION_STATE_WALKING,
        TK_MOTION_STATE_RUNNING,
        TK_MOTION_STATE_FALLING,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct tk_quaternion_t {
        pub w: f32,
        pub x: f32,
        pub y: f32,
        pub z: f32,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct tk_world_state_t {
        pub last_update_timestamp_ns: u64,
        pub orientation: tk_quaternion_t,
        pub motion_state: tk_motion_state_e,
        pub is_speech_detected: bool,
    }

    extern "C" {
        pub fn tk_sensor_fusion_create(
            out_engine: *mut *mut tk_sensor_fusion_t,
            config: *const tk_sensor_fusion_config_t,
        ) -> tk_error_code_t;

        pub fn tk_sensor_fusion_destroy(engine: *mut *mut tk_sensor_fusion_t);

        pub fn tk_sensor_fusion_inject_imu_data(
            engine: *mut tk_sensor_fusion_t,
            imu_data: *const tk_imu_data_t,
        ) -> tk_error_code_t;

        pub fn tk_sensor_fusion_inject_vad_state(engine: *mut tk_sensor_fusion_t, is_speech_active: bool);

        pub fn tk_sensor_fusion_update(engine: *mut tk_sensor_fusion_t, delta_time_s: f32) -> tk_error_code_t;

        pub fn tk_sensor_fusion_get_world_state(
            engine: *mut tk_sensor_fusion_t,
            out_state: *mut tk_world_state_t,
        ) -> tk_error_code_t;
    }
}

// --- Public Module Declarations ---

/// Provides the C-callable FFI functions for sensor processing.
pub mod ffi_bridge;
/// Provides safe wrappers for the sensor fusion engine.
pub mod sensor_fusion;
/// Contains Rust-native implementations of sensor data filters.
pub mod sensor_filters;


// --- Core Public Data Structures & Types ---

use nalgebra::{Quaternion, Vector3};
use thiserror::Error;

/// High-level classification of the user's current motion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionState {
    Unknown,
    Stationary,
    Walking,
    Running,
    Falling,
}

/// The fused, high-level output of the sensor fusion engine.
#[derive(Debug, Clone)]
pub struct WorldState {
    /// The absolute orientation of the device in space, represented as a quaternion.
    pub orientation: Quaternion<f32>,
    /// The user's classified motion state.
    pub motion_state: MotionState,
    /// The current state from the Voice Activity Detector.
    pub is_speech_detected: bool,
}

/// Represents a single, time-stamped reading from an IMU.
#[derive(Debug, Clone, Copy)]
pub struct ImuData {
    /// Accelerometer data (m/s^2).
    pub accelerometer: Vector3<f32>,
    /// Gyroscope data (radians/s).
    pub gyroscope: Vector3<f32>,
    /// Optional: Magnetometer data (microteslas).
    pub magnetometer: Option<Vector3<f32>>,
}

/// The primary error type for all operations within the sensors crate.
#[derive(Debug, Error)]
pub enum SensorsError {
    /// The sensor fusion engine has not been initialized.
    #[error("Sensor fusion engine is not initialized.")]
    NotInitialized,

    /// An FFI call to the underlying C library failed.
    #[error("Sensors FFI call failed: {0}")]
    Ffi(String),
}


// --- Public Prelude ---

pub mod prelude {
    //! A "prelude" for convenient imports of this crate's main types.
    pub use super::{
        sensor_fusion::SensorFusionService, ImuData, MotionState, SensorsError, WorldState,
    };
}
