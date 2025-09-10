/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/sensors/sensor_fusion.rs
 *
 * This file provides a safe Rust wrapper for the Sensor Fusion engine defined
 * in `tk_sensors_fusion.h`. The `SensorFusionService` is the high-level
 * interface that the Cortex interacts with to get a simple, coherent view of
 * the device's physical state.
 *
 * This module encapsulates all the `unsafe` FFI calls required to interact
 * with the C-based fusion engine. It manages the lifecycle of the
 * `tk_sensor_fusion_t` handle using the RAII pattern and translates between
 * rich Rust types (like `nalgebra::Quaternion`) and the plain C structs used
 * by the FFI.
 *
 * Dependencies:
 *   - crate::ffi: For the raw C function bindings.
 *   - crate::{ImuData, WorldState, SensorsError}: For shared types and errors.
 *   - thiserror: For ergonomic error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use super::{ffi, ImuData, MotionState, SensorsError, WorldState};
use nalgebra::Quaternion;
use std::ptr::null_mut;
use thiserror::Error;

/// Represents errors specific to the sensor fusion process.
#[derive(Debug, Error)]
pub enum FusionError {
    /// The underlying C-level context is not initialized.
    #[error("Sensor fusion context is not initialized.")]
    NotInitialized,

    /// An FFI call to the sensor fusion C library failed.
    #[error("Sensor fusion FFI call failed with code: {0}")]
    Ffi(i32),
}

/// Configuration for the `SensorFusionService`.
pub struct SensorFusionConfig {
    /// The target frequency in Hz for sensor processing.
    pub update_rate_hz: f32,
    /// A tuning parameter for the orientation filter.
    pub gyro_trust_factor: f32,
}

/// A safe, high-level interface to the Sensor Fusion Engine.
pub struct SensorFusionService {
    /// The handle to the underlying `tk_sensor_fusion_t` C object.
    engine_handle: *mut ffi::tk_sensor_fusion_t,
}

impl SensorFusionService {
    /// Creates a new `SensorFusionService`.
    pub fn new(config: SensorFusionConfig) -> Result<Self, SensorsError> {
        let mut engine_handle = null_mut();
        let c_config = ffi::tk_sensor_fusion_config_t {
            update_rate_hz: config.update_rate_hz,
            gyro_trust_factor: config.gyro_trust_factor,
        };

        let result = unsafe { ffi::tk_sensor_fusion_create(&mut engine_handle, &c_config) };

        if result != ffi::TK_SUCCESS {
            return Err(SensorsError::Ffi(format!(
                "Failed to create sensor fusion engine: {}",
                result
            )));
        }
        if engine_handle.is_null() {
            return Err(SensorsError::NotInitialized);
        }

        Ok(Self { engine_handle })
    }

    /// Injects a new IMU data sample into the engine.
    pub fn inject_imu_data(&mut self, data: &ImuData) -> Result<(), SensorsError> {
        let c_imu_data = ffi::tk_imu_data_t {
            timestamp_ns: 0, // Placeholder, timestamp is not used in the C code yet
            acc_x: data.accelerometer.x,
            acc_y: data.accelerometer.y,
            acc_z: data.accelerometer.z,
            gyro_x: data.gyroscope.x,
            gyro_y: data.gyroscope.y,
            gyro_z: data.gyroscope.z,
            has_mag_data: data.magnetometer.is_some(),
            mag_x: data.magnetometer.map_or(0.0, |m| m.x),
            mag_y: data.magnetometer.map_or(0.0, |m| m.y),
            mag_z: data.magnetometer.map_or(0.0, |m| m.z),
        };

        let result = unsafe { ffi::tk_sensor_fusion_inject_imu_data(self.engine_handle, &c_imu_data) };
        if result != ffi::TK_SUCCESS {
            return Err(SensorsError::Ffi(format!(
                "Failed to inject IMU data: {}",
                result
            )));
        }
        Ok(())
    }

    /// Injects the VAD (Voice Activity Detection) state.
    pub fn inject_vad_state(&mut self, is_speech_active: bool) {
        unsafe { ffi::tk_sensor_fusion_inject_vad_state(self.engine_handle, is_speech_active) };
    }

    /// Processes all injected data and updates the internal world state.
    pub fn update(&mut self, delta_time_s: f32) -> Result<(), SensorsError> {
        let result = unsafe { ffi::tk_sensor_fusion_update(self.engine_handle, delta_time_s) };
        if result != ffi::TK_SUCCESS {
            return Err(SensorsError::Ffi(format!(
                "Failed to update sensor fusion state: {}",
                result
            )));
        }
        Ok(())
    }

    /// Retrieves the latest, most up-to-date world state.
    pub fn get_world_state(&self) -> Result<WorldState, SensorsError> {
        let mut c_world_state: ffi::tk_world_state_t = unsafe { std::mem::zeroed() };
        let result = unsafe { ffi::tk_sensor_fusion_get_world_state(self.engine_handle, &mut c_world_state) };

        if result != ffi::TK_SUCCESS {
            return Err(SensorsError::Ffi(format!(
                "Failed to get world state: {}",
                result
            )));
        }

        let orientation = Quaternion::new(
            c_world_state.orientation.w,
            c_world_state.orientation.x,
            c_world_state.orientation.y,
            c_world_state.orientation.z,
        );

        let motion_state = match c_world_state.motion_state {
            ffi::tk_motion_state_e::TK_MOTION_STATE_UNKNOWN => MotionState::Unknown,
            ffi::tk_motion_state_e::TK_MOTION_STATE_STATIONARY => MotionState::Stationary,
            ffi::tk_motion_state_e::TK_MOTION_STATE_WALKING => MotionState::Walking,
            ffi::tk_motion_state_e::TK_MOTION_STATE_RUNNING => MotionState::Running,
            ffi::tk_motion_state_e::TK_MOTION_STATE_FALLING => MotionState::Falling,
        };

        Ok(WorldState {
            orientation,
            motion_state,
            is_speech_detected: c_world_state.is_speech_detected,
        })
    }
}

impl Drop for SensorFusionService {
    /// Ensures the C-level sensor fusion engine is always destroyed.
    fn drop(&mut self) {
        if !self.engine_handle.is_null() {
            unsafe { ffi::tk_sensor_fusion_destroy(&mut self.engine_handle) };
        }
    }
}
