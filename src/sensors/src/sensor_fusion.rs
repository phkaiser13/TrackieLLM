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
    #[error("Sensor fusion FFI call failed: {0}")]
    Ffi(String),
}

/// A safe, high-level interface to the Sensor Fusion Engine.
pub struct SensorFusionService {
    /// The handle to the underlying `tk_sensor_fusion_t` C object.
    #[allow(dead_code)]
    engine_handle: *mut ffi::tk_sensor_fusion_t,
}

impl SensorFusionService {
    /// Creates a new `SensorFusionService`.
    ///
    /// This is a simplified constructor. A real implementation would take a
    /// configuration struct and call `tk_sensor_fusion_create`.
    pub fn new() -> Result<Self, SensorsError> {
        // Placeholder for creating the fusion engine.
        Ok(Self {
            engine_handle: null_mut(),
        })
    }

    /// Injects a new IMU data sample into the engine.
    #[allow(dead_code, unused_variables)]
    pub fn inject_imu_data(&mut self, data: &ImuData) -> Result<(), SensorsError> {
        // Mock Implementation:
        // 1. Convert the Rust `ImuData` struct into a C `tk_imu_data_t`.
        // 2. Make the unsafe FFI call to `tk_sensor_fusion_inject_imu_data`.
        // 3. Check the return code.
        log::trace!("Injecting IMU data into fusion engine.");
        Ok(())
    }

    /// Processes all injected data and updates the internal world state.
    #[allow(dead_code, unused_variables)]
    pub fn update(&mut self, delta_time_s: f32) -> Result<(), SensorsError> {
        // Mock Implementation:
        // 1. Call `tk_sensor_fusion_update`.
        // 2. Check the return code.
        log::trace!("Updating sensor fusion state with dt: {}", delta_time_s);
        Ok(())
    }

    /// Retrieves the latest, most up-to-date world state.
    #[allow(dead_code)]
    pub fn get_world_state(&self) -> Result<WorldState, SensorsError> {
        // Mock Implementation:
        // 1. Create a `tk_world_state_t` on the stack.
        // 2. Call `tk_sensor_fusion_get_world_state`.
        // 3. Check the return code.
        // 4. Convert the C struct into the safe Rust `WorldState` struct.
        
        log::debug!("Retrieving world state from fusion engine.");
        let mock_state = WorldState {
            orientation: Quaternion::new(1.0, 0.0, 0.0, 0.0), // Identity quaternion
            motion_state: MotionState::Stationary,
            is_speech_detected: false,
        };
        Ok(mock_state)
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

impl Default for SensorFusionService {
    fn default() -> Self {
        Self::new().expect("Failed to create default SensorFusionService")
    }
}
