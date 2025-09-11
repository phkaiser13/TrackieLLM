/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/sensors/sensor_fusion.rs
 *
 * This file contains the core Rust-native implementation of the sensor fusion
 * engine. It uses the `fusion-ahrs` crate to provide a robust Attitude and
 * Heading Reference System (AHRS) for orientation tracking, and implements a
 * custom state machine for motion classification.
 *
 * The main struct, `SensorFusionEngine`, holds the state for all algorithms
 * and is the single source of truth for the device's physical state. This
 * entire module is designed to be compiled as a static library and linked
 * into the main C/C++ application via the FFI layer in `ffi_bridge.rs`.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use fusion_ahrs::{Ahrs, AhrsSettings, Quaternion, Vector3};
use nalgebra as na;

/// Represents the high-level classification of the user's motion.
/// This enum mirrors the C-level `tk_motion_state_e`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub enum MotionState {
    Unknown,
    Stationary,
    Walking,
    Running,
    Falling,
}

/// The fused, high-level output of the sensor fusion engine.
/// This struct mirrors the C-level `tk_world_state_t`.
#[derive(Debug, Clone, Copy)]
pub struct WorldState {
    pub last_update_timestamp_ns: u64,
    pub orientation: na::Quaternion<f32>,
    pub motion_state: MotionState,
    pub is_speech_detected: bool,
}

/// The main state-bearing struct for the sensor fusion engine.
pub struct SensorFusionEngine {
    ahrs: Ahrs,
    motion_state: MotionState,
    is_speech_detected: bool,
    last_update_timestamp_ns: u64,
}

impl SensorFusionEngine {
    /// Creates a new sensor fusion engine.
    pub fn new(update_rate_hz: f32) -> Self {
        let settings = AhrsSettings::new()
            .gain(0.1) // A common starting point for the filter gain
            .sample_rate(update_rate_hz as u16);

        Self {
            ahrs: Ahrs::new_with_settings(settings),
            motion_state: MotionState::Unknown,
            is_speech_detected: false,
            last_update_timestamp_ns: 0,
        }
    }

    /// Injects a new IMU data sample into the engine.
    pub fn inject_imu_data(&mut self, acc: &na::Vector3<f32>, gyro: &na::Vector3<f32>, mag: &Option<na::Vector3<f32>>, delta_time_s: f32) {
        // The fusion-ahrs library expects gyroscope data in radians/s.
        // The tk_imu_data_t struct already provides it in this unit.
        let gyro_rad = Vector3::new(gyro.x, gyro.y, gyro.z);

        // Accelerometer data is expected in g's. The tk_imu_data_t struct provides
        // it in m/s^2. We need to convert it. 1 g ≈ 9.81 m/s^2.
        const G_FORCE: f32 = 9.81;
        let acc_g = Vector3::new(acc.x / G_FORCE, acc.y / G_FORCE, acc.z / G_FORCE);

        if let Some(mag_data) = mag {
            // Use magnetometer data if available (MARG update)
            let mag_norm = Vector3::new(mag_data.x, mag_data.y, mag_data.z);
            self.ahrs.update(&gyro_rad, &acc_g, &mag_norm).unwrap();
        } else {
            // Otherwise, perform an IMU update (without magnetometer)
            self.ahrs.update_imu(&gyro_rad, &acc_g).unwrap();
        }

        // --- Motion Classification Logic ---
        // A simple classifier based on the magnitude of linear acceleration.
        // The AHRS filter provides an estimate of gravity, which we can remove.
        let gravity = self.ahrs.get_earth_acceleration();
        let linear_acc = acc_g - gravity;
        let linear_acc_magnitude = linear_acc.norm();

        // Simple threshold-based state machine
        const STATIONARY_THRESHOLD: f32 = 0.1; // in g's
        const WALKING_THRESHOLD: f32 = 0.8;    // in g's

        self.motion_state = if linear_acc_magnitude < STATIONARY_THRESHOLD {
            MotionState::Stationary
        } else if linear_acc_magnitude < WALKING_THRESHOLD {
            MotionState::Walking
        } else {
            MotionState::Running
        };
        // Note: Fall detection is more complex and would require checking for
        // a period of near-zero g followed by a large shock. This is a placeholder.

        self.last_update_timestamp_ns = 0; // TODO: Plumb timestamp through
    }

    /// Injects the VAD state.
    pub fn inject_vad_state(&mut self, is_speech_active: bool) {
        self.is_speech_detected = is_speech_active;
    }

    /// Retrieves the current world state.
    pub fn get_world_state(&self) -> WorldState {
        let q = self.ahrs.get_quaternion();
        WorldState {
            last_update_timestamp_ns: self.last_update_timestamp_ns,
            orientation: na::Quaternion::new(q.w, q.x, q.y, q.z),
            motion_state: self.motion_state,
            is_speech_detected: self.is_speech_detected,
        }
    }
}
