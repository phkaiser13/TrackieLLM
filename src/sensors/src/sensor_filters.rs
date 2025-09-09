/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/sensors/sensor_filters.rs
 *
 * This file contains pure Rust implementations of common sensor data filters.
 * These filters are essential for processing raw, noisy sensor data to produce
 * stable and reliable state estimations.
 *
 * This module is designed to be self-contained and does not depend on any C
 * libraries. The implemented filters can be used by the `sensor_fusion` module
 * or any other part of the application that needs to process sensor data.
 *
 * The initial implementation includes:
 * - A `Filter` trait to define a common interface for filters.
 * - A `ComplementaryFilter` for fusing accelerometer and gyroscope data to
 *   estimate orientation. This filter is computationally cheap and provides
 *   good results for many applications.
 *
 * Future additions could include more advanced filters like the Kalman Filter
 * or Madgwick filter.
 *
 * Dependencies:
 *   - nalgebra: For vector and quaternion mathematics.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use nalgebra::{Quaternion, UnitQuaternion, Vector3};

/// A trait for sensor data filters.
///
/// This defines a common interface for filters that take sensor readings
/// and update an internal state.
pub trait Filter {
    /// The output type of the filter (e.g., a Quaternion for orientation).
    type Output;

    /// Updates the filter's state with new sensor readings.
    ///
    /// # Arguments
    /// * `acc` - Accelerometer reading (in m/s^2).
    /// * `gyro` - Gyroscope reading (in radians/s).
    /// * `dt` - The time delta in seconds since the last update.
    fn update(&mut self, acc: Vector3<f32>, gyro: Vector3<f32>, dt: f32);

    /// Returns the current output of the filter.
    fn output(&self) -> &Self::Output;
}

/// A complementary filter for 6-DOF IMU orientation estimation.
///
/// This filter combines the high-frequency orientation data from the gyroscope
/// (which is accurate in the short-term but drifts over time) with the
/// low-frequency orientation data from the accelerometer (which is noisy in the
/// short-term but provides a stable long-term reference to gravity).
///
/// It "complements" the two data sources by high-pass filtering the gyroscope
/// and low-pass filtering the accelerometer.
pub struct ComplementaryFilter {
    /// The current orientation estimate as a quaternion.
    orientation: UnitQuaternion<f32>,
    /// The filter's gain parameter (alpha). A higher value trusts the
    /// accelerometer more, while a lower value trusts the gyroscope more.
    alpha: f32,
}

impl ComplementaryFilter {
    /// Creates a new `ComplementaryFilter`.
    ///
    /// # Arguments
    /// * `alpha` - The filter gain, typically between 0.0 and 1.0. A common
    ///   value is 0.98, meaning the final orientation is 98% from the integrated
    ///   gyroscope and 2% from the accelerometer.
    pub fn new(alpha: f32) -> Self {
        Self {
            orientation: UnitQuaternion::identity(),
            alpha,
        }
    }
}

impl Filter for ComplementaryFilter {
    type Output = UnitQuaternion<f32>;

    fn update(&mut self, acc: Vector3<f32>, gyro: Vector3<f32>, dt: f32) {
        // --- Gyroscope Integration ---
        // Integrate the gyroscope reading to get the change in orientation.
        // This is done by converting the angular velocity vector into a small
        // rotation quaternion.
        let gyro_quat = Quaternion::new(0.0, gyro.x * dt, gyro.y * dt, gyro.z * dt);
        let estimated_orientation = self.orientation * gyro_quat;

        // --- Accelerometer Correction ---
        // Use the accelerometer to get an absolute orientation reference based on gravity.
        // This is only valid when the device is not undergoing linear acceleration.
        let acc_norm = acc.normalize();
        // The accelerometer measures the direction of gravity. We can create a
        // quaternion that represents the rotation from the "up" vector (0, 0, 1)
        // to the measured acceleration vector.
        let acc_orientation = UnitQuaternion::from_vectors(&Vector3::z(), &acc_norm)
            .unwrap_or_else(UnitQuaternion::identity);
        
        // --- Fusion ---
        // Combine the gyroscope-based estimate and the accelerometer-based
        // reference using spherical linear interpolation (slerp). The `alpha`
        // parameter controls the weight of each source.
        self.orientation = estimated_orientation.slerp(&acc_orientation, 1.0 - self.alpha);
        self.orientation.normalize_mut();
    }

    fn output(&self) -> &Self::Output {
        &self.orientation
    }
}

/// A simple first-order IIR (Infinite Impulse Response) low-pass filter.
///
/// This filter smooths a signal by taking a weighted average of the current
/// input and the previous filtered output. It's computationally cheap and
/// effective at reducing high-frequency noise.
///
/// The filter equation is:
///   y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
/// where:
///   y[n] is the current filtered output.
///   x[n] is the current raw input.
///   y[n-1] is the previous filtered output.
///   alpha is the smoothing factor (0.0 < alpha <= 1.0).
pub struct LowPassFilter {
    /// The previous output of the filter.
    prev_output: f32,
    /// The smoothing factor (alpha). A smaller alpha results in more smoothing.
    alpha: f32,
}

impl LowPassFilter {
    /// Creates a new `LowPassFilter`.
    ///
    /// # Arguments
    /// * `alpha` - The smoothing factor. Must be between 0.0 and 1.0.
    /// * `initial_value` - The initial value to prime the filter with.
    pub fn new(alpha: f32, initial_value: f32) -> Self {
        Self {
            prev_output: initial_value,
            alpha: alpha.clamp(0.0, 1.0),
        }
    }

    /// Applies the filter to a new input value.
    ///
    /// # Arguments
    /// * `input` - The new raw value to filter.
    ///
    /// # Returns
    /// The new filtered value.
    pub fn apply(&mut self, input: f32) -> f32 {
        let output = self.alpha * input + (1.0 - self.alpha) * self.prev_output;
        self.prev_output = output;
        output
    }
}


/// A placeholder for a future Kalman Filter implementation.
///
/// A Kalman Filter is a more advanced and mathematically complex filter that
/// can provide more accurate results by modeling the system's state and
/// measurement uncertainty. Implementing it is a significant undertaking.
pub struct KalmanFilter {
    // In a real implementation, this would contain state vectors, covariance
    // matrices, and system models.
}

impl KalmanFilter {
    /// Creates a new `KalmanFilter`.
    #[allow(dead_code)]
    pub fn new() -> Self {
        // Placeholder
        Self {}
    }
}
