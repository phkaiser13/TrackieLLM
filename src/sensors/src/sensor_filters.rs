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
 * The main implementation is the `KalmanFilter`, which provides a sophisticated
 * approach to fusing accelerometer and gyroscope data for orientation tracking.
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


/// An Attitude and Heading Reference System (AHRS) using a Kalman filter.
///
/// This filter provides a more sophisticated approach than the complementary
/// filter by explicitly modeling the system's state and uncertainty. It aims
/// to estimate the orientation (as a quaternion) while correcting for gyroscope
/// bias.
///
/// The implementation is a simplified Kalman filter for educational purposes
/// and may require tuning for a specific IMU sensor.
pub struct KalmanFilter {
    /// The state vector [q_w, q_x, q_y, q_z, bias_x, bias_y, bias_z].
    x: nalgebra::OVector<f32, nalgebra::U7>,
    /// The state covariance matrix.
    p: nalgebra::OMatrix<f32, nalgebra::U7, nalgebra::U7>,
    /// The process noise covariance matrix.
    q: nalgebra::OMatrix<f32, nalgebra::U7, nalgebra::U7>,
    /// The measurement noise covariance matrix.
    r: nalgebra::OMatrix<f32, nalgebra::U3, nalgebra::U3>,
    /// The current orientation estimate.
    orientation: UnitQuaternion<f32>,
}

impl KalmanFilter {
    /// Creates a new `KalmanFilter` with default parameters.
    pub fn new(process_noise: f32, measurement_noise: f32) -> Self {
        let mut x = nalgebra::OVector::<f32, nalgebra::U7>::zeros();
        x[0] = 1.0; // Initial orientation is an identity quaternion

        let p = nalgebra::OMatrix::<f32, nalgebra::U7, nalgebra::U7>::identity() * 1.0;
        let q = nalgebra::OMatrix::<f32, nalgebra::U7, nalgebra::U7>::identity() * process_noise;
        let r = nalgebra::OMatrix::<f32, nalgebra::U3, nalgebra::U3>::identity() * measurement_noise;

        Self {
            x,
            p,
            q,
            r,
            orientation: UnitQuaternion::identity(),
        }
    }
}

impl Filter for KalmanFilter {
    type Output = UnitQuaternion<f32>;

    fn update(&mut self, acc: Vector3<f32>, gyro: Vector3<f32>, dt: f32) {
        // State and current orientation
        let mut state = self.x.clone_owned();
        let q_est = UnitQuaternion::new_normalize(Quaternion::new(state[0], state[1], state[2], state[3]));

        // Gyro bias
        let bias = Vector3::new(state[4], state[5], state[6]);
        let gyro_corrected = gyro - bias;

        // --- PREDICT ---
        let (f, b) = Self::jacobian(&q_est, &gyro_corrected, dt);

        let gyro_quat = Quaternion::from_parts(0.0, gyro_corrected * dt);
        let q_pred = q_est * gyro_quat;

        state[0] = q_pred.w;
        state[1] = q_pred.i;
        state[2] = q_pred.j;
        state[3] = q_pred.k;

        self.p = &f * &self.p * f.transpose() + &b * &self.q * b.transpose();

        // --- UPDATE ---
        let h = Self::measurement_jacobian(&q_pred);

        let z_pred = UnitQuaternion::new_normalize(q_pred).to_rotation_matrix().transform_vector(&Vector3::z());
        let y = acc.normalize() - z_pred; // Innovation

        let s = &h * &self.p * h.transpose() + &self.r;
        let k = self.p.clone_owned() * h.transpose() * s.try_inverse().unwrap_or_else(nalgebra::OMatrix::identity);

        let delta_x = k * y;
        state += delta_x;
        self.p = (nalgebra::OMatrix::identity() - k * h) * self.p.clone_owned();

        self.x = state;
        self.x.fixed_rows_mut::<nalgebra::U4>(0).normalize_mut();

        // Update the stored orientation
        self.orientation = UnitQuaternion::new_normalize(Quaternion::new(
            self.x[0], self.x[1], self.x[2], self.x[3]
        ));
    }

    fn output(&self) -> &Self::Output {
        &self.orientation
    }
}

impl KalmanFilter {
    /// Computes the state transition Jacobian (F) and process noise Jacobian (B).
    fn jacobian(q: &UnitQuaternion<f32>, gyro: &Vector3<f32>, dt: f32) -> (nalgebra::OMatrix<f32, nalgebra::U7, nalgebra::U7>, nalgebra::OMatrix<f32, nalgebra::U7, nalgebra::U7>) {
        let mut f = nalgebra::OMatrix::<f32, nalgebra::U7, nalgebra::U7>::identity();
        let mut b = nalgebra::OMatrix::<f32, nalgebra::U7, nalgebra::U7>::zeros();

        let q_w = q.w;
        let q_x = q.i;
        let q_y = q.j;
        let q_z = q.k;

        let sk_gyro = nalgebra::Matrix3::new(
            0.0, -gyro.z, gyro.y,
            gyro.z, 0.0, -gyro.x,
            -gyro.y, gyro.x, 0.0,
        );

        let mut a_qq = nalgebra::Matrix4::zeros();
        a_qq.fixed_slice_mut::<3, 3>(1, 1).copy_from(&sk_gyro);
        a_qq.fixed_slice_mut::<3, 1>(1, 0).copy_from(&gyro);
        a_qq.fixed_slice_mut::<1, 3>(0, 1).copy_from(&(-gyro.transpose()));

        f.fixed_slice_mut::<4, 4>(0, 0).copy_from(&(nalgebra::Matrix4::identity() + 0.5 * dt * a_qq));

        let mut a_qb = nalgebra::Matrix4x3::zeros();
        a_qb[(0, 0)] = q_x; a_qb[(0, 1)] = q_y; a_qb[(0, 2)] = q_z;
        a_qb[(1, 0)] = -q_w; a_qb[(1, 1)] = q_z; a_qb[(1, 2)] = -q_y;
        a_qb[(2, 0)] = -q_z; a_qb[(2, 1)] = -q_w; a_qb[(2, 2)] = q_x;
        a_qb[(3, 0)] = q_y; a_qb[(3, 1)] = -q_x; a_qb[(3, 2)] = -q_w;
        f.fixed_slice_mut::<4, 3>(0, 4).copy_from(&(-0.5 * dt * a_qb));

        b.fixed_slice_mut::<4, 4>(0, 0).copy_from(&(0.5 * dt * nalgebra::Matrix4::identity()));
        b.fixed_slice_mut::<3, 3>(4, 4).copy_from(&(dt * nalgebra::Matrix3::identity()));

        (f, b)
    }

    /// Computes the measurement Jacobian (H).
    fn measurement_jacobian(q: &UnitQuaternion<f32>) -> nalgebra::OMatrix<f32, nalgebra::U3, nalgebra::U7> {
        let mut h = nalgebra::OMatrix::<f32, nalgebra::U3, nalgebra::U7>::zeros();
        let q_w = q.w;
        let q_x = q.i;
        let q_y = q.j;
        let q_z = q.k;

        h[(0, 0)] = -2.0 * q_y; h[(0, 1)] = 2.0 * q_z; h[(0, 2)] = -2.0 * q_w; h[(0, 3)] = 2.0 * q_x;
        h[(1, 0)] = 2.0 * q_x; h[(1, 1)] = 2.0 * q_w; h[(1, 2)] = 2.0 * q_z; h[(1, 3)] = 2.0 * q_y;
        h[(2, 0)] = 2.0 * q_w; h[(2, 1)] = -2.0 * q_x; h[(2, 2)] = -2.0 * q_y; h[(2, 3)] = 2.0 * q_z;

        h
    }
}
