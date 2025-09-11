/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/workers/sensor_worker.rs
 *
 * This file implements the Sensor Worker, an asynchronous task dedicated
 * to managing the sensor fusion pipeline. It acts as the bridge between
 * the C-based sensor fusion library and the rest of the Rust-based application.
 *
 * Its primary responsibilities are:
 * - Initializing the `tk_sensor_fusion` engine.
 * - Periodically calling the C function to get the latest fused sensor data.
 * - Translating the C-struct results into a safe Rust representation.
 * - Publishing the `SensorFusionResult` event onto the central event bus.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#![deny(missing_docs)]

//! The sensor worker manages the sensor fusion pipeline.

use crate::event_bus::{EventBus, TrackieEvent, SensorFusionData};
use crate::sensors::ffi;
use std::ptr;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

/// A safe wrapper around the C `tk_sensor_fusion_t` handle.
///
/// This struct ensures that the pipeline is always destroyed when it goes out
/// of scope, preventing memory leaks from the C library.
struct SensorFusionWrapper {
    handle: *mut ffi::tk_sensor_fusion_t,
}

impl SensorFusionWrapper {
    /// Creates a new sensor fusion pipeline.
    ///
    /// This function is unsafe because it calls into the C FFI.
    unsafe fn new() -> Result<Self, String> {
        let mut handle: *mut ffi::tk_sensor_fusion_t = ptr::null_mut();

        let config = ffi::tk_sensor_fusion_config_t {
            update_rate_hz: 20.0, // Corresponds to 50ms interval
            gyro_trust_factor: 0.98,
        };

        let err_code = ffi::tk_sensor_fusion_create(&mut handle, &config);
        if err_code == 0 && !handle.is_null() {
            Ok(Self { handle })
        } else {
            Err(format!("Failed to create sensor fusion engine with code {}", err_code))
        }
    }

    /// Retrieves the latest world state from the C engine.
    ///
    /// This is an unsafe FFI call.
    unsafe fn get_world_state(&self) -> Result<ffi::tk_world_state_t, String> {
        // Create a struct on the stack to be filled by the C function.
        let mut out_state: ffi::tk_world_state_t = std::mem::zeroed();

        let err_code = ffi::tk_sensor_fusion_get_world_state(self.handle, &mut out_state);

        if err_code == 0 {
            Ok(out_state)
        } else {
            Err(format!("Failed to get world state with code {}", err_code))
        }
    }
}

impl Drop for SensorFusionWrapper {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                ffi::tk_sensor_fusion_destroy(&mut self.handle);
            }
        }
    }
}


/// The main entry point for the sensor worker task.
///
/// This function will loop indefinitely, polling the sensor fusion engine
/// at a fixed interval.
pub async fn run(event_bus: Arc<EventBus>) {
    println!("[Sensor Worker] Initializing...");

    // Initialize the sensor fusion pipeline.
    let pipeline = match unsafe { SensorFusionWrapper::new() } {
        Ok(p) => {
            println!("[Sensor Worker] Pipeline initialized successfully.");
            p
        },
        Err(e) => {
            eprintln!("[Sensor Worker] Failed to initialize pipeline: {}", e);
            return; // Terminate the worker if initialization fails.
        }
    };

    let mut subscriber = event_bus.subscribe();
    println!("[Sensor Worker] Now listening for events.");

    // Main processing loop
    loop {
        tokio::select! {
            // Branch 1: Wait for the next processing tick.
            _ = sleep(Duration::from_millis(50)) => {
                let c_state = match unsafe { pipeline.get_world_state() } {
                    Ok(state) => state,
                    Err(e) => {
                        eprintln!("[Sensor Worker] Error getting world state: {}", e);
                        continue;
                    }
                };

                // Convert the C struct to the safe Rust struct.
                // The `from` implementation will be in the event_bus module.
                let sensor_data = SensorFusionData::from(c_state);

                // Publish the safe Rust struct to the event bus.
                event_bus.publish(TrackieEvent::SensorFusionResult(Arc::new(sensor_data)));
            }
            // Branch 2: Listen for a shutdown event from the bus.
            Ok(event) = subscriber.next_event() => {
                if let TrackieEvent::Shutdown = event {
                    println!("[Sensor Worker] Shutdown signal received. Terminating.");
                    break; // Exit the loop
                }
            }
        }
    }
}
