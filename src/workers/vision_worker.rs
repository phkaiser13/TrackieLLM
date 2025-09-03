/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/workers/vision_worker.rs
 *
 * This file implements the Vision Worker, an asynchronous task dedicated
 * to managing the vision processing pipeline. It acts as the bridge between
 * the C-based vision library and the rest of the Rust-based application.
 *
 * Its primary responsibilities are:
 * - Initializing the `tk_vision_pipeline`.
 * - Continuously capturing frames from a (currently mocked) camera source.
 * - Invoking the FFI function `tk_vision_pipeline_process_frame` to analyze images.
 * - Translating the C-struct results into a safe Rust representation.
 * - Publishing the `VisionResult` event onto the central event bus.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#![deny(missing_docs)]

//! The vision worker manages the image processing pipeline.

use crate::event_bus::{DetectedObject, EventBus, TrackieEvent, VisionData};
use crate::vision::ffi;
use std::ffi::CStr;
use std::ptr;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::time::sleep;

/// A safe wrapper around the C `tk_vision_pipeline_t` handle.
///
/// This struct ensures that the pipeline is always destroyed when it goes out
/// of scope, preventing memory leaks from the C library.
struct VisionPipelineWrapper {
    handle: *mut ffi::tk_vision_pipeline_t,
}

impl VisionPipelineWrapper {
    /// Creates a new vision pipeline.
    ///
    /// This function is unsafe because it calls into the C FFI.
    unsafe fn new() -> Result<Self, String> {
        let mut handle: *mut ffi::tk_vision_pipeline_t = ptr::null_mut();

        // For this worker, we create a mock configuration. A real application
        // would load this from a file.
        let config = ffi::tk_vision_pipeline_config_t {
            backend: ffi::tk_vision_backend_e::TK_VISION_BACKEND_CPU,
            gpu_device_id: 0,
            object_detection_model_path: ptr::null(), // Mocked
            depth_estimation_model_path: ptr::null(), // Mocked
            tesseract_data_path: ptr::null(),         // Mocked
            object_confidence_threshold: 0.5,
            max_detected_objects: 10,
            focal_length_x: 500.0,
            focal_length_y: 500.0,
        };

        let err_code = ffi::tk_vision_pipeline_create(&mut handle, &config);
        if err_code == 0 && !handle.is_null() {
            Ok(Self { handle })
        } else {
            Err(format!("Failed to create vision pipeline with code {}", err_code))
        }
    }

    /// Processes a single frame.
    ///
    /// This is an unsafe FFI call.
    unsafe fn process_frame(&self) -> *mut ffi::tk_vision_result_t {
        let mut result: *mut ffi::tk_vision_result_t = ptr::null_mut();
        
        // Mock a 640x480 video frame.
        let frame = ffi::tk_video_frame_t {
            width: 640,
            height: 480,
            data: ptr::null(), // Data is not used by the mock C function
        };

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        ffi::tk_vision_pipeline_process_frame(
            self.handle,
            &frame,
            ffi::TK_VISION_ANALYZE_OBJECT_DETECTION,
            timestamp,
            &mut result,
        );
        
        result
    }
}

impl Drop for VisionPipelineWrapper {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                ffi::tk_vision_pipeline_destroy(&mut self.handle);
            }
        }
    }
}

/// The main entry point for the vision worker task.
///
/// This function will loop indefinitely, simulating a camera feed and
/// processing frames at a fixed interval.
pub async fn run(event_bus: Arc<EventBus>) {
    println!("[Vision Worker] Initializing...");

    // Initialize the vision pipeline.
    let pipeline = match unsafe { VisionPipelineWrapper::new() } {
        Ok(p) => {
            println!("[Vision Worker] Pipeline initialized successfully.");
            p
        },
        Err(e) => {
            eprintln!("[Vision Worker] Failed to initialize pipeline: {}", e);
            return; // Terminate the worker if initialization fails.
        }
    };

    // Main processing loop
    loop {
        // Simulate a 10 FPS camera.
        sleep(Duration::from_millis(100)).await;

        let result_ptr = unsafe { pipeline.process_frame() };

        if result_ptr.is_null() {
            // The C function might return null on error.
            continue;
        }
        
        // Unpack the C result into a safe Rust struct.
        // This is highly unsafe and requires careful handling of pointers.
        let vision_data = unsafe {
            let c_result = &*(result_ptr as *const ffi::tk_vision_result_fields);
            
            let mut objects = Vec::new();
            if c_result.object_count > 0 && !c_result.objects.is_null() {
                let c_objects = std::slice::from_raw_parts(c_result.objects, c_result.object_count);
                for c_obj in c_objects {
                    objects.push(DetectedObject {
                        label: CStr::from_ptr(c_obj.label).to_string_lossy().into_owned(),
                        confidence: c_obj.confidence,
                        distance: c_obj.distance_meters,
                    });
                }
            }
            
            VisionData {
                objects,
                timestamp_ns: c_result.source_frame_timestamp_ns,
            }
        };

        // Free the C-allocated result memory *after* we are done with it.
        let mut result_ptr_mut = result_ptr;
        unsafe { ffi::tk_vision_result_destroy(&mut result_ptr_mut) };

        // Publish the safe Rust struct to the event bus.
        event_bus.publish(TrackieEvent::VisionResult(Arc::new(vision_data)));
    }
}
