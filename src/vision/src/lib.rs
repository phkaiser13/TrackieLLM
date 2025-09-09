/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/vision/lib.rs
 *
 * This file is the main library entry point for the 'vision' crate. This
 * crate provides the core computer vision capabilities for the TrackieLLM
 * application, transforming raw video frames into a structured, semantic
 * understanding of the environment.
 *
 * It serves as a high-level, safe Rust abstraction over the comprehensive C/C++
 * vision pipeline. The underlying C/C++ layer handles the complexities of
 * model loading, GPU acceleration (via CUDA), and running multiple inference
 * engines (for object detection, depth estimation, and OCR).
 *
 * The primary interface is the `VisionPipeline` struct, which manages the
 * lifecycle of the underlying `tk_vision_pipeline_t` and provides a safe
 * method for processing video frames.
 *
 * Dependencies:
 *   - internal_tools: For safe path handling.
 *   - log: For structured logging.
 *   - thiserror: For ergonomic error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// === TOC ===
// 1. Crate-level Documentation & Attributes
// 2. FFI Bindings Module
// 3. Public Module Declarations
// 4. Core Public Data Structures & Types
// 5. Public Prelude
// =============

#![deny(unsafe_code)]
#![deny(missing_docs)]
#![deny(warnings)]

//! # TrackieLLM Vision Crate
//!
//! Provides a comprehensive pipeline for real-time computer vision analysis.
//!
//! ## Core Features
//!
//! - **Object Detection**: Identifies and locates objects in a scene.
//! - **Depth Estimation**: Calculates a dense depth map of the environment.
//! - **Text Recognition (OCR)**: Extracts text from images.
//! - **Sensor Fusion**: Combines the above outputs to produce a rich,
//!   contextual understanding (e.g., attaching distances to objects).

// --- FFI Bindings Module ---
// Contains the raw FFI declarations for the vision C/C++ API.
// The implementation is in the `ffi.rs` file.
pub mod ffi;

// --- Public Module Declarations ---

/// Provides a safe wrapper for depth estimation logic.
pub mod depth_processing;

/// Provides a safe wrapper for object detection and analysis.
pub mod object_analysis;


// --- Core Public Data Structures & Types ---

use thiserror::Error;

/// Represents a rectangular region in pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundingBox {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

/// Represents a single detected object.
#[derive(Debug, Clone)]
pub struct DetectedObject {
    pub label: String,
    pub confidence: f32,
    pub bounding_box: BoundingBox,
    pub distance_meters: f32,
}

/// The complete, structured result of a vision pipeline run on a single frame.
#[derive(Debug, Clone)]
pub struct VisionResult {
    pub timestamp_ns: u64,
    pub objects: Vec<DetectedObject>,
    // In a full implementation, this would also include text blocks and the depth map.
}

/// The primary error type for all operations within the vision crate.
#[derive(Debug, Error)]
pub enum VisionError {
    /// The vision pipeline has not been initialized.
    #[error("Vision pipeline is not initialized.")]
    NotInitialized,

    /// An FFI call to the underlying C/C++ library failed.
    #[error("Vision FFI call failed: {0}")]
    Ffi(String),

    /// The model specified for a vision task could not be loaded.
    #[error("Failed to load vision model: {0}")]
    ModelLoadFailed(String),

    /// An error occurred during the inference process.
    #[error("Inference failed: {0}")]
    InferenceFailed(String),
}


// --- Public Prelude ---

pub mod prelude {
    //! A "prelude" for convenient imports of this crate's main types.
    pub use super::{
        depth_processing::DepthEstimator, object_analysis::ObjectDetector, BoundingBox,
        DetectedObject, VisionError, VisionPipeline, VisionResult,
    };
}


// --- Main Service Interface ---

/// The main Vision Pipeline service.
///
/// This struct is the primary public interface to the application's vision capabilities.
/// It wraps the C-level `tk_vision_pipeline_t` and ensures its lifecycle is
/// managed safely via the `Drop` trait.
pub struct VisionPipeline {
    /// A handle to the underlying C object.
    pipeline_handle: *mut ffi::tk_vision_pipeline_t,
}

impl VisionPipeline {
    // A full implementation would have a `new` method that takes a safe
    // `VisionConfig` struct, converts it to the C `tk_vision_pipeline_config_t`,
    // and calls `ffi::tk_vision_pipeline_create`.
}

impl Drop for VisionPipeline {
    /// Ensures the C-level vision pipeline is always destroyed.
    fn drop(&mut self) {
        if !self.pipeline_handle.is_null() {
            unsafe { ffi::tk_vision_pipeline_destroy(&mut self.pipeline_handle) };
        }
    }
}


//------------------------------------------------------------------------------
// FFI Interface for C -> Rust calls
//------------------------------------------------------------------------------

use crate::object_analysis::EnrichedObject;
use std::os::raw::c_void;
use std::slice;

/// The C-compatible struct that is returned to the C layer.
/// It contains a pointer to the fused object data and the count.
#[repr(C)]
pub struct CFusedResult {
    pub objects: *const EnrichedObject,
    pub count: usize,
}

/// Fuses object detections with a depth map. Called from C.
///
/// This function takes raw pointers from the C layer, converts them into safe
/// Rust slices, calls the safe Rust fusion logic, and then prepares the
/// result to be sent back to the C layer.
///
/// # Safety
/// The caller MUST ensure that the pointers `detections_ptr` and `depth_map_ptr`
/// are valid and point to the described data structures, and that `detection_count`
/// is the correct number of elements in the `detections_ptr` array. The caller
/// is also responsible for eventually calling `tk_vision_rust_free_fused_result`
/// on the returned pointer to prevent a memory leak.
#[no_mangle]
pub unsafe extern "C" fn tk_vision_rust_fuse_data(
    detections_ptr: *const ffi::tk_detection_result_t,
    detection_count: usize,
    depth_map_ptr: *const ffi::tk_vision_depth_map_t,
    frame_width: u32,
    frame_height: u32,
    focal_length_x: f32,
    focal_length_y: f32,
) -> *mut CFusedResult {
    // 1. Convert raw C pointers to safe Rust slices and references
    if detections_ptr.is_null() || depth_map_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let detections = slice::from_raw_parts(detections_ptr, detection_count);
    let depth_map = &*depth_map_ptr;

    // 2. Call the safe, core logic
    let enriched_objects_vec = object_analysis::fuse_object_and_depth_data(
        detections,
        depth_map,
        frame_width,
        frame_height,
        focal_length_x,
        focal_length_y
    );

    // 3. Prepare the data to be returned to C
    let count = enriched_objects_vec.len();
    // Prevent the vector's memory from being freed by Rust when it goes out of scope
    let objects_ptr = enriched_objects_vec.leak().as_ptr();

    let result = CFusedResult {
        objects: objects_ptr,
        count,
    };

    // Allocate memory for the result struct and return a pointer to it
    Box::into_raw(Box::new(result))
}

/// Frees the memory allocated by `tk_vision_rust_fuse_data`.
///
/// # Safety
/// This function should only be called with a pointer that was previously
/// returned by `tk_vision_rust_fuse_data`. Calling it with any other pointer
/// will lead to undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn tk_vision_rust_free_fused_result(result_ptr: *mut CFusedResult) {
    if result_ptr.is_null() {
        return;
    }
    // Re-take ownership of the Box to allow it to be dropped, freeing the result struct
    let result = Box::from_raw(result_ptr);

    // Re-take ownership of the Vec to allow it to be dropped, freeing the object array
    let _ = Vec::from_raw_parts(
        result.objects as *mut EnrichedObject,
        result.count,
        result.count,
    );
}
