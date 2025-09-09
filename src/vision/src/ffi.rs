/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/vision/ffi.rs
 *
 * This file contains the raw FFI (Foreign Function Interface) bindings for the
 * C-based vision pipeline library (`tk_vision_pipeline.h`). These bindings
 * allow Rust code to directly call the C functions and interact with the
 * C-defined data structures.
 *
 * The definitions here are intended to be a 1-to-1 mapping of the C header.
 * - Structs are defined with `#[repr(C)]` to ensure a compatible memory layout.
 * - Opaque C types are represented as empty enums (`enum TkTypeName {}`).
 * - Function signatures are declared within an `extern "C"` block.
 *
 * Great care must be taken when using these bindings, as all calls across the
 * FFI boundary are `unsafe`.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#![allow(non_camel_case_types, non_snake_case)]

use std::os::raw::{c_char, c_float, c_int, c_uint, c_void};

// --- Opaque and Forward-Declared Types ---

/// Opaque handle to the main vision pipeline instance.
#[repr(C)]
pub struct tk_vision_pipeline_s {
    _private: [u8; 0],
}
pub type tk_vision_pipeline_t = tk_vision_pipeline_s;

/// Opaque handle to the vision result object.
#[repr(C)]
pub struct tk_vision_result_s {
    _private: [u8; 0],
}
pub type tk_vision_result_t = tk_vision_result_s;

// --- Placeholder for types from other headers ---

/// Placeholder for the video frame struct.
/// In a real implementation, this would be fully defined.
#[repr(C)]
pub struct tk_video_frame_t {
    pub width: u32,
    pub height: u32,
    pub data: *const c_void,
}

/// Placeholder for the path struct.
#[repr(C)]
pub struct tk_path_t {
    _private: [u8; 0],
}

// --- Enums and Constants ---

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum tk_vision_backend_e {
    TK_VISION_BACKEND_CPU,
    TK_VISION_BACKEND_CUDA,
    TK_VISION_BACKEND_METAL,
    TK_VISION_BACKEND_ROCM,
}

pub type tk_vision_analysis_flags_t = u32;
pub const TK_VISION_ANALYZE_OBJECT_DETECTION: u32 = 1 << 0;
pub const TK_VISION_ANALYZE_DEPTH_ESTIMATION: u32 = 1 << 1;
pub const TK_VISION_ANALYZE_OCR: u32 = 1 << 2;
pub const TK_VISION_ANALYZE_FUSION_DISTANCE: u32 = 1 << 3;

// --- C-Compatible Structs ---

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct tk_vision_pipeline_config_t {
    pub backend: tk_vision_backend_e,
    pub gpu_device_id: c_int,
    pub object_detection_model_path: *const tk_path_t,
    pub depth_estimation_model_path: *const tk_path_t,
    pub tesseract_data_path: *const tk_path_t,
    pub object_confidence_threshold: c_float,
    pub max_detected_objects: u32,
    pub focal_length_x: c_float,
    pub focal_length_y: c_float,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct tk_rect_t {
    pub x: c_int,
    pub y: c_int,
    pub w: c_int,
    pub h: c_int,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct tk_vision_object_t {
    pub class_id: u32,
    pub label: *const c_char,
    pub confidence: c_float,
    pub bbox: tk_rect_t,
    pub distance_meters: c_float,
    pub width_meters: c_float,
    pub height_meters: c_float,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct tk_vision_text_block_t {
    pub text: *mut c_char,
    pub confidence: c_float,
    pub bbox: tk_rect_t,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct tk_vision_depth_map_t {
    pub width: u32,
    pub height: u32,
    pub data: *mut c_float,
}

// We cannot define the struct directly because the C header uses a forward
// declaration. We define the fields based on the C source file.
#[repr(C)]
#[derive(Debug)]
pub struct tk_vision_result_fields {
    pub source_frame_timestamp_ns: u64,
    pub object_count: usize,
    pub objects: *mut tk_vision_object_t,
    pub text_block_count: usize,
    pub text_blocks: *mut tk_vision_text_block_t,
    pub depth_map: *mut tk_vision_depth_map_t,
}

// --- Manually defined structs from other C headers ---

/// C-compatible version of a single raw detection result from `tk_object_detector.h`.
/// This is used as input to the Rust fusion logic.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct tk_detection_result_t {
    pub class_id: u32,
    pub label: *const c_char,
    pub confidence: c_float,
    pub bbox: tk_rect_t,
}

// --- FFI Function Declarations ---

// Assuming tk_error_code_t is defined elsewhere, we'll use a placeholder.
pub type tk_error_code_t = c_int;

#[link(name = "trackie")]
extern "C" {
    pub fn tk_vision_pipeline_create(
        out_pipeline: *mut *mut tk_vision_pipeline_t,
        config: *const tk_vision_pipeline_config_t,
    ) -> tk_error_code_t;

    pub fn tk_vision_pipeline_destroy(pipeline: *mut *mut tk_vision_pipeline_t);

    pub fn tk_vision_pipeline_process_frame(
        pipeline: *mut tk_vision_pipeline_t,
        video_frame: *const tk_video_frame_t,
        analysis_flags: tk_vision_analysis_flags_t,
        timestamp_ns: u64,
        out_result: *mut *mut tk_vision_result_t,
    ) -> tk_error_code_t;

    pub fn tk_vision_result_destroy(result: *mut *mut tk_vision_result_t);
}
