/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/cortex/ffi.rs
 *
 * FFI bindings for `tk_cortex_main.h` and related C headers.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#![allow(non_camel_case_types, non_snake_case)]
#![deny(missing_docs)]

//! FFI bindings for the C-Cortex.

use crate::sensors::ffi::tk_world_state_t;
use std::os::raw::{c_char, c_int, c_void, c_ushort};

// Opaque handle for the C-Cortex instance.
pub enum tk_cortex_s {}
pub type tk_cortex_t = tk_cortex_s;

pub type tk_error_code_t = c_int;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum tk_system_state_e {
    TK_STATE_OFF,
    TK_STATE_INITIALIZING,
    TK_STATE_IDLE,
    TK_STATE_LISTENING,
    TK_STATE_PROCESSING,
    TK_STATE_RESPONDING,
    TK_STATE_SHUTDOWN,
    TK_STATE_FATAL_ERROR,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tk_cortex_callbacks_t {
    pub on_state_change: Option<unsafe extern "C" fn(new_state: tk_system_state_e, user_data: *mut c_void)>,
    pub on_tts_audio_ready: Option<unsafe extern "C" fn(audio_data: *const i16, frame_count: usize, sample_rate: u32, user_data: *mut c_void)>,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tk_model_paths_t {
    pub llm_model: *const c_char,
    pub object_detection_model: *const c_char,
    pub depth_estimation_model: *const c_char,
    pub asr_model: *const c_char,
    pub tts_model_dir: *const c_char,
    pub vad_model: *const c_char,
    pub tesseract_data_dir: *const c_char,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tk_cortex_config_t {
    pub model_paths: tk_model_paths_t,
    pub gpu_device_id: c_int,
    pub main_loop_frequency_hz: f32,
    pub user_language: *const c_char,
    pub user_data: *mut c_void,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tk_video_frame_t {
    pub width: c_ushort,
    pub height: c_ushort,
    pub stride: c_ushort,
    pub format: c_int,
    pub data: *const u8,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tk_sensor_event_t {
    pub world_state: tk_world_state_t,
}


extern "C" {
    pub fn tk_cortex_create(
        out_cortex: *mut *mut tk_cortex_t,
        config: *const tk_cortex_config_t,
        callbacks: tk_cortex_callbacks_t,
    ) -> tk_error_code_t;

    pub fn tk_cortex_destroy(cortex: *mut *mut tk_cortex_t);

    pub fn tk_cortex_run(cortex: *mut tk_cortex_t) -> tk_error_code_t;
    pub fn tk_cortex_stop(cortex: *mut tk_cortex_t) -> tk_error_code_t;

    pub fn tk_cortex_inject_video_frame(cortex: *mut tk_cortex_t, frame: *const tk_video_frame_t) -> tk_error_code_t;
    pub fn tk_cortex_inject_audio_frame(cortex: *mut tk_cortex_t, audio_data: *const i16, frame_count: usize) -> tk_error_code_t;
    pub fn tk_cortex_inject_sensor_event(cortex: *mut tk_cortex_t, event: *const tk_sensor_event_t) -> tk_error_code_t;
}
