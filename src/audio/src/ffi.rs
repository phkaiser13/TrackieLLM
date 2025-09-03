/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/audio/ffi.rs
 *
 * This file contains the raw FFI (Foreign Function Interface) bindings for the
 * C-based audio pipeline library (`tk_audio_pipeline.h`). These bindings
 * allow Rust code to directly call the C functions and interact with the
 * C-defined data structures.
 *
 * This is a direct, unsafe mapping of the C header file.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#![allow(non_camel_case_types, non_snake_case)]

use std::os::raw::{c_char, c_float, c_int, c_void};

// --- Opaque and Forward-Declared Types ---

/// Opaque handle to the main audio pipeline instance.
#[repr(C)]
pub struct tk_audio_pipeline_s {
    _private: [u8; 0],
}
pub type tk_audio_pipeline_t = tk_audio_pipeline_s;

// --- Placeholders for types from other headers ---

/// Placeholder for the path struct.
#[repr(C)]
pub struct tk_path_t {
    _private: [u8; 0],
}

/// Placeholder for the response priority enum.
pub type tk_response_priority_e = c_int;

// --- Enums and Structs ---

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct tk_audio_params_t {
    pub sample_rate: u32,
    pub channels: u32,
}

#[repr(C)]
pub struct tk_audio_pipeline_config_t {
    pub input_audio_params: tk_audio_params_t,
    pub asr_model_path: *const tk_path_t,
    pub vad_model_path: *const tk_path_t,
    pub tts_model_dir_path: *const tk_path_t,
    pub user_language: *const c_char,
    pub user_data: *mut c_void,
    pub vad_silence_threshold_ms: c_float,
    pub vad_speech_probability_threshold: c_float,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum tk_vad_event_e {
    TK_VAD_EVENT_SPEECH_STARTED,
    TK_VAD_EVENT_SPEECH_ENDED,
}

#[repr(C)]
#[derive(Debug)]
pub struct tk_transcription_t {
    pub text: *const c_char,
    pub is_final: bool,
    pub confidence: c_float,
}

// --- Callback Function Pointers ---

pub type tk_on_vad_event_cb =
    Option<unsafe extern "C" fn(event: tk_vad_event_e, user_data: *mut c_void)>;

pub type tk_on_transcription_cb =
    Option<unsafe extern "C" fn(result: *const tk_transcription_t, user_data: *mut c_void)>;

pub type tk_on_tts_audio_ready_cb = Option<
    unsafe extern "C" fn(
        audio_data: *const i16,
        frame_count: usize,
        sample_rate: u32,
        user_data: *mut c_void,
    ),
>;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct tk_audio_callbacks_t {
    pub on_vad_event: tk_on_vad_event_cb,
    pub on_transcription: tk_on_transcription_cb,
    pub on_tts_audio_ready: tk_on_tts_audio_ready_cb,
}

// --- FFI Function Declarations ---

pub type tk_error_code_t = c_int;

#[link(name = "trackie")]
extern "C" {
    pub fn tk_audio_pipeline_create(
        out_pipeline: *mut *mut tk_audio_pipeline_t,
        config: *const tk_audio_pipeline_config_t,
        callbacks: tk_audio_callbacks_t,
    ) -> tk_error_code_t;

    pub fn tk_audio_pipeline_destroy(pipeline: *mut *mut tk_audio_pipeline_t);

    pub fn tk_audio_pipeline_process_chunk(
        pipeline: *mut tk_audio_pipeline_t,
        audio_chunk: *const i16,
        frame_count: usize,
    ) -> tk_error_code_t;

    pub fn tk_audio_pipeline_synthesize_text(
        pipeline: *mut tk_audio_pipeline_t,
        text_to_speak: *const c_char,
        priority: tk_response_priority_e,
    ) -> tk_error_code_t;

    pub fn tk_audio_pipeline_force_transcription_end(
        pipeline: *mut tk_audio_pipeline_t,
    ) -> tk_error_code_t;
}
