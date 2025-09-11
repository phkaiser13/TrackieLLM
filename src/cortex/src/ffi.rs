/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/cortex/src/ffi.rs
 *
 * This file contains the foreign function interface (FFI) bindings and safe
 * wrappers for the C-level functions used by the Rust components of the Cortex.
 * It centralizes all `unsafe` FFI declarations and provides safe, idiomatic
 * Rust functions to the rest of the crate.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::Mutex;
use once_cell::sync::Lazy;
use crate::reasoning::ContextualReasoner;

// --- Global State for the Rust-side Reasoner ---
// This static variable holds the Rust-native ContextualReasoner, which contains
// the WorldModel. It's wrapped in a Mutex for thread-safe access from the
// FFI functions, which could be called from any C thread.
static REASONER: Lazy<Mutex<ContextualReasoner>> = Lazy::new(|| {
    Mutex::new(ContextualReasoner::default())
});


// Re-export the C-level types from the main ffi_bridge crate for consistency.
// In a larger project, these might be in a dedicated `ffi-types` crate.
// pub use trackiellm_ffi::{ // Temporarily disabled
//     tk_contextual_reasoner_t,
//     tk_error_code_t,
//     tk_error_code_t_TK_SUCCESS
// };

// Manually define the types we need until the ffi crate is fixed
pub enum tk_contextual_reasoner_s {}
pub type tk_contextual_reasoner_t = tk_contextual_reasoner_s;
pub type tk_error_code_t = i32;
pub const tk_error_code_t_TK_SUCCESS: tk_error_code_t = 0;

// FFI declarations for C functions that we need to call from this module.
#[link(name = "trackiellm_core")]
extern "C" {
    // From tk_contextual_reasoner.c
    pub fn tk_contextual_reasoner_add_conversation_turn(
        reasoner: *mut tk_contextual_reasoner_t,
        is_user_input: bool,
        content: *const c_char,
        confidence: f32,
    ) -> tk_error_code_t;

    pub fn tk_contextual_reasoner_generate_context_string(
        reasoner: *mut tk_contextual_reasoner_t,
        out_context_string: *mut *mut c_char,
        max_token_budget: usize,
    ) -> tk_error_code_t;

    pub fn tk_contextual_reasoner_get_context_summary(
        reasoner: *mut tk_contextual_reasoner_t,
        out_summary: *mut tk_context_summary_t,
    ) -> tk_error_code_t;

    // From the FFI bridge, which forwards to the C implementation.
    fn tk_get_last_error() -> *const c_char;
}

/// Safely gets the last error message from the FFI layer.
pub fn get_last_error_message() -> String {
    unsafe {
        let error_ptr = tk_get_last_error();
        if error_ptr.is_null() {
            return "Unknown FFI error".to_string();
        }
        CStr::from_ptr(error_ptr)
            .to_string_lossy()
            .into_owned()
    }
}

// --- Manual FFI Type Definitions for tk_event_t ---
// These are manually created to match the C headers for Task 1.3.

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum tk_event_type_e {
    TK_EVENT_TYPE_NONE,
    TK_EVENT_TYPE_VISION,
    TK_EVENT_TYPE_AUDIO,
    TK_EVENT_TYPE_SENSORS,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum tk_motion_state_e {
    TK_MOTION_STATE_UNKNOWN,
    TK_MOTION_STATE_STATIONARY,
    TK_MOTION_STATE_WALKING,
    TK_MOTION_STATE_RUNNING,
    TK_MOTION_STATE_FALLING,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum tk_ambient_sound_type_e {
    TK_AMBIENT_SOUND_NONE,
    TK_AMBIENT_SOUND_FIRE_ALARM,
    TK_AMBIENT_SOUND_CAR_HORN,
    TK_AMBIENT_SOUND_SIREN,
    TK_AMBIENT_SOUND_BABY_CRYING,
    TK_AMBIENT_SOUND_DOORBELL,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum tk_navigation_cue_type_e {
    TK_NAVIGATION_CUE_NONE,
    TK_NAVIGATION_CUE_STEP_UP,
    TK_NAVIGATION_CUE_STEP_DOWN,
    TK_NAVIGATION_CUE_DOORWAY,
    TK_NAVIGATION_CUE_STAIRS_UP,
    TK_NAVIGATION_CUE_STAIRS_DOWN,
}

// A placeholder, as the full definition is in another header.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tk_navigation_hazard_t {
    pub hazard_id: u32,
    pub description: *const c_char,
    pub distance_m: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tk_conversation_turn_t {
    pub timestamp_ns: u64,
    pub is_user_input: bool,
    pub content: *const c_char,
    pub confidence: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tk_rect_t {
    pub x: ::std::os::raw::c_int,
    pub y: ::std::os::raw::c_int,
    pub w: ::std::os::raw::c_int,
    pub h: ::std::os::raw::c_int,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tk_vision_object_t {
    pub class_id: u32,
    pub label: *const ::std::os::raw::c_char,
    pub confidence: f32,
    pub bbox: tk_rect_t,
    pub distance_meters: f32,
    pub width_meters: f32,
    pub height_meters: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tk_context_summary_t {
    // Environmental awareness
    pub visible_object_count: usize,
    pub visible_objects: *const tk_vision_object_t,

    // Navigation state
    pub has_clear_path: bool,
    pub clear_path_direction_deg: f32,
    pub clear_path_distance_m: f32,
    pub hazard_count: usize,
    pub hazards: *const tk_navigation_hazard_t,

    // Recent conversation
    pub conversation_turn_count: usize,
    pub recent_conversation: *const tk_conversation_turn_t,

    // Temporal context
    pub recent_events_summary: *const c_char,

    // System state
    pub is_navigation_active: bool,
    pub is_listening_for_commands: bool,
    pub system_confidence: f32,
    pub user_motion_state: tk_motion_state_e,

    // New fields
    pub detected_sound_type: tk_ambient_sound_type_e,
    pub detected_navigation_cue: tk_navigation_cue_type_e,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tk_quaternion_t {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tk_world_state_t {
    pub last_update_timestamp_ns: u64,
    pub orientation: tk_quaternion_t,
    pub motion_state: tk_motion_state_e,
    pub is_speech_detected: bool,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct tk_vision_event_t {
    pub object_count: usize,
    pub objects: *const tk_vision_object_t,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct tk_audio_event_t {
    pub text: *const ::std::os::raw::c_char,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct tk_sensor_event_t {
    pub world_state: tk_world_state_t,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union tk_event_t_data {
    pub vision_event: tk_vision_event_t,
    pub audio_event: tk_audio_event_t,
    pub sensor_event: tk_sensor_event_t,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct tk_event_t {
    pub type_: tk_event_type_e,
    pub timestamp_ns: u64,
    pub data: tk_event_t_data,
}

/// Initializes the Rust-side `ContextualReasoner` with a pointer from C.
///
/// This function should be called once at startup to link the C `tk_contextual_reasoner_t`
/// instance with the Rust-side `ContextualReasoner` that holds the `WorldModel`.
///
/// # Safety
/// The `reasoner_ptr` must be a valid pointer to a `tk_contextual_reasoner_t`
/// instance that will live for the duration of the program.
#[no_mangle]
pub unsafe extern "C" fn tk_cortex_rust_init_reasoner(reasoner_ptr: *mut tk_contextual_reasoner_t) {
    if reasoner_ptr.is_null() {
        println!("[TrackieLLM-Rust-FFI]: Warning - Initializing reasoner with null pointer.");
    }
    let mut reasoner = REASONER.lock().unwrap();
    *reasoner = ContextualReasoner::new(reasoner_ptr);
    println!("[TrackieLLM-Rust-FFI]: Rust Contextual Reasoner Initialized.");
}

/// Processes a generic event from the C side, dispatching it to the correct handler.
///
/// # Safety
/// This function is intended to be called from C code.
/// The `event` pointer must be a valid pointer to a `tk_event_t` struct.
#[no_mangle]
pub unsafe extern "C" fn tk_cortex_rust_process_event(event: *const tk_event_t) {
    if event.is_null() {
        println!("[TrackieLLM-Rust-FFI]: Received a null event pointer.");
        return;
    }

    let safe_event = &*event;

    // Lock the global reasoner to update its state.
    let mut reasoner = REASONER.lock().unwrap();
    if reasoner.ptr.is_null() {
        println!("[TrackieLLM-Rust-FFI]: Cannot process event, reasoner is not initialized.");
        return;
    }

    match safe_event.type_ {
        tk_event_type_e::TK_EVENT_TYPE_VISION => {
            let vision_event = &safe_event.data.vision_event;
            println!(
                "[TrackieLLM-Rust-FFI]: Processing Vision Event with {} objects.",
                vision_event.object_count
            );
            if let Err(e) = reasoner.process_vision_event(vision_event, safe_event.timestamp_ns) {
                println!("[TrackieLLM-Rust-FFI]: Error processing vision event: {}", e);
            }
        }
        tk_event_type_e::TK_EVENT_TYPE_AUDIO => {
            let audio_event = &safe_event.data.audio_event;
            if audio_event.text.is_null() {
                println!("[TrackieLLM-Rust-FFI]: Received Audio Event with null text.");
            } else {
                let c_str = CStr::from_ptr(audio_event.text);
                println!(
                    "[TrackieLLM-Rust-FFI]: Received Audio Event with text: {:?}",
                    c_str
                );
            }
        }
        tk_event_type_e::TK_EVENT_TYPE_SENSORS => {
            let sensor_event = &safe_event.data.sensor_event;
            println!(
                "[TrackieLLM-Rust-FFI]: Received Sensor Event with motion state: {:?}",
                sensor_event.world_state.motion_state
            );
        }
        tk_event_type_e::TK_EVENT_TYPE_NONE => {
            println!("[TrackieLLM-Rust-FFI]: Received a None Event.");
        }
    }
}

/// Executes the simple rules engine and copies any generated alert into a C buffer.
///
/// # Safety
/// `out_alert_buffer` must be a valid, writable pointer to a buffer of at least
/// `buffer_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn tk_cortex_reasoner_run_rules(
    out_alert_buffer: *mut ::std::os::raw::c_char,
    buffer_size: usize,
) -> bool {
    if out_alert_buffer.is_null() || buffer_size == 0 {
        return false;
    }

    let reasoner = REASONER.lock().unwrap();
    if reasoner.ptr.is_null() {
        return false; // Not initialized
    }

    if let Some(alert_string) = reasoner.run_simple_rules() {
        let c_string = match CString::new(alert_string) {
            Ok(s) => s,
            Err(_) => return false, // String had internal null bytes
        };

        // Use libc::strncpy for safe, bounded copying to the C buffer.
        libc::strncpy(out_alert_buffer, c_string.as_ptr(), buffer_size - 1);
        // Ensure null termination, as strncpy might not if the source is too long.
        *out_alert_buffer.add(buffer_size - 1) = 0;

        return true;
    }

    false // No alert was generated
}

/// Generates a contextual prompt and copies it into a C buffer.
///
/// # Safety
/// `prompt_buffer` must be a valid, writable pointer to a buffer of at least
/// `buffer_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn tk_cortex_generate_prompt(
    prompt_buffer: *mut ::std::os::raw::c_char,
    buffer_size: usize,
    user_query: *const c_char,
) -> bool {
    if prompt_buffer.is_null() || buffer_size == 0 {
        return false;
    }

    let mut reasoner = REASONER.lock().unwrap();

    let query_str = if user_query.is_null() {
        ""
    } else {
        match CStr::from_ptr(user_query).to_str() {
            Ok(s) => s,
            Err(_) => {
                println!("[TrackieLLM-Rust-FFI]: Invalid UTF-8 in user_query for generate_prompt.");
                ""
            }
        }
    };

    // Generate the prompt from the reasoner's world model.
    let prompt_result = reasoner.generate_prompt_for_llm(query_str);

    let prompt_string = match prompt_result {
        Ok(s) => s,
        Err(e) => {
            println!("[TrackieLLM-Rust-FFI]: Error generating prompt: {}", e);
            // Provide a fallback prompt
            "An error occurred. Please describe the general situation.".to_string()
        }
    };

    let c_string = match CString::new(prompt_string) {
        Ok(s) => s,
        Err(_) => {
            // This can happen if the generated string has a null byte.
            // In that case, we can't create a valid C-string.
            return false;
        }
    };

    // Safely copy the generated prompt to the C buffer.
    libc::strncpy(prompt_buffer, c_string.as_ptr(), buffer_size - 1);
    // Ensure null termination, as strncpy might not if the source is too long.
    *prompt_buffer.add(buffer_size - 1) = 0;

    true
}

/// Sets a fact in the Rust-side `MemoryManager`.
///
/// # Safety
/// `key` and `value` must be valid, null-terminated C strings.
#[no_mangle]
pub unsafe extern "C" fn tk_cortex_rust_set_fact(key: *const c_char, value: *const c_char) {
    if key.is_null() || value.is_null() {
        return;
    }

    let key_str = match CStr::from_ptr(key).to_str() {
        Ok(s) => s,
        Err(_) => {
            println!("[TrackieLLM-Rust-FFI]: Invalid UTF-8 in key for set_fact.");
            return;
        }
    };

    let value_str = match CStr::from_ptr(value).to_str() {
        Ok(s) => s,
        Err(_) => {
            println!("[TrackieLLM-Rust-FFI]: Invalid UTF-8 in value for set_fact.");
            return;
        }
    };

    let mut reasoner = REASONER.lock().unwrap();
    reasoner.memory_manager.set_fact(key_str, value_str);
    println!("[TrackieLLM-Rust-FFI]: Set fact '{}' = '{}'", key_str, value_str);
}
