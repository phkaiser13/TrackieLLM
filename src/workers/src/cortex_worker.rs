/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/workers/cortex_worker.rs
 *
 * This file implements the Cortex Worker, the asynchronous task that acts as
 * the "brain" of the TrackieLLM application. It serves as the critical bridge
 * between the Rust-based, event-driven workers and the C-based, synchronous
 * Cortex reasoning engine.
 *
 * Key responsibilities:
 * - Initializing and managing the lifecycle of the C-level `tk_cortex_t` object.
 * - Spawning a dedicated OS thread to run the blocking `tk_cortex_run` function.
 * - Subscribing to events on the Rust `EventBus` (`VisionResult`, `TranscriptionResult`).
 * - Translating Rust events into C-compatible data structures and injecting them
 *   into the C Cortex engine.
 * - Handling callbacks from the C engine (e.g., for TTS output) and translating
 *   them back into events on the Rust `EventBus`.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#![deny(missing_docs)]

//! The cortex worker is the central reasoning engine of the application.

use crate::cortex::ffi;
use crate::event_bus::{EventBus, TrackieEvent};
use std::ffi::{c_void, CStr, CString};
use std::ptr;
use std::sync::Arc;
use std::thread;
use tokio::sync::mpsc::{self, Sender};

// --- FFI Callback Bridge ---

/// A context structure to hold an MPSC sender for TTS events. A pointer to
/// this struct is passed as `user_data` to the C library, allowing the
/// `on_tts_audio_ready` callback to send synthesized audio commands back to the async world.
struct CallbackContext {
    tts_sender: Sender<String>,
}

/// FFI-compatible callback for state change notifications from the C Cortex.
unsafe extern "C" fn on_state_change(
    new_state: ffi::tk_system_state_e,
    _user_data: *mut c_void,
) {
    println!("[Cortex-C Callback] State changed to: {:?}", new_state);
}

/// FFI-compatible callback for receiving synthesized speech requests from the C Cortex.
unsafe extern "C" fn on_tts_audio_ready(
    audio_data: *const i16,
    frame_count: usize,
    sample_rate: u32,
    user_data: *mut c_void,
) {
    // This callback would handle raw audio data. For this integration, we assume
    // the C-cortex is orchestrating TTS and this is just a notification.
    // A more advanced implementation would pass the raw audio data.
    // Here, we simulate that the C-side produced a text response and we need to speak it.
    // In a real scenario, the text that was synthesized would be passed along.
    if user_data.is_null() {
        return;
    }
    let context = &*(user_data as *const CallbackContext);
    // For simplicity, we send a confirmation message. The actual text would originate
    // from the C-side decision engine's `on_response_ready` callback, which would
    // then trigger the TTS. This callback signals the *result* of that TTS.
    let _ = context.tts_sender.try_send(
        "The C cortex has generated a response.".to_string()
    );
}


// --- RAII Wrapper for the C-Cortex ---

/// A safe wrapper around the C `tk_cortex_t` handle.
struct CortexWrapper {
    handle: *mut ffi::tk_cortex_t,
    // Keep CStrings alive for the lifetime of the Cortex object.
    _model_paths: Box<ffi::tk_model_paths_t>,
    _lang: CString,
    _llm_path: CString,
    _obj_path: CString,
    _depth_path: CString,
    _asr_path: CString,
    _tts_path: CString,
    _vad_path: CString,
    _tess_path: CString,
}

impl CortexWrapper {
    /// Creates and initializes a new C-Cortex instance.
    unsafe fn new(tts_sender: Sender<String>) -> Result<Self, String> {
        let mut handle: *mut ffi::tk_cortex_t = ptr::null_mut();

        // Box the context so it has a stable memory address.
        let mut context = Box::new(CallbackContext { tts_sender });

        let callbacks = ffi::tk_cortex_callbacks_t {
            on_state_change: Some(on_state_change),
            on_tts_audio_ready: Some(on_tts_audio_ready),
        };

        // Create CStrings for all model paths to ensure they are null-terminated
        // and live long enough for the C library to use them.
        let llm_path = CString::new("assets/models/mistral-7b.gguf").unwrap();
        let obj_path = CString::new("assets/models/yolov5nu.onnx").unwrap();
        let depth_path = CString::new("assets/models/midas_dpt_swin_tiny.onnx").unwrap();
        let asr_path = CString::new("assets/models/whisper_tiny.en.ggml").unwrap();
        let tts_path = CString::new("assets/models/piper_tts_en_us_lessac_medium").unwrap();
        let vad_path = CString::new("assets/models/silero_vad.onnx").unwrap();
        let tess_path = CString::new("assets/tessdata").unwrap();
        let lang = CString::new("en-US").unwrap();

        let mut model_paths = Box::new(ffi::tk_model_paths_t {
            llm_model: llm_path.as_ptr(),
            object_detection_model: obj_path.as_ptr(),
            depth_estimation_model: depth_path.as_ptr(),
            asr_model: asr_path.as_ptr(),
            tts_model_dir: tts_path.as_ptr(),
            vad_model: vad_path.as_ptr(),
            tesseract_data_dir: tess_path.as_ptr(),
        });

        let config = ffi::tk_cortex_config_t {
            model_paths: *model_paths,
            gpu_device_id: -1, // Request CPU
            main_loop_frequency_hz: 10.0,
            user_language: lang.as_ptr(),
            user_data: &mut *context as *mut _ as *mut c_void,
        };

        let err_code = ffi::tk_cortex_create(&mut handle, &config, callbacks);

        if err_code == 0 && !handle.is_null() {
            Ok(Self {
                handle,
                _model_paths: model_paths,
                _lang: lang,
                _llm_path: llm_path,
                _obj_path: obj_path,
                _depth_path: depth_path,
                _asr_path: asr_path,
                _tts_path: tts_path,
                _vad_path: vad_path,
                _tess_path: tess_path,
            })
        } else {
            Err(format!("Failed to create C-Cortex with code {}", err_code))
        }
    }
}

impl Drop for CortexWrapper {
    fn drop(&mut self) {
        println!("[Cortex Worker] CortexWrapper Drop: Stopping C-Cortex...");
        unsafe {
            ffi::tk_cortex_stop(self.handle);
        }
        // The actual destruction will happen after the run thread is joined.
    }
}

/// The main entry point for the cortex worker task.
pub async fn run(event_bus: Arc<EventBus>) {
    println!("[Cortex Worker] Initializing...");

    let (tts_tx, mut tts_rx) = mpsc::channel(32);

    // 1. Initialize the C-Cortex instance.
    let cortex = match unsafe { CortexWrapper::new(tts_tx) } {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("[Cortex Worker] FATAL: {}", e);
            return;
        }
    };

    // 2. Spawn a dedicated OS thread for the blocking C-Cortex run loop.
    let cortex_handle_for_thread = cortex.handle;
    let cortex_thread = thread::spawn(move || {
        println!("[Cortex Worker] C-Cortex thread started.");
        let err_code = unsafe { ffi::tk_cortex_run(cortex_handle_for_thread) };
        println!("[Cortex Worker] C-Cortex thread finished with code: {}", err_code);
    });

    let mut subscriber = event_bus.subscribe();
    println!("[Cortex Worker] Now bridging events to C-Cortex.");

    // 3. Main async loop to bridge events.
    loop {
        tokio::select! {
            // Branch 1: Listen for events from other workers.
            Ok(event) = subscriber.next_event() => {
                match event {
                    TrackieEvent::VisionResult(data) => {
                        // This is a simplification. A real implementation would need
                        // to construct a tk_video_frame_t with the pixel data.
                        // For now, we signal an event with null data, assuming the C-side mock
                        // can handle it.
                        let frame = ffi::tk_video_frame_t {
                            width: 640, height: 480, stride: 640*3,
                            format: 0, // RGB8
                            data: ptr::null(),
                        };
                        unsafe { ffi::tk_cortex_inject_video_frame(cortex.handle, &frame); }
                    }
                    TrackieEvent::TranscriptionResult(text) => {
                        // Injecting audio is also simplified. The C-Cortex expects raw
                        // audio frames, not transcribed text. This call is a placeholder
                        // to show the data flow.
                        let mock_audio = [0i16; 160];
                        unsafe { ffi::tk_cortex_inject_audio_frame(cortex.handle, mock_audio.as_ptr(), mock_audio.len()); }
                    }
                    TrackieEvent::SensorFusionResult(sensor_data) => {
                        // Manually convert from the event bus's safe Rust struct
                        // to the FFI-compatible C struct.
                        let c_world_state = crate::sensors::ffi::tk_world_state_t {
                            last_update_timestamp_ns: sensor_data.last_update_timestamp_ns,
                            orientation: crate::sensors::ffi::tk_quaternion_t {
                                w: sensor_data.orientation.w,
                                x: sensor_data.orientation.x,
                                y: sensor_data.orientation.y,
                                z: sensor_data.orientation.z,
                            },
                            motion_state: match sensor_data.motion_state {
                                crate::event_bus::MotionState::Stationary => crate::sensors::ffi::tk_motion_state_e::TK_MOTION_STATE_STATIONARY,
                                crate::event_bus::MotionState::Walking => crate::sensors::ffi::tk_motion_state_e::TK_MOTION_STATE_WALKING,
                                crate::event_bus::MotionState::Running => crate::sensors::ffi::tk_motion_state_e::TK_MOTION_STATE_RUNNING,
                                crate::event_bus::MotionState::Falling => crate::sensors::ffi::tk_motion_state_e::TK_MOTION_STATE_FALLING,
                                _ => crate::sensors::ffi::tk_motion_state_e::TK_MOTION_STATE_UNKNOWN,
                            },
                            is_speech_detected: sensor_data.is_speech_detected,
                        };

                        let event = ffi::tk_sensor_event_t {
                            world_state: c_world_state,
                        };
                        unsafe { ffi::tk_cortex_inject_sensor_event(cortex.handle, &event); }
                    }
                    TrackieEvent::Shutdown => {
                        println!("[Cortex Worker] Shutdown signal received. Terminating C-Cortex.");
                        // The drop implementation on CortexWrapper will call tk_cortex_stop.
                        // We break the loop, which will drop the wrapper.
                        break;
                    }
                    _ => {} // Ignore other events
                }
            },
            // Branch 2: Listen for TTS requests from the C-Cortex callbacks.
            Some(text_to_speak) = tts_rx.recv() => {
                println!("[Cortex Worker] Relaying TTS request from C-Cortex to EventBus.");
                event_bus.publish(TrackieEvent::Speak(text_to_speak));
            }
        }
    }

    // After the loop breaks, we need to ensure the C-thread is joined and resources are freed.
    println!("[Cortex Worker] Waiting for C-Cortex thread to join...");
    cortex_thread.join().expect("C-Cortex thread panicked!");

    // Explicitly drop the wrapper to call tk_cortex_destroy
    let mut cortex_handle_for_destroy = cortex.handle;
    drop(cortex);
    unsafe {
        ffi::tk_cortex_destroy(&mut cortex_handle_for_destroy);
    }
    println!("[Cortex Worker] Cortex has been destroyed. Terminating.");
}
