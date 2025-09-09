/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/workers/audio_worker.rs
 *
 * This file implements the Audio Worker, an asynchronous task dedicated
 * to managing the audio I/O and processing pipeline. It serves as the bridge
 * between the complex, self-threaded C-based audio library and the main
 * Rust application.
 *
 * Key responsibilities:
 * - Initializing the `tk_audio_pipeline` with Rust-defined FFI callbacks.
 * - Continuously feeding live audio data from a microphone into the pipeline.
 * - Receiving events (like transcriptions) from the C library via the FFI
 *   callbacks and publishing them to the central event bus.
 * - Subscribing to `Speak` events on the bus and commanding the C library's
 *   TTS engine to synthesize and play audio.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#![deny(missing_docs)]

//! The audio worker manages the audio I/O and processing pipeline.

use crate::audio::ffi;
use crate::event_bus::{EventBus, TrackieEvent};
use std::ffi::{c_void, CStr, CString};
use std::ptr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc::{self, Sender}, watch};
use tokio::time::sleep;

// --- FFI Callback Bridge ---

/// A context structure to hold `mpsc` senders. A pointer to this struct
/// is passed as `user_data` to the C library, allowing the C-style callbacks
/// to send data back into the Tokio async world.
struct CallbackContext {
    vad_sender: Sender<ffi::tk_vad_event_e>,
    transcription_sender: Sender<String>,
}

/// FFI-compatible callback for VAD events.
unsafe extern "C" fn on_vad_event(event: ffi::tk_vad_event_e, user_data: *mut c_void) {
    if user_data.is_null() {
        return;
    }
    let context = &*(user_data as *const CallbackContext);
    // `try_send` is used to avoid blocking in this C callback thread.
    let _ = context.vad_sender.try_send(event);
}

/// FFI-compatible callback for transcription results.
unsafe extern "C" fn on_transcription(
    result: *const ffi::tk_transcription_t,
    user_data: *mut c_void,
) {
    if user_data.is_null() || result.is_null() {
        return;
    }
    let context = &*(user_data as *const CallbackContext);
    let result_safe = &*result;

    // We only care about the final transcription for now.
    if result_safe.is_final {
        let text = CStr::from_ptr(result_safe.text).to_string_lossy().into_owned();
        let _ = context.transcription_sender.try_send(text);
    }
}

// --- Pipeline Wrapper ---

/// A safe wrapper around the C `tk_audio_pipeline_t` handle.
struct AudioPipelineWrapper {
    handle: *mut ffi::tk_audio_pipeline_t,
    // Keep the context alive with the pipeline.
    _context: Box<CallbackContext>,
}

impl AudioPipelineWrapper {
    /// Creates a new audio pipeline.
    unsafe fn new(
        vad_sender: Sender<ffi::tk_vad_event_e>,
        transcription_sender: Sender<String>,
    ) -> Result<Self, String> {
        let mut handle: *mut ffi::tk_audio_pipeline_t = ptr::null_mut();

        // Box the context so it has a stable memory address.
        let mut context = Box::new(CallbackContext {
            vad_sender,
            transcription_sender,
        });

        let callbacks = ffi::tk_audio_callbacks_t {
            on_vad_event: Some(on_vad_event),
            on_transcription: Some(on_transcription),
            on_tts_audio_ready: None, // TTS audio is not handled via callback in this worker
        };

        let config = ffi::tk_audio_pipeline_config_t {
            input_audio_params: ffi::tk_audio_params_t {
                sample_rate: 16000,
                channels: 1,
            },
            asr_model_path: ptr::null(),
            vad_model_path: ptr::null(),
            tts_model_dir_path: ptr::null(),
            user_language: CString::new("en").unwrap().into_raw(),
            user_data: &mut *context as *mut _ as *mut c_void,
            vad_silence_threshold_ms: 500.0,
            vad_speech_probability_threshold: 0.5,
        };

        let err_code = ffi::tk_audio_pipeline_create(&mut handle, &config, callbacks);

        // Free the language string we allocated.
        let _ = CString::from_raw(config.user_language as *mut _);

        if err_code == 0 && !handle.is_null() {
            Ok(Self {
                handle,
                _context: context,
            })
        } else {
            Err(format!("Failed to create audio pipeline with code {}", err_code))
        }
    }

    /// Pushes a chunk of audio data to the pipeline.
    unsafe fn process_chunk(&self, audio_chunk: &[i16]) {
        ffi::tk_audio_pipeline_process_chunk(self.handle, audio_chunk.as_ptr(), audio_chunk.len());
    }

    /// Requests TTS synthesis.
    unsafe fn synthesize_text(&self, text: &str) {
        let c_text = CString::new(text).unwrap();
        ffi::tk_audio_pipeline_synthesize_text(self.handle, c_text.as_ptr(), 0); // Normal priority
    }
}

impl Drop for AudioPipelineWrapper {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                ffi::tk_audio_pipeline_destroy(&mut self.handle);
            }
        }
    }
}

/// The main entry point for the audio worker task.
pub async fn run(event_bus: Arc<EventBus>) {
    println!("[Audio Worker] Initializing...");

    // Create MPSC channels for C -> Rust communication.
    let (vad_tx, mut vad_rx) = mpsc::channel(32);
    let (transcription_tx, mut transcription_rx) = mpsc::channel(32);

    // Initialize the audio pipeline.
    let pipeline = match unsafe { AudioPipelineWrapper::new(vad_tx, transcription_tx) } {
        Ok(p) => {
            println!("[Audio Worker] Pipeline initialized successfully.");
            Arc::new(p)
        }
        Err(e) => {
            eprintln!("[Audio Worker] Failed to initialize pipeline: {}", e);
            return;
        }
    };

    // Create a subscriber to listen for TTS commands.
    let mut tts_subscriber = event_bus.subscribe();

    // Create a watch channel for shutting down the mock microphone task.
    let (shutdown_tx, mut shutdown_rx) = watch::channel(());

    // Spawn a separate task to simulate feeding the microphone data.
    let pipeline_clone = pipeline.clone();
    let mic_handle = tokio::spawn(async move {
        let mock_chunk = vec![0i16; 1600]; // 100ms of audio at 16kHz
        loop {
            tokio::select! {
                // Branch 1: Wait for the next tick to send audio data.
                _ = sleep(Duration::from_millis(100)) => {
                    unsafe { pipeline_clone.process_chunk(&mock_chunk) };
                }
                // Branch 2: Listen for the shutdown signal from the main worker.
                _ = shutdown_rx.changed() => {
                    println!("[Audio Worker] Microphone task shutting down.");
                    break;
                }
            }
        }
    });

    println!("[Audio Worker] Now listening for events.");
    // Main worker loop.
    loop {
        tokio::select! {
            // 1. Listen for events from the main event bus.
            Ok(event) = tts_subscriber.next_event() => {
                match event {
                    TrackieEvent::Speak(text) => {
                        println!("[Audio Worker] Received command to speak: '{}'", &text);
                        unsafe { pipeline.synthesize_text(&text) };
                    }
                    TrackieEvent::Shutdown => {
                        println!("[Audio Worker] Shutdown signal received. Terminating.");
                        // Signal the microphone task to shut down.
                        let _ = shutdown_tx.send(());
                        // Wait for the microphone task to finish.
                        let _ = mic_handle.await;
                        break; // Exit the main loop
                    }
                    // Ignore other events like VisionResult, etc.
                    _ => {}
                }
            },
            // 2. Listen for VAD events from the C callback.
            Some(vad_event) = vad_rx.recv() => {
                let is_speaking = vad_event == ffi::tk_vad_event_e::TK_VAD_EVENT_SPEECH_STARTED;
                event_bus.publish(TrackieEvent::VADEvent(is_speaking));
            },
            // 3. Listen for transcription results from the C callback.
            Some(text) = transcription_rx.recv() => {
                println!("[Audio Worker] Publishing final transcription: '{}'", &text);
                event_bus.publish(TrackieEvent::TranscriptionResult(text));
            }
        }
    }
}
