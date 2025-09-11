/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/audio/lib.rs
 *
 * This file is the main library entry point for the 'audio' crate. This crate
 * is responsible for all audio processing in the TrackieLLM application,
 * including speech recognition (ASR) and speech synthesis (TTS).
 *
 * It provides a high-level, safe Rust abstraction over the C-based audio
 * pipeline. The underlying C layer handles the direct interaction with the
 * Whisper.cpp and Piper TTS libraries, and this crate ensures that all
 * interactions with that layer are memory-safe and robust.
 *
 * The primary interface is the `AudioPipeline` struct, which manages the
 * lifecycle of the underlying `tk_audio_pipeline_t` and provides a safe,
 * callback-based or asynchronous API for audio processing.
 *
 * Dependencies:
 *   - internal_tools: For safe path handling.
 *   - log: For structured logging of audio events.
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

#![allow(unsafe_code)] // Unsafe is required for FFI
#![allow(missing_docs)] // TODO: Re-enable this lint and add documentation
#![deny(warnings)]

//! # TrackieLLM Audio Crate
//!
//! Provides a real-time, streaming audio processing pipeline for speech-to-text
//! and text-to-speech.
//!
//! ## Architecture
//!
//! The crate is centered around the `AudioPipeline` service. The application
//! continuously feeds raw microphone audio into the pipeline via the
//! `process_chunk` method. The pipeline internally uses a Voice Activity
//! Detector (VAD) to identify speech. When speech is detected, it is sent to
//! the Automatic Speech Recognition (ASR) engine. The resulting transcription
//! is then sent back to the application via a callback.
//!
//! For outgoing communication, the application can request speech synthesis
//! via the `synthesize_text` method. The pipeline's Text-to-Speech (TTS) engine
//! generates the audio, which is delivered back in chunks via another callback.

// --- FFI Bindings Module ---
// Contains the raw FFI declarations for the audio C API.
// The implementation is in the `ffi.rs` file.
pub mod ffi;


// --- Public Module Declarations ---

/// Provides a safe wrapper for the ASR (Whisper) engine.
pub mod asr_processing;
/// Provides a safe wrapper for the TTS (Piper) engine.
pub mod tts_synthesis;


// --- Core Public Data Structures & Types ---

use thiserror::Error;

/// Represents a finalized transcription of a speech segment.
#[derive(Debug, Clone, PartialEq)]
pub struct Transcription {
    /// The transcribed text.
    pub text: String,
    /// An overall confidence score for the transcription.
    pub confidence: f32,
}

/// The primary error type for all operations within the audio crate.
#[derive(Debug, Error)]
pub enum AudioError {
    /// The audio pipeline has not been initialized.
    #[error("Audio pipeline is not initialized.")]
    NotInitialized,

    /// An FFI call to the underlying C library failed.
    #[error("Audio FFI call failed: {0}")]
    Ffi(String),

    /// A required audio model could not be loaded.
    #[error("Failed to load audio model: {0}")]
    ModelLoadFailed(String),

    /// An error occurred during ASR or TTS inference.
    #[error("Inference failed: {0}")]
    InferenceFailed(String),
}


// --- Public Prelude ---

pub mod prelude {
    //! A "prelude" for convenient imports of this crate's main types.
    pub use super::{
        asr_processing::AsrService, tts_synthesis::TtsService, AudioError, AudioPipeline,
        Transcription,
    };
}


// --- Main Service Interface ---

/// The main Audio Pipeline service.
///
/// This struct is the primary public interface to the application's audio
/// processing capabilities. It wraps the C-level `tk_audio_pipeline_t`.
pub struct AudioPipeline {
    /// A handle to the underlying C object.
    pipeline_handle: *mut ffi::tk_audio_pipeline_t,
}

impl Drop for AudioPipeline {
    /// Ensures the C-level audio pipeline is always destroyed.
    fn drop(&mut self) {
        if !self.pipeline_handle.is_null() {
            unsafe { ffi::tk_audio_pipeline_destroy(&mut self.pipeline_handle) };
        }
    }
}
