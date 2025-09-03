/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/workers/mod.rs
 *
 * This module consolidates all the background worker services of the
 * TrackieLLM application. Each worker runs in its own asynchronous task
 * and is responsible for a specific domain (e.g., vision, audio, cortex).
 *
 * The workers communicate with each other and the rest of the system via
 * the central event bus, ensuring a decoupled and scalable architecture.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#![deny(missing_docs)]

//! # Background Worker Services
//!
//! This module contains the long-running, asynchronous tasks that form the
//! core logic of the application.

/// The worker responsible for the audio input/output pipeline.
pub mod audio_worker;

/// The worker responsible for the core reasoning and decision-making loop.
pub mod cortex_worker;

/// The worker responsible for the visual processing pipeline.
pub mod vision_worker;
