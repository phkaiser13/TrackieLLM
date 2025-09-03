/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/event_bus/mod.rs
 *
 * This file implements the central Event Bus for the TrackieLLM application.
 * It provides a thread-safe, asynchronous, many-to-many communication channel
 * based on the Publisher/Subscriber pattern. This decouples all major
 * components (`vision`, `audio`, `cortex`), allowing them to communicate

 * without direct dependencies.
 *
 * The architecture is built on `tokio::sync::broadcast` channels, which are
 * optimized for the "one-to-many" broadcast use case.
 *
 * Key components:
 * - `TrackieEvent`: An enum that defines the types of messages that can be
 *   sent across the system.
 * - `EventBus`: The main struct that manages the broadcast channel sender.
 * - `EventBusSubscriber`: A wrapper around the broadcast channel receiver.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#![deny(missing_docs)]

//! # The Central Event Bus
//!
//! Provides a system-wide communication channel for asynchronous events.

use std::sync::Arc;
use tokio::sync::broadcast::{self, Receiver, Sender};

// --- Public Data Structures ---

/// A simplified representation of a detected object.
///
/// This struct will be populated from the FFI result of the vision pipeline.
#[derive(Debug, Clone)]
pub struct DetectedObject {
    /// The class label of the detected object (e.g., "cat", "chair").
    pub label: String,
    /// The confidence score of the detection (0.0 to 1.0).
    pub confidence: f32,
    /// The estimated distance to the object in meters.
    pub distance: f32,
}

/// A simplified representation of the vision pipeline's complete output for a frame.
#[derive(Debug, Clone)]
pub struct VisionData {
    /// A list of all objects detected in the frame.
    pub objects: Vec<DetectedObject>,
    /// The timestamp (in nanoseconds) of when the frame was captured.
    pub timestamp_ns: u64,
}

/// Defines all possible events that can be broadcast across the application.
///
/// Using `Arc` for `VisionData` to avoid deep cloning the entire vector of
/// objects for every subscriber.
#[derive(Debug, Clone)]
pub enum TrackieEvent {
    /// Published by the vision worker when a new frame has been analyzed.
    VisionResult(Arc<VisionData>),
    /// Published by the audio worker when a full sentence has been transcribed.
    TranscriptionResult(String),
    /// Published by the audio worker to signal voice activity. `true` for speech
    /// started, `false` for speech ended.
    VADEvent(bool),
    /// Published by the cortex to command the audio worker to speak.
    Speak(String),
    /// Published by the main orchestrator to signal all workers to shut down gracefully.
    Shutdown,
}

// --- Event Bus Implementation ---

const EVENT_BUS_CAPACITY: usize = 256;

/// The central event bus publisher.
///
/// There is typically only one `EventBus` instance, shared across the application
/// (e.g., within an `Arc`).
#[derive(Debug)]
pub struct EventBus {
    sender: Sender<TrackieEvent>,
}

impl EventBus {
    /// Creates a new event bus with a specified channel capacity.
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(EVENT_BUS_CAPACITY);
        Self { sender }
    }

    /// Publishes an event to all active subscribers.
    ///
    /// If there are no subscribers, the event is dropped. This is the expected
    /// behavior of a broadcast channel.
    ///
    /// # Arguments
    /// * `event` - The `TrackieEvent` to broadcast.
    pub fn publish(&self, event: TrackieEvent) {
        // The result of `send` is ignored. If it fails, it's because there are
        // no active receivers, which is a valid state.
        let _ = self.sender.send(event);
    }

    /// Creates a new subscriber to this event bus.
    ///
    /// Each subscriber receives a handle to the `Receiver` end of the channel.
    pub fn subscribe(&self) -> EventBusSubscriber {
        EventBusSubscriber {
            receiver: self.sender.subscribe(),
        }
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

/// A subscriber handle for the event bus.
///
/// This struct wraps the `Receiver` and provides a convenient `next_event` method.
#[derive(Debug)]
pub struct EventBusSubscriber {
    receiver: Receiver<TrackieEvent>,
}

impl EventBusSubscriber {
    /// Waits for the next event to be published.
    ///
    /// This method is asynchronous and will wait until an event is available.
    ///
    /// # Returns
    /// * `Ok(TrackieEvent)` if an event was received.
    /// * `Err(RecvError)` if the channel is lagging or closed. A `Lagged`
    ///   error means the subscriber was too slow and missed some messages.
    pub async fn next_event(&mut self) -> Result<TrackieEvent, broadcast::error::RecvError> {
        self.receiver.recv().await
    }
}
