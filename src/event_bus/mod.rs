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

// --- FFI for C Interoperability ---

/// Creates a new EventBus and returns a pointer to it.
/// The caller is responsible for destroying the bus using `event_bus_destroy`.
#[no_mangle]
pub extern "C" fn event_bus_create() -> *mut EventBus {
    Box::into_raw(Box::new(EventBus::new()))
}

/// Destroys an EventBus instance created by `event_bus_create`.
/// # Safety
/// The provided pointer must be a valid pointer to an EventBus that was
/// allocated by `event_bus_create`.
#[no_mangle]
pub unsafe extern "C" fn event_bus_destroy(bus: *mut EventBus) {
    if !bus.is_null() {
        drop(Box::from_raw(bus));
    }
}

// FFI-safe representation of the vision result.
// This is a simplified version of `tk_vision_result_t` from the C side.
#[repr(C)]
pub struct FfiVisionObject {
    pub label: *const std::os::raw::c_char,
    pub confidence: f32,
    pub distance_meters: f32,
}

#[repr(C)]
pub struct FfiVisionResult {
    pub objects: *const FfiVisionObject,
    pub object_count: usize,
    pub timestamp_ns: u64,
}

/// Publishes a vision result event from C code.
/// # Safety
/// The provided pointers must be valid and point to data with the correct layout.
#[no_mangle]
pub unsafe extern "C" fn vision_publish_result(bus: *const EventBus, result: *const FfiVisionResult) {
    if bus.is_null() || result.is_null() {
        return;
    }

    let bus = &*bus;
    let result = &*result;

    let mut objects = Vec::new();
    if result.object_count > 0 && !result.objects.is_null() {
        let detected_objects = std::slice::from_raw_parts(result.objects, result.object_count);
        for obj in detected_objects {
            objects.push(DetectedObject {
                label: std::ffi::CStr::from_ptr(obj.label).to_string_lossy().into_owned(),
                confidence: obj.confidence,
                distance: obj.distance_meters,
            });
        }
    }

    let vision_data = VisionData {
        objects,
        timestamp_ns: result.timestamp_ns,
    };

    bus.publish(TrackieEvent::VisionResult(Arc::new(vision_data)));
    log::debug!("[Rust FFI] Published VisionResult event to the bus.");
}
