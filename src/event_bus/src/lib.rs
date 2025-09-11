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

use crate::sensors::ffi;
use std::sync::Arc;
use tokio::sync::broadcast::{self, Receiver, Sender};
use trackiellm_navigation::{SpaceSector, TrackedObstacle};


// --- Public Data Structures ---

/// High-level classification of the user's current motion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionState {
    /// The state cannot be determined yet.
    Unknown,
    /// The user is still.
    Stationary,
    /// The user is walking at a steady pace.
    Walking,
    /// The user is running.
    Running,
    /// A potential fall event has been detected (freefall).
    Falling,
}

impl From<ffi::tk_motion_state_e> for MotionState {
    fn from(c_state: ffi::tk_motion_state_e) -> Self {
        match c_state {
            ffi::tk_motion_state_e::TK_MOTION_STATE_STATIONARY => Self::Stationary,
            ffi::tk_motion_state_e::TK_MOTION_STATE_WALKING => Self::Walking,
            ffi::tk_motion_state_e::TK_MOTION_STATE_RUNNING => Self::Running,
            ffi::tk_motion_state_e::TK_MOTION_STATE_FALLING => Self::Falling,
            _ => Self::Unknown,
        }
    }
}

/// Represents orientation in 3D space.
#[derive(Debug, Clone, Copy)]
pub struct Quaternion {
    /// W component of the quaternion.
    pub w: f32,
    /// X component of the quaternion.
    pub x: f32,
    /// Y component of the quaternion.
    pub y: f32,
    /// Z component of the quaternion.
    pub z: f32,
}

impl From<ffi::tk_quaternion_t> for Quaternion {
    fn from(c_quat: ffi::tk_quaternion_t) -> Self {
        Self {
            w: c_quat.w,
            x: c_quat.x,
            y: c_quat.y,
            z: c_quat.z,
        }
    }
}

/// The fused, high-level output of the sensor fusion engine.
#[derive(Debug, Clone, Copy)]
pub struct SensorFusionData {
    /// Timestamp of the last update.
    pub last_update_timestamp_ns: u64,
    /// The absolute orientation of the device in space.
    pub orientation: Quaternion,
    /// The user's classified motion state.
    pub motion_state: MotionState,
    /// The current state from the Voice Activity Detector.
    pub is_speech_detected: bool,
}

impl From<ffi::tk_world_state_t> for SensorFusionData {
    fn from(c_state: ffi::tk_world_state_t) -> Self {
        Self {
            last_update_timestamp_ns: c_state.last_update_timestamp_ns,
            orientation: c_state.orientation.into(),
            motion_state: c_state.motion_state.into(),
            is_speech_detected: c_state.is_speech_detected,
        }
    }
}


/// Contains the combined output of the navigation analysis modules.
#[derive(Debug, Clone)]
pub struct NavigationData {
    /// A sector-based analysis of clear paths.
    pub free_space_sectors: Vec<SpaceSector>,
    /// A list of all currently tracked obstacles.
    pub tracked_obstacles: Vec<TrackedObstacle>,
}

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

/// The classification of a specific area of the ground plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroundPlaneStatus {
    /// The status of this area is unknown.
    Unknown,
    /// This area is considered flat and safe to traverse.
    Flat,
    /// This area contains an obstacle that is too high to be a step.
    Obstacle,
    /// This area is a hole or a sharp drop-off.
    Hole,
    /// This area has a steep incline (ramp up).
    RampUp,
    /// This area has a steep decline (ramp down).
    RampDown,
}

/// Describes a significant vertical change detected between grid cells, like a step or curb.
#[derive(Debug, Clone, Copy)]
pub struct VerticalChange {
    /// The estimated height of the change in meters.
    pub height_m: f32,
    /// The classification of the vertical change.
    pub status: GroundPlaneStatus,
    /// The grid cell index where the change was detected.
    pub grid_index: (u32, u32),
}

/// Represents the analysis of the ground plane for navigation purposes.
#[derive(Debug, Clone, Default)]
pub struct NavigationCues {
    /// A 2D grid representing the traversability of the ground in front of the user.
    pub traversability_grid: Vec<GroundPlaneStatus>,
    /// The dimensions of the grid (width, height).
    pub grid_dimensions: (u32, u32),
    /// A list of detected steps, curbs, or other significant vertical changes.
    pub detected_vertical_changes: Vec<VerticalChange>,
}

/// A simplified representation of the vision pipeline's complete output for a frame.
#[derive(Debug, Clone)]
pub struct VisionData {
    /// A list of all objects detected in the frame.
    pub objects: Vec<DetectedObject>,
    /// The timestamp (in nanoseconds) of when the frame was captured.
    pub timestamp_ns: u64,
    /// Optional navigation cues derived from the depth map analysis.
    pub navigation_cues: Option<Arc<NavigationCues>>,
}

/// Defines all possible events that can be broadcast across the application.
///
/// Using `Arc` for data payloads to avoid deep cloning the entire vector of
/// objects for every subscriber.
#[derive(Debug, Clone)]
pub enum TrackieEvent {
    /// Published by the vision worker when a new frame has been analyzed.
    VisionResult(Arc<VisionData>),
    /// Published by the sensor worker when the world state is updated.
    SensorFusionResult(Arc<SensorFusionData>),
    /// Published by the navigation engines when spatial analysis is complete.
    NavigationResult(Arc<NavigationData>),
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
