/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/workers/cortex_worker.rs
 *
 * This file implements the Cortex Worker, the asynchronous task that acts as
 * the "brain" of the TrackieLLM application. It subscribes to processed
 * sensory information from the event bus and makes decisions.
 *
 * Key responsibilities:
 * - Subscribing to `VisionResult` and `TranscriptionResult` events.
 * - Maintaining and updating the application's state by interacting with the
 *   thread-safe `MemoryManager`.
 * - Performing reasoning based on incoming data and current context.
 * - Publishing new events to trigger actions, such as `Speak` events to
 *   generate a verbal response.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#![deny(missing_docs)]

//! The cortex worker is the central reasoning engine of the application.

use crate::cortex::memory_manager::{MemoryManager, MemoryType};
use crate::event_bus::{EventBus, TrackieEvent};
use std::sync::{Arc, Mutex};
use tokio::time::{Duration, Instant};

/// The state managed by the Cortex worker.
struct CortexState {
    /// Tracks the last time an object was mentioned to avoid repetition.
    last_mention: std::collections::HashMap<String, Instant>,
}

impl CortexState {
    fn new() -> Self {
        Self {
            last_mention: std::collections::HashMap::new(),
        }
    }
}

/// The main entry point for the cortex worker task.
///
/// This function loops indefinitely, waiting for events from the sensory
/// workers and processing them to make decisions.
pub async fn run(event_bus: Arc<EventBus>, memory_manager: Arc<Mutex<MemoryManager>>) {
    println!("[Cortex Worker] Initializing...");

    // Subscribe to all events. The cortex needs to know about everything.
    let mut subscriber = event_bus.subscribe();
    let mut state = CortexState::new();

    println!("[Cortex Worker] Now listening for events.");
    loop {
        match subscriber.next_event().await {
            Ok(TrackieEvent::VisionResult(data)) => {
                // This block handles incoming vision data.
                // It demonstrates context-aware logic to avoid spamming the user.

                for obj in &data.objects {
                    let should_mention = match state.last_mention.get(&obj.label) {
                        // Only mention if it hasn't been mentioned in the last 30 seconds.
                        Some(last_time) => last_time.elapsed() > Duration::from_secs(30),
                        None => true,
                    };

                    if should_mention && obj.confidence > 0.7 {
                        let response = format!("I see a {}.", obj.label);
                        println!("[Cortex Worker] Stating: '{}'", &response);
                        event_bus.publish(TrackieEvent::Speak(response));
                        state.last_mention.insert(obj.label.clone(), Instant::now());

                        // Archive this observation to long-term memory.
                        let mut memory = memory_manager.lock().unwrap();
                        let memory_content = format!(
                            "Saw a {} with {:.2}% confidence at a distance of {:.1}m.",
                            obj.label,
                            obj.confidence * 100.0,
                            obj.distance
                        );
                        memory
                            .archive_memory(
                                MemoryType::Episodic,
                                memory_content,
                                vec!["vision".to_string(), obj.label.clone()],
                                obj.confidence,
                            )
                            .ok();
                    }
                }
            }
            Ok(TrackieEvent::TranscriptionResult(text)) => {
                // This block handles transcribed speech from the user.
                println!("[Cortex Worker] Heard: '{}'", &text);

                // Simple keyword-based response logic.
                let response = if text.to_lowercase().contains("hello") {
                    "Hello to you too!".to_string()
                } else {
                    format!("You said: {}", text)
                };

                event_bus.publish(TrackieEvent::Speak(response));
            }
            Ok(TrackieEvent::VADEvent(is_speaking)) => {
                // The cortex can use VAD events to know when to listen or interrupt.
                if is_speaking {
                    println!("[Cortex Worker] User started speaking.");
                } else {
                    println!("[Cortex Worker] User stopped speaking.");
                }
            }
            // The cortex publishes `Speak` events but does not act on them.
            Ok(TrackieEvent::Speak(_)) => {}
            Err(e) => {
                eprintln!("[Cortex Worker] Event bus error, subscriber lagging: {}", e);
                // A brief pause can help if the system is overloaded.
                sleep(Duration::from_secs(1)).await;
            }
        }
    }
}
