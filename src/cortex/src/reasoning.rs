/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/cortex/reasoning.rs
 *
 * This file implements a safe Rust wrapper for the Contextual Reasoning Engine
 * defined in `tk_contextual_reasoner.h`. It provides a high-level API for
 * managing the application's short-term memory and situational awareness.
 *
 * The `ContextualReasoner` struct is the primary interface. It encapsulates
 * the `unsafe` FFI calls required to interact with the C-based reasoner,
 * handling resource management (via RAII) and error translation automatically.
 *
 * This module is responsible for converting rich Rust types into the C-style
 * structs required by the FFI and vice-versa, ensuring that the rest of the
 * application can operate with type-safe, idiomatic Rust.
 *
 * Dependencies:
 *   - crate::ffi: For the raw C function bindings.
 *   - crate::CortexError: For shared error handling.
 *   - log: For structured logging.
 *   - thiserror: For ergonomic error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use super::{ffi, CortexError};
use super::memory_manager::MemoryManager;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr::null_mut;
use thiserror::Error;
use libc;

/// Represents errors specific to the contextual reasoner.
#[derive(Debug, Error)]
pub enum ReasoningError {
    /// The underlying C-level context is not initialized.
    #[error("Reasoner context is not initialized.")]
    NotInitialized,

    /// An FFI call failed, with a message from the C side.
    #[error("Reasoner FFI call failed: {0}")]
    Ffi(String),

    /// A string passed to an FFI function contained an unexpected null byte.
    #[error("FFI string conversion failed: {0}")]
    NulError(#[from] std::ffi::NulError),

    /// A string returned from an FFI function was not valid UTF-8.
    #[error("FFI string decoding failed: {0}")]
    Utf8Error(#[from] std::str::Utf8Error),
}

// --- World Model Structures ---

/// A simple rectangle, used for bounding boxes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

/// Represents a single object being tracked over time by the reasoner.
#[derive(Debug, Clone)]
pub struct TrackedObject {
    /// A unique, stable ID for this object instance.
    pub id: u64,
    /// The object's class label (e.g., "person", "cup").
    pub label: String,
    /// The timestamp (ns) of the last time this object was seen.
    pub last_seen_timestamp: u64,
    /// The confidence score of the last detection.
    pub last_confidence: f32,
    /// The last known bounding box of the object.
    pub last_bbox: Rect,
    /// The last known distance to the object in meters.
    pub last_known_distance: f32,
}

/// Represents the reasoner's internal understanding of the current environment.
#[derive(Debug, Clone, Default)]
pub struct WorldModel {
    /// A list of all objects currently being tracked.
    pub objects: Vec<TrackedObject>,
    /// A counter to generate unique IDs for new objects.
    next_object_id: u64,
}

impl WorldModel {
    /// Generates a new, unique ID for a `TrackedObject`.
    fn new_object_id(&mut self) -> u64 {
        let id = self.next_object_id;
        self.next_object_id += 1;
        id
    }
}

/// A low-level RAII wrapper for the `tk_contextual_reasoner_t` handle.
/// In a real implementation, this would be managed by the `Cortex` struct.
struct ReasonerContext {
    ptr: *mut ffi::tk_contextual_reasoner_t,
}

impl Drop for ReasonerContext {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // The Cortex owns the reasoner, so it's responsible for destroying it.
            // This Drop impl is just for correctness in case this struct is ever
            // used in a standalone way.
        }
    }
}

/// A safe, high-level interface to the Contextual Reasoning Engine.
pub struct ContextualReasoner {
    // This is not the owner of the pointer. The C-level `tk_cortex_t` is.
    // We just hold a raw pointer to it.
    ptr: *mut ffi::tk_contextual_reasoner_t,
    /// The Rust-native world model, managed by the reasoner.
    world_model: WorldModel,
    /// The memory manager for short-term and long-term memory.
    memory_manager: MemoryManager,
}

impl ContextualReasoner {
    /// Creates a new `ContextualReasoner` wrapper.
    /// This does not create the reasoner, it only wraps an existing one.
    pub fn new(ptr: *mut ffi::tk_contextual_reasoner_t) -> Self {
        Self {
            ptr,
            world_model: WorldModel::default(),
            memory_manager: MemoryManager::default(),
        }
    }

    /// Adds a new turn of conversation to the context.
    pub fn add_conversation_turn(
        &mut self,
        is_user_input: bool,
        content: &str,
        confidence: f32,
    ) -> Result<(), ReasoningError> {
        if self.ptr.is_null() {
            return Err(ReasoningError::NotInitialized);
        }

        let c_content = CString::new(content)?;

        let status = unsafe {
            ffi::tk_contextual_reasoner_add_conversation_turn(
                self.ptr,
                is_user_input,
                c_content.as_ptr(),
                confidence,
            )
        };

        if status != ffi::tk_error_code_t_TK_SUCCESS {
            let error_msg = ffi::get_last_error_message();
            Err(ReasoningError::Ffi(error_msg))
        } else {
            Ok(())
        }
    }

    /// Generates a textual summary of the current context for the LLM.
    pub fn generate_context_string(
        &self,
        max_token_budget: usize,
    ) -> Result<String, ReasoningError> {
        if self.ptr.is_null() {
            return Err(ReasoningError::NotInitialized);
        }

        let mut c_string_ptr: *mut c_char = null_mut();

        let status = unsafe {
            ffi::tk_contextual_reasoner_generate_context_string(
                self.ptr,
                &mut c_string_ptr,
                max_token_budget,
            )
        };

        if status != ffi::tk_error_code_t_TK_SUCCESS {
            return Err(ReasoningError::Ffi(ffi::get_last_error_message()));
        }

        if c_string_ptr.is_null() {
            // C function succeeded but returned a null pointer, which is unexpected.
            // Treat this as an empty string.
            return Ok(String::new());
        }

        // Unsafe block to convert C string to Rust String and then free it.
        let result = unsafe {
            let rust_string = CStr::from_ptr(c_string_ptr).to_str()?.to_owned();
            // We must free the string that was allocated by the C side.
            // Assuming it was allocated with malloc/calloc/strdup.
            libc::free(c_string_ptr as *mut libc::c_void);
            Ok(rust_string)
        };

        result
    }

    /// Processes a vision event, updating the world model with new detections.
    ///
    /// This method contains the core logic for object tracking. It iterates
    /// through detected objects from a vision event and decides whether they
    /// correspond to existing tracked objects or are new ones.
    ///
    /// # Arguments
    /// * `event` - A reference to the C-style vision event from the FFI layer.
    /// * `timestamp_ns` - The timestamp of the event.
    pub fn process_vision_event(
        &mut self,
        event: &ffi::tk_vision_event_t,
        timestamp_ns: u64,
    ) -> Result<(), ReasoningError> {
        let detected_objects = unsafe {
            if event.objects.is_null() {
                &[]
            } else {
                std::slice::from_raw_parts(event.objects, event.object_count)
            }
        };

        let mut matched_world_object_ids = std::collections::HashSet::new();

        for detected_obj in detected_objects {
            let detected_rect = Rect {
                x: detected_obj.bbox.x,
                y: detected_obj.bbox.y,
                w: detected_obj.bbox.w,
                h: detected_obj.bbox.h,
            };

            let label_str = unsafe { CStr::from_ptr(detected_obj.label).to_str()? };

            // Find the best potential match from the existing world objects.
            // A match must have the same label and must not have been already matched in this frame.
            let best_match = self
                .world_model
                .objects
                .iter_mut()
                .filter(|world_obj| {
                    world_obj.label == label_str && !matched_world_object_ids.contains(&world_obj.id)
                })
                .min_by_key(|world_obj| {
                    // Find the closest object by calculating the squared distance between bounding box centers.
                    let world_center_x = world_obj.last_bbox.x + world_obj.last_bbox.w / 2;
                    let world_center_y = world_obj.last_bbox.y + world_obj.last_bbox.h / 2;
                    let detected_center_x = detected_rect.x + detected_rect.w / 2;
                    let detected_center_y = detected_rect.y + detected_rect.h / 2;

                    let dist_x = world_center_x - detected_center_x;
                    let dist_y = world_center_y - detected_center_y;
                    dist_x.pow(2) + dist_y.pow(2)
                });

            let mut was_matched = false;
            if let Some(matched_obj) = best_match {
                // We found a potential match. Use a simple distance heuristic to confirm.
                // A more robust solution would use IoU (Intersection over Union).
                let world_center_x = matched_obj.last_bbox.x + matched_obj.last_bbox.w / 2;
                let detected_center_x = detected_rect.x + detected_rect.w / 2;
                let center_dist_x = (world_center_x - detected_center_x).abs();

                // Heuristic: If horizontal centers are within half the width of the larger box,
                // consider it a match. This is a very basic form of association.
                let max_width = std::cmp::max(matched_obj.last_bbox.w, detected_rect.w);
                if center_dist_x < max_width / 2 {
                    // It's a match, update the object's state.
                    matched_obj.last_seen_timestamp = timestamp_ns;
                    matched_obj.last_confidence = detected_obj.confidence;
                    matched_obj.last_bbox = detected_rect;
                    matched_obj.last_known_distance = detected_obj.distance_meters;
                    matched_world_object_ids.insert(matched_obj.id);
                    was_matched = true;
                }
            }

            if !was_matched {
                // If no suitable match was found, treat this as a new object.
                self.add_new_object(detected_obj, timestamp_ns, label_str, detected_rect);
            }
        }

        Ok(())
    }

    /// A helper function to create and add a new `TrackedObject` to the world model.
    fn add_new_object(
        &mut self,
        detected_obj: &ffi::tk_vision_object_t,
        timestamp_ns: u64,
        label: &str,
        bbox: Rect,
    ) {
        let new_id = self.world_model.new_object_id();
        let new_tracked_obj = TrackedObject {
            id: new_id,
            label: label.to_string(),
            last_seen_timestamp: timestamp_ns,
            last_confidence: detected_obj.confidence,
            last_bbox: bbox,
            last_known_distance: detected_obj.distance_meters,
        };
        self.world_model.objects.push(new_tracked_obj);
    }

    /// Runs a set of simple, hard-coded rules against the current world model.
    ///
    /// This method is a placeholder for a more sophisticated rules engine. It
    /// checks for specific conditions (e.g., a person being too close) and
    /// returns a descriptive alert message if a rule is triggered.
    ///
    /// # Returns
    /// An `Option<String>` containing the alert message if a rule fires, or `None`.
    pub fn run_simple_rules(&mut self) -> Option<String> {
        const PERSON_ALERT_COOLDOWN_SECS: i64 = 10;

        for object in &self.world_model.objects {
            // Rule: Alert if a person is detected less than 1.0 meter away.
            if object.label == "person" && object.last_known_distance < 1.0 {
                let alert_key = format!("person_close_{}", object.id);

                // Check if we have already alerted for this specific person recently.
                if !self.memory_manager.short_term_memory.has_been_alerted_recently(
                    &alert_key,
                    PERSON_ALERT_COOLDOWN_SECS,
                ) {
                    // If not, record the alert and generate the message.
                    self.memory_manager.short_term_memory.record_alert(alert_key);
                    let alert = format!(
                        "Alert: Person detected at {:.1} meters.",
                        object.last_known_distance
                    );
                    return Some(alert);
                }
                // If we have alerted recently, suppress this one.
            }
        }

        // No rules triggered
        None
    }

    /// Generates a prompt for the LLM based on the current world model.
    ///
    /// This function summarizes the tracked objects into a natural language
    /// sentence that can be fed to the LLM to ask for a description of the
    /// environment.
    pub fn generate_environment_summary_prompt(&mut self) -> String {
        const MENTION_COOLDOWN_SECS: i64 = 60; // Consider an object "new" if not mentioned in the last minute.

        if self.world_model.objects.is_empty() {
            return "You are in an environment with no detected objects. What should you say?".to_string();
        }

        let mut new_objects = Vec::new();
        let mut mentioned_objects = Vec::new();

        for object in &self.world_model.objects {
            if self.memory_manager.short_term_memory.was_object_mentioned_recently(object.id, MENTION_COOLDOWN_SECS) {
                mentioned_objects.push(object);
            } else {
                new_objects.push(object);
            }
        }

        // After generating the prompt, we will have "mentioned" the new objects.
        for object in &new_objects {
            self.memory_manager.short_term_memory.record_object_mentioned(object.id);
        }

        // Helper closure to create a descriptive list string from a list of objects.
        let create_list_string = |objects: Vec<&&TrackedObject>| -> String {
            if objects.is_empty() {
                return String::new();
            }
            let mut counts = std::collections::HashMap::new();
            for obj in objects {
                *counts.entry(obj.label.clone()).or_insert(0) += 1;
            }
            let mut descriptions: Vec<String> = counts
                .into_iter()
                .map(|(label, count)| match count {
                    1 => format!("a {}", label),
                    _ => format!("{} {}s", count, label),
                })
                .collect();

            if descriptions.len() > 1 {
                let last = descriptions.pop().unwrap_or_default();
                format!("{}, and {}", descriptions.join(", "), last)
            } else {
                descriptions.get(0).cloned().unwrap_or_default()
            }
        };

        let new_list_str = create_list_string(new_objects);
        let mentioned_list_str = create_list_string(mentioned_objects);

        let mut prompt_parts = Vec::new();
        if !mentioned_list_str.is_empty() {
            prompt_parts.push(format!("You are still seeing {}.", mentioned_list_str));
        }
        if !new_list_str.is_empty() {
            prompt_parts.push(format!("You have now also spotted {}.", new_list_str));
        }

        if prompt_parts.is_empty() {
             "You are in an environment with no detected objects. What should you say?".to_string()
        } else {
            format!(
                "{} Based on this, describe the current situation to the user.",
                prompt_parts.join(" ")
            )
        }
    }
}

impl Default for ContextualReasoner {
    fn default() -> Self {
        Self::new(null_mut())
    }
}
