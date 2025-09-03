/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: audit_helpers.rs
 *
 * This file provides a structured framework for creating security audit trail logs.
 * Unlike diagnostic logs, audit logs are critical for security, compliance, and
 * accountability. This module defines a strict schema for audit events and provides
 * helpers to ensure they are logged consistently and can be easily ingested by
 * Security Information and Event Management (SIEM) systems.
 *
 * Dependencies:
 *  - `serde`, `serde_json`: For serializing the audit event struct.
 *  - `chrono`: For precise event timestamps.
 *  - `log`: To emit the final serialized audit event into the logging pipeline.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Imports ---
use chrono::{DateTime, Utc};
use serde::Serialize;
use std::collections::HashMap;

// --- Core Data Structures ---

/// Represents the actor (who) performing an action.
#[derive(Serialize, Debug, Clone)]
pub struct Actor {
    /// The primary identifier for the actor (e.g., user ID, service name).
    pub id: String,
    /// Additional identifying details (e.g., IP address, process ID).
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub details: HashMap<String, String>,
}

/// Represents the object (what) on which an action was performed.
#[derive(Serialize, Debug, Clone)]
pub struct Object {
    /// The primary identifier for the object (e.g., file path, resource ID).
    pub id: String,
    /// The type of the object (e.g., "file", "database_record").
    #[serde(rename = "type")]
    pub object_type: String,
}

/// The outcome of the audited event.
#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    Success,
    Failure,
    Attempt,
}

/// Defines the strict schema for a single, self-contained audit event.
/// This structure is designed to be serialized directly to JSON for logging.
#[derive(Serialize, Debug, Clone)]
pub struct AuditEvent {
    /// A unique identifier for this specific event instance (e.g., a UUID).
    pub event_id: String,
    /// The precise time the event occurred.
    pub timestamp: DateTime<Utc>,
    /// The actor who initiated the event.
    pub actor: Actor,
    /// The action that was performed (e.g., "user.login", "file.delete").
    /// Uses dot-notation for easy parsing and filtering.
    pub action: String,
    /// The object that was the target of the action.
    pub object: Object,
    /// The result of the action.
    pub outcome: Outcome,
    /// Optional field for additional, action-specific context.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

impl AuditEvent {
    /// A builder-style constructor for creating a new `AuditEvent`.
    pub fn new(actor: Actor, action: &str, object: Object) -> Self {
        Self {
            // In a real system, you'd use a UUID here.
            event_id: format!("{}-{}", Utc::now().timestamp_nanos_opt().unwrap_or(0), rand::random::<u32>()),
            timestamp: Utc::now(),
            actor,
            action: action.to_string(),
            object,
            outcome: Outcome::Attempt, // Default to attempt, should be set explicitly.
            details: None,
        }
    }

    /// Sets the outcome of the event.
    pub fn with_outcome(mut self, outcome: Outcome) -> Self {
        self.outcome = outcome;
        self;
    }

    /// Adds arbitrary JSON-serializable details to the event.
    pub fn with_details<T: Serialize>(mut self, details: T) -> Self {
        self.details = serde_json::to_value(details).ok();
        self;
    }
}

// --- Audit Logger Service ---

/// A helper service for emitting audit events.
/// This service ensures that audit events are logged to a specific, consistent
/// target, allowing them to be easily separated from other application logs.
#[derive(Default)]
pub struct AuditLogger {
    log_target: String,
}

impl AuditLogger {
    /// Creates a new `AuditLogger` that will log to the specified target.
    pub fn new(log_target: &str) -> Self {
        Self {
            log_target: log_target.to_string(),
        }
    }

    /// Logs a structured `AuditEvent`.
    ///
    /// The event is serialized to JSON and then logged at the `INFO` level
    /// using the configured log target. Using a fixed level and target ensures
    /// that audit logs are never accidentally filtered out and can be routed
    /// to a dedicated, secure destination.
    ///
    /// # Arguments
    /// * `event` - The `AuditEvent` to log.
    pub fn log(&self, event: &AuditEvent) {
        match serde_json::to_string(event) {
            Ok(json_string) => {
                // Log to the specific target at the INFO level.
                // This ensures audit logs are consistently captured.
                log::info!(target: &self.log_target, "{}", json_string);
            }
            Err(e) => {
                // If we can't even serialize the audit event, something is very wrong.
                // Log an error to a fallback target.
                log::error!(
                    target: "audit_fallback",
                    "Failed to serialize audit event. Event ID: {}. Error: {}",
                    event.event_id,
                    e
                );
            }
        }
    }
}

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_event_creation() {
        let actor = Actor {
            id: "user-123".to_string(),
            details: HashMap::from([("ip_address".to_string(), "192.168.1.100".to_string())]),
        };
        let object = Object {
            id: "/etc/shadow".to_string(),
            object_type: "file".to_string(),
        };

        let event = AuditEvent::new(actor, "file.read_attempt", object)
            .with_outcome(Outcome::Failure)
            .with_details(json!({ "reason": "Permission denied" }));

        assert_eq!(event.action, "file.read_attempt");
        assert_eq!(event.outcome, Outcome::Failure);
        assert!(event.event_id.len() > 10);
        assert_eq!(event.details, Some(json!({ "reason": "Permission denied" })));
    }

    // To test the `log` macro output, you would typically need a more complex
    // test setup that captures log records. We can simulate the serialization part.
    #[test]
    fn test_audit_event_serialization() {
         let actor = Actor { id: "service-account-abc".to_string(), details: HashMap::new() };
         let object = Object { id: "db-cluster-1".to_string(), object_type: "database".to_string() };

         let mut event = AuditEvent::new(actor.clone(), "db.connect", object.clone());
         // Manually set a predictable event_id for the test
         event.event_id = "test-event-123".to_string();
         let event = event.with_outcome(Outcome::Success);

         let json_string = serde_json::to_string(&event).unwrap();
         let val: serde_json::Value = serde_json::from_str(&json_string).unwrap();

         assert_eq!(val["event_id"], "test-event-123");
         assert_eq!(val["action"], "db.connect");
         assert_eq!(val["outcome"], "success");
         assert_eq!(val["actor"]["id"], "service-account-abc");
         assert_eq!(val["object"]["id"], "db-cluster-1");
         assert_eq!(val["object"]["type"], "database");
    }

    // This test demonstrates how you might use the logger, though it doesn't capture output.
    // In a real application, you'd initialize a logger like `slog` or `env_logger`
    // that could be configured to write the "audit_trail" target to a specific file.
    #[test]
    fn test_audit_logger_usage() {
        // In a test environment, logs might not be initialized, so this just ensures no panics.
        let logger = AuditLogger::new("test_audit_trail");
        let actor = Actor { id: "test-user".to_string(), details: HashMap::new() };
        let object = Object { id: "test-resource".to_string(), object_type: "resource".to_string() };
        let event = AuditEvent::new(actor, "resource.create", object).with_outcome(Outcome::Success);

        // This call will likely do nothing unless a logger is active, but we can
        // call it to ensure it doesn't crash.
        logger.log(&event);
    }
}
