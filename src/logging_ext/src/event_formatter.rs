/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/logging_ext/event_formatter.rs
 *
 * This file provides the logic for formatting log events into a structured
 * JSON format. Structured logging is essential for modern, scalable systems
 * as it allows for easy parsing, filtering, and analysis of logs by automated
 * tools like Elasticsearch, Logstash, and Kibana (ELK stack), or other log
 * aggregation services.
 *
 * The `JsonLogFormatter` provides a `format` function that can be plugged
 * directly into logging frameworks like `fern`. It transforms the standard
 * `log::Record` into a consistent JSON object, including timestamp, level,
 * message, and other metadata.
 *
 * Dependencies:
 *   - crate::LoggingExtError: For error handling.
 *   - log: For the `log::Record` struct.
 *   - serde_json: For building the JSON object.
 *   - chrono: For generating ISO 8601 timestamps.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use crate::LoggingExtError;
use chrono::Utc;
use log::Record;
use serde_json::{json, Value};
use std::fmt::Write;

/// A formatter for creating structured JSON logs.
///
/// This struct is the main entry point for the formatting logic. While it
/// doesn't hold any state itself, it serves as a namespace for the associated
/// `format` function.
pub struct JsonLogFormatter;

impl JsonLogFormatter {
    /// Formats a `log::Record` into a JSON string.
    ///
    /// This function is designed to be compatible with the formatting API of
    /// logging crates like `fern`, which expect a function with a specific
    /// signature.
    ///
    /// # Arguments
    ///
    /// * `out` - A mutable writer where the formatted string will be written.
    /// * `message` - The displayable message from the log record.
    /// * `record` - The `log::Record` containing all metadata about the log event.
    ///
    /// # JSON Output Structure
    ///
    /// The generated JSON object has the following structure:
    ///
    /// ```json
    /// {
    ///   "@timestamp": "2025-09-03T12:34:56.789Z",
    ///   "log.level": "INFO",
    ///   "message": "User successfully authenticated.",
    ///   "ecs.version": "1.6.0",
    ///   "log": {
    ///     "logger": "app::auth",
    ///     "origin": {
    ///       "file": {
    ///         "line": 42,
    ///         "name": "src/auth/service.rs"
    ///       }
    ///     }
    ///   },
    ///   // ... additional fields from the log record's key-values
    /// }
    /// ```
    ///
    /// This structure is inspired by the Elastic Common Schema (ECS), which
    /// provides a standardized way to structure logs for better interoperability.
    pub fn format(
        out: fern::FormatCallback,
        message: &std::fmt::Arguments,
        record: &Record,
    ) {
        let json_log = Self::build_json_log(message, record);
        let formatted_string = match serde_json::to_string(&json_log) {
            Ok(s) => s,
            Err(e) => {
                // As a fallback, if JSON serialization fails, log in a simple text format
                // to avoid losing the log message entirely.
                format!(
                    "FATAL: JSON serialization failed: {}. Original message: {}",
                    e, message
                )
            }
        };
        out.finish(format_args!("{}", formatted_string));
    }

    /// Helper function to construct the `serde_json::Value` for a log record.
    fn build_json_log(message: &std::fmt::Arguments, record: &Record) -> Value {
        let mut log_object = json!({
            "@timestamp": Utc::now().to_rfc3339(),
            "log.level": record.level().to_string(),
            "message": message.to_string(),
        });

        // Add ECS (Elastic Common Schema) fields for better structure.
        if let Some(map) = log_object.as_object_mut() {
            map.insert("ecs.version".to_string(), json!("1.6.0"));

            let mut log_details = json!({ "logger": record.target() });

            if let (Some(file), Some(line)) = (record.file(), record.line()) {
                log_details["origin"] = json!({
                    "file": {
                        "name": file,
                        "line": line,
                    }
                });
            }
            map.insert("log".to_string(), log_details);
        }

        log_object
    }
}
