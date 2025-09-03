/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: event_formatter.rs
 *
 * This file implements a structured JSON log formatter. It is designed to integrate
 * with the standard `log` crate facade and convert log records into a consistent,
 * machine-parseable JSON format. This is crucial for modern observability platforms
 * like Elasticsearch, Splunk, or Datadog. The formatter also supports extracting
 * key-value pairs from the log message for rich, contextual logging.
 *
 * Dependencies:
 *  - `serde`, `serde_json`: For serializing the structured log object to a JSON string.
 *  - `chrono`: For generating ISO 8601 timestamps.
 *  - `log`: To integrate with the standard logging record format.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Imports ---
use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::HashMap;

// --- Custom Error Type ---

/// Represents errors that can occur during log formatting.
#[derive(Debug, thiserror::Error)]
pub enum FormatError {
    #[error("Failed to serialize log record to JSON: {0}")]
    JsonSerializationError(#[from] serde_json::Error),
}

pub type FormatResult<T> = Result<T, FormatError>;

// --- Core Data Structures ---

/// Defines the schema for a structured JSON log entry.
/// This struct is serialized to produce the final log line.
#[derive(Serialize, Debug)]
struct JsonLog<'a> {
    #[serde(rename = "@timestamp")]
    timestamp: DateTime<Utc>,
    level: &'a str,
    message: String,
    target: &'a str,
    #[serde(flatten)] // Flattens the 'data' HashMap into the top-level object
    data: HashMap<String, Value>,
}

// --- JSON Formatter Service ---

/// A service for formatting `log::Record` instances into JSON strings.
#[derive(Default)]
pub struct JsonFormatter;

impl JsonFormatter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Formats a log record into a structured JSON string.
    ///
    /// This function performs two main tasks:
    /// 1. It populates the standard fields of the `JsonLog` struct.
    /// 2. It parses the log message for `key=value` pairs to enrich the structured data.
    ///
    /// # Arguments
    /// * `record` - A reference to a `log::Record` provided by the `log` facade.
    ///
    /// # Returns
    /// A `String` containing the final JSON log line.
    pub fn format(&self, record: &log::Record) -> FormatResult<String> {
        let (message, data) = self.parse_key_value_pairs(record.args().to_string());

        let log_entry = JsonLog {
            timestamp: Utc::now(),
            level: record.level().as_str(),
            target: record.target(),
            message,
            data,
        };

        let json_string = serde_json::to_string(&log_entry)?;
        Ok(json_string)
    }

    /// Parses a log message to separate the core message from structured key-value pairs.
    ///
    /// For a message like "User logged in user_id=123 session_id=abc-123", this function
    /// would return:
    /// - message: "User logged in"
    /// - data: a map containing `{"user_id": 123, "session_id": "abc-123"}`
    ///
    /// It intelligently handles quoted values.
    fn parse_key_value_pairs(&self, full_message: String) -> (String, HashMap<String, Value>) {
        let mut data = HashMap::new();
        let mut core_message_parts = Vec::new();

        for part in full_message.split_whitespace() {
            if let Some(pos) = part.find('=') {
                let (key, mut value_str) = part.split_at(pos);
                value_str = &value_str[1..]; // Remove the '='

                // Attempt to parse the value as a number first, then fallback to a string.
                // This adds semantic value to the JSON output.
                let value = if let Ok(i) = value_str.parse::<i64>() {
                    json!(i)
                } else if let Ok(f) = value_str.parse::<f64>() {
                    json!(f)
                } else if let Ok(b) = value_str.parse::<bool>() {
                    json!(b)
                } else {
                    // Handle quoted strings
                    json!(value_str.trim_matches('"'))
                };

                data.insert(key.to_string(), value);
            } else {
                core_message_parts.push(part);
            }
        }

        (core_message_parts.join(" "), data)
    }
}

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use log::{Level, Record};

    #[test]
    fn test_basic_json_formatting() {
        let formatter = JsonFormatter::new();
        let record = Record::builder()
            .args(format_args!("This is a test message"))
            .level(Level::Info)
            .target("test_target")
            .build();

        let formatted_str = formatter.format(&record).unwrap();
        let json_val: Value = serde_json::from_str(&formatted_str).unwrap();

        assert_eq!(json_val["level"], "INFO");
        assert_eq!(json_val["target"], "test_target");
        assert_eq!(json_val["message"], "This is a test message");
        assert!(json_val["@timestamp"].is_string());
    }

    #[test]
    fn test_key_value_pair_extraction() {
        let formatter = JsonFormatter::new();
        let record = Record::builder()
            .args(format_args!("Request processed status=200 duration_ms=55 user_id=123"))
            .level(Level::Warn)
            .target("api::server")
            .build();

        let formatted_str = formatter.format(&record).unwrap();
        let json_val: Value = serde_json::from_str(&formatted_str).unwrap();

        assert_eq!(json_val["message"], "Request processed");
        assert_eq!(json_val["status"], 200);
        assert_eq!(json_val["duration_ms"], 55);
        assert_eq!(json_val["user_id"], 123);
        assert_eq!(json_val["level"], "WARN");
    }

    #[test]
    fn test_key_value_with_quoted_strings() {
        let formatter = JsonFormatter::new();
        let record = Record::builder()
            .args(format_args!(r#"File uploaded file_name="my test file.txt" size_kb=1024"#))
            .level(Level::Debug)
            .target("fs::uploads")
            .build();

        let formatted_str = formatter.format(&record).unwrap();
        let json_val: Value = serde_json::from_str(&formatted_str).unwrap();

        assert_eq!(json_val["message"], "File uploaded");
        assert_eq!(json_val["file_name"], "my test file.txt");
        assert_eq!(json_val["size_kb"], 1024);
    }

    #[test]
    fn test_message_with_no_kv_pairs() {
        let formatter = JsonFormatter::new();
        let message = "A simple message with no special formatting.";
        let (core_msg, data) = formatter.parse_key_value_pairs(message.to_string());

        assert_eq!(core_msg, message);
        assert!(data.is_empty());
    }
}
