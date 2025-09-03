/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/integration/bridge.rs
 *
 * This file defines the communication bridge between the core application and
 * external components like plugins. The purpose of the bridge is to provide a
 * stable, well-defined, and versioned API for data exchange, decoupling the
 * core logic from the specific implementation details of any single plugin.
 *
 * The bridge uses a request-response pattern. The core application can send a
 * structured `BridgeRequest` to a plugin and will receive a `BridgeResponse`
 * in return. This ensures that all communication is explicit and follows a
 * predefined contract, which is essential for maintaining a stable plugin
 * ecosystem.
 *
 * The data structures are designed to be serializable (e.g., with `serde`),
 * which allows this bridge to be adapted for inter-process communication (IPC)
 * in the future if plugins need to be run in isolated sandboxes.
 *
 * Dependencies:
 *   - serde: For serializing and deserializing request/response data.
 *   - thiserror: For ergonomic error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Represents errors that can occur within the integration bridge.
#[derive(Debug, Error)]
pub enum BridgeError {
    /// The request type is not supported by the target plugin.
    #[error("Request type '{0}' is not supported.")]
    UnsupportedRequest(String),

    /// The response from the plugin was malformed or could not be parsed.
    #[error("Failed to parse response from plugin: {0}")]
    ResponseParseFailed(String),

    /// The plugin failed to handle the request.
    #[error("Plugin failed to handle the request: {0}")]
    PluginExecutionFailed(String),
}

/// Represents a request sent from the core application to a plugin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BridgeRequest {
    /// A request to get basic information about the plugin.
    GetPluginInfo,
    /// A request to perform a specific action, with a generic payload.
    /// The payload is a JSON value to allow for flexibility.
    PerformAction {
        /// The name of the action to perform.
        action_name: String,
        /// The JSON-encoded payload for the action.
        payload: serde_json::Value,
    },
}

/// Represents a response sent from a plugin back to the core application.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BridgeResponse {
    /// A simple acknowledgment of success.
    Ack,
    /// Contains information about the plugin.
    PluginInfo {
        /// The name of the plugin.
        name: String,
        /// The version of the plugin.
        version: String,
        /// The author of the plugin.
        author: String,
    },
    /// Contains the result of a successfully performed action.
    ActionSuccess {
        /// The JSON-encoded result data.
        result: serde_json::Value,
    },
    /// Indicates that an error occurred while the plugin was handling a request.
    ActionError {
        /// A message describing the error.
        error_message: String,
    },
}

/// The `IntegrationBridge` provides the methods for communicating with a plugin.
///
/// In a real implementation, this might hold a handle to a specific plugin
/// instance or a communication channel (e.g., a channel sender).
pub struct IntegrationBridge;

impl IntegrationBridge {
    /// Sends a request to a plugin and awaits a response.
    ///
    /// This is a simplified, synchronous mock of the communication process.
    /// An asynchronous version would be used in a real, non-blocking application.
    ///
    /// # Arguments
    ///
    /// * `plugin_name` - The identifier for the target plugin.
    /// * `request` - The `BridgeRequest` to be sent.
    ///
    /// # Returns
    ///
    /// A `BridgeResponse` from the plugin.
    #[allow(dead_code)] // Mock implementation
    pub fn send_request(
        &self,
        plugin_name: &str,
        request: &BridgeRequest,
    ) -> Result<BridgeResponse, BridgeError> {
        log::info!(
            "Sending request to plugin '{}': {:?}",
            plugin_name,
            request
        );

        // --- Mock Implementation ---
        // This simulates sending the request and receiving a response.
        // A real implementation would involve looking up the plugin by name
        // and calling its `handle_request` method.

        match request {
            BridgeRequest::GetPluginInfo => Ok(BridgeResponse::PluginInfo {
                name: plugin_name.to_string(),
                version: "1.0.0-mock".to_string(),
                author: "Mock Author".to_string(),
            }),
            BridgeRequest::PerformAction { action_name, .. } => {
                if action_name == "do_something" {
                    Ok(BridgeResponse::ActionSuccess {
                        result: serde_json::json!({ "status": "completed" }),
                    })
                } else {
                    Ok(BridgeResponse::ActionError {
                        error_message: format!("Action '{}' not recognized.", action_name),
                    })
                }
            }
        }
    }
}
