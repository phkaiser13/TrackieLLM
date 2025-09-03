/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/monitoring/telemetry.rs
 *
 * This file implements the telemetry reporting logic. It receives metrics
 * collected by the `metrics_collector` module, batches them, and sends them
 * to a configured remote endpoint.
 *
 * The `run_reporter_loop` function is the heart of this module. It runs in a
 * background thread, continuously listening for new metrics on a channel. To
 * optimize network usage, it employs a batching strategy: it collects metrics
 * for a certain period or until a batch size limit is reached, and then sends
 * them all in a single network request.
 *
 * The module includes a simple, asynchronous-like HTTP client for sending
 * data. In a real-world, high-performance application, this would likely be
 * replaced with a more robust async runtime and client (e.g., tokio + reqwest),
 * but for this foundational implementation, a blocking client in a dedicated
 * thread is sufficient to illustrate the architecture.
 *
 * Dependencies:
 *   - crossbeam-channel: For receiving metrics from the collector.
 *   - log: For structured logging.
 *   - thiserror: For ergonomic error handling.
 *   - serde, serde_json: For serializing metrics data into JSON.
 *   - crate::MonitoringConfig: For configuration.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// === TOC ===
// 1. Module-level Imports and Constants
// 2. Error Type for this Module
// 3. Public Configuration Struct
// 4. Core Telemetry Logic
//    - run_reporter_loop (main entry point)
//    - process_and_send_batch (batch handling)
//    - send_telemetry_request (mock HTTP client)
// =============

use crate::metrics_collector::MetricData;
use crate::MonitoringConfig;
use crossbeam_channel::{Receiver, RecvTimeoutError};
use log::{debug, error, info, warn};
use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;

/// The maximum number of metrics to include in a single batch.
const MAX_BATCH_SIZE: usize = 100;
/// The maximum time to wait for a batch to fill up before sending it anyway.
const BATCH_TIMEOUT: Duration = Duration::from_secs(30);

/// Represents all possible errors that can occur within the telemetry module.
#[derive(Debug, Error)]
pub enum TelemetryError {
    /// Failed to serialize metrics data to JSON.
    #[error("Failed to serialize metrics data: {0}")]
    Serialization(#[from] serde_json::Error),

    /// A network error occurred while sending telemetry data.
    #[error("Network request failed: {0}")]
    Network(String),

    /// The remote endpoint returned an error status code.
    #[error("Remote endpoint returned an error: status {0}, body: {1}")]
    EndpointError(u16, String),

    /// The communication channel from the metrics collector is disconnected.
    #[error("Metrics channel is disconnected.")]
    ChannelDisconnected,
}

/// Configuration specific to the telemetry reporter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TelemetryConfig {
    /// The URL of the remote endpoint to which telemetry data will be sent.
    pub endpoint_url: String,
    /// The authentication token to be included in the request headers.
    pub auth_token: String,
}

/// The main entry point for the telemetry reporter thread.
///
/// This function contains the loop that receives metrics, batches them, and
/// sends them. It gracefully handles the batching timeout and shutdown signals.
///
/// # Arguments
///
/// * `config` - The monitoring service configuration.
/// * `receiver` - The channel receiver to get metrics from the collector.
/// * `stop_signal` - An atomic boolean that signals the loop to terminate.
pub(crate) fn run_reporter_loop(
    config: MonitoringConfig,
    receiver: Receiver<MetricData>,
    stop_signal: Arc<AtomicBool>,
) {
    info!("Telemetry reporter thread started.");
    let mut batch: Vec<MetricData> = Vec::with_capacity(MAX_BATCH_SIZE);
    let mut last_send = std::time::Instant::now();

    while !stop_signal.load(Ordering::Relaxed) {
        // Wait for a new metric, but with a timeout so we can check the stop
        // signal and batch timeout periodically.
        match receiver.recv_timeout(Duration::from_millis(500)) {
            Ok(metric) => {
                batch.push(metric);
            }
            Err(RecvTimeoutError::Timeout) => {
                // No new metric, just continue the loop to check other conditions.
            }
            Err(RecvTimeoutError::Disconnected) => {
                error!("Metrics channel disconnected. Shutting down reporter.");
                break;
            }
        }

        let should_send = !batch.is_empty()
            && (batch.len() >= MAX_BATCH_SIZE || last_send.elapsed() >= BATCH_TIMEOUT);

        if should_send {
            if let Err(e) = process_and_send_batch(&mut batch, &config.telemetry) {
                error!("Failed to send telemetry batch: {}", e);
                // In a real implementation, we might implement a retry policy
                // with exponential backoff or save the failed batch to disk.
            }
            batch.clear();
            last_send = std::time::Instant::now();
        }
    }

    // Before shutting down, send any remaining metrics in the batch.
    if !batch.is_empty() {
        info!("Sending final telemetry batch before shutdown...");
        if let Err(e) = process_and_send_batch(&mut batch, &config.telemetry) {
            error!("Failed to send final telemetry batch: {}", e);
        }
    }


    info!("Telemetry reporter thread shutting down.");
}

/// Serializes a batch of metrics and sends them via an HTTP request.
///
/// # Arguments
///
/// * `batch` - A mutable reference to the vector of metrics to be sent.
/// * `config` - The telemetry configuration containing the endpoint and auth info.
fn process_and_send_batch(
    batch: &mut Vec<MetricData>,
    config: &TelemetryConfig,
) -> Result<(), TelemetryError> {
    info!("Sending a batch of {} metrics.", batch.len());

    // Serialize the batch of metrics into a JSON string.
    let json_payload = serde_json::to_string(&batch)?;
    debug!("Serialized payload: {}", json_payload);

    // Send the request.
    send_telemetry_request(&json_payload, config)
}

/// Simulates sending an HTTP POST request to the telemetry endpoint.
///
/// **TODO**: Replace this mock implementation with a real HTTP client like
/// `reqwest`. The use of `std::thread::sleep` simulates network latency.
///
/// # Arguments
///
/// * `payload` - The JSON string to be sent as the request body.
/// * `config` - The telemetry configuration.
fn send_telemetry_request(
    payload: &str,
    config: &TelemetryConfig,
) -> Result<(), TelemetryError> {
    debug!(
        "Simulating HTTP POST to {} with auth token '{}...'",
        config.endpoint_url,
        &config.auth_token.chars().take(8).collect::<String>()
    );

    // Simulate network latency
    std::thread::sleep(Duration::from_millis(
        (rand::random::<f32>() * 1000.0) as u64,
    ));

    // Simulate potential network or endpoint errors
    let random_val = rand::random::<f32>();
    if random_val < 0.1 {
        // Simulate a network failure
        Err(TelemetryError::Network(
            "Failed to resolve host".to_string(),
        ))
    } else if random_val < 0.2 {
        // Simulate a server-side error
        Err(TelemetryError::EndpointError(
            500,
            "Internal Server Error".to_string(),
        ))
    } else {
        // Simulate a successful request
        info!(
            "Successfully sent {} bytes of telemetry data.",
            payload.len()
        );
        Ok(())
    }
}
