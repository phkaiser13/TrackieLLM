/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: telemetry.rs
 *
 * This file provides the telemetry exportation layer. It receives metrics from
 * the collector, batches them, and exports them to a configurable backend over
 * the network. The architecture is designed to be non-blocking, resilient, and
 * extensible to support various observability platforms.
 *
 * Dependencies:
 *  - `tokio`: For the asynchronous runtime to handle network I/O without blocking.
 *  - `reqwest`: A high-level async HTTP client for sending data.
 *  - `crossbeam-channel`: For receiving data from the collector.
 *  - `serde`, `serde_json`: For serializing data to JSON.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Imports ---
use crate::metrics_collector::{SystemMetrics, CustomData};
use crossbeam_channel::Receiver;
use serde::Serialize;
use std::thread::{self, JoinHandle};
use std::time::Duration;
use tokio::runtime::{self, Runtime};
use tokio::time::sleep;

// --- Trait for Backend Extensibility ---

/// Defines a common interface for different telemetry backends.
/// This trait-based design allows for swapping out the export destination
/// (e.g., Prometheus, OpenTelemetry, a custom HTTP endpoint) without changing
/// the core telemetry logic. This is an example of the Strategy pattern.
pub trait TelemetryBackend: Send + Sync + 'static {
    /// Exports a batch of system metrics.
    /// This method must be implemented by concrete backend types.
    fn export_system_metrics(&self, metrics: Vec<SystemMetrics>) -> impl std::future::Future<Output = Result<(), String>> + Send;

    /// Exports a batch of custom data points.
    fn export_custom_data(&self, data: Vec<CustomData>) -> impl std::future::Future<Output = Result<(), String>> + Send;
}

// --- Default HTTP Backend Implementation ---

/// A concrete implementation of `TelemetryBackend` that sends data via HTTP POST
/// to a specified endpoint. This is a common and flexible exportation strategy.
pub struct HttpBackend {
    http_client: reqwest::Client,
    endpoint_url: String,
    api_key: String, // Note: In production, use a secure way to handle secrets.
}

impl HttpBackend {
    /// Creates a new `HttpBackend`.
    pub fn new(endpoint_url: String, api_key: String) -> Self {
        HttpBackend {
            http_client: reqwest::Client::new(),
            endpoint_url,
            api_key,
        }
    }

    /// A helper function to post serialized data to the configured endpoint.
    /// It includes retry logic with exponential backoff for resilience.
    async fn post_data<T: Serialize>(&self, payload: &T, path: &str) -> Result<(), String> {
        let target_url = format!("{}/{}", self.endpoint_url, path);
        let mut attempts = 0;
        let max_attempts = 5;
        let base_delay = Duration::from_millis(500);

        loop {
            attempts += 1;
            let request = self.http_client.post(&target_url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .json(payload);

            match request.send().await {
                Ok(response) if response.status().is_success() => {
                    println!("Successfully exported data to {}", target_url);
                    return Ok(());
                }
                Ok(response) => {
                    let status = response.status();
                    let error_text = response.text().await.unwrap_or_else(|_| "Could not read error body".to_string());
                    let err_msg = format!("Failed to export data to {}. Status: {}. Body: {}", target_url, status, error_text);
                    if attempts >= max_attempts {
                        return Err(err_msg);
                    }
                    eprintln!("{}", err_msg);
                }
                Err(e) => {
                    let err_msg = format!("HTTP request failed for {}: {}", target_url, e);
                    if attempts >= max_attempts {
                        return Err(err_msg);
                    }
                    eprintln!("{}", err_msg);
                }
            }

            // Exponential backoff logic
            let delay = base_delay * 2_u32.pow(attempts - 1);
            println!("Retrying in {:?}...", delay);
            sleep(delay).await;
        }
    }
}

impl TelemetryBackend for HttpBackend {
    /// Asynchronously exports a batch of system metrics.
    async fn export_system_metrics(&self, metrics: Vec<SystemMetrics>) -> Result<(), String> {
        self.post_data(&metrics, "system-metrics").await
    }

    /// Asynchronously exports a batch of custom data.
    async fn export_custom_data(&self, data: Vec<CustomData>) -> Result<(), String> {
        self.post_data(&data, "custom-data").await
    }
}


// --- Telemetry Exporter Engine ---

/// The `TelemetryExporter` manages the lifecycle of receiving and exporting data.
/// It uses a dedicated background thread with an async runtime for non-blocking I/O.
pub struct TelemetryExporter<B: TelemetryBackend> {
    export_thread: Option<JoinHandle<()>>,
    // We don't need a command channel here if the logic is simple,
    // but we need a way to signal shutdown. Dropping the data senders works for this.
}

impl<B: TelemetryBackend> TelemetryExporter<B> {
    /// Creates a new `TelemetryExporter` and starts its background processing thread.
    ///
    /// # Arguments
    ///
    /// * `backend` - A concrete implementation of the `TelemetryBackend` trait.
    /// * `system_metrics_receiver` - Channel to receive `SystemMetrics`.
    /// * `custom_data_receiver` - Channel to receive `CustomData`.
    /// * `batch_size` - The number of items to collect before forcing an export.
    /// * `batch_timeout` - The maximum time to wait before exporting an incomplete batch.
    ///
    /// # Returns
    ///
    /// A new instance of `TelemetryExporter`.
    pub fn new(
        backend: B,
        system_metrics_receiver: Receiver<SystemMetrics>,
        custom_data_receiver: Receiver<CustomData>,
        batch_size: usize,
        batch_timeout: Duration,
    ) -> Self {

        let export_thread = thread::Builder::new()
            .name("telemetry_exporter_thread".to_string())
            .spawn(move || {
                // Create a Tokio runtime within the thread.
                let runtime = runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("Failed to create Tokio runtime for telemetry");

                // Enter the async context.
                runtime.block_on(Self::exporter_loop(
                    backend,
                    system_metrics_receiver,
                    custom_data_receiver,
                    batch_size,
                    batch_timeout,
                ));
            })
            .expect("Failed to spawn telemetry exporter thread");

        TelemetryExporter {
            export_thread: Some(export_thread),
        }
    }

    /// The main async loop for the background exporter thread.
    /// It multiplexes between different data channels and timers to implement batching.
    async fn exporter_loop(
        backend: B,
        system_metrics_rx: Receiver<SystemMetrics>,
        custom_data_rx: Receiver<CustomData>,
        batch_size: usize,
        batch_timeout: Duration,
    ) {
        let mut system_metrics_batch = Vec::with_capacity(batch_size);
        let mut custom_data_batch = Vec::with_capacity(batch_size);

        let timeout = sleep(batch_timeout);
        tokio::pin!(timeout);

        println!("Starting telemetry exporter loop. Batch size: {}, Timeout: {:?}", batch_size, batch_timeout);

        loop {
            tokio::select! {
                // Biased select ensures we drain channels before checking timeout.
                biased;

                // --- Receive System Metrics ---
                maybe_metric = tokio::task::spawn_blocking(move || system_metrics_rx.recv()) => {
                    match maybe_metric {
                        Ok(Ok(metric)) => {
                            system_metrics_batch.push(metric);
                            if system_metrics_batch.len() >= batch_size {
                                println!("System metrics batch is full. Exporting...");
                                Self::flush_system_metrics(&backend, &mut system_metrics_batch).await;
                                timeout.as_mut().reset(tokio::time::Instant::now() + batch_timeout);
                            }
                        },
                        _ => { // Channel disconnected or error
                            println!("System metrics channel closed. Exporter will shut down after flushing.");
                            Self::flush_all(&backend, &mut system_metrics_batch, &mut custom_data_batch).await;
                            return;
                        }
                    }
                },

                // --- Receive Custom Data ---
                maybe_data = tokio::task::spawn_blocking(move || custom_data_rx.recv()) => {
                    match maybe_data {
                        Ok(Ok(data)) => {
                            custom_data_batch.push(data);
                            if custom_data_batch.len() >= batch_size {
                                println!("Custom data batch is full. Exporting...");
                                Self::flush_custom_data(&backend, &mut custom_data_batch).await;
                                timeout.as_mut().reset(tokio::time::Instant::now() + batch_timeout);
                            }
                        },
                        _ => { // Channel disconnected or error
                             println!("Custom data channel closed. Exporter will shut down after flushing.");
                             Self::flush_all(&backend, &mut system_metrics_batch, &mut custom_data_batch).await;
                             return;
                        }
                    }
                },

                // --- Batch Timeout ---
                () = &mut timeout => {
                    // Timeout elapsed, flush whatever we have.
                    println!("Batch timeout reached. Flushing all pending data.");
                    Self::flush_all(&backend, &mut system_metrics_batch, &mut custom_data_batch).await;
                    timeout.as_mut().reset(tokio::time::Instant::now() + batch_timeout);
                }
            }
        }
    }

    /// Flushes all pending data to the backend.
    async fn flush_all(backend: &B, system_metrics_batch: &mut Vec<SystemMetrics>, custom_data_batch: &mut Vec<CustomData>) {
        Self::flush_system_metrics(backend, system_metrics_batch).await;
        Self::flush_custom_data(backend, custom_data_batch).await;
    }

    /// Flushes only the system metrics batch.
    async fn flush_system_metrics(backend: &B, batch: &mut Vec<SystemMetrics>) {
        if !batch.is_empty() {
            // `drain(..)` clears the vector and returns an iterator to the items.
            if let Err(e) = backend.export_system_metrics(batch.drain(..).collect()).await {
                eprintln!("Failed to export system metrics batch: {}", e);
                // Note: Here you could implement a dead-letter queue or other strategies
                // to avoid losing data permanently.
            }
        }
    }

    /// Flushes only the custom data batch.
    async fn flush_custom_data(backend: &B, batch: &mut Vec<CustomData>) {
        if !batch.is_empty() {
            if let Err(e) = backend.export_custom_data(batch.drain(..).collect()).await {
                eprintln!("Failed to export custom data batch: {}", e);
            }
        }
    }
}

impl<B: TelemetryBackend> Drop for TelemetryExporter<B> {
    /// Ensures the background thread is joined upon dropping the exporter.
    /// The dropping of the channel senders in the main application logic is the
    /// signal for the exporter loop to terminate.
    fn drop(&mut self) {
        println!("Shutting down telemetry exporter...");
        if let Some(handle) = self.export_thread.take() {
            if let Err(e) = handle.join() {
                eprintln!("Error joining telemetry exporter thread: {:?}", e);
            }
        }
        println!("Telemetry exporter shut down successfully.");
    }
}

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// A mock backend for testing purposes. It doesn't perform network requests
    /// but instead stores the received data in memory for inspection.
    #[derive(Clone)]
    struct MockBackend {
        system_metrics: Arc<Mutex<Vec<SystemMetrics>>>,
        custom_data: Arc<Mutex<Vec<CustomData>>>,
    }

    impl MockBackend {
        fn new() -> Self {
            MockBackend {
                system_metrics: Arc::new(Mutex::new(Vec::new())),
                custom_data: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    impl TelemetryBackend for MockBackend {
        async fn export_system_metrics(&self, metrics: Vec<SystemMetrics>) -> Result<(), String> {
            println!("MockBackend: Received {} system metrics.", metrics.len());
            self.system_metrics.lock().unwrap().extend(metrics);
            Ok(())
        }

        async fn export_custom_data(&self, data: Vec<CustomData>) -> Result<(), String> {
            println!("MockBackend: Received {} custom data points.", data.len());
            self.custom_data.lock().unwrap().extend(data);
            Ok(())
        }
    }

    #[test]
    fn test_batching_by_size() {
        let backend = MockBackend::new();
        let (sm_tx, sm_rx) = unbounded();
        let (cd_tx, cd_rx) = unbounded();

        let _exporter = TelemetryExporter::new(
            backend.clone(),
            sm_rx,
            cd_rx,
            2, // Batch size of 2
            Duration::from_secs(10),
        );

        // Send one item, should not trigger export yet.
        sm_tx.send(SystemMetrics { timestamp: 1, ..Default::default() }).unwrap();

        // Give it a moment to process, but it shouldn't have exported.
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(backend.system_metrics.lock().unwrap().len(), 0);

        // Send a second item, which should fill the batch and trigger an export.
        sm_tx.send(SystemMetrics { timestamp: 2, ..Default::default() }).unwrap();

        // Wait for the export to happen.
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(backend.system_metrics.lock().unwrap().len(), 2);
    }

    #[test]
    fn test_batching_by_timeout() {
        let backend = MockBackend::new();
        let (sm_tx, sm_rx) = unbounded();
        let (cd_tx, cd_rx) = unbounded();

        let _exporter = TelemetryExporter::new(
            backend.clone(),
            sm_rx,
            cd_rx,
            5, // High batch size
            Duration::from_millis(200), // Short timeout
        );

        // Send one item. It won't fill the batch.
        sm_tx.send(SystemMetrics { timestamp: 1, ..Default::default() }).unwrap();

        // Wait for the timeout to trigger the export.
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(backend.system_metrics.lock().unwrap().len(), 1);
    }

    // Helper to create a default SystemMetrics for tests.
    impl Default for SystemMetrics {
        fn default() -> Self {
            SystemMetrics {
                timestamp: 0,
                cpu_global_usage: 0.0,
                cpu_per_core_usage: vec![],
                memory_total_kb: 0,
                memory_used_kb: 0,
                swap_total_kb: 0,
                swap_used_kb: 0,
                process_metrics: None,
            }
        }
    }
}
