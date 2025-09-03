/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: metrics_collector.rs
 *
 * This file implements the core logic for collecting system and application-specific
 * metrics. It is designed for high performance and minimal overhead, operating on a
 * dedicated background thread to prevent blocking the main application threads.
 * The collected data is sent to the telemetry module for processing and exportation.
 *
 * Dependencies:
 *  - `sysinfo`: For gathering detailed system information (CPU, memory, disk, etc.).
 *  - `crossbeam-channel`: For high-performance, lock-free communication between threads.
 *  - `serde`: For serialization of metric data structures.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Imports ---
use sysinfo::{System, Cpu, Pid, Process};
use crossbeam_channel::{Sender, Receiver, unbounded};
use serde::{Serialize, Deserialize};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

// --- Public Data Structures ---

/// Represents a snapshot of system-wide performance metrics.
/// This struct is designed to be serializable and sent over the network.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SystemMetrics {
    /// Timestamp of the collection in UNIX epoch seconds.
    pub timestamp: u64,
    /// Overall CPU usage as a percentage.
    pub cpu_global_usage: f32,
    /// Per-core CPU usage.
    pub cpu_per_core_usage: Vec<f32>,
    /// Total system memory in kilobytes.
    pub memory_total_kb: u64,
    /// Used system memory in kilobytes.
    pub memory_used_kb: u64,
    /// Total swap space in kilobytes.
    pub swap_total_kb: u64,
    /// Used swap space in kilobytes.
    pub swap_used_kb: u64,
    /// Metrics specific to the current process.
    pub process_metrics: Option<ProcessMetrics>,
}

/// Represents performance metrics for the running process.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProcessMetrics {
    /// The process ID.
    pub pid: u32,
    /// CPU usage of the process as a percentage.
    pub cpu_usage: f32,
    /// Memory usage of the process in kilobytes.
    pub memory_usage_kb: u64,
    /// Disk usage of the process.
    pub disk_usage: DiskUsage,
}

/// Represents the disk I/O of a process.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct DiskUsage {
    /// Total bytes read.
    pub total_read_bytes: u64,
    /// Total bytes written.
    pub total_written_bytes: u64,
    /// Bytes read since the last collection.
    pub read_bytes_delta: u64,
    /// Bytes written since the last collection.
    pub written_bytes_delta: u64,
}

/// Represents a custom, application-defined event or data point.
/// The payload is expected to be a JSON-formatted string for flexibility.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CustomData {
    /// The name identifying the custom event (e.g., "user_login_success").
    pub name: String,
    /// A flexible payload, typically a JSON string, containing event-specific details.
    pub payload: String,
}

/// Internal command enum for controlling the collector's background thread.
/// This demonstrates the Command-Query Separation principle for thread control.
#[derive(Debug)]
enum CollectorCommand {
    /// Command to gracefully shut down the collection thread.
    Shutdown,
    /// Command to trigger an immediate, on-demand metrics collection.
    CollectNow,
}

// --- Core Metrics Collector Engine ---

/// The central struct managing the metrics collection lifecycle.
/// It encapsulates the background thread, communication channels, and shared state.
/// It implements `Drop` to ensure graceful shutdown via RAII.
pub struct MetricsCollector {
    /// Handle to the background collection thread.
    /// It's an `Option` so we can `take` it during shutdown.
    collection_thread: Option<JoinHandle<()>>,

    /// Sender for dispatching commands to the background thread.
    command_sender: Sender<CollectorCommand>,

    /// A shared flag to signal the running state of the thread.
    /// `AtomicBool` is used for safe, lock-free communication.
    is_running: Arc<AtomicBool>,
}

impl MetricsCollector {
    /// Creates a new `MetricsCollector` and starts the background collection thread.
    ///
    /// # Arguments
    ///
    /// * `collection_interval` - The interval at which system metrics are periodically collected.
    /// * `metrics_sender` - A channel sender to forward the collected `SystemMetrics`.
    ///
    /// # Returns
    ///
    /// A `Result` containing the new `MetricsCollector` instance or an error string.
    pub fn new(
        collection_interval: Duration,
        metrics_sender: Sender<SystemMetrics>,
    ) -> Result<Self, String> {

        let (command_sender, command_receiver) = unbounded::<CollectorCommand>();
        let is_running = Arc::new(AtomicBool::new(true));
        let thread_running_flag = Arc::clone(&is_running);

        let thread_handle = thread::Builder::new()
            .name("metrics_collector_thread".to_string())
            .spawn(move || {
                // This is the main loop of the background thread.
                // It listens for commands and periodically collects metrics.
                Self::collection_loop(
                    collection_interval,
                    thread_running_flag,
                    command_receiver,
                    metrics_sender,
                );
            })
            .map_err(|e| format!("Failed to spawn metrics collector thread: {}", e))?;

        println!("Metrics collector background thread started.");

        Ok(MetricsCollector {
            collection_thread: Some(thread_handle),
            command_sender,
            is_running,
        })
    }

    /// The main loop for the background collection thread.
    /// This function contains the core logic for timed and on-demand metric collection.
    fn collection_loop(
        interval: Duration,
        is_running: Arc<AtomicBool>,
        command_receiver: Receiver<CollectorCommand>,
        metrics_sender: Sender<SystemMetrics>,
    ) {
        let mut sys = System::new_all();
        let own_pid = Pid::from_u32(std::process::id());
        let mut last_disk_usage = DiskUsage::default();
        let mut last_collection_time = Instant::now();

        println!("Starting metrics collection loop with interval: {:?}", interval);

        while is_running.load(Ordering::Relaxed) {
            // Wait for a command or a timeout.
            // `recv_timeout` is perfect for this use case, allowing periodic work
            // while remaining responsive to commands.
            match command_receiver.recv_timeout(interval) {
                Ok(CollectorCommand::Shutdown) => {
                    println!("Shutdown command received. Exiting collection loop.");
                    break; // Exit the loop to terminate the thread.
                }
                Ok(CollectorCommand::CollectNow) => {
                    println!("On-demand metrics collection triggered.");
                    // Fall through to collection logic.
                }
                Err(_) => {
                    // Timeout occurred, which is our signal for a periodic collection.
                    // This is the normal execution path.
                }
            }

            let now = Instant::now();
            if now.duration_since(last_collection_time) >= interval {
                sys.refresh_all(); // Refresh system data.

                let metrics = Self::perform_collection(&sys, own_pid, &mut last_disk_usage);

                if let Err(e) = metrics_sender.send(metrics) {
                    eprintln!("Failed to send metrics to telemetry channel: {}. Channel might be closed.", e);
                    // If the receiver is dropped, we can no longer send data,
                    // so we should probably exit.
                    is_running.store(false, Ordering::Relaxed);
                }
                last_collection_time = now;
            }
        }
        println!("Metrics collection loop has finished.");
    }

    /// Performs the actual data collection from the system.
    /// This function is designed to be called from within the `collection_loop`.
    ///
    /// # Arguments
    ///
    /// * `sys` - A mutable reference to the `sysinfo::System` object.
    /// * `pid` - The process ID of the application to monitor.
    /// * `last_disk_usage` - A mutable reference to track disk I/O deltas.
    ///
    /// # Returns
    ///
    /// A `SystemMetrics` struct populated with fresh data.
    fn perform_collection(sys: &System, pid: Pid, last_disk_usage: &mut DiskUsage) -> SystemMetrics {
        // --- Collect Process-Specific Metrics ---
        let process_metrics = if let Some(process) = sys.process(pid) {
            let current_disk_usage = DiskUsage {
                total_read_bytes: process.disk_usage().total_read_bytes(),
                total_written_bytes: process.disk_usage().total_written_bytes(),
                read_bytes_delta: process.disk_usage().total_read_bytes().saturating_sub(last_disk_usage.total_read_bytes),
                written_bytes_delta: process.disk_usage().total_written_bytes().saturating_sub(last_disk_usage.total_written_bytes),
            };
            *last_disk_usage = current_disk_usage.clone();

            Some(ProcessMetrics {
                pid: pid.as_u32(),
                cpu_usage: process.cpu_usage(),
                memory_usage_kb: process.memory(),
                disk_usage: current_disk_usage,
            })
        } else {
            None
        };

        // --- Collect Global System Metrics ---
        let cpu_per_core_usage = sys.cpus().iter().map(Cpu::cpu_usage).collect();

        SystemMetrics {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            cpu_global_usage: sys.global_cpu_info().cpu_usage(),
            cpu_per_core_usage,
            memory_total_kb: sys.total_memory(),
            memory_used_kb: sys.used_memory(),
            swap_total_kb: sys.total_swap(),
            swap_used_kb: sys.used_swap(),
            process_metrics,
        }
    }

    /// Triggers an immediate, on-demand metrics collection.
    pub fn trigger_collection(&self) {
        if let Err(e) = self.command_sender.send(CollectorCommand::CollectNow) {
            eprintln!("Failed to send CollectNow command: {}", e);
        }
    }
}

/// The `Drop` implementation ensures that the background thread is
/// gracefully shut down when the `MetricsCollector` instance goes out of scope.
/// This is a core tenet of the RAII (Resource Acquisition Is Initialization) pattern in Rust.
impl Drop for MetricsCollector {
    fn drop(&mut self) {
        println!("Shutting down metrics collector...");

        // Signal the thread to stop.
        self.is_running.store(false, Ordering::Relaxed);

        // Send a final Shutdown command to unblock the `recv_timeout` call immediately.
        if let Err(e) = self.command_sender.send(CollectorCommand::Shutdown) {
            eprintln!("Failed to send shutdown command to collector thread: {}", e);
        }

        // Wait for the thread to finish its work.
        if let Some(handle) = self.collection_thread.take() {
            if let Err(e) = handle.join() {
                eprintln!("Error joining metrics collector thread: {:?}", e);
            }
        }
        println!("Metrics collector shut down successfully.");
    }
}

// --- Unit Tests ---
// Tests are crucial for verifying the correctness of complex, concurrent code.
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Tests the creation and graceful shutdown of the `MetricsCollector`.
    #[test]
    fn test_collector_creation_and_shutdown() {
        let (tx, _rx) = unbounded();
        let collector = MetricsCollector::new(Duration::from_secs(1), tx)
            .expect("Failed to create collector");
        // The collector is dropped here, triggering the shutdown logic in `Drop`.
        // The test passes if it doesn't panic.
    }

    /// Tests if the collector can receive and process metrics.
    #[test]
    fn test_metric_collection_and_reception() {
        let (tx, rx) = unbounded();
        let collector = MetricsCollector::new(Duration::from_millis(100), tx)
            .expect("Failed to create collector");

        // Wait for at least one metric collection to occur.
        match rx.recv_timeout(Duration::from_secs(1)) {
            Ok(metrics) => {
                assert!(metrics.timestamp > 0);
                assert!(metrics.memory_total_kb > 0);
                assert!(metrics.process_metrics.is_some());
                let process_metrics = metrics.process_metrics.unwrap();
                assert_eq!(process_metrics.pid, std::process::id());
            }
            Err(e) => {
                panic!("Did not receive metrics within the expected time: {}", e);
            }
        }
    }

    /// Tests the on-demand collection trigger.
    #[test]
    fn test_on_demand_collection() {
        let (tx, rx) = unbounded();
        // Use a long interval to ensure collection only happens on demand.
        let collector = MetricsCollector::new(Duration::from_secs(60), tx)
            .expect("Failed to create collector");

        collector.trigger_collection();

        match rx.recv_timeout(Duration::from_secs(2)) {
            Ok(metrics) => {
                println!("Received on-demand metrics: {:?}", metrics);
                assert!(metrics.timestamp > 0);
            }
            Err(e) => {
                panic!("Did not receive on-demand metrics: {}", e);
            }
        }
    }
}
