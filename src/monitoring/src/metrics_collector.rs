/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/monitoring/metrics_collector.rs
 *
 * This file implements the core logic for collecting system and application
 * metrics. It defines the data structures for various metrics (e.g., CPU,
 * memory, disk) and the main collector loop that runs in a background thread.
 *
 * The `MetricsCollector` is designed for low-overhead sampling. It wakes up
 * periodically, gathers a snapshot of the system's state, packages it into a
 * `SystemMetrics` struct, and sends it over a channel for further processing
 * by the telemetry module.
 *
 * The actual data retrieval is abstracted away. In a real implementation, the
 * `gather_*` functions would make FFI calls to the C core or use platform-
 * specific APIs (e.g., /proc on Linux, kstat on Solaris, etc.) to get real
 * data. For now, they generate plausible mock data. This approach allows the
 * core logic and data structures to be developed and tested independently of
 * the underlying data sources.
 *
 * Dependencies:
 *   - crossbeam-channel: For sending collected metrics to the reporter.
 *   - log: For structured logging.
 *   - thiserror: For ergonomic error handling.
 *   - crate::MonitoringConfig: For configuration.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// === TOC ===
// 1. Module-level Imports and Constants
// 2. Error Type for this Module
// 3. Public Metric Data Structures
//    - SystemMetrics
//    - CpuMetrics
//    - MemoryMetrics
//    - DiskMetrics
//    - NetworkMetrics
// 4. The Collector Logic
//    - run_collector_loop (main entry point)
//    - gather_system_metrics (orchestrator)
//    - gather_* (specific metric collectors)
// =============

use crate::MonitoringConfig;
use crossbeam_channel::Sender;
use log::{debug, error, warn};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// The type of metric data sent over the channel.
pub type MetricData = SystemMetrics;

/// Represents all possible errors that can occur within the metrics collector.
#[derive(Debug, Error)]
pub enum MetricsError {
    /// An error occurred while gathering CPU metrics.
    #[error("Failed to collect CPU metrics: {0}")]
    CpuCollection(String),

    /// An error occurred while gathering Memory metrics.
    #[error("Failed to collect Memory metrics: {0}")]
    MemoryCollection(String),

    /// An error occurred while gathering Disk metrics.
    #[error("Failed to collect Disk metrics: {0}")]
    DiskCollection(String),

    /// An error occurred while gathering Network metrics.
    #[error("Failed to collect Network metrics: {0}")]
    NetworkCollection(String),

    /// The communication channel to the telemetry reporter is disconnected.
    #[error("Telemetry channel is disconnected.")]
    ChannelDisconnected,
}

// --- Public Metric Data Structures ---

/// A comprehensive snapshot of system metrics at a single point in time.
///
/// This struct aggregates various metric types into a single package for easy
/// transport and processing.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemMetrics {
    /// The timestamp when the metrics were collected.
    pub timestamp: Instant,
    /// Metrics related to CPU usage.
    pub cpu: CpuMetrics,
    /// Metrics related to memory usage.
    pub memory: MemoryMetrics,
    /// A vector of metrics for each detected disk.
    pub disks: Vec<DiskMetrics>,
    /// A vector of metrics for each detected network interface.
    pub networks: Vec<NetworkMetrics>,
}

/// Represents CPU-related metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct CpuMetrics {
    /// The average CPU load over the last sampling period, as a percentage.
    pub load_avg: f32,
    /// The number of logical cores.
    pub core_count: u32,
    /// Per-core load averages. The length of this vector should match `core_count`.
    pub per_core_load: Vec<f32>,
}

/// Represents memory-related metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryMetrics {
    /// Total physical memory in megabytes.
    pub total_mb: u64,
    /// Used physical memory in megabytes.
    pub used_mb: u64,
    /// Free physical memory in megabytes.
    pub free_mb: u64,
    /// Total swap space in megabytes.
    pub swap_total_mb: u64,
    /// Used swap space in megabytes.
    pub swap_used_mb: u64,
}

/// Represents metrics for a single disk or mount point.
#[derive(Debug, Clone, PartialEq)]
pub struct DiskMetrics {
    /// The name of the disk or mount point (e.g., "/dev/sda1", "/mnt/data").
    pub name: String,
    /// Total disk space in gigabytes.
    pub total_gb: u64,
    /// Used disk space in gigabytes.
    pub used_gb: u64,
    /// Read operations per second since the last sample.
    pub reads_per_sec: f32,
    /// Write operations per second since the last sample.
    pub writes_per_sec: f32,
}

/// Represents metrics for a single network interface.
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkMetrics {
    /// The name of the network interface (e.g., "eth0").
    pub name: String,
    /// Megabits per second received since the last sample.
    pub mbps_in: f32,
    /// Megabits per second sent since the last sample.
    pub mbps_out: f32,
    /// Network packets received per second.
    pub packets_in_per_sec: u64,
    /// Network packets sent per second.
    pub packets_out_per_sec: u64,
}

// --- The Collector Logic ---

/// The main entry point for the metrics collector thread.
///
/// This function contains the primary loop that drives the metric collection
/// process. It wakes up at the interval specified in the `config`, calls
/// `gather_system_metrics` to perform the collection, and then sends the
/// resulting data to the telemetry reporter via the provided channel.
///
/// # Arguments
///
/// * `config` - The monitoring service configuration.
/// * `sender` - The channel sender to transmit collected metrics.
/// * `stop_signal` - An atomic boolean that signals the loop to terminate.
pub(crate) fn run_collector_loop(
    config: MonitoringConfig,
    sender: Sender<MetricData>,
    stop_signal: Arc<AtomicBool>,
) {
    log::info!("Metrics collector thread started.");

    while !stop_signal.load(Ordering::Relaxed) {
        let collection_start = Instant::now();

        match gather_system_metrics() {
            Ok(metrics) => {
                debug!("Successfully gathered system metrics.");
                if let Err(_) = sender.send(metrics) {
                    warn!("Failed to send metrics to reporter; channel may be closed. Shutting down collector.");
                    break;
                }
            }
            Err(e) => {
                error!("Failed to gather system metrics: {}", e);
                // In a real-world scenario, we might want more sophisticated
                // error handling, like exponential backoff on certain errors.
            }
        }

        // Wait for the next collection cycle, accounting for the time spent
        // on the collection itself to maintain a consistent interval.
        let elapsed = collection_start.elapsed();
        if let Some(delay) = config.collection_interval.checked_sub(elapsed) {
            // Use a polling sleep to be responsive to the stop signal.
            let sleep_deadline = Instant::now() + delay;
            while Instant::now() < sleep_deadline {
                if stop_signal.load(Ordering::Relaxed) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }

    log::info!("Metrics collector thread shutting down.");
}

/// Orchestrates the collection of all system metrics.
///
/// This function calls the individual `gather_*` functions for each metric
/// type and aggregates them into a single `SystemMetrics` struct.
///
/// # Errors
///
/// Returns a `MetricsError` if any of the sub-collection functions fail.
fn gather_system_metrics() -> Result<SystemMetrics, MetricsError> {
    let timestamp = Instant::now();

    // In a parallel universe, we could run these concurrently if they are
    // independent and time-consuming. For now, sequential is fine.
    let cpu_metrics = gather_cpu_metrics()?;
    let memory_metrics = gather_memory_metrics()?;
    let disk_metrics = gather_disk_metrics()?;
    let network_metrics = gather_network_metrics()?;

    Ok(SystemMetrics {
        timestamp,
        cpu: cpu_metrics,
        memory: memory_metrics,
        disks: disk_metrics,
        networks: network_metrics,
    }
)
}

// --- Specific Metric Collectors (Mock Implementations) ---

/// Gathers CPU metrics.
///
/// **TODO**: Replace this mock implementation with actual FFI calls to the
/// C core or direct system API calls to get real CPU load data.
fn gather_cpu_metrics() -> Result<CpuMetrics, MetricsError> {
    // MOCK IMPLEMENTATION
    let core_count = 8;
    let per_core_load: Vec<f32> = (0..core_count)
        .map(|_| rand::random::<f32>() * 100.0)
        .collect();
    let load_avg = per_core_load.iter().sum::<f32>() / core_count as f32;

    Ok(CpuMetrics {
        load_avg,
        core_count,
        per_core_load,
    })
}

/// Gathers memory metrics.
///
/// **TODO**: Replace this mock implementation with actual data retrieval.
fn gather_memory_metrics() -> Result<MemoryMetrics, MetricsError> {
    // MOCK IMPLEMENTATION
    let total_mb = 32 * 1024; // 32 GB
    let used_mb = (rand::random::<f32>() * total_mb as f32) as u64;
    let free_mb = total_mb - used_mb;

    Ok(MemoryMetrics {
        total_mb,
        used_mb,
        free_mb,
        swap_total_mb: 16 * 1024,
        swap_used_mb: (rand::random::<f32>() * 1024.0) as u64,
    })
}

/// Gathers disk I/O and usage metrics.
///
/// **TODO**: Replace this mock implementation with a mechanism to discover
/// attached disks and query their statistics.
fn gather_disk_metrics() -> Result<Vec<DiskMetrics>, MetricsError> {
    // MOCK IMPLEMENTATION
    let disk1 = DiskMetrics {
        name: "/dev/sda1".to_string(),
        total_gb: 512,
        used_gb: (rand::random::<f32>() * 400.0) as u64,
        reads_per_sec: rand::random::<f32>() * 100.0,
        writes_per_sec: rand::random::<f32>() * 50.0,
    };

    let disk2 = DiskMetrics {
        name: "/mnt/data".to_string(),
        total_gb: 2048,
        used_gb: (rand::random::<f32>() * 1500.0) as u64,
        reads_per_sec: rand::random::<f32>() * 250.0,
        writes_per_sec: rand::random::<f32>() * 300.0,
    };

    Ok(vec![disk1, disk2])
}

/// Gathers network I/O metrics.
///
/// **TODO**: Replace this mock implementation with a mechanism to discover
/// active network interfaces and query their statistics.
fn gather_network_metrics() -> Result<Vec<NetworkMetrics>, MetricsError> {
    // MOCK IMPLEMENTATION
    let eth0 = NetworkMetrics {
        name: "eth0".to_string(),
        mbps_in: rand::random::<f32>() * 1000.0, // Simulating up to 1 Gbit/s
        mbps_out: rand::random::<f32>() * 500.0,
        packets_in_per_sec: (rand::random::<f32>() * 100_000.0) as u64,
        packets_out_per_sec: (rand::random::<f32>() * 80_000.0) as u64,
    };

    Ok(vec![eth0])
}
