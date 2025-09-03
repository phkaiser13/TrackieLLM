/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/monitoring/lib.rs
 *
 * This file serves as the main library entry point for the 'monitoring' crate.
 * It is responsible for aggregating and exposing the public API of its sub-modules,
 * namely 'metrics_collector' and 'telemetry'. The crate provides a comprehensive
 * solution for system health monitoring, performance metrics collection, and
 * telemetry data reporting.
 *
 * The architecture is designed to be modular, extensible, and performant,
 * following best practices for building robust systems in Rust. It emphasizes

 * clear separation of concerns, where 'metrics_collector' handles the raw data
 * gathering from the system (CPU, memory, GPU, etc.), and 'telemetry' is
 * responsible for processing, formatting, and transmitting this data to a
 * designated endpoint.
 *
 * The primary interface is the `MonitoringService`, a high-level struct that
 * orchestrates the entire monitoring lifecycle. It is configured via a
 * `MonitoringConfig` object, allowing for flexible setup and dependency injection.
 * Error handling is managed through a dedicated `MonitoringError` enum, which
 * provides detailed context for potential failures.
 *
 * Dependencies:
 *   - log: For structured logging.
 *   - crossbeam-channel: For high-performance, multi-threaded message passing.
 *   - ffi: The crate's FFI bridge to the C core, used for accessing low-level
 *     system information where necessary.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// === TOC ===
// 1. Crate-level Documentation
// 2. Module Declarations
// 3. Public Prelude
// 4. Core Public Types (Config, Error)
// 5. Main Service Interface (MonitoringService)
// 6. FFI Interface (if applicable)
// =============

#![deny(unsafe_code)]
#![deny(missing_docs)]
#![deny(warnings)]

//! # TrackieLLM Monitoring Crate
//!
//! This crate provides robust, real-time monitoring and telemetry services
//! for the TrackieLLM application. It is engineered for high performance and
//! low overhead, ensuring that monitoring does not negatively impact the
//! application's primary functions.
//!
//! ## Architecture
//!
//! The monitoring system is built upon two core components:
//!
//! - **Metrics Collector**: A background service that periodically samples
//!   system metrics such as CPU load, memory usage, disk I/O, and network
//!   statistics. It can also be extended to collect custom application-specific
//!   metrics.
//!
//! - **Telemetry Reporter**: A component that consumes the collected metrics,
//_   aggregates them, and sends them to a configured telemetry backend. This
//!   could be a logging service, a time-series database like Prometheus, or a
//!   custom analytics endpoint.
//!
//! Communication between these components is handled by high-performance,
//! lock-free channels to minimize contention and latency in multi-threaded
//! environments.
//!
//! ## Usage
//!
//! To use the monitoring service, first create a `MonitoringConfig` instance
//! to define its behavior, such as the collection interval and telemetry endpoint.
//! Then, instantiate the `MonitoringService` and start it.
//!
//! ```rust,ignore
//! use std::time::Duration;
//! use monitoring::{MonitoringConfig, MonitoringService, TelemetryConfig};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = MonitoringConfig {
//!         collection_interval: Duration::from_secs(5),
//!         telemetry: TelemetryConfig {
//!             endpoint_url: "https://my-telemetry-server.com/api".to_string(),
//!             auth_token: "secret-token".to_string(),
//!         },
//!     };
//!
//!     let mut monitoring_service = MonitoringService::new(config)?;
//!
//!     // Start the monitoring threads
//!     monitoring_service.start()?;
//!
//!     // The service runs in the background. Your application's main logic
//!     // would continue here.
//!     std::thread::sleep(Duration::from_secs(60));
//!
//!     // Stop the service gracefully
//!     monitoring_service.stop()?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Error Handling
//!
//! All fallible operations within this crate return a `Result<T, MonitoringError>`,
//! providing detailed information about the cause of the failure. This allows

//! consumers of the crate to implement robust error handling and recovery logic.

// --- Module Declarations ---
// These modules contain the core logic for the crate. They are kept private
// to encapsulate their implementation details, and their public APIs are
// exposed selectively through this top-level module.

/// Handles the collection of system and application metrics.
pub mod metrics_collector;

/// Responsible for processing and reporting telemetry data.
pub mod telemetry;


// --- Public Prelude ---
// The prelude module provides a convenient way to import the most commonly
// used types from this crate.

pub mod prelude {
    //! A "prelude" for users of the `monitoring` crate.
    //!
    //! This module contains the most commonly used types, traits, and functions,
    //! making them easy to import with a single `use` statement.
    pub use super::{
        MonitoringConfig,
        MonitoringService,
        MonitoringError,
        telemetry::TelemetryConfig
    };
}


// --- Core Public Types ---

use std::time::Duration;
use thiserror::Error;
use crate::telemetry::TelemetryConfig;

/// Configuration for the `MonitoringService`.
///
/// This struct holds all the necessary settings to initialize and run the
/// monitoring services. It is designed to be constructed once at application
//_ startup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MonitoringConfig {
    /// The interval at which system metrics should be collected.
    /// A shorter duration provides more granular data but increases overhead.
    pub collection_interval: Duration,

    /// Configuration for the telemetry reporting sub-system.
    pub telemetry: TelemetryConfig,
}

/// Represents all possible errors that can occur within the monitoring crate.
///
/// This enum is designed to be comprehensive and provide clear, actionable
/// error information. It implements `std::error::Error` to be compatible
/// with Rust's standard error handling mechanisms.
#[derive(Debug, Error)]
pub enum MonitoringError {
    /// Indicates that the service has not been initialized before being used.
    #[error("Service not initialized. Please call `new()` before starting.")]
    NotInitialized,

    /// Indicates that the service has already been started and cannot be started again.
    #[error("Service is already running.")]
    AlreadyRunning,

    /// An error occurred during the configuration of the service.
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// An error originating from the metrics collection sub-system.
    #[error("Metrics collection failed: {0}")]
    MetricsCollection(#[from] metrics_collector::MetricsError),

    /// An error originating from the telemetry reporting sub-system.
    #[error("Telemetry reporting failed: {0}")]
    Telemetry(#[from] telemetry::TelemetryError),

    /// An error occurred in a background thread.
    #[error("A background thread has panicked or exited unexpectedly.")]
    ThreadPanic,

    /// An I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}


// --- Main Service Interface ---

/// The main entry point for the monitoring system.
///
/// This service orchestrates the metrics collection and telemetry reporting
/// in background threads. It provides a simple `start` and `stop` interface
/// for managing the lifecycle of the monitoring tasks.
pub struct MonitoringService {
    /// The configuration for the service.
    config: MonitoringConfig,

    /// A handle to the metrics collector thread.
    /// `Option` is used to allow for graceful shutdown by taking ownership of the handle.
    collector_handle: Option<std::thread::JoinHandle<()>>,

    /// A handle to the telemetry reporter thread.
    reporter_handle: Option<std::thread::JoinHandle<()>>,

    /// A flag to signal the background threads to stop.
    /// `Arc<AtomicBool>` allows for safe, shared access across threads.
    stop_signal: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl MonitoringService {
    /// Creates a new `MonitoringService` instance with the given configuration.
    ///
    /// This function validates the configuration but does not start any background
    /// processes. The `start` method must be called to begin monitoring.
    ///
    /// # Arguments
    ///
    /// * `config` - The `MonitoringConfig` to use for this service instance.
    ///
    /// # Errors
    ///
    /// Returns `MonitoringError::Configuration` if the provided configuration
    /// is invalid.
    pub fn new(config: MonitoringConfig) -> Result<Self, MonitoringError> {
        // Validate the configuration
        if config.collection_interval.as_millis() == 0 {
            return Err(MonitoringError::Configuration(
                "Collection interval must be greater than zero.".to_string(),
            ));
        }

        Ok(Self {
            config,
            collector_handle: None,
            reporter_handle: None,
            stop_signal: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }

    /// Starts the monitoring service.
    ///
    /// This method spawns two background threads:
    /// 1. The metrics collector thread.
    /// 2. The telemetry reporter thread.
    ///
    /// These threads will run until the `stop` method is called.
    ///
    /// # Errors
    ///
    /// * `MonitoringError::AlreadyRunning` if the service has already been started.
    pub fn start(&mut self) -> Result<(), MonitoringError> {
        if self.collector_handle.is_some() || self.reporter_handle.is_some() {
            return Err(MonitoringError::AlreadyRunning);
        }

        log::info!("Starting monitoring service...");

        // Create a channel for communication between the collector and reporter.
        // The channel is bounded to prevent unbounded memory growth if the
        // reporter falls behind.
        let (sender, receiver) = crossbeam_channel::bounded(1024);

        // --- Start the metrics collector thread ---
        let collector_config = self.config.clone();
        let collector_stop_signal = self.stop_signal.clone();
        let collector_handle = std::thread::spawn(move || {
            metrics_collector::run_collector_loop(
                collector_config,
                sender,
                collector_stop_signal,
            );
        });
        self.collector_handle = Some(collector_handle);

        // --- Start the telemetry reporter thread ---
        let reporter_config = self.config.clone();
        let reporter_stop_signal = self.stop_signal.clone();
        let reporter_handle = std::thread::spawn(move || {
            telemetry::run_reporter_loop(
                reporter_config,
                receiver,
                reporter_stop_signal,
            );
        });
        self.reporter_handle = Some(reporter_handle);

        log::info!("Monitoring service started successfully.");
        Ok(())
    }

    /// Stops the monitoring service gracefully.
    ///
    /// This method signals the background threads to stop and waits for them
    /// to finish their current work and exit.
    ///
    /// # Errors
    ///
    /// * `MonitoringError::ThreadPanic` if one of the background threads panicked.
    pub fn stop(&mut self) -> Result<(), MonitoringError> {
        log::info!("Stopping monitoring service...");

        // Signal threads to stop
        self.stop_signal.store(true, std::sync::atomic::Ordering::Relaxed);

        // Wait for the collector thread to finish
        if let Some(handle) = self.collector_handle.take() {
            if handle.join().is_err() {
                log::error!("Metrics collector thread panicked during shutdown.");
                return Err(MonitoringError::ThreadPanic);
            }
        }

        // Wait for the reporter thread to finish
        if let Some(handle) = self.reporter_handle.take() {
            if handle.join().is_err() {
                log::error!("Telemetry reporter thread panicked during shutdown.");
                return Err(MonitoringError::ThreadPanic);
            }
        }

        log::info!("Monitoring service stopped successfully.");
        Ok(())
    }

    /// Checks if the monitoring service is currently running.
    pub fn is_running(&self) -> bool {
        self.collector_handle.is_some() && self.reporter_handle.is_some()
    }
}

/// Implementation of the `Drop` trait for `MonitoringService`.
///
/// This ensures that if the `MonitoringService` instance goes out of scope,
/// it will attempt to shut down the background threads gracefully. This is a
/// fail-safe mechanism to prevent orphaned threads.
impl Drop for MonitoringService {
    fn drop(&mut self) {
        if self.is_running() {
            log::warn!("MonitoringService dropped without being explicitly stopped. Attempting graceful shutdown.");
            if let Err(e) = self.stop() {
                log::error!("Failed to stop MonitoringService during drop: {}", e);
            }
        }
    }
}
