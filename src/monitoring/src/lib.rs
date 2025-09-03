/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: lib.rs
 *
 * This file serves as the main library entry point and the Foreign Function Interface (FFI)
 * boundary for the `monitoring` crate. It is responsible for initializing, managing, and
 * shutting down the monitoring system, and exposing a C-compatible API to the core application.
 *
 * Dependencies:
 *  - `lazy_static`: For thread-safe initialization of global state.
 *  - `crossbeam-channel`: For creating the communication channels between components.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Declarations and Imports ---

// Declare the sub-modules of this crate.
pub mod metrics_collector;
pub mod telemetry;

// Import necessary components from sub-modules and external crates.
use metrics_collector::{MetricsCollector, SystemMetrics, CustomData};
use telemetry::{TelemetryExporter, HttpBackend};
use std::ffi::{CStr, c_char};
use std::sync::Mutex;
use std::time::Duration;
use crossbeam_channel::{Sender, unbounded};
use lazy_static::lazy_static;

// --- Global State Management ---

/// Represents the global state of the monitoring system.
/// This struct holds the core components that run in the background.
/// It is wrapped in a `Mutex` to ensure thread-safe access from the FFI functions.
struct MonitoringSystem {
    /// The collector engine, which runs its own background thread.
    collector: Option<MetricsCollector>,
    /// The telemetry exporter, which also has its own background thread.
    exporter: Option<TelemetryExporter<HttpBackend>>,
    /// A sender for custom data, allowing the C-API to submit events.
    custom_data_sender: Option<Sender<CustomData>>,
}

// Use `lazy_static` to ensure that the global state is initialized only once,
// in a thread-safe manner. This is a common pattern for managing global resources
// exposed via FFI.
lazy_static! {
    static ref MONITORING_SYSTEM: Mutex<MonitoringSystem> = Mutex::new(MonitoringSystem {
        collector: None,
        exporter: None,
        custom_data_sender: None,
    });
}


// --- FFI Public Interface ---
// These functions are exposed to the C/C++ core of the application.
// `#[no_mangle]` prevents the Rust compiler from changing the function names.
// `extern "C"` ensures that the functions use the C calling convention.

/// Initializes the entire monitoring system.
/// This function should be called once at application startup.
/// It sets up the collector, exporter, and the channels connecting them.
///
/// # Arguments
/// - `endpoint_url`: A C-string with the URL for the telemetry backend.
/// - `api_key`: A C-string with the API key for authentication.
///
/// # Safety
/// The caller must ensure that `endpoint_url` and `api_key` are valid, null-terminated
/// C-strings. Passing invalid pointers will result in undefined behavior.
#[no_mangle]
pub extern "C" fn monitoring_initialize(endpoint_url: *const c_char, api_key: *const c_char) {
    // Lock the global state to modify it.
    let mut system = MONITORING_SYSTEM.lock().unwrap();

    if system.collector.is_some() {
        eprintln!("Monitoring system is already initialized. Call shutdown first.");
        return;
    }

    // Safely convert C strings to Rust strings.
    let endpoint = match unsafe { CStr::from_ptr(endpoint_url).to_str() } {
        Ok(s) => s.to_string(),
        Err(e) => {
            eprintln!("Failed to initialize monitoring: invalid endpoint URL. Error: {}", e);
            return;
        }
    };

    let key = match unsafe { CStr::from_ptr(api_key).to_str() } {
        Ok(s) => s.to_string(),
        Err(e) => {
            eprintln!("Failed to initialize monitoring: invalid API key. Error: {}", e);
            return;
        }
    };

    println!("Initializing monitoring system with endpoint: {}", endpoint);

    // --- Create Communication Channels ---
    // These channels will connect the different components of the monitoring system.
    let (system_metrics_sender, system_metrics_receiver) = unbounded::<SystemMetrics>();
    let (custom_data_sender, custom_data_receiver) = unbounded::<CustomData>();

    // --- Instantiate and Start Components ---
    // 1. Create the telemetry backend.
    let backend = HttpBackend::new(endpoint, key);

    // 2. Create the telemetry exporter. It will start its own background thread.
    //    Configuration for batching is hardcoded here but could be passed via FFI.
    let exporter = TelemetryExporter::new(
        backend,
        system_metrics_receiver,
        custom_data_receiver,
        10, // Batch size: export after 10 items
        Duration::from_secs(60), // Batch timeout: export at least once a minute
    );

    // 3. Create the metrics collector. It also starts its own background thread.
    let collector = match MetricsCollector::new(
        Duration::from_secs(15), // Collection interval: collect every 15 seconds
        system_metrics_sender,
    ) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to start metrics collector: {}", e);
            return;
        }
    };

    // --- Store Components in Global State ---
    system.collector = Some(collector);
    system.exporter = Some(exporter);
    system.custom_data_sender = Some(custom_data_sender);

    println!("Monitoring system initialized successfully.");
}

/// Records a custom application event.
/// This function is the FFI entry point for the application to submit its own
/// telemetry data, such as business events or diagnostic information.
///
/// # Arguments
/// - `event_name`: A C-string identifying the event.
/// - `json_payload`: A C-string containing the event data, expected to be in JSON format.
///
/// # Safety
/// The caller must ensure that `event_name` and `json_payload` are valid, null-terminated
/// C-strings.
#[no_mangle]
pub extern "C" fn monitoring_record_custom_event(event_name: *const c_char, json_payload: *const c_char) {
    let system = MONITORING_SYSTEM.lock().unwrap();

    let sender = match &system.custom_data_sender {
        Some(s) => s,
        None => {
            eprintln!("Cannot record custom event: monitoring system not initialized.");
            return;
        }
    };

    let name = unsafe { CStr::from_ptr(event_name).to_string_lossy().into_owned() };
    let payload = unsafe { CStr::from_ptr(json_payload).to_string_lossy().into_owned() };

    let custom_data = CustomData { name, payload };

    if let Err(e) = sender.send(custom_data) {
        eprintln!("Failed to send custom event to exporter channel: {}", e);
    }
}

/// Shuts down the monitoring system gracefully.
/// This function should be called before the application exits. It ensures that
/// all background threads are terminated and any buffered data is flushed.
#[no_mangle]
pub extern "C" fn monitoring_shutdown() {
    println!("Shutting down monitoring system...");
    let mut system = MONITORING_SYSTEM.lock().unwrap();

    // The shutdown is handled by the `Drop` implementations of `MetricsCollector`
    // and `TelemetryExporter`. By setting them to `None`, we trigger the drop.
    // We drop the sender first to signal the exporter to stop waiting for new data.
    system.custom_data_sender = None;

    // We drop the collector, which will stop producing new system metrics.
    system.collector = None;

    // Finally, we drop the exporter. Its `Drop` implementation will wait for the
    // background thread to finish flushing any remaining data.
    system.exporter = None;

    println!("Monitoring system shut down successfully.");
}

// --- Unit Tests (for FFI layer) ---
// Note: Testing FFI functions is complex. These are basic sanity checks.
// A full test suite would involve a C test harness calling these functions.
#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::thread;

    #[test]
    fn test_initialization_and_shutdown_cycle() {
        // Create C-compatible strings for the test.
        let url = CString::new("http://localhost:8080").unwrap();
        let key = CString::new("test-api-key").unwrap();

        // Initialize the system.
        monitoring_initialize(url.as_ptr(), key.as_ptr());

        // Check if the system state reflects initialization.
        {
            let system = MONITORING_SYSTEM.lock().unwrap();
            assert!(system.collector.is_some());
            assert!(system.exporter.is_some());
            assert!(system.custom_data_sender.is_some());
        }

        // Let the system run for a bit.
        thread::sleep(Duration::from_millis(100));

        // Shut down the system.
        monitoring_shutdown();

        // Check if the system state reflects shutdown.
        {
            let system = MONITORING_SYSTEM.lock().unwrap();
            assert!(system.collector.is_none());
            assert!(system.exporter.is_none());
            assert!(system.custom_data_sender.is_none());
        }
    }
}
