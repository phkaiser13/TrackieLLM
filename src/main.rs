/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/main.rs
 *
 * This is the main entry point for the TrackieLLM application. It acts as
 * the "heart" of the system, responsible for initializing all core components
 * and orchestrating the primary background tasks (workers).
 *
 * The application is built on the `tokio` asynchronous runtime. All major
 * functional units (vision, audio, cortex) are spawned into their own
 * long-running, non-blocking tasks. They communicate with each other via a
 * central, asynchronous event bus, ensuring a decoupled and scalable
 * architecture.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// Since this is a binary crate, we declare all our library crates here
// to make them available to the executable.
mod async_tasks;
mod audio;
mod cortex;
mod event_bus;
mod sensors;
mod vision;
mod workers;

use std::sync::{Arc, Mutex};
use tokio;

use async_tasks::prelude::*;
use cortex::memory_manager::MemoryManager;
use event_bus::{EventBus, TrackieEvent};

/// The main function initializes and runs the TrackieLLM application.
#[tokio::main]
async fn main() {
    // --- 1. Initialize Core Shared Components ---

    // The TaskManager is responsible for spawning and managing all async tasks.
    let task_manager = Arc::new(TaskManager::new());

    // The EventBus is the central nervous system for inter-worker communication.
    let event_bus = Arc::new(EventBus::new());

    // The MemoryManager holds the application's state. It's wrapped in a
    // Mutex to ensure thread-safe access from multiple workers.
    let memory_manager = Arc::new(Mutex::new(MemoryManager::new()));

    println!("[Orchestrator] Core components initialized.");

    // --- 2. Spawn Worker Tasks ---

    // Each worker is spawned on the `tokio` runtime using our `TaskManager`.
    // They run concurrently in the background.

    let vision_handle = task_manager.spawn(workers::vision_worker::run(
        event_bus.clone(),
    ));
    println!("[Orchestrator] Vision worker spawned.");

    let audio_handle = task_manager.spawn(workers::audio_worker::run(
        event_bus.clone(),
    ));
    println!("[Orchestrator] Audio worker spawned.");

    let cortex_handle = task_manager.spawn(workers::cortex_worker::run(
        event_bus.clone(),
        memory_manager.clone(),
    ));
    println!("[Orchestrator] Cortex worker spawned.");

    let sensor_handle = task_manager.spawn(workers::sensor_worker::run(
        event_bus.clone(),
    ));
    println!("[Orchestrator] Sensor worker spawned.");

    println!("[Orchestrator] All workers are running.");

    // --- 3. Await and Manage Shutdown ---

    // --- 3. Wait for Shutdown Signal ---

    // The application will run indefinitely until a `Ctrl-C` signal is received.
    if let Err(e) = tokio::signal::ctrl_c().await {
        eprintln!("[Orchestrator] Failed to listen for shutdown signal: {}", e);
        // If we can't listen, we can't gracefully shut down.
        // We'll proceed to a hard shutdown by dropping the handles.
    } else {
        println!("\n[Orchestrator] Shutdown signal received. Broadcasting to all workers...");
        // Publish the shutdown event to notify all workers.
        event_bus.publish(TrackieEvent::Shutdown);
    }

    // --- 4. Await Graceful Worker Termination ---
    println!("[Orchestrator] Waiting for workers to terminate...");

    // We await all handles to ensure they have completed their cleanup.
    // A timeout is added as a safeguard against a worker that fails to
    // terminate.
    let shutdown_timeout = tokio::time::Duration::from_secs(10);
    if tokio::time::timeout(shutdown_timeout, async {
        let (vision_res, audio_res, cortex_res, sensor_res) =
            tokio::join!(vision_handle, audio_handle, cortex_handle, sensor_handle);
        // Check if any worker panicked.
        if let Err(e) = vision_res {
            eprintln!("[Orchestrator] Vision worker panicked: {:?}", e);
        }
        if let Err(e) = audio_res {
            eprintln!("[Orchestrator] Audio worker panicked: {:?}", e);
        }
        if let Err(e) = cortex_res {
            eprintln!("[Orchestrator] Cortex worker panicked: {:?}", e);
        }
        if let Err(e) = sensor_res {
            eprintln!("[Orchestrator] Sensor worker panicked: {:?}", e);
        }
    })
    .await
    .is_err()
    {
        eprintln!("[Orchestrator] Timeout waiting for workers to shut down. Forcing exit.");
    }

    println!("[Orchestrator] All workers have terminated. Shutting down.");
}
