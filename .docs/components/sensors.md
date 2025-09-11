<!-- This documentation was written by Jules - Google labs bot. -->

# Sensors Module

## 1. Overview

The Sensors module is responsible for processing data from the various low-level sensors on the TrackieLLM device, primarily the Inertial Measurement Unit (IMU). It provides the system with crucial information about its own motion and orientation in space.

This data is essential for distinguishing between camera movement and world movement, improving the accuracy of the vision pipeline, and understanding the user's actions (e.g., walking, turning their head).

## 2. Core Responsibilities

-   **Sensor Data Acquisition:** Interfaces with the hardware drivers to read raw data from the IMU (accelerometer, gyroscope).
-   **Sensor Fusion:** Combines data from multiple sensors to produce a more accurate and stable estimate of the device's state. For example, it fuses accelerometer and gyroscope data to combat drift.
-   **High-Level State Classification:** Abstracts raw data into meaningful states, such as classifying the user's motion as `STATIONARY` or `WALKING`.
-   **Data Provisioning:** Makes the fused and classified sensor data available to the rest of the application in a safe and structured way.

## 3. Architecture

The Sensors module follows the hybrid Rust/C architecture used by other components like Vision and Audio. It consists of a C library that handles low-level logic and a Rust "worker" that integrates this logic into the main asynchronous application.

-   **C Layer (`tk_sensors_fusion.c`)**: This layer is responsible for the core sensor fusion algorithms. It exposes a simple API (`tk_sensor_fusion_create`, `tk_sensor_fusion_get_world_state`, etc.) to be consumed by Rust. For testing purposes, this layer contains a `MOCK_SENSORS` compile-time flag that enables it to generate simulated, predictable data without real hardware.

-   **Rust Worker (`src/workers/sensor_worker.rs`)**: This is the bridge between the C layer and the rest of the TrackieLLM application. It is responsible for:
    1.  **Lifecycle Management:** Using a safe RAII `SensorFusionWrapper`, it ensures that the C-level `tk_sensor_fusion_t` object is properly created on startup and destroyed on shutdown.
    2.  **Polling:** It runs in a `tokio` asynchronous loop, periodically (e.g., every 50ms) calling the `tk_sensor_fusion_get_world_state` FFI function to get the latest data.
    3.  **Event Publishing:** After retrieving the data, it converts the FFI-safe C struct (`tk_world_state_t`) into a pure Rust struct (`SensorFusionData`). It then wraps this data in an `Arc` and publishes it on the central `EventBus` as a `TrackieEvent::SensorFusionResult`.

## 4. Data Flow and Integration with Cortex

The data from the sensor module is critical for the Cortex's situational awareness. The end-to-end data flow is as follows:

1.  **Polling:** The `sensor_worker` calls the C function `tk_sensor_fusion_get_world_state()`. In mock mode, this function returns simulated data that alternates between `STATIONARY` and `WALKING`.

2.  **Publication:** The worker publishes the `SensorFusionResult` event to the `EventBus`.

3.  **Consumption by Cortex Worker:** The `cortex_worker` in Rust subscribes to the `EventBus`. When it receives the `SensorFusionResult` event, it converts the data back into a C-compatible struct (`tk_sensor_event_t`).

4.  **Injection into C-Cortex:** The `cortex_worker` calls the C function `tk_cortex_inject_sensor_event`, passing the C-compatible struct.

5.  **Contextual Reasoner Update:** Inside `tk_cortex_main.c`, the injected event is handled. The handler calls `tk_contextual_reasoner_update_motion_context()`, passing the sensor data. This function updates the `motion_state` field inside the `tk_contextual_reasoner_t` object, making the user's current motion state officially part of the Cortex's "world model".

6.  **Intelligent Decision-Making:** The `tk_decision_engine.c` leverages this new information. When processing a response from the LLM, it first checks the user's motion state via the `context_summary`. If the user is moving (`WALKING` or `RUNNING`) and the LLM suggests a simple description of a potential obstacle (e.g., "There is a chair in front of you"), the decision engine enhances the action, transforming it from a simple `SPEAK` action into a higher-priority `NAVIGATE_WARN` action (e.g., "Caution, chair ahead!").

This complete flow ensures that the sensor data is not just collected, but actively used to make the system's responses safer and more context-aware.

## 5. Key Components

-   **`tk_sensors_fusion.c` / `.h`:** The C library for sensor fusion and mocking.
-   **`src/workers/sensor_worker.rs`:** The Rust async worker that drives the polling and event publishing.
-   **`src/event_bus/mod.rs`:** Defines the `SensorFusionData` struct and the `TrackieEvent::SensorFusionResult` variant.
-   **`src/cortex/tk_cortex_main.c`:** Receives the injected event from the `cortex_worker`.
-   **`src/cortex/tk_contextual_reasoner.c`:** Stores the motion state as part of its world model.
-   **`src/cortex/tk_decision_engine.c`:** Uses the motion state to make smarter decisions.
