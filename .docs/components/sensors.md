<!-- This documentation was written by Jules - Google labs bot. -->

# Sensors Module

## 1. Overview

The Sensors module is responsible for processing data from the various low-level sensors on the TrackieLLM device, primarily the Inertial Measurement Unit (IMU). It provides the system with crucial information about its own motion and orientation in space.

This data is essential for distinguishing between camera movement and world movement, improving the accuracy of the vision pipeline, and understanding the user's actions (e.g., walking, turning their head).

## 2. Core Responsibilities

-   **Sensor Data Acquisition:** Interfaces with the hardware drivers to read raw data from the IMU (accelerometer, gyroscope) and any other low-level sensors.
-   **Sensor Fusion:** Combines data from multiple sensors to produce a more accurate and stable estimate of the device's state. For example, it fuses accelerometer and gyroscope data to combat drift.
-   **Motion and Orientation Tracking:** Provides the rest of the system with a real-time understanding of the device's orientation (roll, pitch, yaw) and acceleration.
-   **Event Detection:** (Future implementation) Will detect specific motion events like gestures (e.g., a nod for "yes"), a fall, or a sudden jolt.

## 3. Architecture

The Sensors module is designed to be a lightweight and efficient component, running in the background to provide a continuous stream of state information. It is implemented in C and Rust.

-   **C (`tk_sensors_fusion.c`):** Provides the main interface for the module and handles the direct communication with the sensor hardware drivers.
-   **Rust (`sensor_fusion.rs`, `sensor_filters.rs`):** Implements the core sensor fusion algorithms and filters (e.g., Kalman filters, complementary filters) in a memory-safe environment. This is where the complex mathematical computations happen.

### 3.1. Pipeline

1.  **Raw Data Reading:** The module continuously polls the IMU sensor for new accelerometer and gyroscope readings.
2.  **Filtering (`sensor_filters.rs`):** The raw data is passed through filters to reduce noise and remove biases.
3.  **Fusion (`sensor_fusion.rs`):** The filtered accelerometer and gyroscope data are fused to calculate a stable estimate of the device's orientation (quaternion or Euler angles).
4.  **Data Publishing:** The fused sensor data is published as a `sensor_event_t` to the central event bus, making it available to other modules.

## 4. Key Components

-   **`tk_sensors_fusion.c`:** The central component that orchestrates the data flow within the module.
-   **`sensor_fusion.rs`:** The Rust component containing the core logic for combining sensor data. It likely implements a complementary or Kalman filter to get the best estimate of orientation.
-   **`sensor_filters.rs`:** Contains various digital filters (e.g., low-pass, high-pass) used to clean up the raw sensor signals before fusion.
-   **`tk_vad_silero.c`:** This file seems to be misplaced in the `sensors` directory in the original project structure, as it's related to audio processing (Voice Activity Detection). It is functionally part of the Audio module.

## 5. Integration with other Modules

-   **Cortex:** The Cortex module is a primary consumer of the sensor data. It uses the orientation information to:
    -   Stabilize the world model (e.g., compensating for head movements when tracking an object).
    -   Infer user intent (e.g., a downward head tilt might imply looking at the ground).
    -   Improve navigation by tracking the user's path.

-   **Vision:** The vision pipeline uses the IMU data for tasks like:
    -   **Electronic Image Stabilization (EIS):** Removing jitter from the video feed.
    -   **Sensor-Aided Tracking:** Predicting the location of an object in the next frame based on camera motion.

-   **Audio:** The audio module can use the accelerometer data to detect footsteps and dynamically adjust the noise cancellation profile.
