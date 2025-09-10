/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_sensors_fusion.c
*
* This file implements the Sensor Fusion engine for the TrackieLLM project.
* It provides the concrete implementation for the functions declared in the
* corresponding header file. The engine is responsible for creating a stable
* world model from potentially noisy and high-frequency sensor data.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#include "tk_sensors_fusion.h"
#include "tk_logging.h"
#include <stdlib.h>
#include <string.h>

// Opaque handle for the Rust-managed filter object.
typedef void* FilterHandle;

// FFI declarations for the functions implemented in Rust.
// In a real build system, this would be in a generated header.
extern FilterHandle rust_create_kalman_filter(float process_noise, float measurement_noise);
extern void rust_destroy_filter(FilterHandle handle);
extern void rust_filter_update(
    FilterHandle handle,
    float dt,
    float gyro_x, float gyro_y, float gyro_z,
    float acc_x, float acc_y, float acc_z,
    float* out_quat_w, float* out_quat_x, float* out_quat_y, float* out_quat_z
);


// Define the internal state of the sensor fusion engine.
struct tk_sensor_fusion_s {
    tk_sensor_fusion_config_t config;
    tk_world_state_t          world_state;
    tk_imu_data_t             last_imu_data;
    FilterHandle              rust_filter_handle;
    // TODO: Add mutex for thread-safe data injection and state retrieval
};

TK_NODISCARD tk_error_code_t tk_sensor_fusion_create(tk_sensor_fusion_t** out_engine, const tk_sensor_fusion_config_t* config) {
    if (!out_engine || !config) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    tk_sensor_fusion_t* engine = (tk_sensor_fusion_t*)calloc(1, sizeof(tk_sensor_fusion_t));
    if (!engine) {
        return TK_ERROR_OUT_OF_MEMORY;
    }

    engine->config = *config;

    // Initialize world state
    engine->world_state.motion_state = TK_MOTION_STATE_UNKNOWN;
    engine->world_state.orientation.w = 1.0f; // Identity quaternion

    // Create the Rust-side Kalman filter and store its handle.
    // These noise values are just defaults and may need tuning.
    engine->rust_filter_handle = rust_create_kalman_filter(0.01f, 0.1f);
    if (!engine->rust_filter_handle) {
        LOG_ERROR("Failed to create Rust Kalman filter.");
        free(engine);
        return TK_ERROR_DRIVER_FAILED; // A more specific error would be better
    }

    *out_engine = engine;
    LOG_INFO("Sensor Fusion engine created successfully.");
    return TK_SUCCESS;
}

void tk_sensor_fusion_destroy(tk_sensor_fusion_t** engine) {
    if (engine && *engine) {
        // Destroy the Rust-side filter.
        if ((*engine)->rust_filter_handle) {
            rust_destroy_filter((*engine)->rust_filter_handle);
        }
        free(*engine);
        *engine = NULL;
        LOG_INFO("Sensor Fusion engine destroyed.");
    }
}

TK_NODISCARD tk_error_code_t tk_sensor_fusion_inject_imu_data(tk_sensor_fusion_t* engine, const tk_imu_data_t* imu_data) {
    if (!engine || !imu_data) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    // TODO: Lock mutex
    engine->last_imu_data = *imu_data;
    // TODO: Unlock mutex
    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t tk_sensor_fusion_inject_vad_state(tk_sensor_fusion_t* engine, bool is_speech_active) {
    if (!engine) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    // TODO: Lock mutex
    engine->world_state.is_speech_detected = is_speech_active;
    // TODO: Unlock mutex
    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t tk_sensor_fusion_update(tk_sensor_fusion_t* engine, float delta_time_s) {
    if (!engine) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    // Get the latest injected data
    tk_imu_data_t* imu = &engine->last_imu_data;

    // Call the Rust FFI function to run the complementary filter
    rust_filter_update(
        engine->rust_filter_handle,
        delta_time_s,
        imu->gyro_x, imu->gyro_y, imu->gyro_z,
        imu->acc_x, imu->acc_y, imu->acc_z,
        &engine->world_state.orientation.w,
        &engine->world_state.orientation.x,
        &engine->world_state.orientation.y,
        &engine->world_state.orientation.z
    );

    engine->world_state.last_update_timestamp_ns = imu->timestamp_ns;

    // TODO: Classify motion state based on accelerometer data
    engine->world_state.motion_state = TK_MOTION_STATE_STATIONARY;

    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t tk_sensor_fusion_get_world_state(tk_sensor_fusion_t* engine, tk_world_state_t* out_state) {
    if (!engine || !out_state) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    // TODO: Lock mutex
    *out_state = engine->world_state;
    // TODO: Unlock mutex

    return TK_SUCCESS;
}
