/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_sensors_fusion.h
*
* This header file defines the public API for the Sensor Fusion engine. This
* critical subsystem is responsible for abstracting low-level sensor hardware
* and fusing their disparate data streams into a coherent, high-level
* understanding of the device's physical state and the user's immediate context.
*
* The primary goal is to provide the Cortex with a simplified "world model"
* rather than raw, noisy sensor data. For example, instead of processing raw
* gyroscope data, the Cortex can simply query the device's current orientation
* as a quaternion and its motion state (e.g., stationary, walking, running).
*
* The architecture is based on a stateful object, `tk_sensor_fusion_t`, which
* continuously integrates new sensor readings and updates its internal world
* model. It is designed to run filters (e.g., Kalman, Madgwick, or simpler
* complementary filters) to produce stable and reliable state estimations.
*
* SPDX-License-Identifier:
*/

#ifndef TRACKIELLM_SENSORS_TK_SENSORS_FUSION_H
#define TRACKIELLM_SENSORS_TK_SENSORS_FUSION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"

// Forward-declare the primary fusion engine object as an opaque type.
typedef struct tk_sensor_fusion_s tk_sensor_fusion_t;

/**
 * @struct tk_sensor_fusion_config_t
 * @brief Configuration for initializing the Sensor Fusion engine.
 */
typedef struct {
    float update_rate_hz;       /**< The target frequency at which sensor data is processed. */
    // Parameters for sensor fusion algorithms, e.g., filter gains
    float gyro_trust_factor;    /**< Beta parameter for Madgwick or similar filters. */
} tk_sensor_fusion_config_t;

/**
 * @struct tk_imu_data_t
 * @brief Represents a single, time-stamped reading from an Inertial Measurement Unit (IMU).
 */
typedef struct {
    uint64_t timestamp_ns;      /**< Nanosecond-resolution timestamp of the reading. */
    // Accelerometer data in m/s^2
    float acc_x, acc_y, acc_z;
    // Gyroscope data in radians/s
    float gyro_x, gyro_y, gyro_z;
    // Optional: Magnetometer data in microteslas (uT)
    bool has_mag_data;
    float mag_x, mag_y, mag_z;
} tk_imu_data_t;

/**
 * @enum tk_motion_state_e
 * @brief High-level classification of the user's current motion.
 */
typedef enum {
    TK_MOTION_STATE_UNKNOWN,    /**< The state cannot be determined yet. */
    TK_MOTION_STATE_STATIONARY, /**< The user is still. */
    TK_MOTION_STATE_WALKING,    /**< The user is walking at a steady pace. */
    TK_MOTION_STATE_RUNNING,    /**< The user is running. */
    TK_MOTION_STATE_FALLING     /**< A potential fall event has been detected (freefall). */
} tk_motion_state_e;

/**
 * @struct tk_quaternion_t
 * @brief Represents orientation in 3D space.
 */
typedef struct {
    float w, x, y, z;
} tk_quaternion_t;

/**
 * @struct tk_world_state_t
 * @brief The fused, high-level output of the sensor fusion engine.
 *
 * This structure represents the engine's best estimate of the system's
 * current physical state.
 */
typedef struct {
    uint64_t          last_update_timestamp_ns; /**< Timestamp of the last update. */
    tk_quaternion_t   orientation;              /**< The absolute orientation of the device in space. */
    tk_motion_state_e motion_state;             /**< The user's classified motion state. */
    bool              is_speech_detected;       /**< The current state from the Voice Activity Detector. */
} tk_world_state_t;


#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Engine Lifecycle Management
//------------------------------------------------------------------------------

/**
 * @brief Creates and initializes a new Sensor Fusion engine instance.
 *
 * @param[out] out_engine Pointer to receive the address of the new engine instance.
 * @param[in] config The configuration for the engine.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL.
 * @return TK_ERROR_OUT_OF_MEMORY on memory allocation failure.
 */
TK_NODISCARD tk_error_code_t tk_sensor_fusion_create(tk_sensor_fusion_t** out_engine, const tk_sensor_fusion_config_t* config);

/**
 * @brief Destroys a Sensor Fusion engine instance.
 *
 * @param[in,out] engine Pointer to the engine instance to be destroyed.
 */
void tk_sensor_fusion_destroy(tk_sensor_fusion_t** engine);

//------------------------------------------------------------------------------
// Data Injection and State Processing
//------------------------------------------------------------------------------

/**
 * @brief Injects a new IMU data sample into the engine.
 *
 * The engine will buffer this data and use it in its next processing step.
 *
 * @param[in] engine The sensor fusion engine instance.
 * @param[in] imu_data The new IMU data sample.
 *
 * @return TK_SUCCESS on success.
 * @par Thread-Safety
 * This function should be thread-safe to allow a dedicated sensor thread to
 * push data.
 */
TK_NODISCARD tk_error_code_t tk_sensor_fusion_inject_imu_data(tk_sensor_fusion_t* engine, const tk_imu_data_t* imu_data);

/**
 * @brief Injects the current Voice Activity Detection state.
 *
 * This allows the world model to know if the user is currently speaking.
 *
 * @param[in] engine The sensor fusion engine instance.
 * @param[in] is_speech_active The current VAD state.
 *
 * @return TK_SUCCESS on success.
 * @par Thread-Safety
 * This function should be thread-safe.
 */
TK_NODISCARD tk_error_code_t tk_sensor_fusion_inject_vad_state(tk_sensor_fusion_t* engine, bool is_speech_active);

/**
 * @brief Processes all injected data and updates the internal world state.
 *
 * This function should be called periodically by the Cortex at the frequency
 * defined in the configuration. It executes the core filtering and fusion
 * algorithms.
 *
 * @param[in] engine The sensor fusion engine instance.
 * @param[in] delta_time_s The time elapsed in seconds since the last update.
 *
 * @return TK_SUCCESS on success.
 * @par Thread-Safety
 * This function is NOT thread-safe. It should only be called from the main
 * Cortex processing loop.
 */
TK_NODISCARD tk_error_code_t tk_sensor_fusion_update(tk_sensor_fusion_t* engine, float delta_time_s);

//------------------------------------------------------------------------------
// State Retrieval
//------------------------------------------------------------------------------

/**
 * @brief Retrieves the latest, most up-to-date world state.
 *
 * @param[in] engine The sensor fusion engine instance.
 * @param[out] out_state A pointer to a structure that will be filled with the
 *                       current world state.
 *
 * @return TK_SUCCESS on success.
 * @par Thread-Safety
 * This function is thread-safe, typically using a mutex or seqlock to provide
 * a consistent snapshot of the state.
 */
TK_NODISCARD tk_error_code_t tk_sensor_fusion_get_world_state(tk_sensor_fusion_t* engine, tk_world_state_t* out_state);

#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_SENSORS_TK_SENSORS_FUSION_H