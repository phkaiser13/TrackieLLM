/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_path_planner.h
*
* This header file defines the public API for the Tactical Navigation Engine.
* This module's primary responsibility is to analyze the 3D geometry of the
* immediate environment, derived from depth sensor data, to provide actionable
* insights for safe, short-range navigation.
*
* It transforms a dense depth map into a high-level, semantic understanding of
* the ground plane, identifying traversable areas, obstacles, and critical
* hazards like steps and drop-offs. This is a core component for user safety.
*
* The architecture is centered around a stateful `tk_navigation_engine_t` object
* that requires detailed physical configuration (e.g., camera height) to
* accurately model the relationship between the device and the world.
*
* SPDX-License-Identifier:
*/

#ifndef TRACKIELLM_NAVIGATION_TK_PATH_PLANNER_H
#define TRACKIELLM_NAVIGATION_TK_PATH_PLANNER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "vision/tk_vision_pipeline.h" // For tk_vision_depth_map_t
#include "sensors/tk_sensors_fusion.h" // For tk_quaternion_t

// Forward-declare the primary engine object as an opaque type.
typedef struct tk_navigation_engine_s tk_navigation_engine_t;

/**
 * @struct tk_navigation_config_t
 * @brief Physical and algorithmic configuration for the navigation engine.
 *
 * Accurate physical parameters are CRITICAL for the engine's performance.
 */
typedef struct {
    // --- Physical Camera Setup ---
    float camera_height_m;      /**< Height of the camera lens from the ground in meters. */
    float default_camera_pitch_deg; /**< The typical downward pitch of the camera in degrees. */

    // --- User and Environment Parameters ---
    float step_height_threshold_m; /**< Max height difference considered traversable ground vs. a step (e.g., 0.15m). */
    float user_clearance_width_m;  /**< The minimum width of a path for the user to be considered clear (e.g., 0.8m). */
    float max_analysis_distance_m; /**< The maximum distance from the camera to analyze (e.g., 5.0m). */
} tk_navigation_config_t;

/**
 * @enum tk_traversability_type_e
 * @brief Semantic classification for a cell in the traversability grid.
 */
typedef enum {
    TK_TRAVERSABILITY_UNKNOWN,          /**< Insufficient data or outside analysis range. */
    TK_TRAVERSABILITY_TRAVERSABLE,      /**< Safe, flat, and clear ground. */
    TK_TRAVERSABILITY_OBSTACLE,         /**< An obstruction of significant height. */
    TK_TRAVERSABILITY_HAZARD_STEP_UP,   /**< A curb or step upwards. */
    TK_TRAVERSABILITY_HAZARD_STEP_DOWN, /**< A curb or step downwards (high priority danger). */
    TK_TRAVERSABILITY_HAZARD_HOLE,      /**< A significant drop-off or hole (critical danger). */
} tk_traversability_type_e;

/**
 * @struct tk_traversability_map_t
 * @brief A 2D grid representing the ground plane in front of the user.
 */
typedef struct {
    uint32_t width;                 /**< Width of the grid in cells. */
    uint32_t height;                /**< Height (depth) of the grid in cells. */
    float    resolution_m_per_cell; /**< The real-world size of each cell in meters (e.g., 0.1f for 10cm). */
    tk_traversability_type_e* grid; /**< Pointer to the grid data (row-major order). Owned by the engine. */
} tk_traversability_map_t;

/**
 * @struct tk_navigation_hazard_t
 * @brief Describes a single, high-priority hazard detected in the scene.
 */
typedef struct {
    tk_traversability_type_e type;      /**< The type of hazard. */
    float distance_m;                   /**< Estimated distance to the hazard in meters. */
    float direction_deg;                /**< Direction relative to straight ahead (-90 to +90 degrees). */
} tk_navigation_hazard_t;

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Engine Lifecycle Management
//------------------------------------------------------------------------------

/**
 * @brief Creates and initializes a new Navigation Engine instance.
 *
 * @param[out] out_engine Pointer to receive the address of the new engine instance.
 * @param[in] config The physical and algorithmic configuration.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL or config is invalid.
 * @return TK_ERROR_OUT_OF_MEMORY on memory allocation failure.
 */
TK_NODISCARD tk_error_code_t tk_navigation_engine_create(tk_navigation_engine_t** out_engine, const tk_navigation_config_t* config);

/**
 * @brief Destroys a Navigation Engine instance.
 *
 * @param[in,out] engine Pointer to the engine instance to be destroyed.
 */
void tk_navigation_engine_destroy(tk_navigation_engine_t** engine);

//------------------------------------------------------------------------------
// Core Processing and State Update
//------------------------------------------------------------------------------

/**
 * @brief Updates the navigation engine's world model using a new depth map.
 *
 * This is the core processing function. It takes the latest depth data and
 * device orientation, performs ground plane segmentation, obstacle clustering,
 * and hazard detection, and updates the internal traversability map.
 *
 * @param[in] engine The navigation engine instance.
 * @param[in] depth_map The latest depth map from the vision pipeline.
 * @param[in] orientation The latest device orientation from the sensor fusion engine.
 *
 * @return TK_SUCCESS on successful update.
 * @return TK_ERROR_INVALID_ARGUMENT if required inputs are NULL.
 *
 * @par Thread-Safety
 * This function is NOT thread-safe. It should only be called from the main
 * Cortex processing loop.
 */
TK_NODISCARD tk_error_code_t tk_navigation_engine_update(
    tk_navigation_engine_t* engine,
    const tk_vision_depth_map_t* depth_map,
    const tk_quaternion_t* orientation
);

//------------------------------------------------------------------------------
// High-Level Queries
//------------------------------------------------------------------------------

/**
 * @brief Retrieves a read-only view of the current traversability map.
 *
 * The data pointed to by the returned structure is owned by the engine and is
 * valid only until the next call to `tk_navigation_engine_update`.
 *
 * @param[in] engine The navigation engine instance.
 * @param[out] out_map Pointer to a structure that will be filled with the map's
 *                     properties and a pointer to its data.
 *
 * @return TK_SUCCESS on success.
 */
TK_NODISCARD tk_error_code_t tk_navigation_engine_get_map(tk_navigation_engine_t* engine, tk_traversability_map_t* out_map);

/**
 * @brief Finds the most critical, immediate hazards in front of the user.
 *
 * @param[in] engine The navigation engine instance.
 * @param[in,out] hazards_buffer A caller-provided array to be filled with detected hazards.
 * @param[in] buffer_capacity The maximum number of hazards the buffer can hold.
 * @param[out] out_hazard_count The actual number of hazards written to the buffer.
 *
 * @return TK_SUCCESS on success.
 */
TK_NODISCARD tk_error_code_t tk_navigation_engine_query_hazards(
    tk_navigation_engine_t* engine,
    tk_navigation_hazard_t* hazards_buffer,
    size_t buffer_capacity,
    size_t* out_hazard_count
);

/**
 * @brief Analyzes the traversability map to suggest the clearest path forward.
 *
 * @param[in] engine The navigation engine instance.
 * @param[out] out_clear_path_direction_deg Pointer to a float that will receive the
 *                                          suggested direction in degrees (-90 to +90).
 * @param[out] out_max_clear_distance_m Pointer to a float that will receive the
 *                                      distance of the clear path in that direction.
 *
 * @return TK_SUCCESS if a clear path is found.
 * @return TK_ERROR_NOT_FOUND if no clear path meeting the user's clearance
 *         requirements could be identified.
 */
TK_NODISCARD tk_error_code_t tk_navigation_engine_find_clear_path(
    tk_navigation_engine_t* engine,
    float* out_clear_path_direction_deg,
    float* out_max_clear_distance_m
);

#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_NAVIGATION_TK_PATH_PLANNER_H