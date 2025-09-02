/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_obstacle_avoider.h
*
* This header file defines the public API for the Obstacle Tracking Engine.
* This is a specialized sub-component of the Tactical Navigation Engine, designed
* to process the traversability map and produce a stateful, tracked list of
* discrete obstacles.
*
* While the navigation engine identifies obstacle areas, this module's purpose
* is to cluster those areas into distinct objects, assign them persistent IDs,
* and track their position and velocity over time. This provides the Cortex
* with a much richer, more stable understanding of dynamic and static objects
* in the user's path, enabling more intelligent and context-aware feedback.
*
* The API is centered around the `tk_obstacle_tracker_t` opaque object.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TRACKIELLM_NAVIGATION_TK_OBSTACLE_AVOIDER_H
#define TRACKIELLM_NAVIGATION_TK_OBSTACLE_AVOIDER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "navigation/tk_path_planner.h" // For tk_traversability_map_t

// Forward-declare the primary tracker object as an opaque type.
typedef struct tk_obstacle_tracker_s tk_obstacle_tracker_t;

/**
 * @struct tk_obstacle_tracker_config_t
 * @brief Configuration for initializing the Obstacle Tracking Engine.
 */
typedef struct {
    float    min_obstacle_area_m2;      /**< Minimum area in square meters to be considered a valid obstacle. */
    uint32_t max_tracked_obstacles;     /**< The maximum number of obstacles to track simultaneously. */
    uint32_t max_frames_unseen;         /**< Number of frames an obstacle can be unseen before it's pruned. */
    float    max_match_distance_m;      /**< Maximum distance in meters to associate a new detection with an existing track. */
} tk_obstacle_tracker_config_t;

/**
 * @enum tk_obstacle_status_e
 * @brief The tracking status of a detected obstacle.
 */
typedef enum {
    TK_OBSTACLE_STATUS_NEW,       /**< This obstacle was detected for the first time in the current frame. */
    TK_OBSTACLE_STATUS_TRACKED,   /**< This obstacle was successfully tracked from a previous frame. */
    TK_OBSTACLE_STATUS_COASTED    /**< This obstacle was not seen in the current frame, but its position is being predicted. */
} tk_obstacle_status_e;

/**
 * @struct tk_vector2d_t
 * @brief A simple 2D vector for position and velocity.
 */
typedef struct {
    float x; /**< X-coordinate (e.g., meters left/right). */
    float y; /**< Y-coordinate (e.g., meters forward/backward). */
} tk_vector2d_t;

/**
 * @struct tk_obstacle_t
 * @brief Represents a single, tracked obstacle.
 */
typedef struct {
    uint32_t        id;             /**< A unique and persistent ID for this tracked obstacle. */
    tk_obstacle_status_e status;    /**< The current tracking status. */
    tk_vector2d_t   position_m;     /**< The estimated 2D position of the obstacle's centroid on the ground plane, in meters. */
    tk_vector2d_t   velocity_mps;   /**< The estimated 2D velocity in meters per second. */
    tk_vector2d_t   dimensions_m;   /**< The estimated width and depth of the obstacle's footprint. */
    uint32_t        age_frames;     /**< How many frames this obstacle has been tracked for. */
    uint32_t        unseen_frames;  /**< How many frames this obstacle has been unseen for. */
} tk_obstacle_t;


#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Engine Lifecycle Management
//------------------------------------------------------------------------------

/**
 * @brief Creates and initializes a new Obstacle Tracking Engine instance.
 *
 * @param[out] out_tracker Pointer to receive the address of the new engine instance.
 * @param[in] config The configuration for the tracker.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL.
 * @return TK_ERROR_OUT_OF_MEMORY on memory allocation failure.
 */
TK_NODISCARD tk_error_code_t tk_obstacle_tracker_create(tk_obstacle_tracker_t** out_tracker, const tk_obstacle_tracker_config_t* config);

/**
 * @brief Destroys an Obstacle Tracking Engine instance.
 *
 * @param[in,out] tracker Pointer to the engine instance to be destroyed.
 */
void tk_obstacle_tracker_destroy(tk_obstacle_tracker_t** tracker);

//------------------------------------------------------------------------------
// Core Processing and State Update
//------------------------------------------------------------------------------

/**
 * @brief Updates the obstacle tracker with a new traversability map.
 *
 * This is the core function of the tracker. It performs:
 * 1. Clustering of obstacle cells in the map to form discrete detections.
 * 2. Data association to match new detections with existing tracks.
 * 3. State estimation (e.g., Kalman filter update) for position and velocity.
 * 4. Management of track lifecycle (creation of new tracks, pruning of lost tracks).
 *
 * @param[in] tracker The obstacle tracker instance.
 * @param[in] map The latest traversability map from the navigation engine.
 * @param[in] delta_time_s Time elapsed in seconds since the last update.
 *
 * @return TK_SUCCESS on successful update.
 * @return TK_ERROR_INVALID_ARGUMENT if required inputs are NULL.
 *
 * @par Thread-Safety
 * This function is NOT thread-safe. It should only be called from the main
 * Cortex processing loop.
 */
TK_NODISCARD tk_error_code_t tk_obstacle_tracker_update(
    tk_obstacle_tracker_t* tracker,
    const tk_traversability_map_t* map,
    float delta_time_s
);

//------------------------------------------------------------------------------
// High-Level Queries
//------------------------------------------------------------------------------

/**
 * @brief Retrieves a read-only list of all currently tracked obstacles.
 *
 * @param[in] tracker The obstacle tracker instance.
 * @param[out] out_obstacles Pointer to a constant array of obstacles. The memory
 *                           is owned by the tracker and is valid only until the
 *                           next call to `tk_obstacle_tracker_update`.
 * @param[out] out_count Pointer to receive the number of obstacles in the array.
 *
 * @return TK_SUCCESS on success.
 */
TK_NODISCARD tk_error_code_t tk_obstacle_tracker_get_all(
    tk_obstacle_tracker_t* tracker,
    const tk_obstacle_t** out_obstacles,
    size_t* out_count
);

/**
 * @brief Finds the single closest obstacle to the user.
 *
 * @param[in] tracker The obstacle tracker instance.
 * @param[out] out_closest_obstacle Pointer to a structure to be filled with the
 *                                  closest obstacle's data.
 *
 * @return TK_SUCCESS if an obstacle was found.
 * @return TK_ERROR_NOT_FOUND if there are currently no tracked obstacles.
 */
TK_NODISCARD tk_error_code_t tk_obstacle_tracker_get_closest(
    tk_obstacle_tracker_t* tracker,
    tk_obstacle_t* out_closest_obstacle
);

#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_NAVIGATION_TK_OBSTACLE_AVOIDER_H