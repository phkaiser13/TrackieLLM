/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_obstacle_avoider.c
 * 
 * This file implements the C-side wrapper for the Obstacle Tracking Engine.
 * It acts as a thin Foreign Function Interface (FFI) layer, delegating all
 * tracking logic and state management to the Rust implementation.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "navigation/tk_obstacle_avoider.h"
#include <stdlib.h>

// Opaque handle to the Rust-managed tracker object.
typedef void* ObstacleTrackerHandle;

// FFI declarations for the functions implemented in Rust.
extern ObstacleTrackerHandle rust_obstacle_tracker_create(const tk_obstacle_tracker_config_t* config);
extern void rust_obstacle_tracker_destroy(ObstacleTrackerHandle handle);
extern void rust_obstacle_tracker_update(ObstacleTrackerHandle handle, const tk_traversability_map_t* map, float delta_time_s);
extern void rust_obstacle_tracker_get_all(ObstacleTrackerHandle handle, const tk_obstacle_t** out_obstacles, size_t* out_count);
// Note: get_closest is not implemented in Rust FFI for brevity, it can be implemented C-side by iterating the result of get_all.


// The C-side struct just holds the handle to the Rust object.
struct tk_obstacle_tracker_s {
    ObstacleTrackerHandle rust_handle;
    // Buffer for get_all results to avoid returning a pointer to Rust's internal memory
    const tk_obstacle_t* last_obstacles;
    size_t last_obstacle_count;
};


//------------------------------------------------------------------------------
// Engine Lifecycle Management
//------------------------------------------------------------------------------

tk_error_code_t tk_obstacle_tracker_create(
    tk_obstacle_tracker_t** out_tracker,
    const tk_obstacle_tracker_config_t* config
) {
    if (!out_tracker || !config) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    ObstacleTrackerHandle handle = rust_obstacle_tracker_create(config);
    if (!handle) {
        return TK_ERROR_OUT_OF_MEMORY;
    }

    tk_obstacle_tracker_t* tracker = (tk_obstacle_tracker_t*)calloc(1, sizeof(tk_obstacle_tracker_t));
    if (!tracker) {
        rust_obstacle_tracker_destroy(handle);
        return TK_ERROR_OUT_OF_MEMORY;
    }

    tracker->rust_handle = handle;
    *out_tracker = tracker;
    return TK_SUCCESS;
}

void tk_obstacle_tracker_destroy(tk_obstacle_tracker_t** tracker) {
    if (tracker && *tracker) {
        rust_obstacle_tracker_destroy((*tracker)->rust_handle);
        free(*tracker);
        *tracker = NULL;
    }
}

//------------------------------------------------------------------------------
// Core Processing and State Update
//------------------------------------------------------------------------------

tk_error_code_t tk_obstacle_tracker_update(
    tk_obstacle_tracker_t* tracker,
    const tk_traversability_map_t* map,
    float delta_time_s
) {
    if (!tracker || !map) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    rust_obstacle_tracker_update(tracker->rust_handle, map, delta_time_s);
    return TK_SUCCESS;
}

//------------------------------------------------------------------------------
// High-Level Queries
//------------------------------------------------------------------------------

tk_error_code_t tk_obstacle_tracker_get_all(
    tk_obstacle_tracker_t* tracker,
    const tk_obstacle_t** out_obstacles,
    size_t* out_count
) {
    if (!tracker || !out_obstacles || !out_count) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    // The Rust function currently returns a null pointer as a placeholder.
    // When implemented, it will point to memory managed by Rust.
    rust_obstacle_tracker_get_all(tracker->rust_handle, out_obstacles, out_count);

    // Cache the pointers for other queries like get_closest
    tracker->last_obstacles = *out_obstacles;
    tracker->last_obstacle_count = *out_count;

    return TK_SUCCESS;
}

tk_error_code_t tk_obstacle_tracker_get_closest(
    tk_obstacle_tracker_t* tracker,
    tk_obstacle_t* out_closest_obstacle
) {
    if (!tracker || !out_closest_obstacle) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    if (!tracker->last_obstacles || tracker->last_obstacle_count == 0) {
        return TK_ERROR_NOT_FOUND;
    }
    
    // This logic can stay C-side, as it just iterates over the results from get_all
    float min_dist_sq = -1.0f;
    int closest_idx = -1;

    for (size_t i = 0; i < tracker->last_obstacle_count; i++) {
        const tk_obstacle_t* obs = &tracker->last_obstacles[i];
        float dist_sq = obs->position_m.x * obs->position_m.x + obs->position_m.y * obs->position_m.y;
        if (closest_idx == -1 || dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            closest_idx = i;
        }
    }

    if (closest_idx != -1) {
        *out_closest_obstacle = tracker->last_obstacles[closest_idx];
        return TK_SUCCESS;
    }

    return TK_ERROR_NOT_FOUND;
}
