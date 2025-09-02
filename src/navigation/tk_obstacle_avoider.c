/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_obstacle_avoider.c
 * 
 * This implementation file provides the core logic for the Obstacle Tracking Engine.
 * It processes traversability maps to detect, track, and predict the motion of
 * obstacles in the environment.
 *
 * SPDX-License-Identifier: AGPL-3.0 license AGPL-3.0 license
 */

#include "navigation/tk_obstacle_avoider.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// Constants for internal calculations
#define TK_PI_F 3.14159265358979323846f
#define TK_DEG_TO_RAD_F (TK_PI_F / 180.0f)
#define TK_RAD_TO_DEG_F (180.0f / TK_PI_F)

// Internal structure for tracking a single obstacle detection
typedef struct {
    uint32_t id;
    tk_vector2d_t position_m;
    tk_vector2d_t dimensions_m;
    float area_m2;
} tk_obstacle_detection_t;

// Internal structure for the obstacle tracker state
struct tk_obstacle_tracker_s {
    tk_obstacle_tracker_config_t config;
    tk_obstacle_t* tracked_obstacles;
    uint32_t tracked_count;
    uint32_t next_id;
    bool is_initialized;
};

// Private helper functions
static tk_error_code_t tk_validate_config(const tk_obstacle_tracker_config_t* config);
static tk_error_code_t tk_initialize_tracker(tk_obstacle_tracker_t* tracker);
static void tk_destroy_tracker(tk_obstacle_tracker_t* tracker);
static tk_error_code_t tk_detect_obstacles(
    const tk_obstacle_tracker_t* tracker,
    const tk_traversability_map_t* map,
    tk_obstacle_detection_t** out_detections,
    size_t* out_count
);
static void tk_free_detections(tk_obstacle_detection_t* detections);
static tk_error_code_t tk_cluster_obstacle_cells(
    const tk_traversability_map_t* map,
    tk_obstacle_detection_t* detection
);
static tk_error_code_t tk_associate_detections(
    tk_obstacle_tracker_t* tracker,
    const tk_obstacle_detection_t* detections,
    size_t detection_count
);
static tk_error_code_t tk_update_track_states(
    tk_obstacle_tracker_t* tracker,
    float delta_time_s
);
static tk_error_code_t tk_manage_track_lifecycle(tk_obstacle_tracker_t* tracker);
static float tk_calculate_distance(const tk_vector2d_t* a, const tk_vector2d_t* b);
static int tk_find_closest_obstacle(const tk_obstacle_t* obstacles, size_t count);

//------------------------------------------------------------------------------
// Engine Lifecycle Management
//------------------------------------------------------------------------------

tk_error_code_t tk_obstacle_tracker_create(tk_obstacle_tracker_t** out_tracker, const tk_obstacle_tracker_config_t* config) {
    // Validate input parameters
    if (!out_tracker || !config) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    // Validate configuration
    tk_error_code_t result = tk_validate_config(config);
    if (result != TK_SUCCESS) {
        return result;
    }

    // Allocate memory for the tracker
    tk_obstacle_tracker_t* tracker = (tk_obstacle_tracker_t*)calloc(1, sizeof(tk_obstacle_tracker_t));
    if (!tracker) {
        return TK_ERROR_OUT_OF_MEMORY;
    }

    // Copy configuration
    tracker->config = *config;
    tracker->tracked_count = 0;
    tracker->next_id = 1;
    tracker->is_initialized = false;

    // Initialize tracker memory
    result = tk_initialize_tracker(tracker);
    if (result != TK_SUCCESS) {
        free(tracker);
        return result;
    }

    tracker->is_initialized = true;
    *out_tracker = tracker;
    return TK_SUCCESS;
}

void tk_obstacle_tracker_destroy(tk_obstacle_tracker_t** tracker) {
    if (!tracker || !*tracker) {
        return;
    }

    tk_destroy_tracker(*tracker);
    free(*tracker);
    *tracker = NULL;
}

//------------------------------------------------------------------------------
// Core Processing and State Update
//------------------------------------------------------------------------------

tk_error_code_t tk_obstacle_tracker_update(
    tk_obstacle_tracker_t* tracker,
    const tk_traversability_map_t* map,
    float delta_time_s
) {
    // Validate inputs
    if (!tracker || !map) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    if (!tracker->is_initialized) {
        return TK_ERROR_INVALID_STATE;
    }

    // Detect obstacles in the current map
    tk_obstacle_detection_t* detections = NULL;
    size_t detection_count = 0;
    tk_error_code_t result = tk_detect_obstacles(tracker, map, &detections, &detection_count);
    if (result != TK_SUCCESS) {
        return result;
    }

    // Associate new detections with existing tracks
    result = tk_associate_detections(tracker, detections, detection_count);
    if (result != TK_SUCCESS) {
        tk_free_detections(detections);
        return result;
    }

    // Update track states (position, velocity)
    result = tk_update_track_states(tracker, delta_time_s);
    if (result != TK_SUCCESS) {
        tk_free_detections(detections);
        return result;
    }

    // Manage track lifecycle (create/prune tracks)
    result = tk_manage_track_lifecycle(tracker);
    tk_free_detections(detections);
    return result;
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

    if (!tracker->is_initialized) {
        return TK_ERROR_INVALID_STATE;
    }

    *out_obstacles = tracker->tracked_obstacles;
    *out_count = tracker->tracked_count;
    return TK_SUCCESS;
}

tk_error_code_t tk_obstacle_tracker_get_closest(
    tk_obstacle_tracker_t* tracker,
    tk_obstacle_t* out_closest_obstacle
) {
    if (!tracker || !out_closest_obstacle) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    if (!tracker->is_initialized) {
        return TK_ERROR_INVALID_STATE;
    }

    if (tracker->tracked_count == 0) {
        return TK_ERROR_NOT_FOUND;
    }

    int closest_index = tk_find_closest_obstacle(tracker->tracked_obstacles, tracker->tracked_count);
    if (closest_index < 0) {
        return TK_ERROR_NOT_FOUND;
    }

    *out_closest_obstacle = tracker->tracked_obstacles[closest_index];
    return TK_SUCCESS;
}

//------------------------------------------------------------------------------
// Private Helper Functions
//------------------------------------------------------------------------------

static tk_error_code_t tk_validate_config(const tk_obstacle_tracker_config_t* config) {
    if (config->min_obstacle_area_m2 <= 0.0f) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    if (config->max_tracked_obstacles == 0) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    if (config->max_match_distance_m <= 0.0f) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    return TK_SUCCESS;
}

static tk_error_code_t tk_initialize_tracker(tk_obstacle_tracker_t* tracker) {
    // Allocate memory for tracked obstacles
    size_t obstacles_size = tracker->config.max_tracked_obstacles * sizeof(tk_obstacle_t);
    tracker->tracked_obstacles = (tk_obstacle_t*)calloc(1, obstacles_size);
    if (!tracker->tracked_obstacles) {
        return TK_ERROR_OUT_OF_MEMORY;
    }

    tracker->tracked_count = 0;
    return TK_SUCCESS;
}

static void tk_destroy_tracker(tk_obstacle_tracker_t* tracker) {
    if (tracker && tracker->tracked_obstacles) {
        free(tracker->tracked_obstacles);
        tracker->tracked_obstacles = NULL;
        tracker->tracked_count = 0;
    }
}

static tk_error_code_t tk_detect_obstacles(
    const tk_obstacle_tracker_t* tracker,
    const tk_traversability_map_t* map,
    tk_obstacle_detection_t** out_detections,
    size_t* out_count
) {
    // For this implementation, we'll create a simplified detection approach
    // In a real system, this would involve connected component analysis
    
    // Allocate memory for detections (simplified - fixed number)
    const size_t max_detections = 10;
    tk_obstacle_detection_t* detections = (tk_obstacle_detection_t*)calloc(max_detections, sizeof(tk_obstacle_detection_t));
    if (!detections) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    size_t detection_count = 0;
    
    // Simple detection logic - find clusters of obstacle cells
    // This is a placeholder implementation - in practice this would use
    // connected component analysis or similar techniques
    
    // For demonstration, we'll create some dummy detections
    for (uint32_t i = 0; i < 3 && i < max_detections; ++i) {
        detections[i].id = 0; // Will be assigned during association
        detections[i].position_m.x = (float)(i - 1) * 0.5f; // -0.5, 0.0, 0.5 meters
        detections[i].position_m.y = 2.0f + (float)i * 0.3f; // 2.0, 2.3, 2.6 meters
        detections[i].dimensions_m.x = 0.3f; // Width
        detections[i].dimensions_m.y = 0.4f; // Depth
        detections[i].area_m2 = detections[i].dimensions_m.x * detections[i].dimensions_m.y;
        detection_count++;
    }
    
    *out_detections = detections;
    *out_count = detection_count;
    return TK_SUCCESS;
}

static void tk_free_detections(tk_obstacle_detection_t* detections) {
    if (detections) {
        free(detections);
    }
}

static tk_error_code_t tk_cluster_obstacle_cells(
    const tk_traversability_map_t* map,
    tk_obstacle_detection_t* detection
) {
    // In a real implementation, this function would perform connected component
    // analysis on obstacle cells in the traversability map to form clusters
    // and calculate centroid, dimensions, and area for each cluster.
    
    // For this placeholder, we assume the detection already has valid data
    return TK_SUCCESS;
}

static tk_error_code_t tk_associate_detections(
    tk_obstacle_tracker_t* tracker,
    const tk_obstacle_detection_t* detections,
    size_t detection_count
) {
    // Simple association algorithm:
    // For each detection, find the closest existing track within max_match_distance
    // If found, associate with that track
    // If not found, create a new track
    
    for (size_t i = 0; i < detection_count; ++i) {
        const tk_obstacle_detection_t* detection = &detections[i];
        int best_track_index = -1;
        float min_distance = tracker->config.max_match_distance_m;
        
        // Find closest existing track
        for (uint32_t j = 0; j < tracker->tracked_count; ++j) {
            tk_obstacle_t* track = &tracker->tracked_obstacles[j];
            float distance = tk_calculate_distance(&detection->position_m, &track->position_m);
            
            if (distance < min_distance) {
                min_distance = distance;
                best_track_index = (int)j;
            }
        }
        
        if (best_track_index >= 0) {
            // Associate with existing track
            tk_obstacle_t* track = &tracker->tracked_obstacles[best_track_index];
            track->status = TK_OBSTACLE_STATUS_TRACKED;
            track->position_m = detection->position_m;
            track->dimensions_m = detection->dimensions_m;
            track->unseen_frames = 0;
            track->age_frames++;
        } else {
            // Create new track if we have capacity
            if (tracker->tracked_count < tracker->config.max_tracked_obstacles) {
                tk_obstacle_t* new_track = &tracker->tracked_obstacles[tracker->tracked_count];
                new_track->id = tracker->next_id++;
                new_track->status = TK_OBSTACLE_STATUS_NEW;
                new_track->position_m = detection->position_m;
                new_track->velocity_mps.x = 0.0f;
                new_track->velocity_mps.y = 0.0f;
                new_track->dimensions_m = detection->dimensions_m;
                new_track->age_frames = 1;
                new_track->unseen_frames = 0;
                tracker->tracked_count++;
            }
        }
    }
    
    return TK_SUCCESS;
}

static tk_error_code_t tk_update_track_states(
    tk_obstacle_tracker_t* tracker,
    float delta_time_s
) {
    // Update velocity estimates using simple differencing
    // In a real implementation, this would use a Kalman filter or similar
    
    for (uint32_t i = 0; i < tracker->tracked_count; ++i) {
        tk_obstacle_t* track = &tracker->tracked_obstacles[i];
        
        // For this simplified implementation, we'll just mark velocity as zero
        // A real implementation would calculate velocity based on position changes
        track->velocity_mps.x = 0.0f;
        track->velocity_mps.y = 0.0f;
    }
    
    return TK_SUCCESS;
}

static tk_error_code_t tk_manage_track_lifecycle(tk_obstacle_tracker_t* tracker) {
    // Handle tracks that weren't seen in this frame
    for (uint32_t i = 0; i < tracker->tracked_count; ++i) {
        tk_obstacle_t* track = &tracker->tracked_obstacles[i];
        
        if (track->status != TK_OBSTACLE_STATUS_TRACKED) {
            track->unseen_frames++;
            
            // If track has been unseen for too long, mark for removal
            if (track->unseen_frames > tracker->config.max_frames_unseen) {
                track->status = TK_OBSTACLE_STATUS_COASTED;
            }
        }
    }
    
    // Remove old tracks (move remaining tracks to fill gaps)
    uint32_t write_index = 0;
    for (uint32_t read_index = 0; read_index < tracker->tracked_count; ++read_index) {
        tk_obstacle_t* track = &tracker->tracked_obstacles[read_index];
        
        // Keep track if it's not expired
        if (track->unseen_frames <= tracker->config.max_frames_unseen) {
            if (write_index != read_index) {
                tracker->tracked_obstacles[write_index] = *track;
            }
            write_index++;
        }
    }
    
    tracker->tracked_count = write_index;
    return TK_SUCCESS;
}

static float tk_calculate_distance(const tk_vector2d_t* a, const tk_vector2d_t* b) {
    float dx = a->x - b->x;
    float dy = a->y - b->y;
    return sqrtf(dx * dx + dy * dy);
}

static int tk_find_closest_obstacle(const tk_obstacle_t* obstacles, size_t count) {
    if (count == 0) {
        return -1;
    }
    
    int closest_index = 0;
    float min_distance = tk_calculate_distance(&obstacles[0].position_m, &(tk_vector2d_t){0.0f, 0.0f});
    
    for (size_t i = 1; i < count; ++i) {
        float distance = tk_calculate_distance(&obstacles[i].position_m, &(tk_vector2d_t){0.0f, 0.0f});
        if (distance < min_distance) {
            min_distance = distance;
            closest_index = (int)i;
        }
    }
    
    return closest_index;
}
