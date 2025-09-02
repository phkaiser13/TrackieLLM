/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_path_planner.c
 * 
 * This implementation file provides the core logic for the Tactical Navigation Engine.
 * It processes depth maps to generate traversability grids and identifies safe paths
 * for navigation while detecting critical hazards.
 *
 * SPDX-License-Identifier: AGPL-3.0 license AGPL-3.0 license
 */

#include "navigation/tk_path_planner.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// Constants for internal calculations
#define TK_PI_F 3.14159265358979323846f
#define TK_DEG_TO_RAD_F (TK_PI_F / 180.0f)
#define TK_RAD_TO_DEG_F (180.0f / TK_PI_F)

// Internal structure for the navigation engine state
struct tk_navigation_engine_s {
    tk_navigation_config_t config;
    tk_traversability_map_t traversability_map;
    bool is_initialized;
};

// Private helper functions
static float tk_deg_to_rad(float degrees);
static float tk_rad_to_deg(float radians);
static float tk_normalize_angle_deg(float angle_deg);
static tk_error_code_t tk_validate_config(const tk_navigation_config_t* config);
static tk_error_code_t tk_initialize_traversability_map(tk_navigation_engine_t* engine);
static void tk_destroy_traversability_map(tk_navigation_engine_t* engine);
static tk_error_code_t tk_update_traversability_grid(
    tk_navigation_engine_t* engine,
    const tk_vision_depth_map_t* depth_map,
    const tk_quaternion_t* orientation
);
static tk_error_code_t tk_detect_ground_plane(
    const tk_navigation_engine_t* engine,
    const tk_vision_depth_map_t* depth_map,
    const tk_quaternion_t* orientation,
    float* ground_plane_coeffs,
    uint8_t* ground_mask
);
static tk_error_code_t tk_classify_traversability_cells(
    tk_navigation_engine_t* engine,
    const tk_vision_depth_map_t* depth_map,
    const float* ground_plane_coeffs,
    const uint8_t* ground_mask
);
static tk_error_code_t tk_find_hazards(
    const tk_navigation_engine_t* engine,
    tk_navigation_hazard_t* hazards_buffer,
    size_t buffer_capacity,
    size_t* out_hazard_count
);
static tk_error_code_t tk_find_clear_path(
    const tk_navigation_engine_t* engine,
    float* out_clear_path_direction_deg,
    float* out_max_clear_distance_m
);

//------------------------------------------------------------------------------
// Engine Lifecycle Management
//------------------------------------------------------------------------------

tk_error_code_t tk_navigation_engine_create(tk_navigation_engine_t** out_engine, const tk_navigation_config_t* config) {
    // Validate input parameters
    if (!out_engine || !config) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    // Validate configuration
    tk_error_code_t result = tk_validate_config(config);
    if (result != TK_SUCCESS) {
        return result;
    }

    // Allocate memory for the engine
    tk_navigation_engine_t* engine = (tk_navigation_engine_t*)calloc(1, sizeof(tk_navigation_engine_t));
    if (!engine) {
        return TK_ERROR_OUT_OF_MEMORY;
    }

    // Copy configuration
    engine->config = *config;
    engine->is_initialized = false;

    // Initialize traversability map
    result = tk_initialize_traversability_map(engine);
    if (result != TK_SUCCESS) {
        free(engine);
        return result;
    }

    engine->is_initialized = true;
    *out_engine = engine;
    return TK_SUCCESS;
}

void tk_navigation_engine_destroy(tk_navigation_engine_t** engine) {
    if (!engine || !*engine) {
        return;
    }

    tk_destroy_traversability_map(*engine);
    free(*engine);
    *engine = NULL;
}

//------------------------------------------------------------------------------
// Core Processing and State Update
//------------------------------------------------------------------------------

tk_error_code_t tk_navigation_engine_update(
    tk_navigation_engine_t* engine,
    const tk_vision_depth_map_t* depth_map,
    const tk_quaternion_t* orientation
) {
    // Validate inputs
    if (!engine || !depth_map || !orientation) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    if (!engine->is_initialized) {
        return TK_ERROR_INVALID_STATE;
    }

    // Update the traversability grid based on new depth data
    return tk_update_traversability_grid(engine, depth_map, orientation);
}

//------------------------------------------------------------------------------
// High-Level Queries
//------------------------------------------------------------------------------

tk_error_code_t tk_navigation_engine_get_map(tk_navigation_engine_t* engine, tk_traversability_map_t* out_map) {
    if (!engine || !out_map) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    if (!engine->is_initialized) {
        return TK_ERROR_INVALID_STATE;
    }

    *out_map = engine->traversability_map;
    return TK_SUCCESS;
}

tk_error_code_t tk_navigation_engine_query_hazards(
    tk_navigation_engine_t* engine,
    tk_navigation_hazard_t* hazards_buffer,
    size_t buffer_capacity,
    size_t* out_hazard_count
) {
    if (!engine || !hazards_buffer || !out_hazard_count) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    if (!engine->is_initialized) {
        return TK_ERROR_INVALID_STATE;
    }

    return tk_find_hazards(engine, hazards_buffer, buffer_capacity, out_hazard_count);
}

tk_error_code_t tk_navigation_engine_find_clear_path(
    tk_navigation_engine_t* engine,
    float* out_clear_path_direction_deg,
    float* out_max_clear_distance_m
) {
    if (!engine || !out_clear_path_direction_deg || !out_max_clear_distance_m) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    if (!engine->is_initialized) {
        return TK_ERROR_INVALID_STATE;
    }

    return tk_find_clear_path(engine, out_clear_path_direction_deg, out_max_clear_distance_m);
}

//------------------------------------------------------------------------------
// Private Helper Functions
//------------------------------------------------------------------------------

static float tk_deg_to_rad(float degrees) {
    return degrees * TK_DEG_TO_RAD_F;
}

static float tk_rad_to_deg(float radians) {
    return radians * TK_RAD_TO_DEG_F;
}

static float tk_normalize_angle_deg(float angle_deg) {
    while (angle_deg > 180.0f) angle_deg -= 360.0f;
    while (angle_deg < -180.0f) angle_deg += 360.0f;
    return angle_deg;
}

static tk_error_code_t tk_validate_config(const tk_navigation_config_t* config) {
    if (config->camera_height_m <= 0.0f) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    if (config->default_camera_pitch_deg < -90.0f || config->default_camera_pitch_deg > 90.0f) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    if (config->step_height_threshold_m <= 0.0f) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    if (config->user_clearance_width_m <= 0.0f) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    if (config->max_analysis_distance_m <= 0.0f) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    return TK_SUCCESS;
}

static tk_error_code_t tk_initialize_traversability_map(tk_navigation_engine_t* engine) {
    // Define grid dimensions (example values - should be configurable)
    const uint32_t grid_width = 64;   // cells
    const uint32_t grid_height = 64;  // cells
    const float resolution_m_per_cell = 0.1f; // 10cm per cell

    // Allocate grid memory
    size_t grid_size = grid_width * grid_height * sizeof(tk_traversability_type_e);
    tk_traversability_type_e* grid = (tk_traversability_type_e*)calloc(1, grid_size);
    if (!grid) {
        return TK_ERROR_OUT_OF_MEMORY;
    }

    // Initialize grid to unknown state
    for (size_t i = 0; i < grid_width * grid_height; ++i) {
        grid[i] = TK_TRAVERSABILITY_UNKNOWN;
    }

    // Set up traversability map structure
    engine->traversability_map.width = grid_width;
    engine->traversability_map.height = grid_height;
    engine->traversability_map.resolution_m_per_cell = resolution_m_per_cell;
    engine->traversability_map.grid = grid;

    return TK_SUCCESS;
}

static void tk_destroy_traversability_map(tk_navigation_engine_t* engine) {
    if (engine && engine->traversability_map.grid) {
        free(engine->traversability_map.grid);
        engine->traversability_map.grid = NULL;
        engine->traversability_map.width = 0;
        engine->traversability_map.height = 0;
    }
}

static tk_error_code_t tk_update_traversability_grid(
    tk_navigation_engine_t* engine,
    const tk_vision_depth_map_t* depth_map,
    const tk_quaternion_t* orientation
) {
    // Detect ground plane
    float ground_plane_coeffs[4]; // ax + by + cz + d = 0
    uint8_t* ground_mask = (uint8_t*)calloc(depth_map->width * depth_map->height, sizeof(uint8_t));
    if (!ground_mask) {
        return TK_ERROR_OUT_OF_MEMORY;
    }

    tk_error_code_t result = tk_detect_ground_plane(engine, depth_map, orientation, ground_plane_coeffs, ground_mask);
    if (result != TK_SUCCESS) {
        free(ground_mask);
        return result;
    }

    // Classify traversability cells
    result = tk_classify_traversability_cells(engine, depth_map, ground_plane_coeffs, ground_mask);

    free(ground_mask);
    return result;
}

static tk_error_code_t tk_detect_ground_plane(
    const tk_navigation_engine_t* engine,
    const tk_vision_depth_map_t* depth_map,
    const tk_quaternion_t* orientation,
    float* ground_plane_coeffs,
    uint8_t* ground_mask
) {
    // In a real implementation, this would use RANSAC or similar algorithm
    // to fit a plane to the ground points in the depth map.
    // For now, we'll use a simplified approach assuming a flat ground plane
    
    // Extract camera parameters
    const float camera_height = engine->config.camera_height_m;
    const float camera_pitch = tk_deg_to_rad(engine->config.default_camera_pitch_deg);
    
    // Calculate ground plane coefficients based on camera setup
    // For a flat ground plane at z=0 with camera at height h looking down at angle p:
    // Normal vector: (0, sin(p), cos(p))
    // Point on plane: (0, 0, 0)
    // Plane equation: 0*x + sin(p)*y + cos(p)*z + 0 = 0
    // Simplified: sin(p)*y + cos(p)*z = 0
    
    ground_plane_coeffs[0] = 0.0f; // a
    ground_plane_coeffs[1] = sinf(camera_pitch); // b
    ground_plane_coeffs[2] = cosf(camera_pitch); // c
    ground_plane_coeffs[3] = -camera_height * cosf(camera_pitch); // d
    
    // Create ground mask (simplified - all points are considered ground in this basic version)
    memset(ground_mask, 1, depth_map->width * depth_map->height);
    
    return TK_SUCCESS;
}

static tk_error_code_t tk_classify_traversability_cells(
    tk_navigation_engine_t* engine,
    const tk_vision_depth_map_t* depth_map,
    const float* ground_plane_coeffs,
    const uint8_t* ground_mask
) {
    // Get map properties
    const uint32_t grid_width = engine->traversability_map.width;
    const uint32_t grid_height = engine->traversability_map.height;
    const float resolution = engine->traversability_map.resolution_m_per_cell;
    
    // Reset grid to unknown
    memset(engine->traversability_map.grid, 
           TK_TRAVERSABILITY_UNKNOWN, 
           grid_width * grid_height * sizeof(tk_traversability_type_e));
    
    // Process each cell in the traversability grid
    for (uint32_t y = 0; y < grid_height; ++y) {
        for (uint32_t x = 0; x < grid_width; ++x) {
            // Convert grid cell to world coordinates (simplified projection)
            float world_x = (x - grid_width/2.0f) * resolution;
            float world_y = y * resolution;
            
            // In a real implementation, we would project this world point back
            // to the depth map and check the actual depth values.
            // For this example, we'll use a simplified heuristic:
            
            // Check if point is within analysis range
            float distance = sqrtf(world_x * world_x + world_y * world_y);
            if (distance > engine->config.max_analysis_distance_m) {
                engine->traversability_map.grid[y * grid_width + x] = TK_TRAVERSABILITY_UNKNOWN;
                continue;
            }
            
            // Simple classification based on distance and position
            if (distance < 1.0f) {
                // Close range - assume traversable
                engine->traversability_map.grid[y * grid_width + x] = TK_TRAVERSABILITY_TRAVERSABLE;
            } else if (world_x > 0.5f) {
                // Right side obstacle
                engine->traversability_map.grid[y * grid_width + x] = TK_TRAVERSABILITY_OBSTACLE;
            } else if (world_x < -0.5f) {
                // Left side obstacle
                engine->traversability_map.grid[y * grid_width + x] = TK_TRAVERSABILITY_OBSTACLE;
            } else {
                // Center path - check for steps
                if (y > grid_height * 0.7f) {
                    engine->traversability_map.grid[y * grid_width + x] = TK_TRAVERSABILITY_HAZARD_STEP_DOWN;
                } else {
                    engine->traversability_map.grid[y * grid_width + x] = TK_TRAVERSABILITY_TRAVERSABLE;
                }
            }
        }
    }
    
    return TK_SUCCESS;
}

static tk_error_code_t tk_find_hazards(
    const tk_navigation_engine_t* engine,
    tk_navigation_hazard_t* hazards_buffer,
    size_t buffer_capacity,
    size_t* out_hazard_count
) {
    *out_hazard_count = 0;
    
    // Get map properties
    const uint32_t grid_width = engine->traversability_map.width;
    const uint32_t grid_height = engine->traversability_map.height;
    const float resolution = engine->traversability_map.resolution_m_per_cell;
    
    // Scan the grid for hazards
    for (uint32_t y = 0; y < grid_height && *out_hazard_count < buffer_capacity; ++y) {
        for (uint32_t x = 0; x < grid_width && *out_hazard_count < buffer_capacity; ++x) {
            tk_traversability_type_e type = engine->traversability_map.grid[y * grid_width + x];
            
            // Only report critical hazards
            if (type == TK_TRAVERSABILITY_HAZARD_STEP_DOWN || 
                type == TK_TRAVERSABILITY_HAZARD_HOLE) {
                
                // Calculate hazard position
                float world_x = (x - grid_width/2.0f) * resolution;
                float world_y = y * resolution;
                
                // Convert to polar coordinates
                float distance = sqrtf(world_x * world_x + world_y * world_y);
                float direction = tk_rad_to_deg(atan2f(world_x, world_y));
                
                // Add to buffer
                hazards_buffer[*out_hazard_count].type = type;
                hazards_buffer[*out_hazard_count].distance_m = distance;
                hazards_buffer[*out_hazard_count].direction_deg = direction;
                (*out_hazard_count)++;
            }
        }
    }
    
    return TK_SUCCESS;
}

static tk_error_code_t tk_find_clear_path(
    const tk_navigation_engine_t* engine,
    float* out_clear_path_direction_deg,
    float* out_max_clear_distance_m
) {
    // Get map properties
    const uint32_t grid_width = engine->traversability_map.width;
    const uint32_t grid_height = engine->traversability_map.height;
    const float resolution = engine->traversability_map.resolution_m_per_cell;
    const float min_clearance = engine->config.user_clearance_width_m;
    
    // Find the clearest path by scanning sectors
    float best_direction = 0.0f;
    float max_distance = 0.0f;
    bool found_clear_path = false;
    
    // Scan in 10-degree increments
    for (int angle_deg = -45; angle_deg <= 45; angle_deg += 10) {
        float angle_rad = tk_deg_to_rad(angle_deg);
        float sin_angle = sinf(angle_rad);
        float cos_angle = cosf(angle_rad);
        
        // Check path clearance
        float current_clearance = 0.0f;
        float path_distance = 0.0f;
        bool path_blocked = false;
        
        // Scan along the path
        for (uint32_t step = 0; step < grid_height; ++step) {
            float distance = step * resolution;
            if (distance > engine->config.max_analysis_distance_m) break;
            
            // Calculate grid position
            float world_x = distance * sin_angle;
            float world_y = distance * cos_angle;
            
            // Convert to grid coordinates
            int grid_x = (int)(world_x / resolution + grid_width/2.0f);
            int grid_y = (int)(world_y / resolution);
            
            // Check bounds
            if (grid_x < 0 || grid_x >= (int)grid_width || 
                grid_y < 0 || grid_y >= (int)grid_height) {
                break;
            }
            
            // Check cell type
            tk_traversability_type_e type = engine->traversability_map.grid[grid_y * grid_width + grid_x];
            if (type == TK_TRAVERSABILITY_OBSTACLE || 
                type == TK_TRAVERSABILITY_HAZARD_STEP_DOWN || 
                type == TK_TRAVERSABILITY_HAZARD_HOLE) {
                path_blocked = true;
                break;
            }
            
            path_distance = distance;
        }
        
        // If path is clear and longer than previous best
        if (!path_blocked && path_distance > max_distance) {
            max_distance = path_distance;
            best_direction = angle_deg;
            found_clear_path = true;
        }
    }
    
    if (found_clear_path) {
        *out_clear_path_direction_deg = best_direction;
        *out_max_clear_distance_m = max_distance;
        return TK_SUCCESS;
    } else {
        return TK_ERROR_NOT_FOUND;
    }
}
