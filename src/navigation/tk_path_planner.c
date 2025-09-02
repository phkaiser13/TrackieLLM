/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_path_planner.c
 * 
 * This implementation file provides the core logic for the Tactical Navigation Engine.
 * It processes depth maps to generate traversability grids and identifies safe paths
 * for navigation while detecting critical hazards.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "navigation/tk_path_planner.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

// Constants for internal calculations
#define TK_PI_F 3.14159265358979323846f
#define TK_DEG_TO_RAD_F (TK_PI_F / 180.0f)
#define TK_RAD_TO_DEG_F (180.0f / TK_PI_F)
#define RANSAC_MAX_ITERATIONS 100
#define RANSAC_DISTANCE_THRESHOLD 0.05f // 5cm threshold for ground points

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

// Point3D structure for 3D point cloud processing
typedef struct {
    float x, y, z;
} point3d_t;

// RANSAC helper functions
static void tk_unproject_depth_to_point_cloud(
    const tk_vision_depth_map_t* depth_map,
    const tk_quaternion_t* orientation,
    const tk_navigation_config_t* config,
    point3d_t* point_cloud,
    size_t* out_point_count
);
static void tk_apply_orientation_correction(
    point3d_t* points,
    size_t point_count,
    const tk_quaternion_t* orientation
);
static int tk_ransac_fit_plane(
    const point3d_t* points,
    size_t point_count,
    float* plane_coeffs,
    uint8_t* inlier_mask,
    int max_iterations,
    float distance_threshold
);
static float tk_point_to_plane_distance(
    const point3d_t* point,
    const float* plane_coeffs
);
static void tk_normalize_plane_coefficients(float* coeffs);

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
    // Create point cloud from depth map
    point3d_t* point_cloud = (point3d_t*)malloc(depth_map->width * depth_map->height * sizeof(point3d_t));
    if (!point_cloud) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    size_t point_count = 0;
    tk_unproject_depth_to_point_cloud(depth_map, orientation, &engine->config, point_cloud, &point_count);
    
    // Apply orientation correction to align points with world coordinates
    tk_apply_orientation_correction(point_cloud, point_count, orientation);
    
    // Use RANSAC to fit a plane to the ground points
    int inlier_count = tk_ransac_fit_plane(
        point_cloud, 
        point_count, 
        ground_plane_coeffs, 
        ground_mask, 
        RANSAC_MAX_ITERATIONS, 
        RANSAC_DISTANCE_THRESHOLD
    );
    
    free(point_cloud);
    
    if (inlier_count < 10) { // Need at least 10 points to consider valid ground plane
        // Fallback to default ground plane if RANSAC fails
        const float camera_height = engine->config.camera_height_m;
        const float camera_pitch = tk_deg_to_rad(engine->config.default_camera_pitch_deg);
        
        ground_plane_coeffs[0] = 0.0f; // a
        ground_plane_coeffs[1] = sinf(camera_pitch); // b
        ground_plane_coeffs[2] = cosf(camera_pitch); // c
        ground_plane_coeffs[3] = -camera_height * cosf(camera_pitch); // d
        
        // Mark all points as potential ground in fallback mode
        memset(ground_mask, 1, depth_map->width * depth_map->height);
        return TK_SUCCESS;
    }
    
    // Normalize plane coefficients
    tk_normalize_plane_coefficients(ground_plane_coeffs);
    
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
    
    // Create point cloud for processing
    point3d_t* point_cloud = (point3d_t*)malloc(depth_map->width * depth_map->height * sizeof(point3d_t));
    if (!point_cloud) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    size_t point_count = 0;
    tk_unproject_depth_to_point_cloud(depth_map, NULL, &engine->config, point_cloud, &point_count);
    
    // Process each point in the point cloud
    for (size_t i = 0; i < point_count; ++i) {
        const point3d_t* point = &point_cloud[i];
        
        // Calculate distance to ground plane
        float distance = tk_point_to_plane_distance(point, ground_plane_coeffs);
        
        // Convert 3D point to grid coordinates
        // Assuming camera is at origin looking along positive Y axis
        int grid_x = (int)((point->x / resolution) + grid_width/2.0f);
        int grid_y = (int)(point->z / resolution); // Using z as forward direction
        
        // Check bounds
        if (grid_x >= 0 && grid_x < (int)grid_width && 
            grid_y >= 0 && grid_y < (int)grid_height) {
            
            // Classify based on distance to ground plane
            if (fabsf(distance) <= 0.05f) { // Within 5cm of ground plane
                engine->traversability_map.grid[grid_y * grid_width + grid_x] = TK_TRAVERSABILITY_TRAVERSABLE;
            } else if (distance > engine->config.step_height_threshold_m) {
                engine->traversability_map.grid[grid_y * grid_width + grid_x] = TK_TRAVERSABILITY_OBSTACLE;
            } else if (distance < -engine->config.step_height_threshold_m) {
                engine->traversability_map.grid[grid_y * grid_width + grid_x] = TK_TRAVERSABILITY_HAZARD_STEP_DOWN;
            } else {
                engine->traversability_map.grid[grid_y * grid_width + grid_x] = TK_TRAVERSABILITY_TRAVERSABLE;
            }
        }
    }
    
    free(point_cloud);
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

//------------------------------------------------------------------------------
// RANSAC Implementation for Ground Plane Detection
//------------------------------------------------------------------------------

static void tk_unproject_depth_to_point_cloud(
    const tk_vision_depth_map_t* depth_map,
    const tk_quaternion_t* orientation,
    const tk_navigation_config_t* config,
    point3d_t* point_cloud,
    size_t* out_point_count
) {
    // Simplified camera intrinsic parameters (should be configurable)
    const float fx = depth_map->width * 0.8f;  // Focal length x
    const float fy = depth_map->height * 0.8f; // Focal length y
    const float cx = depth_map->width / 2.0f;  // Principal point x
    const float cy = depth_map->height / 2.0f; // Principal point y
    
    size_t point_idx = 0;
    
    for (uint32_t v = 0; v < depth_map->height; ++v) {
        for (uint32_t u = 0; u < depth_map->width; ++u) {
            size_t idx = v * depth_map->width + u;
            float depth = depth_map->data[idx];
            
            // Skip invalid depth values
            if (depth <= 0.0f || depth > config->max_analysis_distance_m) {
                continue;
            }
            
            // Convert pixel coordinates to 3D world coordinates
            // Using pinhole camera model
            point_cloud[point_idx].x = (u - cx) * depth / fx;
            point_cloud[point_idx].y = (v - cy) * depth / fy;
            point_cloud[point_idx].z = depth;
            
            point_idx++;
        }
    }
    
    *out_point_count = point_idx;
}

static void tk_apply_orientation_correction(
    point3d_t* points,
    size_t point_count,
    const tk_quaternion_t* orientation
) {
    // Convert quaternion to rotation matrix
    float w = orientation->w;
    float x = orientation->x;
    float y = orientation->y;
    float z = orientation->z;
    
    // Rotation matrix elements
    float r11 = 1 - 2*(y*y + z*z);
    float r12 = 2*(x*y - w*z);
    float r13 = 2*(x*z + w*y);
    
    float r21 = 2*(x*y + w*z);
    float r22 = 1 - 2*(x*x + z*z);
    float r23 = 2*(y*z - w*x);
    
    float r31 = 2*(x*z - w*y);
    float r32 = 2*(y*z + w*x);
    float r33 = 1 - 2*(x*x + y*y);
    
    // Apply rotation to each point
    for (size_t i = 0; i < point_count; ++i) {
        point3d_t* p = &points[i];
        float px = p->x;
        float py = p->y;
        float pz = p->z;
        
        p->x = r11*px + r12*py + r13*pz;
        p->y = r21*px + r22*py + r23*pz;
        p->z = r31*px + r32*py + r33*pz;
    }
}

static int tk_ransac_fit_plane(
    const point3d_t* points,
    size_t point_count,
    float* plane_coeffs,
    uint8_t* inlier_mask,
    int max_iterations,
    float distance_threshold
) {
    if (point_count < 3) {
        return 0;
    }
    
    int best_inlier_count = 0;
    float best_plane[4] = {0};
    
    // Initialize inlier mask
    memset(inlier_mask, 0, point_count * sizeof(uint8_t));
    
    // RANSAC iterations
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Randomly select 3 points
        int idx1 = rand() % point_count;
        int idx2 = rand() % point_count;
        int idx3 = rand() % point_count;
        
        // Ensure we have 3 distinct points
        if (idx1 == idx2 || idx2 == idx3 || idx1 == idx3) {
            continue;
        }
        
        const point3d_t* p1 = &points[idx1];
        const point3d_t* p2 = &points[idx2];
        const point3d_t* p3 = &points[idx3];
        
        // Calculate plane normal using cross product
        float v1x = p2->x - p1->x;
        float v1y = p2->y - p1->y;
        float v1z = p2->z - p1->z;
        
        float v2x = p3->x - p1->x;
        float v2y = p3->y - p1->y;
        float v2z = p3->z - p1->z;
        
        // Cross product v1 x v2
        float nx = v1y*v2z - v1z*v2y;
        float ny = v1z*v2x - v1x*v2z;
        float nz = v1x*v2y - v1y*v2x;
        
        // Normalize normal vector
        float norm = sqrtf(nx*nx + ny*ny + nz*nz);
        if (norm < 1e-6) continue; // Degenerate case
        
        nx /= norm;
        ny /= norm;
        nz /= norm;
        
        // Plane equation: nx*x + ny*y + nz*z + d = 0
        // Calculate d using point p1
        float d = -(nx*p1->x + ny*p1->y + nz*p1->z);
        
        // Count inliers
        int inlier_count = 0;
        for (size_t i = 0; i < point_count; ++i) {
            float dist = fabsf(nx*points[i].x + ny*points[i].y + nz*points[i].z + d);
            if (dist < distance_threshold) {
                inlier_count++;
            }
        }
        
        // Update best model if this one is better
        if (inlier_count > best_inlier_count) {
            best_inlier_count = inlier_count;
            best_plane[0] = nx;
            best_plane[1] = ny;
            best_plane[2] = nz;
            best_plane[3] = d;
            
            // Update inlier mask
            memset(inlier_mask, 0, point_count * sizeof(uint8_t));
            for (size_t i = 0; i < point_count; ++i) {
                float dist = fabsf(nx*points[i].x + ny*points[i].y + nz*points[i].z + d);
                if (dist < distance_threshold) {
                    inlier_mask[i] = 1;
                }
            }
        }
    }
    
    // Copy best plane coefficients
    plane_coeffs[0] = best_plane[0];
    plane_coeffs[1] = best_plane[1];
    plane_coeffs[2] = best_plane[2];
    plane_coeffs[3] = best_plane[3];
    
    return best_inlier_count;
}

static float tk_point_to_plane_distance(
    const point3d_t* point,
    const float* plane_coeffs
) {
    // Plane equation: ax + by + cz + d = 0
    // Distance = |ax + by + cz + d| / sqrt(a^2 + b^2 + c^2)
    float a = plane_coeffs[0];
    float b = plane_coeffs[1];
    float c = plane_coeffs[2];
    float d = plane_coeffs[3];
    
    float numerator = fabsf(a*point->x + b*point->y + c*point->z + d);
    float denominator = sqrtf(a*a + b*b + c*c);
    
    if (denominator < 1e-6) {
        return 0.0f;
    }
    
    return numerator / denominator;
}

static void tk_normalize_plane_coefficients(float* coeffs) {
    float norm = sqrtf(coeffs[0]*coeffs[0] + coeffs[1]*coeffs[1] + coeffs[2]*coeffs[2]);
    if (norm > 1e-6) {
        coeffs[0] /= norm;
        coeffs[1] /= norm;
        coeffs[2] /= norm;
        coeffs[3] /= norm;
    }
}
