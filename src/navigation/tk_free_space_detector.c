/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_free_space_detector.c
 * 
 * This implementation file provides the core logic for the Free Space Detector.
 * It analyzes traversability maps to identify clear paths and characterize the
 * navigable space around the user.
 *
 * SPDX-License-Identifier: AGPL-3.0 license AGPL-3.0 license
 */

#include "navigation/tk_free_space_detector.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// Constants for internal calculations
#define TK_PI_F 3.14159265358979323846f
#define TK_DEG_TO_RAD_F (TK_PI_F / 180.0f)
#define TK_RAD_TO_DEG_F (180.0f / TK_PI_F)

// Internal structure for the free space detector state
struct tk_free_space_detector_s {
    tk_free_space_config_t config;
    tk_space_sector_t* sectors;
    bool is_initialized;
    bool has_analyzed;
    
    // Cached analysis results
    tk_free_space_analysis_t analysis;
};

// Private helper functions
static tk_error_code_t tk_validate_config(const tk_free_space_config_t* config);
static tk_error_code_t tk_initialize_sectors(tk_free_space_detector_t* detector);
static void tk_destroy_sectors(tk_free_space_detector_t* detector);
static tk_error_code_t tk_perform_sector_analysis(
    tk_free_space_detector_t* detector,
    const tk_traversability_map_t* map
);
static float tk_calculate_sector_width_deg(const tk_free_space_config_t* config);
static int tk_get_sector_index(float angle_deg, const tk_free_space_config_t* config);
static float tk_normalize_angle_deg(float angle_deg);

//------------------------------------------------------------------------------
// Engine Lifecycle Management
//------------------------------------------------------------------------------

tk_error_code_t tk_free_space_detector_create(tk_free_space_detector_t** out_detector, const tk_free_space_config_t* config) {
    // Validate input parameters
    if (!out_detector || !config) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    // Validate configuration
    tk_error_code_t result = tk_validate_config(config);
    if (result != TK_SUCCESS) {
        return result;
    }

    // Allocate memory for the detector
    tk_free_space_detector_t* detector = (tk_free_space_detector_t*)calloc(1, sizeof(tk_free_space_detector_t));
    if (!detector) {
        return TK_ERROR_OUT_OF_MEMORY;
    }

    // Copy configuration
    detector->config = *config;
    detector->is_initialized = false;
    detector->has_analyzed = false;

    // Initialize sectors
    result = tk_initialize_sectors(detector);
    if (result != TK_SUCCESS) {
        free(detector);
        return result;
    }

    detector->is_initialized = true;
    detector->has_analyzed = false;
    
    // Initialize analysis structure
    detector->analysis.sectors = detector->sectors;
    detector->analysis.sector_count = detector->config.num_angular_sectors;
    detector->analysis.is_any_path_clear = false;
    detector->analysis.clearest_path_angle_deg = 0.0f;
    detector->analysis.clearest_path_distance_m = 0.0f;

    *out_detector = detector;
    return TK_SUCCESS;
}

void tk_free_space_detector_destroy(tk_free_space_detector_t** detector) {
    if (!detector || !*detector) {
        return;
    }

    tk_destroy_sectors(*detector);
    free(*detector);
    *detector = NULL;
}

//------------------------------------------------------------------------------
// Core Processing and Analysis
//------------------------------------------------------------------------------

tk_error_code_t tk_free_space_detector_analyze(
    tk_free_space_detector_t* detector,
    const tk_traversability_map_t* map
) {
    // Validate inputs
    if (!detector || !map) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    if (!detector->is_initialized) {
        return TK_ERROR_INVALID_STATE;
    }

    // Perform sector-based analysis of the traversability map
    tk_error_code_t result = tk_perform_sector_analysis(detector, map);
    if (result != TK_SUCCESS) {
        return result;
    }

    detector->has_analyzed = true;
    return TK_SUCCESS;
}

//------------------------------------------------------------------------------
// High-Level Queries
//------------------------------------------------------------------------------

tk_error_code_t tk_free_space_detector_get_analysis(
    tk_free_space_detector_t* detector,
    tk_free_space_analysis_t* out_analysis
) {
    if (!detector || !out_analysis) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    if (!detector->is_initialized || !detector->has_analyzed) {
        return TK_ERROR_INVALID_STATE;
    }

    *out_analysis = detector->analysis;
    return TK_SUCCESS;
}

//------------------------------------------------------------------------------
// Private Helper Functions
//------------------------------------------------------------------------------

static tk_error_code_t tk_validate_config(const tk_free_space_config_t* config) {
    if (config->num_angular_sectors == 0) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    if (config->analysis_fov_deg <= 0.0f || config->analysis_fov_deg > 360.0f) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    if (config->user_clearance_width_m <= 0.0f) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    return TK_SUCCESS;
}

static tk_error_code_t tk_initialize_sectors(tk_free_space_detector_t* detector) {
    // Allocate memory for sectors
    size_t sectors_size = detector->config.num_angular_sectors * sizeof(tk_space_sector_t);
    detector->sectors = (tk_space_sector_t*)calloc(1, sectors_size);
    if (!detector->sectors) {
        return TK_ERROR_OUT_OF_MEMORY;
    }

    // Initialize sector angles
    float sector_width = tk_calculate_sector_width_deg(&detector->config);
    float start_angle = -detector->config.analysis_fov_deg / 2.0f;
    
    for (uint32_t i = 0; i < detector->config.num_angular_sectors; ++i) {
        detector->sectors[i].center_angle_deg = start_angle + (i + 0.5f) * sector_width;
        detector->sectors[i].max_clear_distance_m = 0.0f;
        detector->sectors[i].is_clear = false;
    }

    return TK_SUCCESS;
}

static void tk_destroy_sectors(tk_free_space_detector_t* detector) {
    if (detector && detector->sectors) {
        free(detector->sectors);
        detector->sectors = NULL;
    }
}

static tk_error_code_t tk_perform_sector_analysis(
    tk_free_space_detector_t* detector,
    const tk_traversability_map_t* map
) {
    // Reset sector data
    float sector_width = tk_calculate_sector_width_deg(&detector->config);
    float start_angle = -detector->config.analysis_fov_deg / 2.0f;
    
    for (uint32_t i = 0; i < detector->config.num_angular_sectors; ++i) {
        detector->sectors[i].center_angle_deg = start_angle + (i + 0.5f) * sector_width;
        detector->sectors[i].max_clear_distance_m = 0.0f;
        detector->sectors[i].is_clear = false;
    }

    // Initialize analysis summary
    detector->analysis.is_any_path_clear = false;
    detector->analysis.clearest_path_angle_deg = 0.0f;
    detector->analysis.clearest_path_distance_m = 0.0f;

    // Get map properties
    const uint32_t grid_width = map->width;
    const uint32_t grid_height = map->height;
    const float resolution = map->resolution_m_per_cell;
    
    // Analyze each cell in the traversability map
    for (uint32_t y = 0; y < grid_height; ++y) {
        for (uint32_t x = 0; x < grid_width; ++x) {
            // Get cell type
            tk_traversability_type_e cell_type = map->grid[y * grid_width + x];
            
            // Skip non-traversable cells
            if (cell_type != TK_TRAVERSABILITY_TRAVERSABLE) {
                continue;
            }
            
            // Convert grid cell to world coordinates
            float world_x = (x - grid_width/2.0f) * resolution;
            float world_y = y * resolution;
            
            // Calculate distance and angle
            float distance = sqrtf(world_x * world_x + world_y * world_y);
            float angle_deg = tk_normalize_angle_deg(tk_rad_to_deg(atan2f(world_x, world_y)));
            
            // Check if angle is within analysis FOV
            if (fabsf(angle_deg) > detector->config.analysis_fov_deg / 2.0f) {
                continue;
            }
            
            // Determine which sector this point belongs to
            int sector_index = tk_get_sector_index(angle_deg, &detector->config);
            if (sector_index < 0 || sector_index >= (int)detector->config.num_angular_sectors) {
                continue;
            }
            
            // Update sector maximum distance
            if (distance > detector->sectors[sector_index].max_clear_distance_m) {
                detector->sectors[sector_index].max_clear_distance_m = distance;
            }
        }
    }
    
    // Determine which sectors are clear based on user clearance width
    float half_clearance = detector->config.user_clearance_width_m / 2.0f;
    for (uint32_t i = 0; i < detector->config.num_angular_sectors; ++i) {
        // A sector is considered clear if it has sufficient distance
        // In a real implementation, we would also check for lateral clearance
        detector->sectors[i].is_clear = (detector->sectors[i].max_clear_distance_m > 0.5f);
        
        // Update analysis summary
        if (detector->sectors[i].is_clear) {
            detector->analysis.is_any_path_clear = true;
            
            if (detector->sectors[i].max_clear_distance_m > detector->analysis.clearest_path_distance_m) {
                detector->analysis.clearest_path_distance_m = detector->sectors[i].max_clear_distance_m;
                detector->analysis.clearest_path_angle_deg = detector->sectors[i].center_angle_deg;
            }
        }
    }
    
    return TK_SUCCESS;
}

static float tk_calculate_sector_width_deg(const tk_free_space_config_t* config) {
    return config->analysis_fov_deg / (float)config->num_angular_sectors;
}

static int tk_get_sector_index(float angle_deg, const tk_free_space_config_t* config) {
    float sector_width = tk_calculate_sector_width_deg(config);
    float start_angle = -config->analysis_fov_deg / 2.0f;
    
    if (angle_deg < start_angle || angle_deg > (start_angle + config->analysis_fov_deg)) {
        return -1; // Angle outside analysis range
    }
    
    return (int)((angle_deg - start_angle) / sector_width);
}

static float tk_normalize_angle_deg(float angle_deg) {
    while (angle_deg > 180.0f) angle_deg -= 360.0f;
    while (angle_deg < -180.0f) angle_deg += 360.0f;
    return angle_deg;
}
