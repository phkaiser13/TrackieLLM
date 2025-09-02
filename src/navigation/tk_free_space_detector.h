/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_free_space_detector.h
*
* This header file defines the public API for the Free Space Detector. This is a
* specialized analysis component within the Tactical Navigation Engine. Its sole
* purpose is to analyze the traversability map and produce a high-level,
* quantitative summary of the safe navigation paths available to the user.
*
* Instead of simply identifying obstacles, this module focuses on characterizing
* the "negative space". It employs a sector-based analysis to provide the Cortex
* with a panoramic understanding of open corridors, enabling directional guidance
* like "clear path slightly to your left".
*
* This module consumes the output of the main navigation engine (`tk_path_planner`)
* and provides its analysis to the Cortex.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TRACKIELLM_NAVIGATION_TK_FREE_SPACE_DETECTOR_H
#define TRACKIELLM_NAVIGATION_TK_FREE_SPACE_DETECTOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "navigation/tk_path_planner.h" // For tk_traversability_map_t

// Forward-declare the primary detector object as an opaque type.
typedef struct tk_free_space_detector_s tk_free_space_detector_t;

/**
 * @struct tk_free_space_config_t
 * @brief Configuration for initializing the Free Space Detector.
 */
typedef struct {
    uint32_t num_angular_sectors;   /**< The number of angular sectors to divide the analysis into (e.g., 7). */
    float    analysis_fov_deg;      /**< The total horizontal field of view to analyze in degrees (e.g., 90.0). */
    float    user_clearance_width_m;/**< The minimum width of a path for it to be considered clear.
                                         This should match the navigation engine's config. */
} tk_free_space_config_t;

/**
 * @struct tk_space_sector_t
 * @brief Represents the analysis result for a single angular sector.
 */
typedef struct {
    float center_angle_deg;         /**< The center angle of this sector in degrees (0 is straight ahead). */
    float max_clear_distance_m;     /**< The maximum traversable distance within this sector. */
    bool  is_clear;                 /**< True if a path of `user_clearance_width_m` exists in this sector. */
} tk_space_sector_t;

/**
 * @struct tk_free_space_analysis_t
 * @brief Contains the complete, structured analysis of the free space.
 */
typedef struct {
    const tk_space_sector_t* sectors;       /**< Pointer to an array of sector analysis results. Owned by the detector. */
    size_t                   sector_count;  /**< The number of sectors in the array. */
    
    // --- High-level summary ---
    bool                     is_any_path_clear; /**< True if at least one clear path was found. */
    float                    clearest_path_angle_deg; /**< The angle of the sector with the greatest clear distance. */
    float                    clearest_path_distance_m;/**< The maximum clear distance found. */
} tk_free_space_analysis_t;


#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Engine Lifecycle Management
//------------------------------------------------------------------------------

/**
 * @brief Creates and initializes a new Free Space Detector instance.
 *
 * @param[out] out_detector Pointer to receive the address of the new detector instance.
 * @param[in] config The configuration for the analysis.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL or config is invalid.
 * @return TK_ERROR_OUT_OF_MEMORY on memory allocation failure.
 */
TK_NODISCARD tk_error_code_t tk_free_space_detector_create(tk_free_space_detector_t** out_detector, const tk_free_space_config_t* config);

/**
 * @brief Destroys a Free Space Detector instance.
 *
 * @param[in,out] detector Pointer to the detector instance to be destroyed.
 */
void tk_free_space_detector_destroy(tk_free_space_detector_t** detector);

//------------------------------------------------------------------------------
// Core Processing and Analysis
//------------------------------------------------------------------------------

/**
 * @brief Analyzes a traversability map to characterize the free space.
 *
 * This is the core function of the detector. It takes the grid-based map and
 * performs a geometric analysis to populate its internal sector-based model
 * of the environment.
 *
 * @param[in] detector The free space detector instance.
 * @param[in] map The latest traversability map from the navigation engine.
 *
 * @return TK_SUCCESS on successful analysis.
 * @return TK_ERROR_INVALID_ARGUMENT if required inputs are NULL.
 *
 * @par Thread-Safety
 * This function is NOT thread-safe. It should only be called from the main
 * Cortex processing loop.
 */
TK_NODISCARD tk_error_code_t tk_free_space_detector_analyze(
    tk_free_space_detector_t* detector,
    const tk_traversability_map_t* map
);

//------------------------------------------------------------------------------
// High-Level Queries
//------------------------------------------------------------------------------

/**
 * @brief Retrieves the latest, most up-to-date analysis of the free space.
 *
 * The data pointed to by the returned structure is owned by the detector and is
 * valid only until the next call to `tk_free_space_detector_analyze`.
 *
 * @param[in] detector The free space detector instance.
 * @param[out] out_analysis Pointer to a structure that will be filled with the
 *                          analysis results, including a pointer to the sector data.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_INVALID_STATE if `analyze` has not been called yet.
 */
TK_NODISCARD tk_error_code_t tk_free_space_detector_get_analysis(
    tk_free_space_detector_t* detector,
    tk_free_space_analysis_t* out_analysis
);


#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_NAVIGATION_TK_FREE_SPACE_DETECTOR_H