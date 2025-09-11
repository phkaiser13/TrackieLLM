/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/vision/depth_processing.rs
 *
 * This file contains advanced analysis logic for depth maps, focusing on
 * extracting navigation-related cues for the TrackieLLM system. It is
 * responsible for identifying traversable ground, potential hazards like
 * holes and steps, and other features that can inform the Cortex about
 * the safety of the environment.
 *
 * This logic is designed to be called from the C-level pipeline via FFI.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use super::ffi;
use trackiellm_event_bus::{GroundPlaneStatus, NavigationCues, VerticalChange};
use std::slice;

// --- Constants for Navigation Analysis ---

/// The dimensions of the grid to impose on the ground plane (width, height).
const GRID_DIMS: (u32, u32) = (5, 5);
/// The vertical portion of the depth map to analyze (e.g., 0.5 means lower half).
const GROUND_PLANE_VERTICAL_RATIO: f32 = 0.5;
/// The minimum number of valid depth points required in a cell to consider it for analysis.
const MIN_VALID_POINTS_PER_CELL: usize = 10;
/// The depth difference (in meters) to classify something as a "hole" or "obstacle".
const HOLE_OBSTACLE_THRESHOLD_M: f32 = 0.5;
/// The depth difference (in meters) to classify a vertical change as a "step".
const STEP_THRESHOLD_M: f32 = 0.1;
/// The maximum depth difference to be considered a "ramp" instead of a step.
const RAMP_THRESHOLD_M: f32 = 0.05;


/// Analyzes a depth map to extract navigation cues like traversable area and hazards.
///
/// # Arguments
/// * `depth_map` - A pointer to the C-level depth map struct.
///
/// # Returns
/// An `Option<NavigationCues>` containing the analysis results. Returns `None` if the
/// depth map is invalid or analysis cannot be performed.
pub fn analyze_navigation_cues(depth_map: &ffi::tk_vision_depth_map_t) -> Option<NavigationCues> {
    if depth_map.data.is_null() || depth_map.width == 0 || depth_map.height == 0 {
        return None;
    }

    let depth_data = unsafe {
        slice::from_raw_parts(depth_map.data, (depth_map.width * depth_map.height) as usize)
    };

    // Define the region of interest (lower part of the image)
    let roi_y_start = (depth_map.height as f32 * (1.0 - GROUND_PLANE_VERTICAL_RATIO)) as u32;
    let roi_height = depth_map.height - roi_y_start;

    let cell_width = depth_map.width / GRID_DIMS.0;
    let cell_height = roi_height / GRID_DIMS.1;

    let mut avg_depths = vec![0.0; (GRID_DIMS.0 * GRID_DIMS.1) as usize];
    let mut traversability_grid = vec![GroundPlaneStatus::Unknown; avg_depths.len()];
    let mut detected_vertical_changes = Vec::new();

    // 1. Calculate the average depth for each cell in the grid
    for gy in 0..GRID_DIMS.1 {
        for gx in 0..GRID_DIMS.0 {
            let x_start = gx * cell_width;
            let y_start = roi_y_start + (gy * cell_height);
            let x_end = x_start + cell_width;
            let y_end = y_start + cell_height;

            let mut valid_points = Vec::new();
            for y in y_start..y_end {
                for x in x_start..x_end {
                    let idx = (y * depth_map.width + x) as usize;
                    if idx < depth_data.len() && depth_data[idx] > 0.0 {
                        valid_points.push(depth_data[idx]);
                    }
                }
            }

            if valid_points.len() >= MIN_VALID_POINTS_PER_CELL {
                let sum: f32 = valid_points.iter().sum();
                avg_depths[(gy * GRID_DIMS.0 + gx) as usize] = sum / valid_points.len() as f32;
            }
        }
    }

    // 2. Analyze the grid to determine traversability and find hazards
    // We assume the closest cell (bottom center) is our reference ground plane
    let reference_cell_gx = GRID_DIMS.0 / 2;
    let reference_cell_gy = GRID_DIMS.1 - 1; // Last row
    let reference_idx = (reference_cell_gy * GRID_DIMS.0 + reference_cell_gx) as usize;
    let reference_depth = avg_depths[reference_idx];

    if reference_depth <= 0.0 {
        // Cannot perform analysis without a valid reference point
        return None;
    }

    for gy in 0..GRID_DIMS.1 {
        for gx in 0..GRID_DIMS.0 {
            let current_idx = (gy * GRID_DIMS.0 + gx) as usize;
            let current_depth = avg_depths[current_idx];

            if current_depth <= 0.0 {
                traversability_grid[current_idx] = GroundPlaneStatus::Unknown;
                continue;
            }

            let depth_diff = current_depth - reference_depth;

            // Basic classification
            if depth_diff.abs() < RAMP_THRESHOLD_M {
                traversability_grid[current_idx] = GroundPlaneStatus::Flat;
            } else if depth_diff > HOLE_OBSTACLE_THRESHOLD_M {
                traversability_grid[current_idx] = GroundPlaneStatus::Hole;
            } else if depth_diff < -HOLE_OBSTACLE_THRESHOLD_M {
                traversability_grid[current_idx] = GroundPlaneStatus::Obstacle;
            }

            // Check for vertical changes with the cell directly "below" (closer to the viewer)
            if gy < GRID_DIMS.1 - 1 {
                let below_idx = ((gy + 1) * GRID_DIMS.0 + gx) as usize;
                let below_depth = avg_depths[below_idx];

                if below_depth > 0.0 {
                    let vertical_diff = current_depth - below_depth;

                    if vertical_diff.abs() > STEP_THRESHOLD_M {
                        let status = if vertical_diff > 0.0 {
                            GroundPlaneStatus::Hole // A drop-off
                        } else {
                            GroundPlaneStatus::Obstacle // A step up
                        };

                        detected_vertical_changes.push(VerticalChange {
                            height_m: vertical_diff.abs(),
                            status,
                            grid_index: (gx, gy),
                        });
                    } else if vertical_diff.abs() > RAMP_THRESHOLD_M {
                         let status = if vertical_diff > 0.0 {
                            GroundPlaneStatus::RampDown
                        } else {
                            GroundPlaneStatus::RampUp
                        };
                         traversability_grid[current_idx] = status;
                    }
                }
            }
        }
    }


    Some(NavigationCues {
        traversability_grid,
        grid_dimensions: GRID_DIMS,
        detected_vertical_changes,
    })
}
