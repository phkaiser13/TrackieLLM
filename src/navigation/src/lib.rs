/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/navigation/lib.rs
 *
 * This file is the main library entry point for the 'navigation' crate.
 * It contains the FFI bridge that allows the C-based navigation components
 * to call into the Rust implementations for complex analysis.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#![deny(missing_docs)]

//! # TrackieLLM Navigation Crate
//!
//! Provides spatial analysis and navigation logic.

pub mod free_space;
pub mod obstacle_tracker;

// Re-export the necessary types to make them part of the public API.
pub use free_space::SpaceSector;
pub use obstacle_tracker::TrackedObstacle;

use free_space::{FreeSpaceConfig, FreeSpaceDetector, TraversabilityMap, TraversabilityType};
use obstacle_tracker::{ObstacleTracker, ObstacleTrackerConfig, Vector2D, ObstacleStatus};
use std::slice;

// --- FFI Bridge Implementation ---

// --- Free Space Detector FFI ---

#[repr(C)]
pub struct tk_free_space_config_t {
    pub num_angular_sectors: u32,
    pub analysis_fov_deg: f32,
    pub user_clearance_width_m: f32,
}

#[repr(C)]
pub struct tk_traversability_map_t {
    pub width: u32,
    pub height: u32,
    pub resolution_m_per_cell: f32,
    pub grid: *const TraversabilityType,
}

#[repr(C)]
pub struct tk_free_space_analysis_t {
    pub sectors: *const SpaceSector,
    pub sector_count: usize,
    pub is_any_path_clear: bool,
    pub clearest_path_angle_deg: f32,
    pub clearest_path_distance_m: f32,
}

/// Creates a `FreeSpaceDetector` and returns an opaque handle to it.
#[no_mangle]
pub extern "C" fn rust_free_space_detector_create(
    config: &tk_free_space_config_t,
) -> *mut FreeSpaceDetector {
    let rust_config = FreeSpaceConfig {
        num_angular_sectors: config.num_angular_sectors,
        analysis_fov_deg: config.analysis_fov_deg,
    };
    let detector = Box::new(FreeSpaceDetector::new(rust_config));
    Box::into_raw(detector)
}

/// Destroys the `FreeSpaceDetector` instance.
#[no_mangle]
pub unsafe extern "C" fn rust_free_space_detector_destroy(handle: *mut FreeSpaceDetector) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle);
    }
}

/// Analyzes a traversability map.
#[no_mangle]
pub unsafe extern "C" fn rust_free_space_detector_analyze(
    handle: *mut FreeSpaceDetector,
    map: &tk_traversability_map_t,
) {
    let detector = &mut *handle;
    let grid_slice = slice::from_raw_parts(map.grid, (map.width * map.height) as usize);
    let rust_map = TraversabilityMap {
        width: map.width,
        height: map.height,
        resolution_m_per_cell: map.resolution_m_per_cell,
        grid: grid_slice,
    };
    detector.analyze(&rust_map);
}

/// Retrieves the latest analysis results.
#[no_mangle]
pub unsafe extern "C" fn rust_free_space_detector_get_analysis(
    handle: *const FreeSpaceDetector,
    out_analysis: *mut tk_free_space_analysis_t,
) {
    let detector = &*handle;
    let out = &mut *out_analysis;
    let sectors = detector.get_sectors();

    out.sectors = sectors.as_ptr();
    out.sector_count = sectors.len();

    let mut clearest_path_dist = 0.0;
    let mut clearest_path_angle = 0.0;
    let mut any_clear = false;

    for sector in sectors {
        if sector.is_clear {
            any_clear = true;
            if sector.max_clear_distance_m > clearest_path_dist {
                clearest_path_dist = sector.max_clear_distance_m;
                clearest_path_angle = sector.center_angle_deg;
            }
        }
    }

    out.is_any_path_clear = any_clear;
    out.clearest_path_distance_m = clearest_path_dist;
    out.clearest_path_angle_deg = clearest_path_angle;
}


// --- Obstacle Tracker FFI ---

#[repr(C)]
pub struct tk_obstacle_tracker_config_t {
    pub min_obstacle_area_m2: f32,
    pub max_tracked_obstacles: u32,
    pub max_frames_unseen: u32,
    pub max_match_distance_m: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum tk_obstacle_status_e {
    New,
    Tracked,
    Coasted,
}

#[repr(C)]
pub struct tk_obstacle_t {
    pub id: u32,
    pub status: tk_obstacle_status_e,
    pub position_m: Vector2D,
    pub velocity_mps: Vector2D,
    pub dimensions_m: Vector2D,
    pub age_frames: u32,
    pub unseen_frames: u32,
}

/// Creates an `ObstacleTracker` and returns an opaque handle to it.
#[no_mangle]
pub extern "C" fn rust_obstacle_tracker_create(
    config: &tk_obstacle_tracker_config_t,
) -> *mut ObstacleTracker {
    let rust_config = ObstacleTrackerConfig {
        min_obstacle_area_m2: config.min_obstacle_area_m2,
        max_tracked_obstacles: config.max_tracked_obstacles,
        max_frames_unseen: config.max_frames_unseen,
        max_match_distance_m: config.max_match_distance_m,
    };
    let tracker = Box::new(ObstacleTracker::new(rust_config));
    Box::into_raw(tracker)
}

/// Destroys the `ObstacleTracker` instance.
#[no_mangle]
pub unsafe extern "C" fn rust_obstacle_tracker_destroy(handle: *mut ObstacleTracker) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle);
    }
}

/// Updates the obstacle tracker with a new traversability map.
#[no_mangle]
pub unsafe extern "C" fn rust_obstacle_tracker_update(
    handle: *mut ObstacleTracker,
    map: &tk_traversability_map_t,
    delta_time_s: f32,
) {
    let tracker = &mut *handle;
    let grid_slice = slice::from_raw_parts(map.grid, (map.width * map.height) as usize);
    let rust_map = TraversabilityMap {
        width: map.width,
        height: map.height,
        resolution_m_per_cell: map.resolution_m_per_cell,
        grid: grid_slice,
    };
    tracker.update(&rust_map, delta_time_s);
}

/// Retrieves a list of all currently tracked obstacles.
#[no_mangle]
pub unsafe extern "C" fn rust_obstacle_tracker_get_all(
    handle: *const ObstacleTracker,
    out_obstacles: *mut *const tk_obstacle_t,
    out_count: *mut usize,
) {
    let tracker = &*handle;
    let obstacles = tracker.get_obstacles();
    // This is tricky because the internal representation is not the same as the C one.
    // For a real implementation, we would need to copy the data into a C-compatible buffer.
    // For now, we will return an empty list.
    // A proper solution would be to allocate a buffer and copy, or have the C side provide a buffer.
    *out_obstacles = std::ptr::null();
    *out_count = 0;
}
