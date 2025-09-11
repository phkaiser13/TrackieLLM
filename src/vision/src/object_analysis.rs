/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/vision/object_analysis.rs
 *
 * This file contains the core data fusion logic for the vision pipeline,
 * written in safe Rust. Its primary responsibility is to take the raw outputs
 * from the object detector and the depth estimator and combine them into a
 * single, more meaningful representation: a list of "enriched" objects that
 * include an estimated distance.
 *
 * This logic is designed to be called from the C-level pipeline via FFI.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use super::ffi;
use std::slice;

/// Represents a detected object with its distance calculated.
/// This struct is C-compatible and will be returned to the C layer.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EnrichedObject {
    pub class_id: u32,
    pub confidence: f32,
    pub bbox: ffi::tk_rect_t,
    pub distance_meters: f32,
    pub width_meters: f32,
    pub height_meters: f32,
    pub is_partially_occluded: bool,
}

/// Fuses object detection results with a depth map to calculate the distance
/// to each object. This is a safe Rust implementation.
///
/// # Arguments
/// * `detections` - A slice of raw detection results from the object detector.
/// * `depth_map` - The depth map data from the depth estimator.
/// * `frame_width`, `frame_height` - Dimensions of the original video frame.
/// * `focal_length_x`, `focal_length_y` - Camera focal lengths for size estimation.
///
/// # Returns
/// A `Vec<EnrichedObject>` containing the fused data.
pub fn fuse_object_and_depth_data(
    detections: &[ffi::tk_detection_result_t],
    depth_map: &ffi::tk_vision_depth_map_t,
    frame_width: u32,
    frame_height: u32,
    focal_length_x: f32,
    focal_length_y: f32,
) -> Vec<EnrichedObject> {
    const MIN_DEPTH_POINTS_FOR_STATS: usize = 10;
    const OCCLUSION_STD_DEV_THRESHOLD: f32 = 0.5; // in meters

    let depth_data = unsafe {
        slice::from_raw_parts(depth_map.data, (depth_map.width * depth_map.height) as usize)
    };

    let mut enriched_objects = Vec::with_capacity(detections.len());

    for detection in detections {
        let bbox = detection.bbox;

        // --- This logic is ported directly from the C implementation in `fuse_object_depth` ---

        // Normalize bounding box coordinates to depth map space
        let norm_x_min = bbox.x as f32 / frame_width as f32;
        let norm_y_min = bbox.y as f32 / frame_height as f32;
        let norm_x_max = (bbox.x + bbox.w) as f32 / frame_width as f32;
        let norm_y_max = (bbox.y + bbox.h) as f32 / frame_height as f32;

        // Map to depth coordinates
        let depth_x_min = (norm_x_min * (depth_map.width - 1) as f32).round() as u32;
        let depth_y_min = (norm_y_min * (depth_map.height - 1) as f32).round() as u32;
        let depth_x_max = (norm_x_max * (depth_map.width - 1) as f32).round() as u32;
        let depth_y_max = (norm_y_max * (depth_map.height - 1) as f32).round() as u32;

        // Boundary checks
        if depth_x_min >= depth_map.width || depth_y_min >= depth_map.height ||
           depth_x_max >= depth_map.width || depth_y_max >= depth_map.height ||
           depth_x_min >= depth_x_max || depth_y_min >= depth_y_max {
            enriched_objects.push(EnrichedObject {
                class_id: detection.class_id,
                confidence: detection.confidence,
                bbox: detection.bbox,
                distance_meters: -1.0,
                width_meters: -1.0,
                height_meters: -1.0,
                is_partially_occluded: false,
            });
            continue;
        }

        // Collect valid depth values from the entire bounding box
        let mut valid_depths = Vec::new();
        for y in depth_y_min..=depth_y_max {
            for x in depth_x_min..=depth_x_max {
                let index = (y * depth_map.width + x) as usize;
                if index < depth_data.len() {
                    let depth_value = depth_data[index];
                    if depth_value > 0.1 && depth_value < 100.0 { // Reasonable range
                        valid_depths.push(depth_value);
                    }
                }
            }
        }

        let mut robust_distance = -1.0;
        let mut is_occluded = false;

        if valid_depths.len() >= MIN_DEPTH_POINTS_FOR_STATS {
            // Sort to calculate median and IQR
            valid_depths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // --- Outlier removal using IQR ---
            let q1_index = valid_depths.len() / 4;
            let q3_index = valid_depths.len() * 3 / 4;
            let q1 = valid_depths[q1_index];
            let q3 = valid_depths[q3_index];
            let iqr = q3 - q1;
            let lower_bound = q1 - 1.5 * iqr;
            let upper_bound = q3 + 1.5 * iqr;

            let filtered_depths: Vec<f32> = valid_depths
                .into_iter()
                .filter(|&d| d >= lower_bound && d <= upper_bound)
                .collect();

            if !filtered_depths.is_empty() {
                // --- Calculate robust distance (mean of filtered depths) ---
                let sum: f32 = filtered_depths.iter().sum();
                robust_distance = sum / filtered_depths.len() as f32;

                // --- Occlusion detection based on standard deviation ---
                let mean = robust_distance;
                let variance = filtered_depths.iter().map(|value| {
                    let diff = mean - *value;
                    diff * diff
                }).sum::<f32>() / filtered_depths.len() as f32;
                let std_dev = variance.sqrt();

                if std_dev > OCCLUSION_STD_DEV_THRESHOLD {
                    is_occluded = true;
                }
            }
        } else if !valid_depths.is_empty() {
            // Fallback for when there are not enough points for statistical analysis
            // We use the median in this case, which is more robust than the mean with few points.
            let median_index = valid_depths.len() / 2;
            robust_distance = valid_depths[median_index];
        }


        let mut width_meters = -1.0;
        let mut height_meters = -1.0;

        if robust_distance > 0.0 && focal_length_x > 0.0 && focal_length_y > 0.0 {
            width_meters = (bbox.w as f32 * robust_distance) / focal_length_x;
            height_meters = (bbox.h as f32 * robust_distance) / focal_length_y;
        }

        enriched_objects.push(EnrichedObject {
            class_id: detection.class_id,
            confidence: detection.confidence,
            bbox: detection.bbox,
            distance_meters: robust_distance,
            width_meters,
            height_meters,
            is_partially_occluded: is_occluded,
        });
    }

    enriched_objects
}
