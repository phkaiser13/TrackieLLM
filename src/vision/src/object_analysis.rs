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
        let depth_x_min = (norm_x_min * (depth_map.width - 1) as f32) as u32;
        let depth_y_min = (norm_y_min * (depth_map.height - 1) as f32) as u32;
        let depth_x_max = (norm_x_max * (depth_map.width - 1) as f32) as u32;
        let depth_y_max = (norm_y_max * (depth_map.height - 1) as f32) as u32;

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
            });
            continue;
        }

        // Focus on the central 25% of the bounding box
        let center_width = depth_x_max - depth_x_min;
        let center_height = depth_y_max - depth_y_min;
        let crop_margin_x = center_width / 4;
        let crop_margin_y = center_height / 4;
        
        let center_x_min = depth_x_min + crop_margin_x;
        let center_y_min = depth_y_min + crop_margin_y;
        let center_x_max = depth_x_max - crop_margin_x;
        let center_y_max = depth_y_max - crop_margin_y;

        // Collect valid depth values from the central region
        let mut valid_depths = Vec::new();
        for y in center_y_min..=center_y_max {
            for x in center_x_min..=center_x_max {
                let index = (y * depth_map.width + x) as usize;
                let depth_value = depth_data[index];
                if depth_value > 0.0 && depth_value < 100.0 { // Reasonable range
                    valid_depths.push(depth_value);
                }
            }
        }

        let mut robust_distance = -1.0;
        if !valid_depths.is_empty() {
            // Sort to find the minimum (closest point)
            valid_depths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            robust_distance = valid_depths[0];
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
        });
    }

    enriched_objects
}
