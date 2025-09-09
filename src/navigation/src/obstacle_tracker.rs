/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/navigation/obstacle_tracker.rs
 *
 * This file provides the primary Rust implementation for the Obstacle Tracker.
 * It is called by the C FFI wrappers in `tk_obstacle_avoider.c`.
 *
 * The logic here is responsible for:
 * 1. Clustering obstacle cells from a traversability map into discrete objects.
 * 2. Tracking these objects over time, maintaining a persistent ID.
 * 3. Providing a list of tracked obstacles to the rest of the system.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use super::free_space::{TraversabilityMap, TraversabilityType};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Copy)]
pub struct ObstacleTrackerConfig {
    pub min_obstacle_area_m2: f32,
    pub max_tracked_obstacles: u32,
    pub max_frames_unseen: u32,
    pub max_match_distance_m: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObstacleStatus {
    New,
    Tracked,
    Coasted,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Vector2D {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone)]
pub struct TrackedObstacle {
    pub id: u32,
    pub status: ObstacleStatus,
    pub position_m: Vector2D,
    pub velocity_mps: Vector2D,
    pub dimensions_m: Vector2D,
    pub age_frames: u32,
    pub unseen_frames: u32,
}

#[derive(Debug, Clone)]
struct ObstacleDetection {
    pub position_m: Vector2D,
    pub cell_indices: Vec<(u32, u32)>,
}

pub struct ObstacleTracker {
    config: ObstacleTrackerConfig,
    tracked_obstacles: Vec<TrackedObstacle>,
    next_id: u32,
}

impl ObstacleTracker {
    pub fn new(config: ObstacleTrackerConfig) -> Self {
        Self {
            config,
            tracked_obstacles: Vec::with_capacity(config.max_tracked_obstacles as usize),
            next_id: 1,
        }
    }

    /// Main update function for the tracker.
    pub fn update(&mut self, map: &TraversabilityMap, delta_time_s: f32) {
        let detections = self.detect_obstacles(map);
        self.associate_and_update(&detections);
        self.manage_lifecycle();
    }

    /// Performs clustering on the traversability map to find discrete obstacles.
    fn detect_obstacles(&self, map: &TraversabilityMap) -> Vec<ObstacleDetection> {
        let mut detections = Vec::new();
        let mut visited = vec![false; (map.width * map.height) as usize];

        for y in 0..map.height {
            for x in 0..map.width {
                let idx = (y * map.width + x) as usize;
                if map.grid[idx] == TraversabilityType::Obstacle && !visited[idx] {
                    // Start of a new potential obstacle, begin flood fill (BFS)
                    let mut cluster_cells = Vec::new();
                    let mut queue = VecDeque::new();

                    visited[idx] = true;
                    queue.push_back((x, y));
                    cluster_cells.push((x, y));

                    while let Some((cx, cy)) = queue.pop_front() {
                        // Check neighbors
                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                if dx == 0 && dy == 0 { continue; }

                                let nx = cx as i32 + dx;
                                let ny = cy as i32 + dy;

                                if nx >= 0 && nx < map.width as i32 && ny >= 0 && ny < map.height as i32 {
                                    let n_idx = (ny as u32 * map.width + nx as u32) as usize;
                                    if map.grid[n_idx] == TraversabilityType::Obstacle && !visited[n_idx] {
                                        visited[n_idx] = true;
                                        queue.push_back((nx as u32, ny as u32));
                                        cluster_cells.push((nx as u32, ny as u32));
                                    }
                                }
                            }
                        }
                    }

                    // Convert cluster to a detection
                    if !cluster_cells.is_empty() {
                         let mut sum_x = 0.0;
                         let mut sum_y = 0.0;
                         for (cell_x, cell_y) in &cluster_cells {
                             sum_x += *cell_x as f32;
                             sum_y += *cell_y as f32;
                         }
                         let count = cluster_cells.len() as f32;
                         let center_x = (sum_x / count - (map.width as f32 / 2.0)) * map.resolution_m_per_cell;
                         let center_y = (sum_y / count) * map.resolution_m_per_cell;

                         detections.push(ObstacleDetection {
                             position_m: Vector2D { x: center_x, y: center_y },
                             cell_indices: cluster_cells,
                         });
                    }
                }
            }
        }
        detections
    }

    /// Associates new detections with existing tracks.
    fn associate_and_update(&mut self, detections: &[ObstacleDetection]) {
        // Mark all tracks as unseen initially
        for track in self.tracked_obstacles.iter_mut() {
            track.status = ObstacleStatus::Coasted;
        }

        let mut used_detections = vec![false; detections.len()];

        // Simple nearest-neighbor association
        for track in self.tracked_obstacles.iter_mut() {
            let mut best_match_idx = None;
            let mut min_dist_sq = self.config.max_match_distance_m.powi(2);

            for (i, detection) in detections.iter().enumerate() {
                if !used_detections[i] {
                    let dist_sq = (track.position_m.x - detection.position_m.x).powi(2) +
                                  (track.position_m.y - detection.position_m.y).powi(2);
                    if dist_sq < min_dist_sq {
                        min_dist_sq = dist_sq;
                        best_match_idx = Some(i);
                    }
                }
            }

            if let Some(idx) = best_match_idx {
                used_detections[idx] = true;
                track.status = ObstacleStatus::Tracked;
                track.position_m = detections[idx].position_m;
                track.unseen_frames = 0;
                track.age_frames += 1;
            }
        }

        // Create new tracks for unused detections
        for (i, detection) in detections.iter().enumerate() {
            if !used_detections[i] && self.tracked_obstacles.len() < self.config.max_tracked_obstacles as usize {
                let new_track = TrackedObstacle {
                    id: self.next_id,
                    status: ObstacleStatus::New,
                    position_m: detection.position_m,
                    velocity_mps: Default::default(),
                    dimensions_m: Default::default(), // TODO: calculate dimensions
                    age_frames: 1,
                    unseen_frames: 0,
                };
                self.tracked_obstacles.push(new_track);
                self.next_id += 1;
            }
        }
    }

    /// Prunes tracks that have been unseen for too long.
    fn manage_lifecycle(&mut self) {
        for track in self.tracked_obstacles.iter_mut() {
            if track.status == ObstacleStatus::Coasted {
                track.unseen_frames += 1;
            }
        }
        self.tracked_obstacles.retain(|t| t.unseen_frames <= self.config.max_frames_unseen);
    }

    pub fn get_obstacles(&self) -> &[TrackedObstacle] {
        &self.tracked_obstacles
    }
}
