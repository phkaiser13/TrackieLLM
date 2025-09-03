/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: fs_utils.rs
 *
 * This file provides a collection of high-level, robust file system utilities.
 * These functions are designed to simplify common tasks like finding files based on
 * patterns, recursively copying directories, and ensuring directory existence, which
 * are often required for internal build scripts, test setups, and application logic.
 *
 * Dependencies:
 *  - `walkdir`: For efficient recursive directory traversal.
 *  - `glob`: For matching file paths against glob patterns.
 *  - `thiserror`: For structured error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Imports ---
use glob::{Pattern, PatternError};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use walkdir::{DirEntry, WalkDir};

// --- Custom Error and Result Types ---

/// Represents errors that can occur during file system operations.
#[derive(Debug, thiserror::Error)]
pub enum FsError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("Glob pattern '{pattern}' is invalid: {error}")]
    InvalidGlobPattern {
        pattern: String,
        #[source]
        error: PatternError,
    },
    #[error("WalkDir error: {0}")]
    WalkDir(#[from] walkdir::Error),
}

pub type FsResult<T> = Result<T, FsError>;

// --- Options and Configuration Structs ---

/// Defines the options for a file search operation.
/// This allows for fine-grained control over the search criteria.
#[derive(Debug, Clone, Default)]
pub struct FindOptions {
    pub max_depth: Option<usize>,
    pub min_size: Option<u64>,
    pub max_size: Option<u64>,
    pub follow_symlinks: bool,
}

// --- File System Utilities Service ---

/// A service providing high-level file system operations.
#[derive(Default)]
pub struct FileSystemUtils;

impl FileSystemUtils {
    pub fn new() -> Self {
        Self::default()
    }

    /// Recursively finds files in a directory that match a set of glob patterns.
    ///
    /// # Arguments
    /// * `root` - The directory to start the search from.
    /// * `patterns` - A slice of glob patterns to match against file names.
    /// * `options` - A `FindOptions` struct to constrain the search.
    ///
    /// # Returns
    /// A `Vec<PathBuf>` containing the paths of all matching files.
    pub fn find_files(
        &self,
        root: &Path,
        patterns: &[&str],
        options: &FindOptions,
    ) -> FsResult<Vec<PathBuf>> {
        // 1. Compile glob patterns once for efficiency.
        let glob_patterns: Vec<Pattern> = patterns
            .iter()
            .map(|p| {
                Pattern::new(p).map_err(|e| FsError::InvalidGlobPattern {
                    pattern: p.to_string(),
                    error: e,
                })
            })
            .collect::<Result<_, _>>()?;

        // 2. Configure the directory walker.
        let mut walker = WalkDir::new(root).follow_links(options.follow_symlinks);
        if let Some(depth) = options.max_depth {
            walker = walker.max_depth(depth);
        }

        let mut found_files = Vec::new();

        // 3. Iterate through directory entries.
        for entry_result in walker {
            let entry = entry_result?;
            if !entry.file_type().is_file() {
                continue;
            }

            // 4. Apply filters from `FindOptions`.
            if self.is_entry_filtered(&entry, options)? {
                continue;
            }

            // 5. Check if the file name matches any of the glob patterns.
            let file_name = entry.file_name().to_string_lossy();
            if glob_patterns.iter().any(|p| p.matches(&file_name)) {
                found_files.push(entry.into_path());
            }
        }

        Ok(found_files)
    }

    /// Helper to check if a directory entry should be filtered out based on `FindOptions`.
    fn is_entry_filtered(&self, entry: &DirEntry, options: &FindOptions) -> FsResult<bool> {
        let metadata = entry.metadata()?;
        if let Some(min) = options.min_size {
            if metadata.len() < min {
                return Ok(true);
            }
        }
        if let Some(max) = options.max_size {
            if metadata.len() > max {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Recursively copies a directory from a source to a destination.
    /// Creates the destination directory if it does not exist.
    pub fn copy_dir_all(&self, src: &Path, dst: &Path) -> FsResult<()> {
        fs::create_dir_all(dst)?;
        for entry_result in fs::read_dir(src)? {
            let entry = entry_result?;
            let file_type = entry.file_type()?;
            let dst_path = dst.join(entry.file_name());

            if file_type.is_dir() {
                self.copy_dir_all(&entry.path(), &dst_path)?;
            } else {
                fs::copy(&entry.path(), &dst_path)?;
            }
        }
        Ok(())
    }

    /// Ensures a directory and all its parent components exist.
    /// This is a simple wrapper around `std::fs::create_dir_all`.
    pub fn ensure_dir_exists(&self, path: &Path) -> FsResult<()> {
        fs::create_dir_all(path)?;
        Ok(())
    }
}

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use assert_fs::prelude::*;
    use assert_fs::TempDir;

    #[test]
    fn test_find_files_with_glob_patterns() {
        let dir = TempDir::new().unwrap();
        dir.child("a.log").touch().unwrap();
        dir.child("b.txt").touch().unwrap();
        dir.child("c.log").touch().unwrap();
        let sub = dir.child("sub");
        sub.create_dir_all().unwrap();
        sub.child("d.log").touch().unwrap();

        let utils = FileSystemUtils::new();
        let options = FindOptions::default();
        let results = utils.find_files(dir.path(), &["*.log"], &options).unwrap();

        assert_eq!(results.len(), 3);
        assert!(results.iter().any(|p| p.ends_with("a.log")));
        assert!(results.iter().any(|p| p.ends_with("c.log")));
        assert!(results.iter().any(|p| p.ends_with("d.log")));
    }

    #[test]
    fn test_find_files_with_max_depth() {
        let dir = TempDir::new().unwrap();
        dir.child("a.txt").touch().unwrap();
        let sub = dir.child("sub");
        sub.create_dir_all().unwrap();
        sub.child("b.txt").touch().unwrap();

        let utils = FileSystemUtils::new();
        let options = FindOptions { max_depth: Some(1), ..Default::default() };
        let results = utils.find_files(dir.path(), &["*.txt"], &options).unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].ends_with("a.txt"));
    }

    #[test]
    fn test_find_files_with_size_constraints() {
        let dir = TempDir::new().unwrap();
        let small_file = dir.child("small.txt");
        small_file.write_str("small").unwrap(); // 5 bytes
        let large_file = dir.child("large.txt");
        large_file.write_str("very large content").unwrap(); // 18 bytes

        let utils = FileSystemUtils::new();
        let options = FindOptions { min_size: Some(10), ..Default::default() };
        let results = utils.find_files(dir.path(), &["*.txt"], &options).unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].ends_with("large.txt"));
    }

    #[test]
    fn test_copy_dir_all() {
        let src = TempDir::new().unwrap();
        src.child("file.txt").write_str("hello").unwrap();
        let sub = src.child("sub");
        sub.create_dir_all().unwrap();
        sub.child("nested.txt").write_str("world").unwrap();

        let dst = TempDir::new().unwrap();

        let utils = FileSystemUtils::new();
        utils.copy_dir_all(src.path(), dst.path()).unwrap();

        dst.child("file.txt").assert("hello");
        dst.child("sub/nested.txt").assert("world");
    }
}
