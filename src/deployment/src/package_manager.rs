/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: package_manager.rs
 *
 * This file implements a package management system. It handles parsing package
 * metadata from manifest files, resolving dependencies against a set of available
 * packages, and simulating the installation process (e.g., by extracting an archive).
 * The design is modular to separate parsing, resolution, and installation logic.
 *
 * Dependencies:
 *  - `serde`: For deserializing package manifests.
 *  - `toml`: For parsing TOML-formatted manifest files.
 *  - `semver`: For checking version compatibility of dependencies.
 *  - `tar`, `flate2`: For handling `.tar.gz` package archives.
 *  - `thiserror`: For structured error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Imports ---
use semver::{Version, VersionReq};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use tar::Archive;
use flate2::read::GzDecoder;

// --- Custom Error and Result Types ---

/// Represents errors that can occur during package management operations.
#[derive(Debug, thiserror::Error)]
pub enum PackageError {
    #[error("Failed to read package file at '{path}': {error}")]
    IoError {
        path: PathBuf,
        #[source]
        error: std::io::Error,
    },
    #[error("Failed to parse package manifest '{path}': {error}")]
    ManifestParseError {
        path: PathBuf,
        #[source]
        error: toml::de::Error,
    },
    #[error("The package manifest at '{path}' is missing the required '{field}' field.")]
    MissingManifestField { path: PathBuf, field: String },
    #[error("Dependency '{name}' with requirement '{requirement}' could not be found.")]
    DependencyNotFound { name: String, requirement: VersionReq },
    #[error("A circular dependency was detected involving package '{package_name}'.")]
    CircularDependency { package_name: String },
    #[error("Package archive extraction failed for '{path}': {error}")]
    ExtractionError {
        path: PathBuf,
        error: String,
    },
}

type PackageResult<T> = Result<T, PackageError>;

// --- Core Data Structures ---

/// Represents a dependency requirement for a package.
#[derive(Deserialize, Debug, Clone)]
pub struct Dependency {
    pub name: String,
    pub version_req: VersionReq,
}

/// Represents the metadata for a package, typically loaded from a manifest file.
#[derive(Deserialize, Debug, Clone)]
pub struct PackageMetadata {
    pub name: String,
    pub version: Version,
    pub description: Option<String>,
    #[serde(default)]
    pub dependencies: Vec<Dependency>,
}

/// Represents a software package, including its metadata and location.
#[derive(Debug, Clone)]
pub struct Package {
    pub metadata: PackageMetadata,
    /// Path to the package archive file (e.g., a `.tar.gz`).
    pub archive_path: PathBuf,
}


// --- Package Management Service ---

/// A service for handling package operations like loading, resolution, and installation.
#[derive(Default)]
pub struct PackageManager;

impl PackageManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads a package by reading its manifest from a `.tar.gz` archive.
    /// The manifest is expected to be a file named `package.toml` at the root of the archive.
    ///
    /// # Arguments
    /// * `archive_path` - The path to the package's `.tar.gz` file.
    pub fn load_from_archive(&self, archive_path: &Path) -> PackageResult<Package> {
        let file = File::open(archive_path).map_err(|e| PackageError::IoError {
            path: archive_path.to_path_buf(),
            error: e,
        })?;
        let mut archive = Archive::new(GzDecoder::new(file));

        let manifest_str = archive.entries()
            .map_err(|e| PackageError::ExtractionError { path: archive_path.to_path_buf(), error: e.to_string() })?
            .filter_map(Result::ok)
            .find(|entry| entry.path().map_or(false, |p| p.to_str() == Some("package.toml")))
            .map_or(Err(PackageError::MissingManifestField {
                path: archive_path.to_path_buf(),
                field: "package.toml".to_string(),
            }), |mut entry| {
                let mut contents = String::new();
                entry.read_to_string(&mut contents).map(|_| contents).map_err(|e| PackageError::IoError {
                    path: archive_path.to_path_buf(),
                    error: e,
                })
            })?;

        let metadata: PackageMetadata = toml::from_str(&manifest_str).map_err(|e| PackageError::ManifestParseError {
            path: archive_path.to_path_buf(),
            error: e,
        })?;

        Ok(Package {
            metadata,
            archive_path: archive_path.to_path_buf(),
        })
    }

    /// Resolves the dependency graph for a root package against a list of available packages.
    ///
    /// This implements a basic topological sort-style dependency resolution.
    ///
    /// # Arguments
    /// * `root_package` - The main package for which to resolve dependencies.
    /// * `available_packages` - A slice of all packages available for installation.
    ///
    /// # Returns
    /// A `Vec<Package>` containing the resolved dependency tree, in an order that
    /// is safe for installation (dependencies come before dependents).
    pub fn resolve_dependencies<'a>(
        &self,
        root_package: &'a Package,
        available_packages: &'a [Package],
    ) -> Result<Vec<&'a Package>, PackageError> {

        let available_map: HashMap<_, _> = available_packages.iter().map(|p| (p.metadata.name.as_str(), p)).collect();
        let mut resolved = Vec::new();
        let mut visited = HashSet::new(); // For tracking resolved packages
        let mut resolving_stack = HashSet::new(); // For detecting circular dependencies

        self.resolve_recursive(root_package, &available_map, &mut resolved, &mut visited, &mut resolving_stack)?;

        Ok(resolved)
    }

    /// Helper function for recursive dependency resolution.
    fn resolve_recursive<'a>(
        &self,
        package: &'a Package,
        available_map: &HashMap<&str, &'a Package>,
        resolved: &mut Vec<&'a Package>,
        visited: &mut HashSet<&'a str>,
        resolving_stack: &mut HashSet<&'a str>,
    ) -> Result<(), PackageError> {

        // 1. Detect circular dependencies.
        if !resolving_stack.insert(package.metadata.name.as_str()) {
            return Err(PackageError::CircularDependency { package_name: package.metadata.name.clone() });
        }

        // 2. Resolve dependencies of the current package.
        for dep in &package.metadata.dependencies {
            if visited.contains(dep.name.as_str()) {
                continue; // Already resolved this dependency.
            }

            // Find a suitable package from the available list.
            let dep_package = available_map.get(dep.name.as_str())
                .filter(|p| dep.version_req.matches(&p.metadata.version))
                .ok_or_else(|| PackageError::DependencyNotFound {
                    name: dep.name.clone(),
                    requirement: dep.version_req.clone(),
                })?;

            self.resolve_recursive(dep_package, available_map, resolved, visited, resolving_stack)?;
        }

        // 3. Add the current package to the resolved list after its dependencies are resolved.
        resolving_stack.remove(package.metadata.name.as_str());
        if visited.insert(package.metadata.name.as_str()) {
            resolved.push(package);
        }

        Ok(())
    }

    /// Simulates the installation of a package by extracting its archive.
    ///
    /// # Arguments
    /// * `package` - The package to install.
    /// * `install_dir` - The directory where the package contents should be extracted.
    pub fn install_package(&self, package: &Package, install_dir: &Path) -> PackageResult<()> {
        let file = File::open(&package.archive_path).map_err(|e| PackageError::IoError {
            path: package.archive_path.clone(),
            error: e,
        })?;

        let mut archive = Archive::new(GzDecoder::new(file));

        // Ensure the installation directory exists.
        fs::create_dir_all(install_dir).map_err(|e| PackageError::IoError {
            path: install_dir.to_path_buf(),
            error: e,
        })?;

        archive.unpack(install_dir).map_err(|e| PackageError::ExtractionError {
            path: package.archive_path.clone(),
            error: e.to_string(),
        })?;

        println!("Successfully installed package '{}' to '{}'", package.metadata.name, install_dir.display());
        Ok(())
    }
}

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tar::Builder;
    use flate2::write::GzEncoder;
    use flate2::Compression;

    // Helper to create a dummy package archive for testing.
    fn create_test_archive(path: &Path, manifest_content: &str) {
        let file = File::create(path).unwrap();
        let enc = GzEncoder::new(file, Compression::default());
        let mut tar = Builder::new(enc);

        let mut header = tar::Header::new_gnu();
        header.set_size(manifest_content.len() as u64);
        header.set_cksum();
        tar.append_data(&mut header, "package.toml", manifest_content.as_bytes()).unwrap();
        tar.finish().unwrap();
    }

    #[test]
    fn test_load_package_from_archive() {
        let dir = tempfile::tempdir().unwrap();
        let archive_path = dir.path().join("test-pkg.tar.gz");
        let manifest = r#"
            name = "test-pkg"
            version = "1.0.0"
            description = "A test package"
        "#;
        create_test_archive(&archive_path, manifest);

        let pm = PackageManager::new();
        let package = pm.load_from_archive(&archive_path).unwrap();

        assert_eq!(package.metadata.name, "test-pkg");
        assert_eq!(package.metadata.version, Version::new(1, 0, 0));
    }

    #[test]
    fn test_dependency_resolution_success() {
        let pm = PackageManager::new();
        let pkg_a = Package {
            metadata: PackageMetadata {
                name: "A".to_string(), version: Version::new(1, 0, 0), description: None,
                dependencies: vec![Dependency { name: "B".to_string(), version_req: VersionReq::parse("=1.0.0").unwrap() }]
            },
            archive_path: PathBuf::new(),
        };
        let pkg_b = Package {
            metadata: PackageMetadata { name: "B".to_string(), version: Version::new(1, 0, 0), description: None, dependencies: vec![] },
            archive_path: PathBuf::new(),
        };

        let available = vec![pkg_a.clone(), pkg_b.clone()];
        let resolved = pm.resolve_dependencies(&pkg_a, &available).unwrap();

        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved[0].metadata.name, "B"); // Dependency 'B' should come first.
        assert_eq!(resolved[1].metadata.name, "A");
    }

    #[test]
    fn test_dependency_not_found() {
        let pm = PackageManager::new();
        let pkg_a = Package {
            metadata: PackageMetadata {
                name: "A".to_string(), version: Version::new(1, 0, 0), description: None,
                dependencies: vec![Dependency { name: "C".to_string(), version_req: VersionReq::parse("=1.0.0").unwrap() }]
            },
            archive_path: PathBuf::new(),
        };

        let result = pm.resolve_dependencies(&pkg_a, &[pkg_a.clone()]);
        assert!(matches!(result, Err(PackageError::DependencyNotFound { .. })));
    }
}
