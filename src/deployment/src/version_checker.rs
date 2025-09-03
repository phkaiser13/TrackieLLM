/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: version_checker.rs
 *
 * This file implements a robust version checking mechanism for the application and its
 * components. It uses semantic versioning (SemVer) to parse, compare, and validate
 * versions against a set of defined requirements. The design allows for loading version
 * manifests and checking them for compatibility, which is crucial for safe deployments
 * and updates.
 *
 * Dependencies:
 *  - `semver`: For all SemVer parsing and requirement matching logic.
 *  - `serde`: For deserializing version manifests from formats like JSON or TOML.
 *  - `thiserror`: For structured and descriptive error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Imports ---
use semver::{Version, VersionReq};
use serde::Deserialize;
use std::collections::HashMap;

// --- Custom Error and Result Types ---

/// Represents a specific compatibility error found during version checking.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum CompatibilityError {
    #[error("Component '{component}' is missing from the manifest but is required.")]
    MissingComponent { component: String },
    #[error("Component '{component}' version '{actual}' does not meet requirement '{required}'.")]
    VersionMismatch {
        component: String,
        actual: Version,
        required: VersionReq,
    },
    #[error("Failed to parse version requirement for component '{component}': {error}")]
    InvalidRequirement {
        component: String,
        error: String,
    },
}

/// A specialized `Result` type for version checking operations.
pub type CheckResult<T> = Result<T, Vec<CompatibilityError>>;

// --- Core Data Structures ---

/// Represents a single component with its current version.
/// This can be deserialized from a manifest file.
#[derive(Deserialize, Debug, Clone)]
pub struct Component {
    pub name: String,
    pub version: Version,
}

/// Represents a version manifest, which is a collection of components.
/// This structure can be loaded directly from a JSON or TOML file.
#[derive(Deserialize, Debug, Clone)]
pub struct VersionManifest {
    #[serde(rename = "components")]
    component_list: Vec<Component>,

    // A map for quick, O(1) average time complexity lookups.
    #[serde(skip)]
    component_map: HashMap<String, Version>,
}

impl VersionManifest {
    /// Creates a new `VersionManifest` from a list of components.
    /// It also builds the internal hash map for efficient lookups.
    pub fn new(components: Vec<Component>) -> Self {
        let component_map = components
            .iter()
            .map(|c| (c.name.clone(), c.version.clone()))
            .collect();

        VersionManifest {
            component_list: components,
            component_map,
        }
    }

    /// Attempts to load a `VersionManifest` from a JSON string.
    pub fn from_json(json_data: &str) -> Result<Self, serde_json::Error> {
        let components: Vec<Component> = serde_json::from_str(json_data)?;
        Ok(Self::new(components))
    }

    /// Attempts to load a `VersionManifest` from a TOML string.
    pub fn from_toml(toml_data: &str) -> Result<Self, toml::de::Error> {
        #[derive(Deserialize)]
        struct TomlRoot { components: Vec<Component> }
        let root: TomlRoot = toml::from_str(toml_data)?;
        Ok(Self::new(root.components))
    }

    /// Retrieves the version of a component by its name.
    pub fn get_version(&self, component_name: &str) -> Option<&Version> {
        self.component_map.get(component_name)
    }
}


// --- Version Checking Service ---

/// A service dedicated to checking version compatibility.
/// It is stateless and can be used to validate different manifests against
/// different sets of requirements.
#[derive(Default)]
pub struct VersionChecker;

impl VersionChecker {
    /// Creates a new `VersionChecker`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Checks a `VersionManifest` against a map of requirements.
    ///
    /// This function performs a comprehensive check and returns a `Vec` of all
    /// compatibility errors found, rather than failing on the first one. This
    /// provides a full picture of all issues at once.
    ///
    /// # Arguments
    /// * `manifest` - The `VersionManifest` containing the actual versions of components.
    /// * `requirements` - A map where the key is the component name and the value
    ///   is the required version string (e.g., ">=1.2.0, <2.0.0").
    ///
    /// # Returns
    /// An empty `Ok(())` if all components are compatible, otherwise an `Err`
    /// containing a vector of all `CompatibilityError`s found.
    pub fn check_compatibility(
        &self,
        manifest: &VersionManifest,
        requirements: &HashMap<String, String>,
    ) -> CheckResult<()> {
        let mut errors = Vec::new();

        for (component_name, req_str) in requirements {
            // 1. Parse the requirement string into a `VersionReq`.
            let version_req = match VersionReq::parse(req_str) {
                Ok(req) => req,
                Err(e) => {
                    errors.push(CompatibilityError::InvalidRequirement {
                        component: component_name.clone(),
                        error: e.to_string(),
                    });
                    continue; // Can't proceed with this component if requirement is invalid.
                }
            };

            // 2. Look up the component in the manifest.
            match manifest.get_version(component_name) {
                Some(actual_version) => {
                    // 3. If found, check if the actual version matches the requirement.
                    if !version_req.matches(actual_version) {
                        errors.push(CompatibilityError::VersionMismatch {
                            component: component_name.clone(),
                            actual: actual_version.clone(),
                            required: version_req,
                        });
                    }
                }
                None => {
                    // 4. If not found, record a missing component error.
                    errors.push(CompatibilityError::MissingComponent {
                        component: component_name.clone(),
                    });
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_manifest() -> VersionManifest {
        VersionManifest::new(vec![
            Component { name: "core-engine".to_string(), version: Version::new(1, 5, 2) },
            Component { name: "networking".to_string(), version: Version::new(2, 0, 1) },
            Component { name: "security-module".to_string(), version: Version::parse("3.1.0-alpha.1").unwrap() },
        ])
    }

    #[test]
    fn test_successful_compatibility_check() {
        let checker = VersionChecker::new();
        let manifest = create_test_manifest();
        let requirements = HashMap::from([
            ("core-engine".to_string(), ">=1.5.0, <2.0.0".to_string()),
            ("networking".to_string(), "^2.0.0".to_string()), // Compatible with 2.0.1
        ]);

        let result = checker.check_compatibility(&manifest, &requirements);
        assert!(result.is_ok());
    }

    #[test]
    fn test_version_mismatch_error() {
        let checker = VersionChecker::new();
        let manifest = create_test_manifest();
        let requirements = HashMap::from([
            ("core-engine".to_string(), "==1.6.0".to_string()), // Fails, actual is 1.5.2
        ]);

        let result = checker.check_compatibility(&manifest, &requirements);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0], CompatibilityError::VersionMismatch {
            component: "core-engine".to_string(),
            actual: Version::new(1, 5, 2),
            required: VersionReq::parse("==1.6.0").unwrap(),
        });
    }

    #[test]
    fn test_missing_component_error() {
        let checker = VersionChecker::new();
        let manifest = create_test_manifest();
        let requirements = HashMap::from([
            ("ai-module".to_string(), ">=1.0.0".to_string()), // Not in manifest
        ]);

        let result = checker.check_compatibility(&manifest, &requirements);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0], CompatibilityError::MissingComponent {
            component: "ai-module".to_string(),
        });
    }

    #[test]
    fn test_multiple_errors_are_collected() {
        let checker = VersionChecker::new();
        let manifest = create_test_manifest();
        let requirements = HashMap::from([
            ("core-engine".to_string(), ">2.0.0".to_string()),      // Mismatch
            ("ai-module".to_string(), ">=1.0.0".to_string()),      // Missing
            ("networking".to_string(), "2.0.1".to_string()),       // Success
        ]);

        let result = checker.check_compatibility(&manifest, &requirements);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 2); // Should find both errors
    }

    #[test]
    fn test_semver_prerelease_handling() {
        let checker = VersionChecker::new();
        let manifest = create_test_manifest(); // security-module is 3.1.0-alpha.1

        // A requirement without a prerelease tag will not match a version with one by default.
        let requirements1 = HashMap::from([
            ("security-module".to_string(), ">=3.1.0".to_string()),
        ]);
        let result1 = checker.check_compatibility(&manifest, &requirements1);
        assert!(result1.is_err(), "Should not match prerelease version by default");

        // To match a prerelease, the requirement must also include one.
        let requirements2 = HashMap::from([
            ("security-module".to_string(), ">=3.1.0-alpha.0".to_string()),
        ]);
        let result2 = checker.check_compatibility(&manifest, &requirements2);
        assert!(result2.is_ok(), "Should match prerelease when specified in requirement");
    }

    #[test]
    fn test_manifest_loading_from_json() {
        let json_data = r#"[
            {"name": "component-a", "version": "1.2.3"},
            {"name": "component-b", "version": "4.5.6"}
        ]"#;
        let manifest = VersionManifest::from_json(json_data).unwrap();
        assert_eq!(manifest.get_version("component-a"), Some(&Version::new(1, 2, 3)));
        assert_eq!(manifest.get_version("component-b"), Some(&Version::new(4, 5, 6)));
    }
}
