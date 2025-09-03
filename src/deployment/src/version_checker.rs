/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/deployment/version_checker.rs
 *
 * This file implements the logic for checking for new application versions.
 * It defines a `Version` struct that adheres to semantic versioning principles,
 * allowing for robust comparison between versions. It also defines the logic
 * for fetching and parsing an `UpdateInfo` manifest from a remote server.
 *
 * The core function, `fetch_update_info`, simulates a network request to an
 * update server, retrieves a version manifest (as a JSON string), and then
 * parses it. It compares the latest version from the manifest with the
 * currently running application version to determine if an update is available.
 *
 * Dependencies:
 *   - log: For structured logging.
 *   - thiserror: For ergonomic error handling.
 *   - serde: For serializing and deserializing the Version and UpdateInfo structs.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// Represents a semantic version (Major.Minor.Patch).
///
/// This struct provides functionality for parsing, displaying, and comparing
/// versions, which is crucial for determining update eligibility.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Version {
    /// Major version: increases for incompatible API changes.
    pub major: u32,
    /// Minor version: increases for new, backward-compatible functionality.
    pub minor: u32,
    /// Patch version: increases for backward-compatible bug fixes.
    pub patch: u32,
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Contains information about an available update.
///
/// This struct is typically deserialized from a JSON manifest file fetched
/// from an update server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpdateInfo {
    /// The latest available version.
    pub latest_version: Version,
    /// The URL from which the update package can be downloaded.
    pub download_url: String,
    /// A checksum (e.g., SHA-256) of the update package for integrity verification.
    pub checksum: String,
    /// Release notes or a summary of changes in the new version.
    pub release_notes: String,
}

/// Represents errors that can occur during the version checking process.
#[derive(Debug, Error)]
pub enum VersionCheckError {
    /// A network error occurred while trying to fetch the update manifest.
    #[error("Network error while fetching update info: {0}")]
    Network(String),

    /// The update manifest could not be parsed or was malformed.
    #[error("Failed to parse update manifest: {0}")]
    ManifestParse(#[from] serde_json::Error),

    /// The remote server returned an error.
    #[error("Update server returned an error: status {0}")]
    ServerError(u16),
}

/// Fetches update information from a remote server and checks for a new version.
///
/// This function simulates a network call to the provided URL, parses the
/// returned manifest, and compares the manifest's version with the current
/// application version.
///
/// # Arguments
///
/// * `server_url` - The URL of the update manifest server.
/// * `current_version` - The version of the currently running application.
///
/// # Returns
///
/// * `Ok(Some(UpdateInfo))` if a newer version is available.
/// * `Ok(None)` if the current version is up-to-date or newer.
/// * `Err(VersionCheckError)` if an error occurs.
pub fn fetch_update_info(
    server_url: &str,
    current_version: &Version,
) -> Result<Option<UpdateInfo>, VersionCheckError> {
    log::debug!(
        "Fetching update manifest from {} for current version {}",
        server_url,
        current_version
    );

    // --- Mock Implementation ---
    // In a real application, this would use an HTTP client like `reqwest`.
    // Here, we simulate the process.

    // 1. Simulate the network request.
    let response_body = match simulate_network_fetch(server_url) {
        Ok(body) => body,
        Err(e) => return Err(e),
    };

    // 2. Parse the manifest from the response body.
    let update_info: UpdateInfo = serde_json::from_str(&response_body)?;
    log::debug!("Successfully parsed update manifest for version {}", update_info.latest_version);

    // 3. Compare the versions.
    if update_info.latest_version > *current_version {
        log::info!(
            "A new version is available: {} (current: {})",
            update_info.latest_version,
            current_version
        );
        Ok(Some(update_info))
    } else {
        log::info!("Current version {} is up-to-date.", current_version);
        Ok(None)
    }
}

/// Simulates fetching a resource over the network.
///
/// This mock function returns a hardcoded JSON manifest for a hypothetical
/// new version. It also simulates potential network failures.
fn simulate_network_fetch(url: &str) -> Result<String, VersionCheckError> {
    // Simulate latency
    std::thread::sleep(std::time::Duration::from_millis(300));

    // Simulate a chance of network failure.
    if url.contains("fail") {
        return Err(VersionCheckError::Network("Connection timed out".to_string()));
    }

    // Return a mock manifest file.
    let manifest = UpdateInfo {
        latest_version: Version { major: 1, minor: 1, patch: 0 },
        download_url: "https://example.com/downloads/trackiellm-v1.1.0.pkg".to_string(),
        checksum: "a1b2c3d4e5f6...".to_string(), // In reality, a full SHA-256 hash
        release_notes: "Version 1.1.0 includes performance improvements and bug fixes.".to_string(),
    };

    Ok(serde_json::to_string_pretty(&manifest).unwrap())
}
