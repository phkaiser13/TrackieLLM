/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/deployment/lib.rs
 *
 * This file serves as the main library entry point for the 'deployment' crate.
 * It is responsible for orchestrating application updates, version checking,
 * and package management. The crate provides a high-level API to abstract
 * the complexities of securely downloading, verifying, and applying updates.
 *
 * The architecture is designed around a central `DeploymentService`, which
 * coordinates the operations of the two main sub-modules:
 * - `version_checker`: Responsible for querying a remote update server to
 *   determine if a new version of the application is available.
 * - `package_manager`: Responsible for downloading, verifying the integrity
 *   and signature of, and installing update packages.
 *
 * Since the corresponding C header files (`tk_updater.h`,
 * `tk_package_installer.h`) are empty, this implementation is primarily
 * Rust-native, focusing on building a robust and secure deployment pipeline.
 * Future integration with a C core would involve adding FFI calls where
*  necessary, for example, to a C-based package installer.
 *
 * Dependencies:
 *   - log: For structured logging.
 *   - thiserror: For ergonomic error handling.
 *   - serde: For serializing and deserializing manifest files.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// === TOC ===
// 1. Crate-level Documentation & Attributes
// 2. Public Module Declarations
// 3. Public Prelude
// 4. Core Public Types (Config, Error)
// 5. Main Service Interface (DeploymentService)
// =============

#![deny(unsafe_code)]
#![deny(missing_docs)]
#![deny(warnings)]

//! # TrackieLLM Deployment Crate
//!
//! Manages the lifecycle of application deployment, including checking for,
//! downloading, and installing updates.
//!
//! ## Architecture
//!
//! The crate is centered around the `DeploymentService`, which can be run
//! periodically in a background thread to check for updates automatically, or
//_ used manually to trigger update checks and installations.
//!
//! ### Version Checking
//!
//! The `version_checker` module handles communication with the update server.
//! It fetches a version manifest, compares it with the current application
//! version, and determines if an update is available.
//!
//! ### Package Management
//!
//! The `package_manager` module handles the download and installation of
//! update packages. It includes critical security features such as checksum
//! verification and digital signature validation to ensure that packages are
//! authentic and have not been tampered with.

// --- Public Module Declarations ---

/// Handles checking for new application versions.
pub mod version_checker;

/// Handles downloading and installing update packages.
pub mod package_manager;


// --- Public Prelude ---

pub mod prelude {
    //! A "prelude" for convenient imports of the deployment crate's main types.
    pub use super::{
        DeploymentConfig, DeploymentError, DeploymentService,
        version_checker::{UpdateInfo, Version},
        package_manager::Package,
    };
}


// --- Core Public Types ---

use thiserror::Error;
use crate::version_checker::{UpdateInfo, Version};

/// Configuration for the `DeploymentService`.
///
/// This struct defines the necessary parameters for the deployment services,
/// such as the current application version and the update server URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeploymentConfig {
    /// The version of the currently running application.
    pub current_version: Version,
    /// The URL of the update manifest server.
    pub update_server_url: String,
    /// The public key used to verify the signature of update packages.
    /// This should be a PEM-encoded public key.
    pub package_verification_key: String,
}

/// Represents all possible errors that can occur within the deployment crate.
#[derive(Debug, Error)]
pub enum DeploymentError {
    /// The service is already performing an operation and cannot start another.
    #[error("Deployment service is already busy.")]
    ServiceBusy,

    /// An error occurred during the version checking process.
    #[error("Version check failed: {0}")]
    VersionCheck(#[from] version_checker::VersionCheckError),

    /// An error occurred during package management.
    #[error("Package management failed: {0}")]
    Package(#[from] package_manager::PackageError),

    /// The downloaded update is not applicable for the current version.
    #[error("Update from version {current} to {update} is not supported.")]
    UnsupportedUpdatePath {
        /// The current application version.
        current: Version,
        /// The version of the attempted update.
        update: Version,
    },

    /// A generic I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}


// --- Main Service Interface ---

/// The main service for handling application deployment tasks.
///
/// This service provides a high-level API to check for and apply updates.
/// It is designed to be used as a singleton or a long-lived object within
/// the main application.
pub struct DeploymentService {
    /// The configuration for the service.
    config: DeploymentConfig,
    /// An internal state to prevent concurrent operations.
    /// In a real implementation, this would be an `Arc<Mutex<State>>` or similar
    /// to allow for safe multi-threaded access.
    is_busy: bool,
}

impl DeploymentService {
    /// Creates a new `DeploymentService` with the given configuration.
    pub fn new(config: DeploymentConfig) -> Self {
        Self {
            config,
            is_busy: false,
        }
    }

    /// Checks for updates and returns information about the latest version if available.
    ///
    /// This method contacts the update server, fetches the version manifest,
    /// and compares it against the application's current version.
    ///
    /// # Returns
    ///
    /// `Ok(Some(UpdateInfo))` if a new version is available.
    /// `Ok(None)` if the application is already up-to-date.
    /// `Err(DeploymentError)` if the check fails.
    pub fn check_for_updates(&mut self) -> Result<Option<UpdateInfo>, DeploymentError> {
        if self.is_busy {
            return Err(DeploymentError::ServiceBusy);
        }
        self.is_busy = true;

        log::info!("Checking for updates from '{}'...", self.config.update_server_url);

        let result = version_checker::fetch_update_info(
            &self.config.update_server_url,
            &self.config.current_version,
        );

        self.is_busy = false;

        match result {
            Ok(Some(info)) => {
                log::info!(
                    "New version available: {}. Current version: {}.",
                    info.latest_version,
                    self.config.current_version
                );
                Ok(Some(info))
            }
            Ok(None) => {
                log::info!("Application is up-to-date.");
                Ok(None)
            }
            Err(e) => {
                log::error!("Failed to check for updates: {}", e);
                Err(e.into())
            }
        }
    }

    /// Downloads and applies an update.
    ///
    /// This is a comprehensive operation that:
    /// 1. Downloads the update package described by `update_info`.
    /// 2. Verifies the package's checksum and digital signature.
    /// 3. Installs the package.
    ///
    /// # Arguments
    ///
    /// * `update_info` - The `UpdateInfo` obtained from `check_for_updates`.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the update was successful. The application will likely
    /// need to be restarted to use the new version.
    pub fn apply_update(&mut self, update_info: &UpdateInfo) -> Result<(), DeploymentError> {
        if self.is_busy {
            return Err(DeploymentError::ServiceBusy);
        }
        self.is_busy = true;

        log::info!("Attempting to apply update to version {}...", update_info.latest_version);

        // Download the package
        let package = package_manager::download_package(&update_info.download_url)?;

        // Verify the package
        package_manager::verify_package(&package, &update_info.checksum, &self.config.package_verification_key)?;

        // Install the package
        package_manager::install_package(&package)?;

        self.is_busy = false;
        log::info!("Update to version {} applied successfully. Please restart the application.", update_info.latest_version);
        Ok(())
    }
}
