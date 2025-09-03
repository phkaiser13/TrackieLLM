/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: lib.rs
 *
 * This file is the main library entry point and FFI boundary for the `deployment`
 * crate. It exposes high-level functions to the C/C++ core for checking version
 * compatibility and managing package installations. It orchestrates the functionality
 * provided by the `version_checker` and `package_manager` modules.
 *
 * Dependencies:
 *  - `lazy_static`: For thread-safe initialization of the global manager instance.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Declarations and Imports ---
pub mod package_manager;
pub mod version_checker;

use lazy_static::lazy_static;
use package_manager::PackageManager;
use std::collections::HashMap;
use std::ffi::{c_char, CStr};
use std::path::Path;
use std::sync::Mutex;
use version_checker::{VersionChecker, VersionManifest};

// --- Global State Management ---

/// A manager struct to hold instances of our services.
/// This makes it easy to manage the state of the deployment module.
struct DeploymentManager {
    version_checker: VersionChecker,
    package_manager: PackageManager,
}

// A thread-safe, lazily-initialized global instance of our manager.
// This provides a single, safe entry point for all FFI calls.
lazy_static! {
    static ref DEPLOYMENT_MANAGER: Mutex<DeploymentManager> = Mutex::new(DeploymentManager {
        version_checker: VersionChecker::new(),
        package_manager: PackageManager::new(),
    });
}

// --- FFI Helper Functions ---

/// A helper to safely convert a C string pointer to a Rust `&str`.
/// Prints an error and returns `None` if the pointer is null or the string is invalid UTF-8.
fn c_str_to_rust_str<'a>(c_string: *const c_char, param_name: &str) -> Option<&'a str> {
    if c_string.is_null() {
        eprintln!("Error: FFI parameter '{}' was a null pointer.", param_name);
        return None;
    }
    unsafe { CStr::from_ptr(c_string).to_str().ok() }
}

// --- FFI Public Interface ---

/// Checks component versions from a JSON manifest against requirements from another JSON string.
///
/// This function provides a C-compatible way to perform a full version compatibility check.
///
/// # Arguments
/// - `manifest_json`: A C-string containing a JSON array of `Component` objects.
/// - `requirements_json`: A C-string containing a JSON object where keys are component
///   names and values are SemVer version requirement strings.
///
/// # Returns
/// - `0`: If all checks passed successfully.
/// - `> 0`: The number of compatibility errors found. Errors are printed to stderr.
/// - `-1`: On critical failure (e.g., null pointer, invalid JSON).
///
/// # Safety
/// The caller must ensure that `manifest_json` and `requirements_json` are valid,
/// null-terminated C-strings containing well-formed JSON.
#[no_mangle]
pub extern "C" fn deployment_check_versions_from_json(
    manifest_json: *const c_char,
    requirements_json: *const c_char,
) -> i32 {
    // 1. Safely convert C strings to Rust strings.
    let manifest_str = match c_str_to_rust_str(manifest_json, "manifest_json") {
        Some(s) => s,
        None => return -1,
    };
    let reqs_str = match c_str_to_rust_str(requirements_json, "requirements_json") {
        Some(s) => s,
        None => return -1,
    };

    // 2. Parse the JSON inputs into our Rust data structures.
    let manifest = match VersionManifest::from_json(manifest_str) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to parse version manifest JSON: {}", e);
            return -1;
        }
    };
    let requirements: HashMap<String, String> = match serde_json::from_str(reqs_str) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to parse requirements JSON: {}", e);
            return -1;
        }
    };

    // 3. Lock the global manager and perform the check.
    let manager = DEPLOYMENT_MANAGER.lock().unwrap();
    match manager.version_checker.check_compatibility(&manifest, &requirements) {
        Ok(()) => {
            println!("Version compatibility check passed successfully.");
            0 // Success
        }
        Err(errors) => {
            eprintln!("Version compatibility check failed with {} error(s):", errors.len());
            for error in &errors {
                eprintln!("- {}", error);
            }
            errors.len() as i32 // Return the number of errors found.
        }
    }
}

/// Installs a package from a local `.tar.gz` archive to a specified directory.
///
/// # Arguments
/// - `archive_path_c`: A C-string with the path to the package archive.
/// - `install_dir_c`: A C-string with the path to the target installation directory.
///
/// # Returns
/// - `0`: On success.
/// - `-1`: On failure. Details are printed to stderr.
///
/// # Safety
/// The caller must ensure that `archive_path_c` and `install_dir_c` are valid,
/// null-terminated C-strings representing valid paths.
#[no_mangle]
pub extern "C" fn deployment_install_package_from_archive(
    archive_path_c: *const c_char,
    install_dir_c: *const c_char,
) -> i32 {
    // 1. Safely convert paths.
    let archive_path_str = match c_str_to_rust_str(archive_path_c, "archive_path_c") {
        Some(s) => s,
        None => return -1,
    };
    let install_dir_str = match c_str_to_rust_str(install_dir_c, "install_dir_c") {
        Some(s) => s,
        None => return -1,
    };

    let archive_path = Path::new(archive_path_str);
    let install_dir = Path::new(install_dir_str);

    // 2. Lock the manager and perform the installation.
    let manager = DEPLOYMENT_MANAGER.lock().unwrap();

    // In a real scenario, you'd load multiple packages and resolve dependencies first.
    // Here, we simplify to loading and installing a single package.
    match manager.package_manager.load_from_archive(archive_path) {
        Ok(package) => {
            match manager.package_manager.install_package(&package, install_dir) {
                Ok(()) => 0, // Success
                Err(e) => {
                    eprintln!("Failed to install package '{}': {}", package.metadata.name, e);
                    -1
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to load package from archive '{}': {}", archive_path.display(), e);
            -1
        }
    }
}

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::io::Write;
    use tar::Builder;
    use flate2::write::GzEncoder;
    use flate2::Compression;

    #[test]
    fn test_ffi_version_check_success() {
        let manifest = r#"[{"name": "core", "version": "1.2.3"}]"#;
        let reqs = r#"{"core": ">=1.2.0"}"#;

        let manifest_c = CString::new(manifest).unwrap();
        let reqs_c = CString::new(reqs).unwrap();

        let result = deployment_check_versions_from_json(manifest_c.as_ptr(), reqs_c.as_ptr());
        assert_eq!(result, 0);
    }

    #[test]
    fn test_ffi_version_check_failure() {
        let manifest = r#"[{"name": "core", "version": "1.2.3"}]"#;
        let reqs = r#"{"core": "<1.2.0"}"#;

        let manifest_c = CString::new(manifest).unwrap();
        let reqs_c = CString::new(reqs).unwrap();

        let result = deployment_check_versions_from_json(manifest_c.as_ptr(), reqs_c.as_ptr());
        assert_eq!(result, 1); // One compatibility error
    }

    #[test]
    fn test_ffi_install_package() {
        // 1. Create a dummy package archive.
        let dir = tempfile::tempdir().unwrap();
        let archive_path = dir.path().join("my-pkg.tar.gz");
        let manifest_content = r#"
            name = "my-pkg"
            version = "1.0.0"
        "#;
        let file_content = "hello from package";

        let file = File::create(&archive_path).unwrap();
        let enc = GzEncoder::new(file, Compression::default());
        let mut tar = Builder::new(enc);

        let mut header = tar::Header::new_gnu();
        header.set_size(manifest_content.len() as u64);
        header.set_cksum();
        tar.append_data(&mut header, "package.toml", manifest_content.as_bytes()).unwrap();

        let mut header = tar::Header::new_gnu();
        header.set_size(file_content.len() as u64);
        header.set_cksum();
        tar.append_data(&mut header, "data.txt", file_content.as_bytes()).unwrap();

        tar.finish().unwrap();

        // 2. Prepare paths for FFI call.
        let install_dir = dir.path().join("install");
        let archive_path_c = CString::new(archive_path.to_str().unwrap()).unwrap();
        let install_dir_c = CString::new(install_dir.to_str().unwrap()).unwrap();

        // 3. Call the FFI function.
        let result = deployment_install_package_from_archive(archive_path_c.as_ptr(), install_dir_c.as_ptr());
        assert_eq!(result, 0);

        // 4. Verify that the file was extracted.
        let installed_file_path = install_dir.join("data.txt");
        assert!(installed_file_path.exists());
        let content = std::fs::read_to_string(installed_file_path).unwrap();
        assert_eq!(content, file_content);
    }
}
