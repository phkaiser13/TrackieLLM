/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/integration/plugin_manager.rs
 *
 * This file implements the `PluginManager`, a service responsible for the
 * discovery, loading, and lifecycle management of dynamic plugins. Plugins
 * allow the application to be extended with new functionality at runtime
 * without requiring a recompile of the core application.
 *
 * The plugin system is designed around a core `Plugin` trait. Each plugin,
 * compiled as a shared library (e.g., .so, .dll), must export a special
 * constructor function (e.g., `_plugin_create`) that returns a pointer to a
 * struct implementing this trait.
 *
 * The `PluginManager` scans a designated directory, loads each shared library,
 * resolves the constructor symbol, and initializes the plugin. It maintains a
 * registry of active plugins and orchestrates their lifecycle.
 *
 * This implementation conceptually uses the `libloading` crate for cross-platform
 * dynamic library loading.
 *
 * Dependencies:
 *   - crate::bridge::{BridgeRequest, BridgeResponse}: For plugin communication.
 *   - libloading: For loading shared libraries.
 *   - log: For structured logging.
 *   - thiserror: For ergonomic error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use crate::bridge::{BridgeRequest, BridgeResponse};
use libloading::{Library, Symbol};
use std::collections::HashMap;
use thiserror::Error;

/// The common interface that all plugins must implement.
///
/// This trait defines the contract between the core application and a plugin.
pub trait Plugin {
    /// Called immediately after the plugin is loaded.
    /// Use this for one-time initialization.
    fn on_load(&mut self);

    /// Called just before the plugin is unloaded.
    /// Use this for graceful cleanup.
    fn on_unload(&mut self);

    /// The primary entry point for handling requests from the core application.
    fn handle_request(&mut self, request: &BridgeRequest) -> BridgeResponse;
}

/// Represents errors that can occur within the plugin manager.
#[derive(Debug, Error)]
pub enum PluginError {
    /// An error occurred while loading a shared library.
    #[error("Failed to load plugin library from '{path}': {source}")]
    LibraryLoad {
        path: String,
        #[source]
        source: libloading::Error,
    },

    /// The required constructor symbol was not found in the plugin library.
    #[error("Constructor symbol '{symbol}' not found in plugin '{path}'")]
    SymbolNotFound {
        path: String,
        symbol: String,
    },

    /// The specified plugin directory could not be read.
    #[error("Failed to read plugin directory '{path}': {source}")]
    DirectoryRead {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

// A type alias for the plugin's constructor function.
// The function takes no arguments and returns a raw pointer to a heap-allocated
// object that implements the `Plugin` trait.
type PluginCreate = unsafe extern "C" fn() -> *mut dyn Plugin;

/// Manages the lifecycle of all loaded plugins.
pub struct PluginManager {
    /// A collection of loaded plugins, mapping a plugin's name to its handle.
    /// The `PluginHandle` contains both the plugin trait object and the library
    /// it was loaded from, ensuring the library stays loaded as long as the
    /// plugin is in use.
    plugins: HashMap<String, PluginHandle>,
}

/// A handle that holds a loaded plugin and its associated library.
struct PluginHandle {
    /// The plugin trait object, boxed to be stored on the heap.
    #[allow(dead_code)]
    plugin: Box<dyn Plugin>,
    /// The library the plugin was loaded from. This field must be kept
    /// to ensure the library's code is not unloaded from memory while the
    /// plugin is still in use.
    #[allow(dead_code)]
    library: Library,
}

impl PluginManager {
    /// Creates a new, empty `PluginManager`.
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }

    /// Scans a directory for shared libraries and loads them as plugins.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the directory containing plugin files.
    ///
    /// # Safety
    ///
    /// This function is `unsafe` because it loads and executes code from
    /// external libraries, which is an inherently unsafe operation. The caller
    /// must trust the plugins in the specified directory.
    pub unsafe fn load_plugins_from_dir(&mut self, path: &str) -> Result<(), PluginError> {
        log::info!("Loading plugins from directory: {}", path);

        for entry in std::fs::read_dir(path).map_err(|e| PluginError::DirectoryRead {
            path: path.to_string(),
            source: e,
        })? {
            let entry = entry.map_err(|e| PluginError::DirectoryRead {
                path: path.to_string(),
                source: e,
            })?;
            let file_path = entry.path();

            // Check for platform-specific shared library extensions.
            if let Some(ext) = file_path.extension() {
                if ext == "so" || ext == "dll" || ext == "dylib" {
                    let path_str = file_path.to_str().unwrap_or_default().to_string();
                    log::debug!("Attempting to load plugin from: {}", path_str);

                    // Load the library.
                    let library = Library::new(&file_path).map_err(|e| PluginError::LibraryLoad {
                        path: path_str.clone(),
                        source: e,
                    })?;

                    // Find the constructor symbol.
                    let constructor: Symbol<PluginCreate> =
                        library.get(b"_plugin_create\0").map_err(|e| PluginError::SymbolNotFound {
                            path: path_str.clone(),
                            symbol: "_plugin_create".to_string(),
                        })?;

                    // Create the plugin instance and take ownership of the raw pointer.
                    let mut plugin = Box::from_raw(constructor());
                    plugin.on_load();

                    // For now, use the file name as the plugin's unique name.
                    let plugin_name = file_path.file_stem().unwrap().to_str().unwrap().to_string();
                    log::info!("Successfully loaded plugin: {}", plugin_name);
                    
                    self.plugins.insert(
                        plugin_name,
                        PluginHandle { plugin, library },
                    );
                }
            }
        }

        Ok(())
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for PluginManager {
    /// Ensures all plugins are gracefully unloaded when the manager is dropped.
    fn drop(&mut self) {
        log::info!("Shutting down PluginManager and unloading all plugins.");
        for (name, handle) in self.plugins.iter_mut() {
            log::debug!("Unloading plugin: {}", name);
            handle.plugin.on_unload();
        }
        self.plugins.clear();
    }
}
