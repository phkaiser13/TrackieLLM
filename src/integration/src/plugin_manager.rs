/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: plugin_manager.rs
 *
 * This file implements the core `PluginManager`, which is responsible for the
 * lifecycle of dynamic plugins. It handles loading shared libraries (.so, .dll),
 * resolving the plugin entry point symbol, initializing the plugin, and safely
 * unloading it. The design ensures that library handles are kept alive for as
 * long as the plugin is in use, preventing memory safety issues.
 *
 * Dependencies:
 *  - `libloading`: For cross-platform dynamic library loading.
 *  - `thiserror`: For structured error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Imports ---
use crate::bridge::{BridgeContext, Plugin, PluginError};
use libloading::{Library, Symbol};
use std::collections::HashMap;
use std::path::Path;

// --- Custom Error and Result Types ---

#[derive(Debug, thiserror::Error)]
pub enum PluginManagerError {
    #[error("Failed to load shared library from '{path}': {error}")]
    LibraryLoadFailed {
        path: String,
        #[source]
        error: libloading::Error,
    },
    #[error("Could not find the required entry point symbol '_plugin_create' in '{path}'.")]
    SymbolNotFound {
        path: String,
        #[source]
        error: libloading::Error,
    },
    #[error("Plugin '{name}' failed to load: {error}")]
    PluginLoadFailed {
        name: String,
        #[source]
        error: PluginError,
    },
    #[error("A plugin with the name '{0}' is already loaded.")]
    PluginAlreadyLoaded(String),
    #[error("Plugin with name '{0}' not found.")]
    PluginNotFound(String),
}

pub type PluginManagerResult<T> = Result<T, PluginManagerError>;

// --- Plugin Management Structures ---

/// A container for a loaded plugin, holding both the plugin trait object
/// and the library handle. The library handle *must* be kept alive for the
/// duration of the plugin's life to prevent the library from being unloaded.
struct LoadedPlugin {
    plugin: Box<dyn Plugin>,
    // This field is crucial. When `LoadedPlugin` is dropped, `library` is
    // dropped, which unloads the dynamic library from memory.
    library: Library,
}

/// The entry point symbol that every plugin must export.
type PluginCreate = unsafe extern "C" fn() -> *mut dyn Plugin;


/// The main service for managing the plugin lifecycle.
#[derive(Default)]
pub struct PluginManager {
    // Plugins are stored in a map, keyed by the name they report.
    plugins: HashMap<String, LoadedPlugin>,
}

impl PluginManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads a plugin from a shared library file.
    ///
    /// # Arguments
    /// * `library_path` - The path to the dynamic library (.so, .dll, .dylib).
    /// * `context` - The `BridgeContext` to pass to the plugin's `on_load` method.
    ///
    /// # Returns
    /// The name of the loaded plugin on success.
    pub fn load(&mut self, library_path: &Path, context: BridgeContext) -> PluginManagerResult<String> {
        // 1. Load the dynamic library into memory. This is an `unsafe` operation
        //    because it involves interacting with the OS's library loader.
        let library = unsafe {
            Library::new(library_path).map_err(|e| PluginManagerError::LibraryLoadFailed {
                path: library_path.to_string_lossy().into_owned(),
                error: e,
            })?
        };

        // 2. Get a reference to the `_plugin_create` function symbol.
        let create_func: Symbol<PluginCreate> = unsafe {
            library
                .get(b"_plugin_create\0") // The symbol name must be null-terminated.
                .map_err(|e| PluginManagerError::SymbolNotFound {
                    path: library_path.to_string_lossy().into_owned(),
                    error: e,
                })?
        };

        // 3. Call the constructor function to get a raw pointer to the plugin object.
        //    The plugin crate "leaks" a Box, and we reconstruct it here to take ownership.
        let mut plugin = unsafe { Box::from_raw(create_func()) };
        let plugin_name = plugin.name().to_string();

        if self.plugins.contains_key(&plugin_name) {
            return Err(PluginManagerError::PluginAlreadyLoaded(plugin_name));
        }

        // 4. Call the plugin's `on_load` lifecycle hook.
        plugin
            .on_load(context)
            .map_err(|e| PluginManagerError::PluginLoadFailed {
                name: plugin_name.clone(),
                error: e,
            })?;

        // 5. Store the loaded plugin and its library handle.
        let loaded_plugin = LoadedPlugin { plugin, library };
        self.plugins.insert(plugin_name.clone(), loaded_plugin);

        println!("Successfully loaded plugin '{}' from '{}'", plugin_name, library_path.display());
        Ok(plugin_name)
    }

    /// Unloads a plugin by its name.
    pub fn unload(&mut self, name: &str) -> PluginManagerResult<()> {
        if let Some(mut loaded_plugin) = self.plugins.remove(name) {
            // Call the cleanup hook before dropping the plugin.
            loaded_plugin.plugin.on_unload();
            println!("Successfully unloaded plugin '{}'.", name);
            // `loaded_plugin` is dropped here, which in turn drops its `library`
            // field, safely unloading the shared library from memory.
            Ok(())
        } else {
            Err(PluginManagerError::PluginNotFound(name.to_string()))
        }
    }

    /// Returns a reference to a loaded plugin by its name.
    pub fn get_plugin(&self, name: &str) -> Option<&dyn Plugin> {
        self.plugins.get(name).map(|lp| lp.plugin.as_ref())
    }

    /// Returns a list of the names of all currently loaded plugins.
    pub fn list_plugins(&self) -> Vec<String> {
        self.plugins.keys().cloned().collect()
    }
}

impl Drop for PluginManager {
    /// Ensures all plugins are unloaded when the manager is dropped.
    fn drop(&mut self) {
        let names: Vec<String> = self.plugins.keys().cloned().collect();
        for name in names {
            println!("Manager is dropping. Unloading plugin '{}'...", name);
            // We ignore the result here as we can't do much about an error during drop.
            let _ = self.unload(&name);
        }
    }
}

// --- Unit Tests ---
// Note: Testing dynamic library loading is complex. It would typically require
// compiling a separate dummy plugin crate as part of the test process. These
// tests are conceptual and focus on the manager's state logic.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_initialization_and_drop() {
        let mut manager = PluginManager::new();
        // Conceptually, we'd load a plugin here.
        // For this test, we just ensure it can be created and dropped without panicking.
    }

    #[test]
    fn test_list_plugins_is_initially_empty() {
        let manager = PluginManager::new();
        assert!(manager.list_plugins().is_empty());
    }

    // The following test would require a compiled dummy plugin library.
    // We will ignore it for now as it cannot run in this environment.
    #[test]
    #[ignore]
    fn conceptual_test_load_and_unload_plugin() {
        // 1. Assume `dummy_plugin.so` exists and was compiled from a crate
        //    that uses our `declare_plugin!` macro.
        let plugin_path = Path::new("path/to/dummy_plugin.so");
        let context = BridgeContext::default();

        let mut manager = PluginManager::new();

        // 2. Load the plugin.
        let plugin_name = manager.load(plugin_path, context).unwrap();
        assert_eq!(plugin_name, "DummyPlugin");
        assert_eq!(manager.list_plugins(), vec!["DummyPlugin"]);

        // 3. Get the plugin.
        let plugin_ref = manager.get_plugin("DummyPlugin").unwrap();
        assert_eq!(plugin_ref.name(), "DummyPlugin");

        // 4. Unload the plugin.
        manager.unload("DummyPlugin").unwrap();
        assert!(manager.list_plugins().is_empty());
        assert!(manager.get_plugin("DummyPlugin").is_none());
    }
}
