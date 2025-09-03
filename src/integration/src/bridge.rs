/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: bridge.rs
 *
 * This file defines the core contract for the plugin system. It contains the `Plugin`
 * trait that all dynamic plugins must implement. It also defines the `BridgeContext`,
 * which acts as a controlled channel for communication from the host application to
 * the plugin, enabling a secure and decoupled integration.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Imports ---
use std::any::Any;

// --- Custom Error Type ---
// We use `anyhow::Error` for the plugin interface. It provides a flexible and
// ergonomic way for plugins to return arbitrary errors, while the host can
// still interact with them as a standard error type.
pub type PluginError = anyhow::Error;

// --- Plugin Contract and Context ---

/// A context object passed from the host application to the plugin upon loading.
///
/// This struct acts as a "bridge," providing plugins with safe, controlled access
/// to host functionality or shared state without exposing internal details.
/// For example, it could contain a logging callback or a handle to a host service.
#[derive(Default)]
pub struct BridgeContext {
    // In a real application, this could hold `Arc<Logger>`, `Sender<HostMessage>`, etc.
    // For this example, we'll keep it simple.
    pub host_version: String,
}

/// The main trait that all plugins must implement.
///
/// This defines the lifecycle and basic identity of a plugin. The host application
/// will interact with the loaded plugin object through this trait.
/// The `Send + Sync` bounds are crucial to ensure that plugins can be used safely
/// in a multi-threaded environment. The `'static` lifetime ensures that the plugin
/// does not hold any non-static references.
pub trait Plugin: Any + Send + Sync + 'static {
    /// Returns the unique, human-readable name of the plugin.
    fn name(&self) -> &str;

    /// Called exactly once when the plugin is loaded by the `PluginManager`.
    ///
    /// This is the plugin's opportunity to initialize its state, start background
    /// tasks, and register itself with any host services provided in the `context`.
    ///
    /// # Arguments
    /// * `context` - The `BridgeContext` providing access to host functionality.
    ///
    /// # Returns
    /// `Ok(())` on successful initialization, or an error if setup fails.
    /// A failure here will prevent the plugin from being loaded.
    fn on_load(&mut self, context: BridgeContext) -> Result<(), PluginError>;

    /// Called exactly once just before the plugin is unloaded.
    ///
    /// This method should be used to perform any necessary cleanup, such as
    /// shutting down background threads, flushing buffers, or saving state.
    /// After this method returns, the plugin's library will be released from memory.
    fn on_unload(&mut self);

    /// Provides a mechanism for downcasting, allowing the host to interact with
    /// concrete plugin types if necessary. This is an advanced pattern that adds
    /// flexibility for plugins that expose more than the basic `Plugin` interface.
    fn as_any(&self) -> &dyn Any;
}

// --- Plugin Declaration Macro ---

/// A macro to be used within a plugin crate to declare the plugin's existence.
///
/// This macro creates the `_plugin_create` symbol that the `PluginManager` will
/// look for. It handles the boilerplate of creating an instance of the plugin
/// struct and boxing it up as a `Box<dyn Plugin>`.
///
/// # Example Usage (in a plugin crate)
///
/// ```rust,ignore
/// use trackiellm_integration::bridge::{Plugin, declare_plugin};
///
/// struct MyCoolPlugin;
/// impl Plugin for MyCoolPlugin { /* ... */ }
///
/// declare_plugin!(MyCoolPlugin, MyCoolPlugin::new);
/// ```
#[macro_export]
macro_rules! declare_plugin {
    ($plugin_type:ty, $constructor:path) => {
        #[no_mangle]
        pub extern "C" fn _plugin_create() -> *mut dyn $crate::bridge::Plugin {
            // The constructor is called to create an instance of the plugin.
            let constructor: fn() -> $plugin_type = $constructor;
            let object = constructor();
            let boxed: Box<dyn $crate::bridge::Plugin> = Box::new(object);
            // The Box is leaked, and a raw pointer is returned to the host.
            // The host is now responsible for this memory and must reconstruct
            // the Box to properly drop the plugin later.
            Box::into_raw(boxed)
        }
    };
}
