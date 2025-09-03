/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: config_loader.rs
 *
 * This file implements a flexible and powerful configuration loader. It is designed
 * to load configuration from various sources (e.g., TOML, JSON files), and supports
 * being overridden by environment variables, which is a common pattern for modern,
 * containerized applications. The loader is generic and can deserialize into any
 * struct that implements `serde::Deserialize`.
 *
 * Dependencies:
 *  - `serde`: For deserialization of configuration data.
 *  - `toml`, `serde_json`: For parsing specific configuration formats.
 *  - `thiserror`: For structured error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Imports ---
use serde::de::DeserializeOwned;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

// --- Custom Error and Result Types ---

/// Represents errors that can occur during configuration loading.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Failed to read configuration file at '{path}': {error}")]
    FileReadError {
        path: PathBuf,
        #[source]
        error: std::io::Error,
    },
    #[error("Failed to parse TOML configuration: {0}")]
    TomlParseError(#[from] toml::de::Error),
    #[error("Failed to parse JSON configuration: {0}")]
    JsonParseError(#[from] serde_json::Error),
    #[error("Unsupported configuration file format for path: {path}")]
    UnsupportedFormat { path: PathBuf },
    #[error("Environment variable '{key}' could not be read: {error}")]
    EnvVarError {
        key: String,
        #[source]
        error: env::VarError,
    },
}

pub type ConfigResult<T> = Result<T, ConfigError>;

// --- Example Configuration Structure ---

/// An example of a nested application configuration structure.
/// This demonstrates how a complex configuration can be organized.
/// The `#[serde(default)]` attribute is useful for making entire sections optional.
#[derive(serde::Deserialize, Debug, Clone)]
pub struct AppConfig {
    #[serde(default)]
    pub database: DatabaseConfig,
    #[serde(default)]
    pub api: ApiConfig,
    pub log_level: Option<String>,
}

#[derive(serde::Deserialize, Debug, Clone)]
#[serde(default)]
pub struct DatabaseConfig {
    pub host: String,
    pub port: u16,
    pub user: String,
    pub password: Option<String>, // Option for values that might come from env vars
}

impl Default for DatabaseConfig {
    fn default() -> Self { Self { host: "localhost".to_string(), port: 5432, user: "root".to_string(), password: None } }
}

#[derive(serde::Deserialize, Debug, Clone)]
#[serde(default)]
pub struct ApiConfig {
    pub listen_address: String,
    pub timeout_seconds: u64,
}

impl Default for ApiConfig {
    fn default() -> Self { Self { listen_address: "127.0.0.1:8080".to_string(), timeout_seconds: 30 } }
}


// --- Configuration Loading Service ---

/// A service for loading configuration from files and environment variables.
#[derive(Default)]
pub struct ConfigLoader;

impl ConfigLoader {
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads configuration from a file path, automatically detecting the format.
    /// It then applies environment variable overrides.
    ///
    /// # Arguments
    /// * `path` - The path to the configuration file (e.g., `config.toml`).
    ///
    /// # Returns
    /// A fully resolved configuration struct of type `T`.
    pub fn load_from_file<T: DeserializeOwned + Configurable>(&self, path: &Path) -> ConfigResult<T> {
        let contents = fs::read_to_string(path).map_err(|e| ConfigError::FileReadError {
            path: path.to_path_buf(),
            error: e,
        })?;

        let mut config: T = match path.extension().and_then(|s| s.to_str()) {
            Some("toml") => toml::from_str(&contents)?,
            Some("json") => serde_json::from_str(&contents)?,
            _ => {
                return Err(ConfigError::UnsupportedFormat {
                    path: path.to_path_buf(),
                })
            }
        };

        // Apply environment variable overrides
        config.apply_env_overrides()?;

        Ok(config)
    }
}

/// A trait that allows a configuration struct to define its own logic
/// for being overridden by environment variables.
pub trait Configurable {
    fn apply_env_overrides(&mut self) -> ConfigResult<()>;
}

// Example implementation of overrides for our `AppConfig`.
// This pattern keeps the override logic coupled with the config struct itself.
impl Configurable for AppConfig {
    fn apply_env_overrides(&mut self) -> ConfigResult<()> {
        // Override nested structs
        self.database.apply_env_overrides()?;
        self.api.apply_env_overrides()?;

        // Override top-level fields
        if let Some(val) = read_env_var("APP_LOG_LEVEL")? {
            self.log_level = Some(val);
        }

        Ok(())
    }
}

impl Configurable for DatabaseConfig {
    fn apply_env_overrides(&mut self) -> ConfigResult<()> {
        if let Some(val) = read_env_var("APP_DATABASE_HOST")? { self.host = val; }
        if let Some(val) = read_env_var("APP_DATABASE_PORT")? { self.port = val.parse().unwrap_or(self.port); }
        if let Some(val) = read_env_var("APP_DATABASE_USER")? { self.user = val; }
        if let Some(val) = read_env_var("APP_DATABASE_PASSWORD")? { self.password = Some(val); }
        Ok(())
    }
}

impl Configurable for ApiConfig {
    fn apply_env_overrides(&mut self) -> ConfigResult<()> {
        if let Some(val) = read_env_var("APP_API_LISTEN_ADDRESS")? { self.listen_address = val; }
        if let Some(val) = read_env_var("APP_API_TIMEOUT_SECONDS")? { self.timeout_seconds = val.parse().unwrap_or(self.timeout_seconds); }
        Ok(())
    }
}

/// Helper function to read an environment variable, returning `Ok(None)` if not set.
fn read_env_var(key: &str) -> ConfigResult<Option<String>> {
    match env::var(key) {
        Ok(val) => Ok(Some(val)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(ConfigError::EnvVarError { key: key.to_string(), error }),
    }
}


// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use assert_fs::prelude::*;
    use assert_fs::TempDir;

    #[test]
    fn test_load_from_toml_file() {
        let dir = TempDir::new().unwrap();
        let config_file = dir.child("config.toml");
        config_file.write_str(r#"
            log_level = "debug"
            [database]
            host = "db.example.com"
            user = "testuser"
        "#).unwrap();

        let loader = ConfigLoader::new();
        let config = loader.load_from_file::<AppConfig>(config_file.path()).unwrap();

        assert_eq!(config.log_level, Some("debug".to_string()));
        assert_eq!(config.database.host, "db.example.com");
        assert_eq!(config.database.port, 5432); // From default
        assert_eq!(config.database.user, "testuser");
    }

    #[test]
    fn test_env_variable_overrides() {
        let dir = TempDir::new().unwrap();
        let config_file = dir.child("config.toml");
        config_file.write_str(r#"
            [database]
            host = "file_host"
            port = 1111
        "#).unwrap();

        // Set environment variables to override the file
        env::set_var("APP_DATABASE_HOST", "env_host");
        env::set_var("APP_DATABASE_PORT", "9999");
        env::set_var("APP_API_TIMEOUT_SECONDS", "120"); // Override a default

        let loader = ConfigLoader::new();
        let config = loader.load_from_file::<AppConfig>(config_file.path()).unwrap();

        assert_eq!(config.database.host, "env_host");
        assert_eq!(config.database.port, 9999);
        assert_eq!(config.api.timeout_seconds, 120);

        // Clean up environment variables
        env::remove_var("APP_DATABASE_HOST");
        env::remove_var("APP_DATABASE_PORT");
        env::remove_var("APP_API_TIMEOUT_SECONDS");
    }

    #[test]
    fn test_password_from_env() {
        let dir = TempDir::new().unwrap();
        let config_file = dir.child("config.toml");
        config_file.write_str(r#"
            [database]
            user = "some_user"
        "#).unwrap();

        env::set_var("APP_DATABASE_PASSWORD", "supersecret");

        let loader = ConfigLoader::new();
        let config = loader.load_from_file::<AppConfig>(config_file.path()).unwrap();

        assert_eq!(config.database.password, Some("supersecret".to_string()));

        env::remove_var("APP_DATABASE_PASSWORD");
    }
}
