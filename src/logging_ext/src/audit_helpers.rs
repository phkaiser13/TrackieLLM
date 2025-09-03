/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/logging_ext/audit_helpers.rs
 *
 * This file provides a set of high-level helper functions for generating
 * standardized audit logs. Audit logging is a critical security practice that
 * involves recording a chronological sequence of events related to security,
 * compliance, and system integrity.
 *
 * By providing simple, clear functions for common audit events (e.g., user
 * login, file access, configuration changes), this module helps ensure that
 * all parts of the application log these events in a consistent and
 * machine-parsable format. This is vital for security information and event
 * management (SIEM) systems.
 *
 * The helpers use the standard `log::info!` macro but with a specific target
 * (`"audit"`) and a structured message format, making them easy to filter
 * and process by the `JsonLogFormatter` or other logging backends.
 *
 * Dependencies:
 *   - log: The logging facade used to emit the audit events.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use log::{info, warn};

/// Represents the category of an audit event.
/// This is used to classify events for easier filtering and analysis.
#[derive(Debug)]
pub enum AuditCategory {
    /// Events related to user authentication.
    Authentication,
    /// Events related to authorization and access control.
    Authorization,
    /// Events related to changes in system configuration.
    Configuration,
    /// Events related to filesystem access.
    DataAccess,
}

impl std::fmt::Display for AuditCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Represents a specific audit event with its associated metadata.
#[derive(Debug)]
pub struct AuditEvent<'a> {
    /// The category of the event.
    pub category: AuditCategory,
    /// A specific identifier for the action being performed (e.g., "USER_LOGIN").
    pub action: &'a str,
    /// The outcome of the event (e.g., "SUCCESS", "FAILURE").
    pub outcome: &'a str,
    /// The user, service, or entity that initiated the event.
    pub actor: &'a str,
    /// The resource or object that was the target of the event.
    pub target: Option<&'a str>,
    /// Additional, human-readable details about the event.
    pub details: &'a str,
}

/// The core function for logging an audit event.
///
/// It takes an `AuditEvent` and logs it as a structured message using `log::info!`.
/// The log's target is always set to "audit" to allow for specific routing
/// of audit logs.
///
/// # Arguments
///
/// * `event` - The `AuditEvent` to be logged.
fn log_audit_event(event: AuditEvent) {
    // The format is designed to be key-value like for easy parsing, even if
    // the final output is not JSON. The JSON formatter will handle this more
    // robustly, but this provides a decent fallback.
    info!(
        target: "audit",
        "category=\"{}\" action=\"{}\" outcome=\"{}\" actor=\"{}\" target=\"{}\" details=\"{}\"",
        event.category,
        event.action,
        event.outcome,
        event.actor,
        event.target.unwrap_or("N/A"),
        event.details
    );
}

// --- Public Helper Functions ---

/// Logs a successful user authentication event.
///
/// # Arguments
///
/// * `username` - The name of the user who successfully authenticated.
/// * `source_ip` - The IP address from which the authentication request originated.
pub fn log_authentication_success(username: &str, source_ip: &str) {
    log_audit_event(AuditEvent {
        category: AuditCategory::Authentication,
        action: "USER_LOGIN",
        outcome: "SUCCESS",
        actor: username,
        target: Some(source_ip),
        details: &format!("User '{}' successfully logged in from IP '{}'.", username, source_ip),
    });
}

/// Logs a failed user authentication attempt.
///
/// # Arguments
///
/// * `username` - The username used in the failed attempt.
/// * `source_ip` - The IP address from which the attempt originated.
/// * `reason` - The reason for the failure (e.g., "InvalidPassword", "UserNotFound").
pub fn log_authentication_failure(username: &str, source_ip: &str, reason: &str) {
    // Failures are often more security-sensitive, so we log them with a `warn` level.
    warn!(
        target: "audit",
        "category=\"{}\" action=\"{}\" outcome=\"{}\" actor=\"{}\" target=\"{}\" details=\"{}\"",
        AuditCategory::Authentication,
        "USER_LOGIN",
        "FAILURE",
        username,
        source_ip,
        &format!("Failed login attempt for user '{}' from IP '{}'. Reason: {}", username, source_ip, reason)
    );
}

/// Logs a change in system configuration.
///
/// # Arguments
///
/// * `actor` - The user or process that made the change.
/// * `setting_changed` - The name of the configuration key that was changed.
/// * `old_value` - The value of the setting before the change.
/// * `new_value` - The value of the setting after the change.
pub fn log_config_change(actor: &str, setting_changed: &str, old_value: &str, new_value: &str) {
    log_audit_event(AuditEvent {
        category: AuditCategory::Configuration,
        action: "CONFIG_UPDATE",
        outcome: "SUCCESS",
        actor,
        target: Some(setting_changed),
        details: &format!(
            "Configuration setting '{}' changed from '{}' to '{}'.",
            setting_changed, old_value, new_value
        ),
    });
}

/// Logs access to a sensitive file or resource.
///
/// # Arguments
///
/// * `actor` - The user or process that accessed the resource.
/// * `resource_path` - The path to the resource that was accessed.
/// * `operation` - The operation performed (e.g., "READ", "WRITE").
pub fn log_file_access(actor: &str, resource_path: &str, operation: &str) {
    log_audit_event(AuditEvent {
        category: AuditCategory::DataAccess,
        action: "FILE_ACCESS",
        outcome: "SUCCESS",
        actor,
        target: Some(resource_path),
        details: &format!(
            "Performed {} operation on resource '{}'.",
            operation, resource_path
        ),
    });
}
