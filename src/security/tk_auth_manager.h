/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_auth_manager.h
 *
 * This header defines the authentication and session management system for
 * the TrackieLLM project. It provides secure user/device identity management,
 * session token generation and validation, and authorization for sensitive
 * operations. All sensitive data is encrypted using the tk_encryption module
 * and stored securely using the tk_file_manager.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#ifndef TK_AUTH_MANAGER_H
#define TK_AUTH_MANAGER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <time.h>

#include "utils/tk_error_handling.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations of opaque context structures
typedef struct tk_auth_manager_s tk_auth_manager_t;
typedef struct tk_auth_session_s tk_auth_session_t;

/* ---------------------------------------------------------------------------
 * Authentication Manager Lifecycle
 * --------------------------------------------------------------------------- */

/**
 * @brief Initializes the authentication manager.
 *
 * Creates and initializes the authentication manager, loading existing
 * identity and credentials from secure storage if available. If no existing
 * identity is found, it will generate a new device identity.
 *
 * @param[out] auth_mgr Pointer to store the newly created auth manager.
 * @return TK_SUCCESS on success, or an appropriate error code on failure.
 * @pre auth_mgr != NULL
 * @post *auth_mgr is a valid pointer to a tk_auth_manager_t on success.
 */
TK_NODISCARD tk_error_code_t tk_auth_manager_init(tk_auth_manager_t** auth_mgr);

/**
 * @brief Destroys the authentication manager and clears all sensitive data.
 *
 * Securely wipes all authentication data from memory and frees all resources.
 * Any active sessions are automatically invalidated.
 *
 * @param[in,out] auth_mgr Pointer to the auth manager to destroy. Set to NULL on return.
 * @pre auth_mgr != NULL && *auth_mgr != NULL
 * @post *auth_mgr == NULL
 */
void tk_auth_manager_destroy(tk_auth_manager_t** auth_mgr);

/* ---------------------------------------------------------------------------
 * Device Identity Management
 * --------------------------------------------------------------------------- */

/**
 * @brief Gets the unique device identifier.
 *
 * Returns the unique identifier for this device/installation. This ID is
 * generated once and persisted securely across sessions.
 *
 * @param[in] auth_mgr The authentication manager.
 * @param[out] device_id Buffer to store the device ID string.
 * @param[in,out] buffer_size Size of the buffer. Updated with actual size needed.
 * @return TK_SUCCESS on success, TK_ERROR_BUFFER_TOO_SMALL if buffer too small,
 *         or another error code on failure.
 * @pre auth_mgr != NULL && device_id != NULL && buffer_size != NULL
 */
TK_NODISCARD tk_error_code_t tk_auth_manager_get_device_id(const tk_auth_manager_t* auth_mgr,
                                                          char* device_id,
                                                          size_t* buffer_size);

/**
 * @brief Checks if the device has been properly initialized with credentials.
 *
 * @param[in] auth_mgr The authentication manager.
 * @param[out] is_initialized Pointer to store the initialization status.
 * @return TK_SUCCESS on success, or an appropriate error code on failure.
 * @pre auth_mgr != NULL && is_initialized != NULL
 */
TK_NODISCARD tk_error_code_t tk_auth_manager_is_device_initialized(const tk_auth_manager_t* auth_mgr,
                                                                  bool* is_initialized);

/* ---------------------------------------------------------------------------
 * User Authentication
 * --------------------------------------------------------------------------- */

/**
 * @brief Sets a user PIN/password for authentication.
 *
 * Establishes or updates the user's authentication credential. The PIN is
 * hashed and stored securely. This is typically called during first setup
 * or when changing credentials.
 *
 * @param[in] auth_mgr The authentication manager.
 * @param[in] pin The PIN/password to set (null-terminated string).
 * @return TK_SUCCESS on success, or an appropriate error code on failure.
 * @pre auth_mgr != NULL && pin != NULL
 */
TK_NODISCARD tk_error_code_t tk_auth_manager_set_user_pin(tk_auth_manager_t* auth_mgr,
                                                         const char* pin);

/**
 * @brief Verifies a user PIN/password.
 *
 * Validates the provided PIN against the stored credential hash.
 *
 * @param[in] auth_mgr The authentication manager.
 * @param[in] pin The PIN/password to verify (null-terminated string).
 * @param[out] is_valid Pointer to store the verification result.
 * @return TK_SUCCESS on success, or an appropriate error code on failure.
 * @pre auth_mgr != NULL && pin != NULL && is_valid != NULL
 */
TK_NODISCARD tk_error_code_t tk_auth_manager_verify_pin(const tk_auth_manager_t* auth_mgr,
                                                       const char* pin,
                                                       bool* is_valid);

/* ---------------------------------------------------------------------------
 * Session Management
 * --------------------------------------------------------------------------- */

/**
 * @brief Creates a new authenticated session.
 *
 * Generates a new session token with specified permissions and expiration time.
 * The session must be created after successful PIN verification.
 *
 * @param[in] auth_mgr The authentication manager.
 * @param[in] permissions Bitmask of permissions for this session.
 * @param[in] duration_seconds Session duration in seconds (0 for default).
 * @param[out] session Pointer to store the newly created session.
 * @return TK_SUCCESS on success, or an appropriate error code on failure.
 * @pre auth_mgr != NULL && session != NULL
 */
TK_NODISCARD tk_error_code_t tk_auth_manager_create_session(tk_auth_manager_t* auth_mgr,
                                                           uint32_t permissions,
                                                           uint32_t duration_seconds,
                                                           tk_auth_session_t** session);

/**
 * @brief Validates an existing session.
 *
 * Checks if a session token is valid and not expired.
 *
 * @param[in] auth_mgr The authentication manager.
 * @param[in] session The session to validate.
 * @param[out] is_valid Pointer to store the validation result.
 * @return TK_SUCCESS on success, or an appropriate error code on failure.
 * @pre auth_mgr != NULL && session != NULL && is_valid != NULL
 */
TK_NODISCARD tk_error_code_t tk_auth_manager_validate_session(const tk_auth_manager_t* auth_mgr,
                                                             const tk_auth_session_t* session,
                                                             bool* is_valid);

/**
 * @brief Invalidates a session.
 *
 * Marks a session as invalid, effectively logging out the user.
 *
 * @param[in] auth_mgr The authentication manager.
 * @param[in,out] session The session to invalidate.
 * @return TK_SUCCESS on success, or an appropriate error code on failure.
 * @pre auth_mgr != NULL && session != NULL
 */
TK_NODISCARD tk_error_code_t tk_auth_manager_invalidate_session(tk_auth_manager_t* auth_mgr,
                                                               tk_auth_session_t* session);

/**
 * @brief Destroys a session and frees its resources.
 *
 * Securely wipes session data and frees memory.
 *
 * @param[in,out] session Pointer to the session to destroy. Set to NULL on return.
 * @pre session != NULL && *session != NULL
 * @post *session == NULL
 */
void tk_auth_session_destroy(tk_auth_session_t** session);

/* ---------------------------------------------------------------------------
 * Permission and Authorization
 * --------------------------------------------------------------------------- */

/**
 * @brief Checks if a session has specific permissions.
 *
 * Validates that the current session has the required permissions for an operation.
 *
 * @param[in] session The session to check.
 * @param[in] required_permissions Bitmask of required permissions.
 * @param[out] has_permission Pointer to store the permission check result.
 * @return TK_SUCCESS on success, or an appropriate error code on failure.
 * @pre session != NULL && has_permission != NULL
 */
TK_NODISCARD tk_error_code_t tk_auth_session_has_permission(const tk_auth_session_t* session,
                                                           uint32_t required_permissions,
                                                           bool* has_permission);

/**
 * @brief Gets the remaining time for a session.
 *
 * Returns the number of seconds until the session expires.
 *
 * @param[in] session The session to check.
 * @param[out] seconds_remaining Pointer to store remaining seconds (0 if expired).
 * @return TK_SUCCESS on success, or an appropriate error code on failure.
 * @pre session != NULL && seconds_remaining != NULL
 */
TK_NODISCARD tk_error_code_t tk_auth_session_time_remaining(const tk_auth_session_t* session,
                                                           uint32_t* seconds_remaining);

/* ---------------------------------------------------------------------------
 * Secure Storage Operations
 * --------------------------------------------------------------------------- */

/**
 * @brief Saves current authentication state to secure storage.
 *
 * Persists device identity, credentials, and configuration to encrypted files.
 *
 * @param[in] auth_mgr The authentication manager.
 * @return TK_SUCCESS on success, or an appropriate error code on failure.
 * @pre auth_mgr != NULL
 */
TK_NODISCARD tk_error_code_t tk_auth_manager_save_state(const tk_auth_manager_t* auth_mgr);

/**
 * @brief Loads authentication state from secure storage.
 *
 * Restores device identity and credentials from encrypted files.
 *
 * @param[in] auth_mgr The authentication manager.
 * @return TK_SUCCESS on success, TK_ERROR_NOT_FOUND if no saved state exists,
 *         or another error code on failure.
 * @pre auth_mgr != NULL
 */
TK_NODISCARD tk_error_code_t tk_auth_manager_load_state(tk_auth_manager_t* auth_mgr);

/* ---------------------------------------------------------------------------
 * Constants and Enumerations
 * --------------------------------------------------------------------------- */

/** Maximum length for device ID string (including null terminator) */
#define TK_AUTH_DEVICE_ID_MAX_LENGTH    64

/** Maximum length for PIN (including null terminator) */
#define TK_AUTH_PIN_MAX_LENGTH         128

/** Default session duration in seconds (1 hour) */
#define TK_AUTH_DEFAULT_SESSION_DURATION  3600

/** Maximum session duration in seconds (24 hours) */
#define TK_AUTH_MAX_SESSION_DURATION     86400

/** Permission flags for session authorization */
typedef enum {
    TK_AUTH_PERM_NONE               = 0x00000000,  /**< No special permissions */
    TK_AUTH_PERM_READ_CONFIG        = 0x00000001,  /**< Read configuration files */
    TK_AUTH_PERM_WRITE_CONFIG       = 0x00000002,  /**< Write configuration files */
    TK_AUTH_PERM_SYSTEM_UPDATE      = 0x00000004,  /**< Perform system updates */
    TK_AUTH_PERM_USER_DATA_ACCESS   = 0x00000008,  /**< Access user personal data */
    TK_AUTH_PERM_NETWORK_ACCESS     = 0x00000010,  /**< Access network resources */
    TK_AUTH_PERM_DEVICE_CONTROL     = 0x00000020,  /**< Control hardware devices */
    TK_AUTH_PERM_ADMIN              = 0xFFFFFFFF   /**< Full administrative access */
} tk_auth_permission_t;

#ifdef __cplusplus
}
#endif

#endif // TK_AUTH_MANAGER_H