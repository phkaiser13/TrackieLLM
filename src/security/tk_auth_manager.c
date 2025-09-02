/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_auth_manager.c
 *
 * This source file implements the authentication and session management system
 * for the TrackieLLM project. It provides secure device identity management,
 * user authentication with PIN/password, session token generation and validation,
 * and authorization controls. All sensitive data is encrypted using libsodium
 * through the tk_encryption module and stored securely using tk_file_manager.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>

#include <sodium.h>

#include "security/tk_auth_manager.h"
#include "security/tk_encryption.h"
#include "utils/tk_error_handling.h"
#include "utils/tk_logging.h"

/* TODO: Include these when available */
/* #include "internal_tools/tk_file_manager.h" */

/* ---------------------------------------------------------------------------
 * Internal data structures
 * --------------------------------------------------------------------------- */

/** Internal structure for authentication manager */
struct tk_auth_manager_s {
    bool                    is_initialized;
    char                    device_id[TK_AUTH_DEVICE_ID_MAX_LENGTH];
    uint8_t                 pin_hash[crypto_pwhash_STRBYTES];
    bool                    has_pin;
    tk_encryption_ctx_t*    crypto_ctx;
    uint32_t                next_session_id;
};

/** Internal structure for authentication session */
struct tk_auth_session_s {
    uint32_t    session_id;
    time_t      created_time;
    time_t      expires_time;
    uint32_t    permissions;
    bool        is_valid;
    uint8_t     token_hash[32];  /* SHA256 hash of session token */
};

/* ---------------------------------------------------------------------------
 * Internal helper functions
 * --------------------------------------------------------------------------- */

/**
 * @brief Generates a cryptographically secure device ID.
 */
static tk_error_code_t generate_device_id(char* device_id, size_t buffer_size) {
    if (device_id == NULL || buffer_size < TK_AUTH_DEVICE_ID_MAX_LENGTH) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    uint8_t random_bytes[16];
    randombytes_buf(random_bytes, sizeof(random_bytes));

    /* Format as hex string with prefix */
    snprintf(device_id, buffer_size, "TKD-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             random_bytes[0], random_bytes[1], random_bytes[2], random_bytes[3],
             random_bytes[4], random_bytes[5], random_bytes[6], random_bytes[7],
             random_bytes[8], random_bytes[9], random_bytes[10], random_bytes[11],
             random_bytes[12], random_bytes[13], random_bytes[14], random_bytes[15]);

    return TK_SUCCESS;
}

/**
 * @brief Hashes a PIN using Argon2id.
 */
static tk_error_code_t hash_pin(const char* pin, uint8_t* hash_output) {
    if (pin == NULL || hash_output == NULL) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    if (crypto_pwhash_str((char*)hash_output, pin, strlen(pin),
                         crypto_pwhash_OPSLIMIT_INTERACTIVE,
                         crypto_pwhash_MEMLIMIT_INTERACTIVE) != 0) {
        tk_log_error("hash_pin: crypto_pwhash_str failed");
        return TK_ERROR_INTERNAL;
    }

    return TK_SUCCESS;
}

/**
 * @brief Verifies a PIN against its hash.
 */
static bool verify_pin_hash(const char* pin, const uint8_t* hash) {
    if (pin == NULL || hash == NULL) {
        return false;
    }

    return crypto_pwhash_str_verify((const char*)hash, pin, strlen(pin)) == 0;
}

/**
 * @brief Generates a session token hash.
 */
static tk_error_code_t generate_session_token_hash(uint32_t session_id, time_t created_time, 
                                                  uint8_t* token_hash) {
    if (token_hash == NULL) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    /* Create token data from session ID, timestamp and random data */
    uint8_t token_data[64];
    uint8_t random_data[32];
    
    randombytes_buf(random_data, sizeof(random_data));
    
    memcpy(token_data, &session_id, sizeof(session_id));
    memcpy(token_data + sizeof(session_id), &created_time, sizeof(created_time));
    memcpy(token_data + sizeof(session_id) + sizeof(created_time), random_data, 
           sizeof(token_data) - sizeof(session_id) - sizeof(created_time));

    /* Hash the token data */
    crypto_hash_sha256(token_hash, token_data, sizeof(token_data));
    
    /* Securely wipe the temporary data */
    sodium_memzero(token_data, sizeof(token_data));
    sodium_memzero(random_data, sizeof(random_data));

    return TK_SUCCESS;
}

/* ---------------------------------------------------------------------------
 * Public API implementation
 * --------------------------------------------------------------------------- */

TK_NODISCARD tk_error_code_t tk_auth_manager_init(tk_auth_manager_t** auth_mgr) {
    if (auth_mgr == NULL) {
        tk_log_error("tk_auth_manager_init: auth_mgr is NULL");
        return TK_ERROR_INVALID_ARGUMENT;
    }

    /* Initialize libsodium */
    if (sodium_init() == -1) {
        tk_log_error("tk_auth_manager_init: libsodium initialization failed");
        return TK_ERROR_INTERNAL;
    }

    /* Allocate the manager structure */
    tk_auth_manager_t* mgr = (tk_auth_manager_t*)calloc(1, sizeof(tk_auth_manager_t));
    if (mgr == NULL) {
        tk_log_error("tk_auth_manager_init: out of memory");
        return TK_ERROR_OUT_OF_MEMORY;
    }

    /* Initialize encryption context */
    tk_error_code_t err = tk_encryption_ctx_create(&mgr->crypto_ctx);
    if (err != TK_SUCCESS) {
        tk_log_error("tk_auth_manager_init: failed to create crypto context: %s", 
                     tk_error_to_string(err));
        free(mgr);
        return err;
    }

    /* Try to load existing state */
    err = tk_auth_manager_load_state(mgr);
    if (err == TK_ERROR_NOT_FOUND) {
        /* First time setup - generate new device ID */
        err = generate_device_id(mgr->device_id, sizeof(mgr->device_id));
        if (err != TK_SUCCESS) {
            tk_log_error("tk_auth_manager_init: failed to generate device ID: %s",
                         tk_error_to_string(err));
            tk_encryption_ctx_destroy(&mgr->crypto_ctx);
            free(mgr);
            return err;
        }
        
        /* Generate encryption key */
        err = tk_encryption_generate_key(mgr->crypto_ctx);
        if (err != TK_SUCCESS) {
            tk_log_error("tk_auth_manager_init: failed to generate encryption key: %s",
                         tk_error_to_string(err));
            tk_encryption_ctx_destroy(&mgr->crypto_ctx);
            free(mgr);
            return err;
        }

        tk_log_info("tk_auth_manager_init: created new device identity: %s", mgr->device_id);
    } else if (err != TK_SUCCESS) {
        tk_log_error("tk_auth_manager_init: failed to load state: %s", 
                     tk_error_to_string(err));
        tk_encryption_ctx_destroy(&mgr->crypto_ctx);
        free(mgr);
        return err;
    }

    mgr->is_initialized = true;
    mgr->next_session_id = 1;
    *auth_mgr = mgr;

    tk_log_debug("tk_auth_manager_init: authentication manager initialized");
    return TK_SUCCESS;
}

void tk_auth_manager_destroy(tk_auth_manager_t** auth_mgr) {
    if (auth_mgr == NULL || *auth_mgr == NULL) {
        return;
    }

    tk_auth_manager_t* mgr = *auth_mgr;

    /* Destroy crypto context */
    if (mgr->crypto_ctx != NULL) {
        tk_encryption_ctx_destroy(&mgr->crypto_ctx);
    }

    /* Securely wipe sensitive data */
    sodium_memzero(mgr->pin_hash, sizeof(mgr->pin_hash));
    sodium_memzero(mgr->device_id, sizeof(mgr->device_id));

    /* Free the structure */
    sodium_memzero(mgr, sizeof(*mgr));
    free(mgr);
    *auth_mgr = NULL;

    tk_log_debug("tk_auth_manager_destroy: authentication manager destroyed");
}

TK_NODISCARD tk_error_code_t tk_auth_manager_get_device_id(const tk_auth_manager_t* auth_mgr,
                                                          char* device_id,
                                                          size_t* buffer_size) {
    if (auth_mgr == NULL || device_id == NULL || buffer_size == NULL) {
        tk_log_error("tk_auth_manager_get_device_id: invalid arguments");
        return TK_ERROR_INVALID_ARGUMENT;
    }

    if (!auth_mgr->is_initialized) {
        tk_log_error("tk_auth_manager_get_device_id: manager not initialized");
        return TK_ERROR_INVALID_STATE;
    }

    size_t required_size = strlen(auth_mgr->device_id) + 1;
    if (*buffer_size < required_size) {
        *buffer_size = required_size;
        return TK_ERROR_BUFFER_TOO_SMALL;
    }

    strcpy(device_id, auth_mgr->device_id);
    *buffer_size = required_size;
    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t tk_auth_manager_is_device_initialized(const tk_auth_manager_t* auth_mgr,
                                                                  bool* is_initialized) {
    if (auth_mgr == NULL || is_initialized == NULL) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    *is_initialized = auth_mgr->is_initialized && (strlen(auth_mgr->device_id) > 0);
    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t tk_auth_manager_set_user_pin(tk_auth_manager_t* auth_mgr,
                                                         const char* pin) {
    if (auth_mgr == NULL || pin == NULL) {
        tk_log_error("tk_auth_manager_set_user_pin: invalid arguments");
        return TK_ERROR_INVALID_ARGUMENT;
    }

    if (!auth_mgr->is_initialized) {
        tk_log_error("tk_auth_manager_set_user_pin: manager not initialized");
        return TK_ERROR_INVALID_STATE;
    }

    size_t pin_len = strlen(pin);
    if (pin_len == 0 || pin_len >= TK_AUTH_PIN_MAX_LENGTH) {
        tk_log_error("tk_auth_manager_set_user_pin: invalid PIN length");
        return TK_ERROR_INVALID_ARGUMENT;
    }

    /* Hash the PIN */
    tk_error_code_t err = hash_pin(pin, auth_mgr->pin_hash);
    if (err != TK_SUCCESS) {
        tk_log_error("tk_auth_manager_set_user_pin: failed to hash PIN: %s",
                     tk_error_to_string(err));
        return err;
    }

    auth_mgr->has_pin = true;

    /* Save the updated state */
    err = tk_auth_manager_save_state(auth_mgr);
    if (err != TK_SUCCESS) {
        tk_log_warning("tk_auth_manager_set_user_pin: failed to save state: %s",
                      tk_error_to_string(err));
        /* Continue anyway - PIN is set in memory */
    }

    tk_log_info("tk_auth_manager_set_user_pin: user PIN updated successfully");
    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t tk_auth_manager_verify_pin(const tk_auth_manager_t* auth_mgr,
                                                       const char* pin,
                                                       bool* is_valid) {
    if (auth_mgr == NULL || pin == NULL || is_valid == NULL) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    *is_valid = false;

    if (!auth_mgr->is_initialized || !auth_mgr->has_pin) {
        tk_log_debug("tk_auth_manager_verify_pin: no PIN set");
        return TK_SUCCESS;
    }

    *is_valid = verify_pin_hash(pin, auth_mgr->pin_hash);
    
    if (*is_valid) {
        tk_log_debug("tk_auth_manager_verify_pin: PIN verification successful");
    } else {
        tk_log_warning("tk_auth_manager_verify_pin: PIN verification failed");
    }

    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t tk_auth_manager_create_session(tk_auth_manager_t* auth_mgr,
                                                           uint32_t permissions,
                                                           uint32_t duration_seconds,
                                                           tk_auth_session_t** session) {
    if (auth_mgr == NULL || session == NULL) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    if (!auth_mgr->is_initialized) {
        return TK_ERROR_INVALID_STATE;
    }

    /* Validate duration */
    if (duration_seconds == 0) {
        duration_seconds = TK_AUTH_DEFAULT_SESSION_DURATION;
    } else if (duration_seconds > TK_AUTH_MAX_SESSION_DURATION) {
        duration_seconds = TK_AUTH_MAX_SESSION_DURATION;
    }

    /* Allocate session structure */
    tk_auth_session_t* sess = (tk_auth_session_t*)calloc(1, sizeof(tk_auth_session_t));
    if (sess == NULL) {
        return TK_ERROR_OUT_OF_MEMORY;
    }

    /* Initialize session */
    time_t now = time(NULL);
    sess->session_id = auth_mgr->next_session_id++;
    sess->created_time = now;
    sess->expires_time = now + duration_seconds;
    sess->permissions = permissions;
    sess->is_valid = true;

    /* Generate session token hash */
    tk_error_code_t err = generate_session_token_hash(sess->session_id, sess->created_time, 
                                                     sess->token_hash);
    if (err != TK_SUCCESS) {
        free(sess);
        return err;
    }

    *session = sess;
    
    tk_log_debug("tk_auth_manager_create_session: created session %u with permissions 0x%08x",
                sess->session_id, permissions);
    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t tk_auth_manager_validate_session(const tk_auth_manager_t* auth_mgr,
                                                             const tk_auth_session_t* session,
                                                             bool* is_valid) {
    if (auth_mgr == NULL || session == NULL || is_valid == NULL) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    *is_valid = false;

    if (!auth_mgr->is_initialized) {
        return TK_ERROR_INVALID_STATE;
    }

    /* Check if session is marked as valid */
    if (!session->is_valid) {
        return TK_SUCCESS;
    }

    /* Check expiration */
    time_t now = time(NULL);
    if (now > session->expires_time) {
        tk_log_debug("tk_auth_manager_validate_session: session %u expired", 
                    session->session_id);
        return TK_SUCCESS;
    }

    *is_valid = true;
    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t tk_auth_manager_invalidate_session(tk_auth_manager_t* auth_mgr,
                                                               tk_auth_session_t* session) {
    if (auth_mgr == NULL || session == NULL) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    session->is_valid = false;
    sodium_memzero(session->token_hash, sizeof(session->token_hash));

    tk_log_debug("tk_auth_manager_invalidate_session: invalidated session %u", 
                session->session_id);
    return TK_SUCCESS;
}

void tk_auth_session_destroy(tk_auth_session_t** session) {
    if (session == NULL || *session == NULL) {
        return;
    }

    tk_auth_session_t* sess = *session;

    /* Securely wipe sensitive data */
    sodium_memzero(sess->token_hash, sizeof(sess->token_hash));
    sodium_memzero(sess, sizeof(*sess));
    
    free(sess);
    *session = NULL;
}

TK_NODISCARD tk_error_code_t tk_auth_session_has_permission(const tk_auth_session_t* session,
                                                           uint32_t required_permissions,
                                                           bool* has_permission) {
    if (session == NULL || has_permission == NULL) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    *has_permission = false;

    if (!session->is_valid) {
        return TK_SUCCESS;
    }

    /* Check expiration */
    time_t now = time(NULL);
    if (now > session->expires_time) {
        return TK_SUCCESS;
    }

    /* Check permissions */
    *has_permission = (session->permissions & required_permissions) == required_permissions;
    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t tk_auth_session_time_remaining(const tk_auth_session_t* session,
                                                           uint32_t* seconds_remaining) {
    if (session == NULL || seconds_remaining == NULL) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    *seconds_remaining = 0;

    if (!session->is_valid) {
        return TK_SUCCESS;
    }

    time_t now = time(NULL);
    if (now >= session->expires_time) {
        return TK_SUCCESS;
    }

    *seconds_remaining = (uint32_t)(session->expires_time - now);
    return TK_SUCCESS;
}

/* ---------------------------------------------------------------------------
 * Secure Storage Operations (Placeholder Implementation)
 * --------------------------------------------------------------------------- */

TK_NODISCARD tk_error_code_t tk_auth_manager_save_state(const tk_auth_manager_t* auth_mgr) {
    if (auth_mgr == NULL) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    if (!auth_mgr->is_initialized) {
        return TK_ERROR_INVALID_STATE;
    }

    /* 
     * TODO: Implement actual file persistence using tk_file_manager when available.
     * This is a placeholder implementation that logs the operation.
     * 
     * The real implementation should:
     * 1. Create a structure containing device_id, pin_hash, has_pin
     * 2. Serialize this data
     * 3. Encrypt it using auth_mgr->crypto_ctx
     * 4. Write to ~/.config/trackiellm/auth_state.enc using tk_file_manager
     */
    
    tk_log_info("tk_auth_manager_save_state: saving authentication state for device %s", 
                auth_mgr->device_id);
    
    /* Simulate successful save for now */
    return TK_SUCCESS;
}

TK_NODISCARD tk_error_code_t tk_auth_manager_load_state(tk_auth_manager_t* auth_mgr) {
    if (auth_mgr == NULL) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    /* 
     * TODO: Implement actual file loading using tk_file_manager when available.
     * This is a placeholder implementation.
     * 
     * The real implementation should:
     * 1. Check if ~/.config/trackiellm/auth_state.enc exists using tk_file_manager
     * 2. If not found, return TK_ERROR_NOT_FOUND
     * 3. Read the encrypted file
     * 4. Decrypt using auth_mgr->crypto_ctx (key derived from hardware/OS info)
     * 5. Deserialize and populate auth_mgr fields
     */
    
    tk_log_debug("tk_auth_manager_load_state: attempting to load authentication state");
    
    /* Simulate file not found for first-time setup */
    return TK_ERROR_NOT_FOUND;
}

/* ---------------------------------------------------------------------------
 * Additional Utility Functions for Integration
 * --------------------------------------------------------------------------- */

/**
 * @brief Convenience function to create a session with standard user permissions.
 *
 * This function creates a session suitable for normal user operations.
 *
 * @param[in] auth_mgr The authentication manager.
 * @param[out] session Pointer to store the newly created session.
 * @return TK_SUCCESS on success, or an appropriate error code on failure.
 */
TK_NODISCARD tk_error_code_t tk_auth_manager_create_user_session(tk_auth_manager_t* auth_mgr,
                                                                tk_auth_session_t** session) {
    uint32_t user_permissions = TK_AUTH_PERM_READ_CONFIG | 
                               TK_AUTH_PERM_USER_DATA_ACCESS |
                               TK_AUTH_PERM_DEVICE_CONTROL;
    
    return tk_auth_manager_create_session(auth_mgr, user_permissions, 0, session);
}

/**
 * @brief Convenience function to create a session with administrative permissions.
 *
 * This function creates a session suitable for system administration tasks.
 * Should only be used after proper authentication.
 *
 * @param[in] auth_mgr The authentication manager.
 * @param[in] duration_seconds Session duration (0 for default, limited to 1 hour max for admin).
 * @param[out] session Pointer to store the newly created session.
 * @return TK_SUCCESS on success, or an appropriate error code on failure.
 */
TK_NODISCARD tk_error_code_t tk_auth_manager_create_admin_session(tk_auth_manager_t* auth_mgr,
                                                                 uint32_t duration_seconds,
                                                                 tk_auth_session_t** session) {
    /* Limit admin sessions to maximum 1 hour for security */
    if (duration_seconds == 0 || duration_seconds > 3600) {
        duration_seconds = 3600;
    }
    
    return tk_auth_manager_create_session(auth_mgr, TK_AUTH_PERM_ADMIN, duration_seconds, session);
}

/**
 * @brief Validates that the authentication manager is ready for use.
 *
 * This function performs comprehensive validation of the auth manager state.
 *
 * @param[in] auth_mgr The authentication manager to validate.
 * @param[out] status_message Buffer to store human-readable status (optional).
 * @param[in] message_size Size of the status message buffer.
 * @return TK_SUCCESS if ready, or an appropriate error code if not ready.
 */
TK_NODISCARD tk_error_code_t tk_auth_manager_validate_ready(const tk_auth_manager_t* auth_mgr,
                                                           char* status_message,
                                                           size_t message_size) {
    if (auth_mgr == NULL) {
        if (status_message && message_size > 0) {
            snprintf(status_message, message_size, "Authentication manager is NULL");
        }
        return TK_ERROR_INVALID_ARGUMENT;
    }

    if (!auth_mgr->is_initialized) {
        if (status_message && message_size > 0) {
            snprintf(status_message, message_size, "Authentication manager not initialized");
        }
        return TK_ERROR_INVALID_STATE;
    }

    if (strlen(auth_mgr->device_id) == 0) {
        if (status_message && message_size > 0) {
            snprintf(status_message, message_size, "Device ID not generated");
        }
        return TK_ERROR_INVALID_STATE;
    }

    if (auth_mgr->crypto_ctx == NULL) {
        if (status_message && message_size > 0) {
            snprintf(status_message, message_size, "Encryption context not available");
        }
        return TK_ERROR_INVALID_STATE;
    }

    if (status_message && message_size > 0) {
        snprintf(status_message, message_size, "Authentication manager ready (Device: %.20s...)", 
                auth_mgr->device_id);
    }

    return TK_SUCCESS;
}