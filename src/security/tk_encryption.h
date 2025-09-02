/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_encryption.h
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TK_ENCRYPTION_H
#define TK_ENCRYPTION_H

#include <stdint.h>
#include <stddef.h>

#include "utils/tk_error_handling.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration of the opaque context structure
typedef struct tk_encryption_ctx_s tk_encryption_ctx_t;

/**
 * @brief Creates a new encryption context.
 *
 * Allocates and initializes a new encryption context. This context must be
 * destroyed using tk_encryption_ctx_destroy() when no longer needed.
 *
 * @param[out] ctx Pointer to store the newly created context.
 * @return TK_SUCCESS on success, or an appropriate error code on failure.
 * @pre ctx != NULL
 * @post *ctx is a valid pointer to a tk_encryption_ctx_t on success.
 */
TK_NODISCARD tk_error_code_t tk_encryption_ctx_create(tk_encryption_ctx_t** ctx);

/**
 * @brief Destroys an encryption context and securely wipes any sensitive data.
 *
 * Frees all resources associated with the context, including any stored keys.
 *
 * @param[in,out] ctx Pointer to the context to destroy. Set to NULL on return.
 * @pre ctx != NULL && *ctx != NULL
 * @post *ctx == NULL
 */
void tk_encryption_ctx_destroy(tk_encryption_ctx_t** ctx);

/**
 * @brief Generates a new random encryption key and stores it in the context.
 *
 * This function generates a cryptographically secure random key suitable for
 * use with the encryption algorithm and stores it internally in the context.
 *
 * @param[in,out] ctx The encryption context.
 * @return TK_SUCCESS on success, or an appropriate error code on failure.
 * @pre ctx != NULL
 */
TK_NODISCARD tk_error_code_t tk_encryption_generate_key(tk_encryption_ctx_t* ctx);

/**
 * @brief Sets a custom encryption key from a buffer.
 *
 * Allows setting a specific key for encryption/decryption. The key must be
 * exactly TK_ENCRYPTION_KEY_SIZE bytes long.
 *
 * @param[in,out] ctx The encryption context.
 * @param[in] key_buffer Buffer containing the key.
 * @param[in] key_size Size of the key buffer. Must be TK_ENCRYPTION_KEY_SIZE.
 * @return TK_SUCCESS on success, TK_ERROR_INVALID_ARGUMENT if key_size is incorrect,
 *         or another error code on failure.
 * @pre ctx != NULL && key_buffer != NULL
 */
TK_NODISCARD tk_error_code_t tk_encryption_set_key(tk_encryption_ctx_t* ctx,
                                                   const uint8_t* key_buffer,
                                                   size_t key_size);

/**
 * @brief Encrypts plaintext data.
 *
 * Encrypts the provided plaintext using the key stored in the context. A random
 * nonce is generated for each encryption operation and prepended to the ciphertext.
 * The output buffer must be large enough to hold the nonce, ciphertext, and tag.
 *
 * @param[in] ctx The encryption context.
 * @param[in] plaintext The data to encrypt.
 * @param[in] plaintext_size Size of the plaintext in bytes.
 * @param[out] ciphertext Buffer to store the encrypted data (nonce + ciphertext + tag).
 * @param[in,out] ciphertext_size Pointer to the size of the ciphertext buffer.
 *                On input, it specifies the buffer size. On output, it contains
 *                the actual size of the encrypted data written.
 * @return TK_SUCCESS on success, TK_ERROR_BUFFER_TOO_SMALL if the buffer is too small,
 *         or another error code on failure.
 * @pre ctx != NULL && plaintext != NULL && ciphertext != NULL && ciphertext_size != NULL
 * @post On success, *ciphertext_size <= plaintext_size + overhead.
 */
TK_NODISCARD tk_error_code_t tk_encryption_encrypt(tk_encryption_ctx_t* ctx,
                                                   const uint8_t* plaintext,
                                                   size_t plaintext_size,
                                                   uint8_t* ciphertext,
                                                   size_t* ciphertext_size);

/**
 * @brief Decrypts ciphertext data.
 *
 * Decrypts the provided ciphertext using the key stored in the context. The nonce
 * is expected to be prepended to the ciphertext.
 *
 * @param[in] ctx The encryption context.
 * @param[in] ciphertext The data to decrypt (nonce + ciphertext + tag).
 * @param[in] ciphertext_size Size of the ciphertext in bytes.
 * @param[out] plaintext Buffer to store the decrypted data.
 * @param[in,out] plaintext_size Pointer to the size of the plaintext buffer.
 *                On input, it specifies the buffer size. On output, it contains
 *                the actual size of the decrypted data written.
 * @return TK_SUCCESS on success, TK_ERROR_DECRYPTION_FAILED if decryption fails (e.g., tampered data),
 *         TK_ERROR_BUFFER_TOO_SMALL if the buffer is too small, or another error code on failure.
 * @pre ctx != NULL && ciphertext != NULL && plaintext != NULL && plaintext_size != NULL
 */
TK_NODISCARD tk_error_code_t tk_encryption_decrypt(tk_encryption_ctx_t* ctx,
                                                   const uint8_t* ciphertext,
                                                   size_t ciphertext_size,
                                                   uint8_t* plaintext,
                                                   size_t* plaintext_size);

// Constants for key and nonce sizes
/** Size of the encryption key in bytes (256 bits). */
#define TK_ENCRYPTION_KEY_SIZE   32

/** Size of the nonce in bytes (96 bits). */
#define TK_ENCRYPTION_NONCE_SIZE 12

/** Size of the authentication tag in bytes (128 bits). */
#define TK_ENCRYPTION_TAG_SIZE   16

/** Total overhead added by encryption (nonce + tag). */
#define TK_ENCRYPTION_OVERHEAD   (TK_ENCRYPTION_NONCE_SIZE + TK_ENCRYPTION_TAG_SIZE)

#ifdef __cplusplus
}
#endif

#endif // TK_ENCRYPTION_H
