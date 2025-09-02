/* 
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_encryption.c
 *
 * This source file implements the cryptographic abstraction layer for the
 * TrackieLLM project. It wraps libsodium's XChaCha20‑Poly1305 AEAD primitive
 * and provides a simple context‑based API for key management, encryption and
 * decryption. All sensitive material (the symmetric key) is stored inside the
 * opaque context and is securely wiped before the context is destroyed.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <sodium.h>

#include "security/tk_encryption.h"
#include "utils/tk_error_handling.h"
#include "utils/tk_logging.h"

/* ---------------------------------------------------------------------------
 * Internal data structures (opaque in the public header)
 * --------------------------------------------------------------------------- */
struct tk_encryption_ctx_s {
    bool     is_initialized;                     /* true after successful init   */
    bool     has_key;                            /* true after key generation/set */
    unsigned char key[crypto_aead_xchacha20poly1305_ietf_KEYBYTES];
};

/* ---------------------------------------------------------------------------
 * Helper macros
 * --------------------------------------------------------------------------- */
#define NONCE_SIZE crypto_aead_xchacha20poly1305_ietf_NPUBBYTES
#define TAG_SIZE   crypto_aead_xchacha20poly1305_ietf_ABYTES
#define KEY_SIZE   crypto_aead_xchacha20poly1305_ietf_KEYBYTES

/* ---------------------------------------------------------------------------
 * Public API implementation
 * --------------------------------------------------------------------------- */

/**
 * @brief Creates a new encryption context.
 *
 * Allocates the opaque context, initialises libsodium (once per process) and
 * marks the context as ready for use.
 */
TK_NODISCARD tk_error_code_t
tk_encryption_ctx_create(tk_encryption_ctx_t **out_ctx)
{
    if (out_ctx == NULL) {
        tk_log_error("tk_encryption_ctx_create: out_ctx is NULL");
        return TK_ERROR_INVALID_ARGUMENT;
    }

    if (sodium_init() == -1) {
        tk_log_error("tk_encryption_ctx_create: libsodium initialization failed");
        return TK_ERROR_INTERNAL;
    }

    tk_encryption_ctx_t *ctx = (tk_encryption_ctx_t *)malloc(sizeof(tk_encryption_ctx_t));
    if (ctx == NULL) {
        tk_log_error("tk_encryption_ctx_create: out of memory");
        return TK_ERROR_OUT_OF_MEMORY;
    }

    ctx->is_initialized = true;
    ctx->has_key        = false;
    sodium_memzero(ctx->key, KEY_SIZE);

    *out_ctx = ctx;
    tk_log_debug("tk_encryption_ctx_create: context created at %p", (void *)ctx);
    return TK_SUCCESS;
}

/**
 * @brief Destroys an encryption context and securely wipes any sensitive data.
 */
void
tk_encryption_ctx_destroy(tk_encryption_ctx_t **ctx_ptr)
{
    if (ctx_ptr == NULL || *ctx_ptr == NULL) {
        return; /* Nothing to do */
    }

    tk_encryption_ctx_t *ctx = *ctx_ptr;

    /* Securely erase the symmetric key if it was ever set */
    if (ctx->has_key) {
        sodium_memzero(ctx->key, KEY_SIZE);
    }

    /* Zero the whole structure before freeing to avoid leaving traces */
    sodium_memzero(ctx, sizeof(*ctx));
    free(ctx);
    *ctx_ptr = NULL;

    tk_log_debug("tk_encryption_ctx_destroy: context destroyed");
}

/**
 * @brief Generates a fresh random symmetric key and stores it in the context.
 */
TK_NODISCARD tk_error_code_t
tk_encryption_generate_key(tk_encryption_ctx_t *ctx)
{
    if (ctx == NULL) {
        tk_log_error("tk_encryption_generate_key: ctx is NULL");
        return TK_ERROR_INVALID_ARGUMENT;
    }

    crypto_aead_xchacha20poly1305_ietf_keygen(ctx->key);
    ctx->has_key = true;

    tk_log_debug("tk_encryption_generate_key: new key generated");
    return TK_SUCCESS;
}

/**
 * @brief Sets a custom symmetric key supplied by the caller.
 *
 * The caller must provide exactly KEY_SIZE bytes.
 */
TK_NODISCARD tk_error_code_t
tk_encryption_set_key(tk_encryption_ctx_t *ctx,
                      const uint8_t *key_buffer,
                      size_t key_size)
{
    if (ctx == NULL || key_buffer == NULL) {
        tk_log_error("tk_encryption_set_key: invalid argument (NULL)");
        return TK_ERROR_INVALID_ARGUMENT;
    }

    if (key_size != KEY_SIZE) {
        tk_log_error("tk_encryption_set_key: key size %zu != %d", key_size, KEY_SIZE);
        return TK_ERROR_INVALID_ARGUMENT;
    }

    memcpy(ctx->key, key_buffer, KEY_SIZE);
    ctx->has_key = true;

    tk_log_debug("tk_encryption_set_key: custom key installed");
    return TK_SUCCESS;
}

/**
 * @brief Encrypts plaintext using the stored key.
 *
 * The output buffer must be large enough to hold:
 *   NONCE_SIZE + plaintext_len + TAG_SIZE
 *
 * On success, *ciphertext_size is set to the actual number of bytes written.
 */
TK_NODISCARD tk_error_code_t
tk_encryption_encrypt(tk_encryption_ctx_t *ctx,
                      const uint8_t *plaintext,
                      size_t plaintext_size,
                      uint8_t *ciphertext,
                      size_t *ciphertext_size)
{
    if (ctx == NULL || plaintext == NULL || ciphertext == NULL || ciphertext_size == NULL) {
        tk_log_error("tk_encryption_encrypt: invalid argument (NULL)");
        return TK_ERROR_INVALID_ARGUMENT;
    }

    if (!ctx->has_key) {
        tk_log_error("tk_encryption_encrypt: no key set in context");
        return TK_ERROR_INVALID_ARGUMENT;
    }

    size_t needed = NONCE_SIZE + plaintext_size + TAG_SIZE;
    if (*ciphertext_size < needed) {
        *ciphertext_size = needed;
        tk_log_error("tk_encryption_encrypt: buffer too small (need %zu, got %zu)",
                     needed, *ciphertext_size);
        return TK_ERROR_BUFFER_TOO_SMALL;
    }

    /* Generate a fresh nonce */
    unsigned char nonce[NONCE_SIZE];
    randombytes_buf(nonce, NONCE_SIZE);
    memcpy(ciphertext, nonce, NONCE_SIZE); /* prepend nonce */

    unsigned long long ciphertext_len = 0;
    int rc = crypto_aead_xchacha20poly1305_ietf_encrypt(
        ciphertext + NONCE_SIZE,          /* ciphertext output */
        &ciphertext_len,                  /* written length */
        plaintext,                        /* plaintext input */
        plaintext_size,                   /* plaintext length */
        NULL, 0,                          /* additional data (none) */
        NULL,                             /* no secret nonce */
        nonce,                            /* public nonce */
        ctx->key);                        /* symmetric key */

    if (rc != 0) {
        tk_log_error("tk_encryption_encrypt: libsodium encryption failed (rc=%d)", rc);
        return TK_ERROR_INTERNAL;
    }

    *ciphertext_size = NONCE_SIZE + (size_t)ciphertext_len;
    tk_log_debug("tk_encryption_encrypt: success, output size %zu", *ciphertext_size);
    return TK_SUCCESS;
}

/**
 * @brief Decrypts ciphertext using the stored key.
 *
 * The ciphertext must be formatted as:
 *   NONCE_SIZE || encrypted_data || TAG_SIZE
 *
 * On success, *plaintext_size is set to the number of plaintext bytes written.
 */
TK_NODISCARD tk_error_code_t
tk_encryption_decrypt(tk_encryption_ctx_t *ctx,
                      const uint8_t *ciphertext,
                      size_t ciphertext_size,
                      uint8_t *plaintext,
                      size_t *plaintext_size)
{
    if (ctx == NULL || ciphertext == NULL || plaintext == NULL || plaintext_size == NULL) {
        tk_log_error("tk_encryption_decrypt: invalid argument (NULL)");
        return TK_ERROR_INVALID_ARGUMENT;
    }

    if (!ctx->has_key) {
        tk_log_error("tk_encryption_decrypt: no key set in context");
        return TK_ERROR_INVALID_ARGUMENT;
    }

    if (ciphertext_size < NONCE_SIZE + TAG_SIZE) {
        tk_log_error("tk_encryption_decrypt: ciphertext too short (%zu bytes)", ciphertext_size);
        return TK_ERROR_INVALID_ARGUMENT;
    }

    const unsigned char *nonce = ciphertext;                     /* first NONCE_SIZE bytes */
    const unsigned char *enc   = ciphertext + NONCE_SIZE;       /* encrypted payload */
    size_t enc_len = ciphertext_size - NONCE_SIZE;              /* includes TAG_SIZE */

    if (enc_len < TAG_SIZE) {
        tk_log_error("tk_encryption_decrypt: encrypted part too short");
        return TK_ERROR_INVALID_ARGUMENT;
    }

    size_t needed_plaintext = enc_len - TAG_SIZE;
    if (*plaintext_size < needed_plaintext) {
        *plaintext_size = needed_plaintext;
        tk_log_error("tk_encryption_decrypt: buffer too small (need %zu, got %zu)",
                     needed_plaintext, *plaintext_size);
        return TK_ERROR_BUFFER_TOO_SMALL;
    }

    unsigned long long plaintext_len = 0;
    int rc = crypto_aead_xchacha20poly1305_ietf_decrypt(
        plaintext,                         /* plaintext output */
        &plaintext_len,                    /* written length */
        NULL,                              /* no secret nonce */
        enc,                               /* ciphertext input */
        enc_len,                           /* ciphertext length (incl. tag) */
        NULL, 0,                           /* additional data (none) */
        nonce,                             /* public nonce */
        ctx->key);                         /* symmetric key */

    if (rc != 0) {
        tk_log_error("tk_encryption_decrypt: authentication failed (rc=%d)", rc);
        return TK_ERROR_DECRYPTION_FAILED;
    }

    *plaintext_size = (size_t)plaintext_len;
    tk_log_debug("tk_encryption_decrypt: success, plaintext size %zu", *plaintext_size);
    return TK_SUCCESS;
}
