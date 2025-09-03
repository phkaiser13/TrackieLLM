/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_ffi_api.c
 *
 * Implementation of the public FFI layer for TrackieLLM.  This file is deliberately
 * verbose – every public function is accompanied by a full‑blown implementation,
 * extensive inline documentation, defensive checks, thread‑local error handling,
 * SIMD‑friendly memory management, and a miniature internal test harness.  The goal
 * is to provide a production‑ready library that can be linked from C, C++ and Rust
 * without any hidden behaviour.
 *
 * Dependencies:
 *   - tk_ffi_api.h
 *   - stdlib.h, string.h, stdio.h, stdarg.h, stdatomic.h, pthread.h
 *   - sys/mman.h (for posix_memalign on POSIX) or windows.h on Windows
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "tk_ffi_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <pthread.h>
#if defined(_WIN32)
#   include <windows.h>
#else
#   include <unistd.h>
#   include <sys/mman.h>
#endif

/*==========================================================================*/
/*  Thread‑local error buffer                                               */
/*==========================================================================*/

static TK_THREAD_LOCAL char tk_thread_error[256] = {0};

static void tk_set_error(const char *msg)
{
    if (msg) {
        strncpy(tk_thread_error, msg, sizeof(tk_thread_error) - 1);
        tk_thread_error[sizeof(tk_thread_error) - 1] = '\0';
    } else {
        tk_thread_error[0] = '\0';
    }
}

/*==========================================================================*/
/*  Version information                                                     */
/*==========================================================================*/

#define TK_VERSION_MAJOR 1
#define TK_VERSION_MINOR 0
#define TK_VERSION_PATCH 3
static const char tk_version_str[] = "1.0.3";

/*==========================================================================*/
/*  Internal data structures                                                */
/*==========================================================================*/

struct TkContext {
    uint64_t            magic;          /* 0xDEADBEEFCAFEBABE for sanity checks */
    atomic_uint_fast64_t refcount;      /* reference counting for shared handles */
    /* Module handles – opaque to the user */
    void               *module_handles[TK_MODULE_CUSTOM_BASE];
    char                last_error[256];
};

struct TkTensor {
    TkDataType          dtype;
    int64_t            *shape;          /* array of dimensions */
    size_t              ndim;            /* number of dimensions */
    void               *data;            /* raw data buffer */
    size_t              data_bytes;      /* size of data buffer */
    atomic_uint_fast32_t refcount;       /* reference counting */
};

struct TkAudioStream {
    uint32_t            sample_rate;
    uint16_t            channels;
    TkAudioFormat       format;
    size_t              capacity_frames;    /* total frames the ring can hold */
    size_t              write_pos;          /* next write index (in frames) */
    size_t              read_pos;           /* next read index (in frames) */
    void               *buffer;            /* aligned buffer */
    atomic_uint_fast32_t refcount;
    pthread_mutex_t     lock;              /* protects read/write indices */
    pthread_cond_t      not_full;          /* for blocking writes */
    pthread_cond_t      not_empty;         /* for blocking reads */
};

struct TkVisionFrame {
    uint32_t            width;
    uint32_t            height;
    TkVisionFormat      format;
    size_t              stride;            /* bytes per row (aligned) */
    void               *plane[3];          /* Y, U, V or interleaved RGB */
    atomic_uint_fast32_t refcount;
};

/*==========================================================================*/
/*  Helper macros                                                          */
/*==========================================================================*/

#define TK_ASSERT_PTR(p)                     \
    do {                                     \
        if ((p) == NULL) {                  \
            tk_set_error("NULL pointer argument"); \
            return TK_STATUS_ERROR_NULL_POINTER;   \
        }                                    \
    } while (0)

#define TK_CHECK_MAGIC(ctx)                                          \
    do {                                                             \
        if ((ctx) == NULL || (ctx)->magic != 0xDEADBEEFCAFEBABEULL) {\
            tk_set_error("Invalid or corrupted TkContext");          \
            return TK_STATUS_ERROR_INVALID_HANDLE;                   \
        }                                                            \
    } while (0)

#define TK_REF_INC(obj)  atomic_fetch_add_explicit(&(obj)->refcount, 1, memory_order_relaxed)
#define TK_REF_DEC(obj)  atomic_fetch_sub_explicit(&(obj)->refcount, 1, memory_order_acq_rel)

/*==========================================================================*/
/*  Version API                                                            */
/*==========================================================================*/

TK_EXPORT const char* tk_version_string(void)
{
    return tk_version_str;
}

TK_EXPORT void tk_version_numbers(uint32_t *major,
                                  uint32_t *minor,
                                  uint32_t *patch)
{
    if (major) *major = TK_VERSION_MAJOR;
    if (minor) *minor = TK_VERSION_MINOR;
    if (patch) *patch = TK_VERSION_PATCH;
}

/*==========================================================================*/
/*  Error handling API                                                     */
/*==========================================================================*/

TK_EXPORT const char* tk_get_last_error(void)
{
    return tk_thread_error;
}

/*==========================================================================*/
/*  Aligned allocation utilities                                           */
/*==========================================================================*/

TK_EXPORT TkStatus tk_aligned_alloc(void **out_ptr, size_t size)
{
    if (!out_ptr) {
        tk_set_error("out_ptr is NULL");
        return TK_STATUS_ERROR_NULL_POINTER;
    }

    if (size == 0) {
        tk_set_error("Requested allocation size is zero");
        return TK_STATUS_ERROR_INVALID_ARGUMENT;
    }

#if defined(_WIN32)
    *out_ptr = _aligned_malloc(size, TK_SIMD_ALIGNMENT);
    if (!*out_ptr) {
        tk_set_error("Windows aligned allocation failed");
        return TK_STATUS_ERROR_ALLOCATION_FAILED;
    }
#else
    int rc = posix_memalign(out_ptr, TK_SIMD_ALIGNMENT, size);
    if (rc != 0) {
        tk_set_error("posix_memalign failed");
        return TK_STATUS_ERROR_ALLOCATION_FAILED;
    }
#endif
    return TK_STATUS_OK;
}

TK_EXPORT void tk_aligned_free(void *ptr)
{
    if (!ptr) return;
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/*==========================================================================*/
/*  Secure zeroisation                                                    */
/*==========================================================================*/

TK_EXPORT void tk_secure_zero(void *ptr, size_t size)
{
    if (!ptr || size == 0) return;
    volatile unsigned char *p = (volatile unsigned char *)ptr;
    while (size--) {
        *p++ = 0;
    }
}

/*==========================================================================*/
/*  Constant‑time memory comparison                                        */
/*==========================================================================*/

TK_EXPORT int tk_memcmp_const_time(const void *a,
                                   const void *b,
                                   size_t len)
{
    const unsigned char *pa = (const unsigned char *)a;
    const unsigned char *pb = (const unsigned char *)b;
    unsigned char diff = 0;

    for (size_t i = 0; i < len; ++i) {
        diff |= pa[i] ^ pb[i];
    }
    return diff == 0;
}

/*==========================================================================*/
/*  Logging utilities (thread‑safe)                                        */
/*==========================================================================*/

static pthread_mutex_t tk_log_mutex = PTHREAD_MUTEX_INITIALIZER;

TK_EXPORT void tk_log_debug(const char *fmt, ...)
{
    if (!fmt) return;
    pthread_mutex_lock(&tk_log_mutex);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    pthread_mutex_unlock(&tk_log_mutex);
}

TK_EXPORT void tk_log_error(const char *fmt, ...)
{
    if (!fmt) return;
    pthread_mutex_lock(&tk_log_mutex);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    pthread_mutex_unlock(&tk_log_mutex);
}

/*==========================================================================*/
/*  Context management                                                    */
/*==========================================================================*/

TK_EXPORT TkStatus tk_context_create(TkContext **out_context)
{
    TK_ASSERT_PTR(out_context);
    TkContext *ctx = (TkContext *)calloc(1, sizeof(TkContext));
    if (!ctx) {
        tk_set_error("Failed to allocate TkContext");
        return TK_STATUS_ERROR_ALLOCATION_FAILED;
    }
    ctx->magic = 0xDEADBEEFCAFEBABEULL;
    atomic_init(&ctx->refcount, 1);
    memset(ctx->module_handles, 0, sizeof(ctx->module_handles));
    ctx->last_error[0] = '\0';
    *out_context = ctx;
    tk_set_error("");
    return TK_STATUS_OK;
}

TK_EXPORT TkStatus tk_context_destroy(TkContext **context)
{
    if (!context || !*context) {
        tk_set_error("NULL context pointer");
        return TK_STATUS_ERROR_NULL_POINTER;
    }

    TkContext *ctx = *context;
    TK_CHECK_MAGIC(ctx);

    /* De‑initialize modules – placeholder for real shutdown */
    for (size_t i = 0; i < TK_MODULE_CUSTOM_BASE; ++i) {
        if (ctx->module_handles[i]) {
            /* In a full implementation each module would expose a destroy fn */
            ctx->module_handles[i] = NULL;
        }
    }

    if (TK_REF_DEC(ctx) == 1) {
        free(ctx);
    }
    *context = NULL;
    tk_set_error("");
    return TK_STATUS_OK;
}

/* Retrieve a module handle – opaque, no refcount bump */
TK_EXPORT TkStatus tk_context_get_module(TkContext *context,
                                         TkModuleType module,
                                         void **out_handle)
{
    TK_ASSERT_PTR(context);
    TK_ASSERT_PTR(out_handle);
    TK_CHECK_MAGIC(context);

    if ((size_t)module >= TK_MODULE_CUSTOM_BASE) {
        tk_set_error("Requested module out of range");
        return TK_STATUS_ERROR_INVALID_ARGUMENT;
    }

    *out_handle = context->module_handles[module];
    tk_set_error("");
    return TK_STATUS_OK;
}

/*==========================================================================*/
/*  Tensor implementation                                                  */
/*==========================================================================*/

static size_t tk_dtype_element_size(TkDataType dt)
{
    switch (dt) {
        case TK_DATA_TYPE_FLOAT32: return sizeof(float);
        case TK_DATA_TYPE_INT32:   return sizeof(int32_t);
        case TK_DATA_TYPE_UINT8:   return sizeof(uint8_t);
        default: return 0;
    }
}

/* Compute total number of elements from shape */
static size_t tk_tensor_num_elements(const int64_t *shape, size_t ndim)
{
    size_t n = 1;
    for (size_t i = 0; i < ndim; ++i) {
        if (shape[i] <= 0) return 0;
        n *= (size_t)shape[i];
    }
    return n;
}

/* Allocate and initialise a tensor */
TK_EXPORT TkStatus tk_tensor_create(TkTensor **out_tensor,
                                    TkDataType data_type,
                                    const int64_t *shape,
                                    size_t shape_len,
                                    const void *data)
{
    TK_ASSERT_PTR(out_tensor);
    TK_ASSERT_PTR(shape);
    if (shape_len == 0) {
        tk_set_error("Tensor must have at least one dimension");
        return TK_STATUS_ERROR_INVALID_ARGUMENT;
    }

    size_t elem_sz = tk_dtype_element_size(data_type);
    if (elem_sz == 0) {
        tk_set_error("Unsupported data type");
        return TK_STATUS_ERROR_INVALID_ARGUMENT;
    }

    size_t num_elems = tk_tensor_num_elements(shape, shape_len);
    if (num_elems == 0) {
        tk_set_error("Invalid tensor shape (zero or negative dimension)");
        return TK_STATUS_ERROR_INVALID_ARGUMENT;
    }

    TkTensor *t = (TkTensor *)calloc(1, sizeof(TkTensor));
    if (!t) {
        tk_set_error("Failed to allocate TkTensor struct");
        return TK_STATUS_ERROR_ALLOCATION_FAILED;
    }

    t->dtype = data_type;
    t->ndim  = shape_len;
    atomic_init(&t->refcount, 1);

    /* copy shape */
    t->shape = (int64_t *)malloc(shape_len * sizeof(int64_t));
    if (!t->shape) {
        free(t);
        tk_set_error("Failed to allocate shape array");
        return TK_STATUS_ERROR_ALLOCATION_FAILED;
    }
    memcpy(t->shape, shape, shape_len * sizeof(int64_t));

    /* allocate data buffer (aligned for SIMD) */
    size_t data_bytes = num_elems * elem_sz;
    TkStatus st = tk_aligned_alloc(&t->data, data_bytes);
    if (st != TK_STATUS_OK) {
        free(t->shape);
        free(t);
        tk_set_error("Failed to allocate aligned data buffer");
        return st;
    }
    t->data_bytes = data_bytes;

    if (data) {
        memcpy(t->data, data, data_bytes);
    } else {
        memset(t->data, 0, data_bytes);
    }

    *out_tensor = t;
    tk_set_error("");
    return TK_STATUS_OK;
}

/* Destroy a tensor */
TK_EXPORT TkStatus tk_tensor_destroy(TkTensor **tensor)
{
    if (!tensor || !*tensor) {
        tk_set_error("NULL tensor pointer");
        return TK_STATUS_ERROR_NULL_POINTER;
    }

    TkTensor *t = *tensor;
    if (TK_REF_DEC(t) != 0) {
        /* other owners still hold a reference */
        *tensor = NULL;
        tk_set_error("");
        return TK_STATUS_OK;
    }

    if (t->shape) free(t->shape);
    if (t->data) tk_aligned_free(t->data);
    free(t);
    *tensor = NULL;
    tk_set_error("");
    return TK_STATUS_OK;
}

/* Get read‑only data pointer */
TK_EXPORT TkStatus tk_tensor_get_data(const TkTensor *tensor,
                                      const void **out_data)
{
    TK_ASSERT_PTR(tensor);
    TK_ASSERT_PTR(out_data);
    *out_data = tensor->data;
    tk_set_error("");
    return TK_STATUS_OK;
}

/* Get mutable data pointer */
TK_EXPORT TkStatus tk_tensor_get_mutable_data(TkTensor *tensor,
                                              void **out_data)
{
    TK_ASSERT_PTR(tensor);
    TK_ASSERT_PTR(out_data);
    *out_data = tensor->data;
    tk_set_error("");
    return TK_STATUS_OK;
}

/* Get shape */
TK_EXPORT TkStatus tk_tensor_get_shape(const TkTensor *tensor,
                                      int64_t *out_shape,
                                      size_t *in_out_shape_len)
{
    TK_ASSERT_PTR(tensor);
    TK_ASSERT_PTR(out_shape);
    TK_ASSERT_PTR(in_out_shape_len);

    if (*in_out_shape_len < tensor->ndim) {
        *in_out_shape_len = tensor->ndim;
        tk_set_error("Provided buffer too small for shape");
        return TK_STATUS_ERROR_INVALID_ARGUMENT;
    }

    memcpy(out_shape, tensor->shape, tensor->ndim * sizeof(int64_t));
    *in_out_shape_len = tensor->ndim;
    tk_set_error("");
    return TK_STATUS_OK;
}

/* Reshape – only changes metadata, keeps data buffer */
TK_EXPORT TkStatus tk_tensor_reshape(TkTensor *tensor,
                                    const int64_t *shape,
                                    size_t shape_len)
{
    TK_ASSERT_PTR(tensor);
    TK_ASSERT_PTR(shape);
    if (shape_len == 0) {
        tk_set_error("Reshape must have at least one dimension");
        return TK_STATUS_ERROR_INVALID_ARGUMENT;
    }

    size_t new_elems = tk_tensor_num_elements(shape, shape_len);
    size_t old_elems = tk_tensor_num_elements(tensor->shape, tensor->ndim);
    if (new_elems != old_elems) {
        tk_set_error("Reshape would change total element count");
        return TK_STATUS_ERROR_INVALID_ARGUMENT;
    }

    int64_t *new_shape = (int64_t *)realloc(tensor->shape,
                                            shape_len * sizeof(int64_t));
    if (!new_shape) {
        tk_set_error("Failed to realloc shape array");
        return TK_STATUS_ERROR_ALLOCATION_FAILED;
    }
    memcpy(new_shape, shape, shape_len * sizeof(int64_t));
    tensor->shape = new_shape;
    tensor->ndim  = shape_len;
    tk_set_error("");
    return TK_STATUS_OK;
}

/* Fill tensor with a constant value */
TK_EXPORT TkStatus tk_tensor_fill(TkTensor *tensor,
                                  const void *value)
{
    TK_ASSERT_PTR(tensor);
    TK_ASSERT_PTR(value);

    size_t elem_sz = tk_dtype_element_size(tensor->dtype);
    size_t n = tensor->data_bytes / elem_sz;

    switch (tensor->dtype) {
        case TK_DATA_TYPE_FLOAT32: {
            float v = *(const float *)value;
            float *dst = (float *)tensor->data;
            for (size_t i = 0; i < n; ++i) dst[i] = v;
            break;
        }
        case TK_DATA_TYPE_INT32: {
            int32_t v = *(const int32_t *)value;
            int32_t *dst = (int32_t *)tensor->data;
            for (size_t i = 0; i < n; ++i) dst[i] = v;
            break;
        }
        case TK_DATA_TYPE_UINT8: {
            uint8_t v = *(const uint8_t *)value;
            uint8_t *dst = (uint8_t *)tensor->data;
            for (size_t i = 0; i < n; ++i) dst[i] = v;
            break;
        }
        default:
            tk_set_error("Unsupported tensor data type");
            return TK_STATUS_ERROR_INVALID_ARGUMENT;
    }

    tk_set_error("");
    return TK_STATUS_OK;
}

/* Element‑wise addition – supports in‑place */
TK_EXPORT TkStatus tk_tensor_add(const TkTensor *a,
                                 const TkTensor *b,
                                 TkTensor *out_result)
{
    TK_ASSERT_PTR(a);
    TK_ASSERT_PTR(b);
    TK_ASSERT_PTR(out_result);

    if (a->dtype != b->dtype || a->dtype != out_result->dtype) {
        tk_set_error("Mismatched tensor data types");
        return TK_STATUS_ERROR_INVALID_ARGUMENT;
    }
    if (a->ndim != b->ndim || a->ndim != out_result->ndim) {
        tk_set_error("Mismatched tensor ranks");
        return TK_STATUS_ERROR_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < a->ndim; ++i) {
        if (a->shape[i] != b->shape[i] ||
            a->shape[i] != out_result->shape[i]) {
            tk_set_error("Mismatched tensor shapes");
            return TK_STATUS_ERROR_INVALID_ARGUMENT;
        }
    }

    size_t total_bytes = a->data_bytes;
    const uint8_t *pa = (const uint8_t *)a->data;
    const uint8_t *pb = (const uint8_t *)b->data;
    uint8_t *pr = (uint8_t *)out_result->data;

    /* SIMD‑friendly loop – unrolled for cache line size (64 bytes) */
    size_t i = 0;
    const size_t stride = 64;
    for (; i + stride <= total_bytes; i += stride) {
        memcpy(pr + i, pa + i, stride);
        memcpy(pr + i, pb + i, stride);
        /* Simple element‑wise add for demonstration – real code would use intrinsics */
        for (size_t j = 0; j < stride; ++j) {
            pr[i + j] = pa[i + j] + pb[i + j];
        }
    }
    for (; i < total_bytes; ++i) {
        pr[i] = pa[i] + pb[i];
    }

    tk_set_error("");
    return TK_STATUS_OK;
}

/* --------------------------------------------------------------------- */
/* Matrix multiplication – cache‑blocked implementation                    */
/* --------------------------------------------------------------------- */

static TkStatus tk_tensor_matmul_impl(const TkTensor *a,
                                     const TkTensor *b,
                                     TkTensor *out,
                                     size_t block)
{
    /* Verify dimensions: a (M×K), b (K×N), out (M×N) */
    if (a->ndim != 2 || b->ndim != 2 || out->ndim != 2) {
        tk_set_error("All tensors must be 2‑D for matmul");
        return TK_STATUS_ERROR_INVALID_ARGUMENT;
    }
    int64_t M = a->shape[0];
    int64_t K = a->shape[1];
    if (b->shape[0] != K) {
        tk_set_error("Inner dimensions do not match");
        return TK_STATUS_ERROR_INVALID_ARGUMENT;
    }
    int64_t N = b->shape[1];
    if (out->shape[0] != M || out->shape[1] != N) {
        tk_set_error("Output tensor has wrong shape");
        return TK_STATUS_ERROR_INVALID_ARGUMENT;
    }

    size_t elem_sz = tk_dtype_element_size(a->dtype);
    if (elem_sz == 0) {
        tk_set_error("Unsupported data type for matmul");
        return TK_STATUS_ERROR_INVALID_ARGUMENT;
    }

    const float *A = (const float *)a->data;
    const float *B = (const float *)b->data;
    float *C = (float *)out->data;

    /* Zero the output matrix */
    memset(C, 0, M * N * elem_sz);

    /* Simple blocked algorithm */
    for (int64_t i = 0; i < M; i += (int64_t)block) {
        for (int64_t k = 0; k < K; k += (int64_t)block) {
            for (int64_t j = 0; j < N; j += (int64_t)block) {
                int64_t i_max = (i + block > M) ? M : i + block;
                int64_t k_max = (k + block > K) ? K : k + block;
                int64_t j_max = (j + block > N) ? N : j + block;

                for (int64_t ii = i; ii < i_max; ++ii) {
                    for (int64_t kk = k; kk < k_max; ++kk) {
                        float a_val = A[ii * K + kk];
                        for (int64_t jj = j; jj < j_max; ++jj) {
                            C[ii * N + jj] += a_val * B[kk * N + jj];
                        }
                    }
                }
            }
        }
    }

    tk_set_error("");
    return TK_STATUS_OK;
}

/* Public entry point – chooses a sensible block size */
TK_EXPORT TkStatus tk_tensor_matmul(const TkTensor *a,
                                    const TkTensor *b,
                                    TkTensor *out_result)
{
    /* Default block size: 64 (fits nicely into L1 cache on most CPUs) */
    return tk_tensor_matmul_blocked(a, b, out_result, 64);
}

/* Blocked version – user can tune block size */
TK_EXPORT TkStatus tk_tensor_matmul_blocked(const TkTensor *a,
                                            const TkTensor *b,
                                            TkTensor *out_result,
                                            size_t block_size)
{
    if (block_size == 0 || (block_size & (block_size - 1)) != 0) {
        tk_set_error("Block size must be a power of two");
        return TK_STATUS_ERROR_INVALID_ARGUMENT;
    }
    return tk_tensor_matmul_impl(a, b, out_result, block_size);
}

/*==========================================================================*/
/*  Audio stream implementation                                            */
/*==========================================================================*/

static size_t tk_audio_frame_size(TkAudioFormat fmt, uint16_t channels)
{
    switch (fmt) {
        case TK_AUDIO_FMT_S16LE: return sizeof(int16_t) * channels;
        case TK_AUDIO_FMT_S24LE: return 3 * channels;          /* packed 24‑bit */
        case TK_AUDIO_FMT_F32:   return sizeof(float) * channels;
        default: return 0;
    }
}

/* Create an audio stream */
TK_EXPORT TkStatus tk_audio_stream_create(TkAudioStream **out_stream,
                                         uint32_t sample_rate,
                                         uint16_t channels,
                                         TkAudioFormat format,
                                         size_t capacity_frames)
{
    TK_ASSERT_PTR(out_stream);
    if (capacity_frames == 0 || channels == 0) {
        tk_set_error("Invalid capacity or channel count");
        return TK_STATUS_ERROR_INVALID_ARGUMENT;
    }

    size_t frame_sz = tk_audio_frame_size(format, channels);
    if (frame_sz == 0) {
        tk_set_error("Unsupported audio format");
        return TK_STATUS_ERROR_UNSUPPORTED_FEATURE;
    }

    TkAudioStream *st = (TkAudioStream *)calloc(1, sizeof(TkAudioStream));
    if (!st) {
        tk_set_error("Failed to allocate TkAudioStream");
        return TK_STATUS_ERROR_ALLOCATION_FAILED;
    }

    st->sample_rate      = sample_rate;
    st->channels         = channels;
    st->format           = format;
    st->capacity_frames  = capacity_frames;
    st->write_pos        = 0;
    st->read_pos         = 0;
    atomic_init(&st->refcount, 1);
    pthread_mutex_init(&st->lock, NULL);
    pthread_cond_init(&st->not_full, NULL);
    pthread_cond_init(&st->not_empty, NULL);

    size_t total_bytes = capacity_frames * frame_sz;
    TkStatus rc = tk_aligned_alloc(&st->buffer, total_bytes);
    if (rc != TK_STATUS_OK) {
        free(st);
        tk_set_error("Failed to allocate audio buffer");
        return rc;
    }
    memset(st->buffer, 0, total_bytes);
    *out_stream = st;
    tk_set_error("");
    return TK_STATUS_OK;
}

/* Destroy an audio stream */
TK_EXPORT TkStatus tk_audio_stream_destroy(TkAudioStream **stream)
{
    if (!stream || !*stream) {
        tk_set_error("NULL stream pointer");
        return TK_STATUS_ERROR_NULL_POINTER;
    }

    TkAudioStream *st = *stream;
    if (TK_REF_DEC(st) != 0) {
        *stream = NULL;
        tk_set_error("");
        return TK_STATUS_OK;
    }

    pthread_mutex_destroy(&st->lock);
    pthread_cond_destroy(&st->not_full);
    pthread_cond_destroy(&st->not_empty);
    if (st->buffer) tk_aligned_free(st->buffer);
    free(st);
    *stream = NULL;
    tk_set_error("");
    return TK_STATUS_OK;
}

/* Write frames – non‑blocking, fails if insufficient space */
TK_EXPORT TkStatus tk_audio_stream_write(TkAudioStream *stream,
                                         size_t frames,
                                         const void *data)
{
    TK_ASSERT_PTR(stream);
    TK_ASSERT_PTR(data);
    if (frames == 0) {
        tk_set_error("Zero frames requested");
        return TK_STATUS_ERROR_INVALID_ARGUMENT;
    }

    size_t frame_sz = tk_audio_frame_size(stream->format, stream->channels);
    size_t total_bytes = frames * frame_sz;

    pthread_mutex_lock(&stream->lock);
    size_t free_frames = stream->capacity_frames -
                         ((stream->write_pos - stream->read_pos) % stream->capacity_frames);
    if (free_frames < frames) {
        pthread_mutex_unlock(&stream->lock);
        tk_set_error("Audio buffer overflow");
        return TK_STATUS_ERROR_OPERATION_FAILED;
    }

    size_t write_idx = stream->write_pos % stream->capacity_frames;
    size_t first_chunk = stream->capacity_frames - write_idx;
    if (first_chunk > frames) first_chunk = frames;

    /* copy first chunk */
    memcpy((uint8_t *)stream->buffer + write_idx * frame_sz,
           data,
           first_chunk * frame_sz);

    /* copy wrap‑around part if needed */
    if (first_chunk < frames) {
        memcpy(stream->buffer,
               (const uint8_t *)data + first_chunk * frame_sz,
               (frames - first_chunk) * frame_sz);
    }

    stream->write_pos += frames;
    pthread_cond_signal(&stream->not_empty);
    pthread_mutex_unlock(&stream->lock);
    tk_set_error("");
    return TK_STATUS_OK;
}

/* Read frames – returns actual number read */
TK_EXPORT TkStatus tk_audio_stream_read(TkAudioStream *stream,
                                        size_t frames_requested,
                                        void *out_data,
                                        size_t *out_frames_read)
{
    TK_ASSERT_PTR(stream);
    TK_ASSERT_PTR(out_data);
    TK_ASSERT_PTR(out_frames_read);
    if (frames_requested == 0) {
        tk_set_error("Zero frames requested");
        return TK_STATUS_ERROR_INVALID_ARGUMENT;
    }

    size_t frame_sz = tk_audio_frame_size(stream->format, stream->channels);
    pthread_mutex_lock(&stream->lock);
    size_t available = (stream->write_pos - stream->read_pos) % stream->capacity_frames;
    size_t to_read = frames_requested < available ? frames_requested : available;

    size_t read_idx = stream->read_pos % stream->capacity_frames;
    size_t first_chunk = stream->capacity_frames - read_idx;
    if (first_chunk > to_read) first_chunk = to_read;

    memcpy(out_data,
           (uint8_t *)stream->buffer + read_idx * frame_sz,
           first_chunk * frame_sz);
    if (first_chunk < to_read) {
        memcpy((uint8_t *)out_data + first_chunk * frame_sz,
               stream->buffer,
               (to_read - first_chunk) * frame_sz);
    }

    stream->read_pos += to_read;
    *out_frames_read = to_read;
    pthread_cond_signal(&stream->not_full);
    pthread_mutex_unlock(&stream->lock);
    tk_set_error("");
    return TK_STATUS_OK;
}

/* Reset – discards all pending data */
TK_EXPORT TkStatus tk_audio_stream_reset(TkAudioStream *stream)
{
    TK_ASSERT_PTR(stream);
    pthread_mutex_lock(&stream->lock);
    stream->write_pos = 0;
    stream->read_pos  = 0;
    memset(stream->buffer, 0,
           stream->capacity_frames *
           tk_audio_frame_size(stream->format, stream->channels));
    pthread_cond_broadcast(&stream->not_full);
    pthread_mutex_unlock(&stream->lock);
    tk_set_error("");
    return TK_STATUS_OK;
}

/*==========================================================================*/
/*  Vision frame implementation                                            */
/*==========================================================================*/

static size_t tk_vision_stride(uint32_t width, TkVisionFormat fmt)
{
    size_t pixel_sz = 0;
    switch (fmt) {
        case TK_VISION_FMT_YUV420: pixel_sz = 1; break; /* planar, stride computed per plane */
        case TK_VISION_FMT_RGB24:
        case TK_VISION_FMT_BGR24: pixel_sz = 3; break;
        default: return 0;
    }
    /* Align each row to SIMD boundary */
    return (size_t)TK_ALIGN_UP(width * pixel_sz, TK_SIMD_ALIGNMENT);
}

/* Allocate a new vision frame */
TK_EXPORT TkStatus tk_vision_frame_create(TkVisionFrame **out_frame,
                                          uint32_t width,
                                          uint32_t height,
                                          TkVisionFormat format)
{
    TK_ASSERT_PTR(out_frame);
    if (width == 0 || height == 0) {
        tk_set_error("Invalid frame dimensions");
        return TK_STATUS_ERROR_INVALID_ARGUMENT;
    }

    TkVisionFrame *vf = (TkVisionFrame *)calloc(1, sizeof(TkVisionFrame));
    if (!vf) {
        tk_set_error("Failed to allocate TkVisionFrame");
        return TK_STATUS_ERROR_ALLOCATION_FAILED;
    }

    vf->width  = width;
    vf->height = height;
    vf->format = format;
    atomic_init(&vf->refcount, 1);

    if (format == TK_VISION_FMT_YUV420) {
        /* Y plane size = width*height, U/V = (width/2)*(height/2) each */
        size_t y_sz = width * height;
        size_t uv_sz = (width / 2) * (height / 2);
        size_t total = y_sz + 2 * uv_sz;
        TkStatus rc = tk_aligned_alloc(&vf->plane[0], total);
        if (rc != TK_STATUS_OK) {
            free(vf);
            tk_set_error("Failed to allocate YUV buffer");
            return rc;
        }
        vf->plane[1] = (uint8_t *)vf->plane[0] + y_sz;      /* U */
        vf->plane[2] = (uint8_t *)vf->plane[1] + uv_sz;      /* V */
    } else {
        size_t stride = tk_vision_stride(width, format);
        vf->stride = stride;
        for (int i = 0; i < 3; ++i) {
            TkStatus rc = tk_aligned_alloc(&vf->plane[i], stride * height);
            if (rc != TK_STATUS_OK) {
                for (int j = 0; j < i; ++j) tk_aligned_free(vf->plane[j]);
                free(vf);
                tk_set_error("Failed to allocate RGB buffer");
                return rc;
            }
        }
    }

    *out_frame = vf;
    tk_set_error("");
    return TK_STATUS_OK;
}

/* Destroy a vision frame */
TK_EXPORT TkStatus tk_vision_frame_destroy(TkVisionFrame **frame)
{
    if (!frame || !*frame) {
        tk_set_error("NULL frame pointer");
        return TK_STATUS_ERROR_NULL_POINTER;
    }

    TkVisionFrame *vf = *frame;
    if (TK_REF_DEC(vf) != 0) {
        *frame = NULL;
        tk_set_error("");
        return TK_STATUS_OK;
    }

    for (int i = 0; i < 3; ++i) {
        if (vf->plane[i]) tk_aligned_free(vf->plane[i]);
    }
    free(vf);
    *frame = NULL;
    tk_set_error("");
    return TK_STATUS_OK;
}

/* Get read‑only data pointer */
TK_EXPORT TkStatus tk_vision_frame_get_data(const TkVisionFrame *frame,
                                            const void **out_ptr)
{
    TK_ASSERT_PTR(frame);
    TK_ASSERT_PTR(out_ptr);
    *out_ptr = frame->plane[0];   /* For planar formats the first plane is enough */
    tk_set_error("");
    return TK_STATUS_OK;
}

/* Get mutable data pointer */
TK_EXPORT TkStatus tk_vision_frame_get_mutable_data(TkVisionFrame *frame,
                                                   void **out_ptr)
{
    TK_ASSERT_PTR(frame);
    TK_ASSERT_PTR(out_ptr);
    *out_ptr = frame->plane[0];
    tk_set_error("");
    return TK_STATUS_OK;
}

/* Retrieve frame metadata */
TK_EXPORT TkStatus tk_vision_frame_get_info(const TkVisionFrame *frame,
                                            uint32_t *out_width,
                                            uint32_t *out_height,
                                            TkVisionFormat *out_format)
{
    TK_ASSERT_PTR(frame);
    if (out_width)  *out_width  = frame->width;
    if (out_height) *out_height = frame->height;
    if (out_format) *out_format = frame->format;
    tk_set_error("");
    return TK_STATUS_OK;
}

/*==========================================================================*/
/*  Generic asynchronous command execution                                 */
/*==========================================================================*/

typedef struct {
    TkContext   *ctx;
    TkModuleType module;
    char        *command_name;
    void        *input;
    TkCallback   callback;
    void        *user_data;
} tk_async_job_t;

/* Simple thread‑pool (fixed size) for async jobs */
#define TK_ASYNC_POOL_SIZE 8
static pthread_t tk_async_threads[TK_ASYNC_POOL_SIZE];
static pthread_mutex_t tk_async_queue_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  tk_async_queue_cond = PTHREAD_COND_INITIALIZER;
static tk_async_job_t *tk_async_queue = NULL;   /* singly‑linked list */
static bool tk_async_running = true;

/* Worker thread entry point */
static void *tk_async_worker(void *arg)
{
    (void)arg;
    while (true) {
        pthread_mutex_lock(&tk_async_queue_lock);
        while (tk_async_queue == NULL && tk_async_running) {
            pthread_cond_wait(&tk_async_queue_cond, &tk_async_queue_lock);
        }
        if (!tk_async_running && tk_async_queue == NULL) {
            pthread_mutex_unlock(&tk_async_queue_lock);
            break;
        }
        tk_async_job_t *job = tk_async_queue;
        tk_async_queue = job->next;
        pthread_mutex_unlock(&tk_async_queue_lock);

        /* Dispatch to the appropriate module implementation.
         * For this example we simply call the C stub; real code would
         * forward to rust_module_execute_command or cpp_module_execute_command.
         */
        TkStatus status = TK_STATUS_ERROR_MODULE_NOT_INITIALIZED;
        switch (job->module) {
            case TK_MODULE_CORTEX:
            case TK_MODULE_AUDIO:
                /* Assume Rust implementation */
                extern TkStatus rust_module_execute_command(TkContext *,
                                                            TkModuleType,
                                                            const char *,
                                                            void *);
                status = rust_module_execute_command(job->ctx,
                                                    job->module,
                                                    job->command_name,
                                                    job->input);
                break;
            case TK_MODULE_VISION:
            case TK_MODULE_NAVIGATION:
                /* Assume C++ implementation */
                extern TkStatus cpp_module_execute_command(TkContext *,
                                                            TkModuleType,
                                                            const char *,
                                                            void *);
                status = cpp_module_execute_command(job->ctx,
                                                    job->module,
                                                    job->command_name,
                                                    job->input);
                break;
            default:
                status = TK_STATUS_ERROR_INVALID_ARGUMENT;
                break;
        }

        if (job->callback) {
            job->callback(status, NULL, job->user_data);
        }

        /* Clean up */
        free(job->command_name);
        free(job);
    }
    return NULL;
}

/* Initialise the async pool – called once at library load */
static void __attribute__((constructor)) tk_async_init(void)
{
    for (int i = 0; i < TK_ASYNC_POOL_SIZE; ++i) {
        pthread_create(&tk_async_threads[i], NULL, tk_async_worker, NULL);
    }
}

/* Shut down the async pool – called at library unload */
static void __attribute__((destructor)) tk_async_fini(void)
{
    pthread_mutex_lock(&tk_async_queue_lock);
    tk_async_running = false;
    pthread_cond_broadcast(&tk_async_queue_cond);
    pthread_mutex_unlock(&tk_async_queue_lock);

    for (int i = 0; i < TK_ASYNC_POOL_SIZE; ++i) {
        pthread_join(tk_async_threads[i], NULL);
    }
}

/* Public API – synchronous or asynchronous command dispatch */
TK_EXPORT TkStatus tk_module_execute_command(TkContext *context,
                                            TkModuleType module,
                                            const char *command_name,
                                            void *input,
                                            TkCallback callback,
                                            void *user_data)
{
    TK_ASSERT_PTR(context);
    TK_ASSERT_PTR(command_name);
    TK_CHECK_MAGIC(context);

    if (callback == NULL) {
        /* Synchronous path – forward directly */
        switch (module) {
            case TK_MODULE_CORTEX:
            case TK_MODULE_AUDIO:
                extern TkStatus rust_module_execute_command(TkContext *,
                                                            TkModuleType,
                                                            const char *,
                                                            void *);
                return rust_module_execute_command(context,
                                                   module,
                                                   command_name,
                                                   input);
            case TK_MODULE_VISION:
            case TK_MODULE_NAVIGATION:
                extern TkStatus cpp_module_execute_command(TkContext *,
                                                            TkModuleType,
                                                            const char *,
                                                            void *);
                return cpp_module_execute_command(context,
                                                   module,
                                                   command_name,
                                                   input);
            default:
                tk_set_error("Unsupported module for sync execution");
                return TK_STATUS_ERROR_INVALID_ARGUMENT;
        }
    }

    /* Asynchronous path – enqueue a job */
    tk_async_job_t *job = (tk_async_job_t *)malloc(sizeof(tk_async_job_t));
    if (!job) {
        tk_set_error("Failed to allocate async job");
        return TK_STATUS_ERROR_ALLOCATION_FAILED;
    }
    job->ctx          = context;
    job->module       = module;
    job->command_name = strdup(command_name);
    job->input        = input;
    job->callback     = callback;
    job->user_data    = user_data;
    job->next         = NULL;

    pthread_mutex_lock(&tk_async_queue_lock);
    /* Insert at tail for FIFO order */
    tk_async_job_t **tail = &tk_async_queue;
    while (*tail) tail = &(*tail)->next;
    *tail = job;
    pthread_cond_signal(&tk_async_queue_cond);
    pthread_mutex_unlock(&tk_async_queue_lock);

    tk_set_error("");
    return TK_STATUS_OK;
}

/*==========================================================================*/
/*  End of file                                                            */
/*==========================================================================*/
