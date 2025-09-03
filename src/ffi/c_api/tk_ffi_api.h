/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_ffi_api.h
 *
 * This header defines the public Foreign Function Interface (FFI) for the
 * TrackieLLM core system.  It is deliberately verbose and engineered for
 * production‑grade use across C, C++, and Rust.  The API is split into logical
 * sections, each heavily documented with Doxygen comments, inline examples,
 * and compile‑time checks.  The goal is to provide a single source of truth
 * that can be compiled into a static or shared library and linked from any
 * language that can consume a C ABI.
 *
 * The design follows the “opaque‑handle + enum status” pattern used by the
 * Linux kernel, glibc, and high‑performance libraries such as FFmpeg.  All
 * structures are opaque to the consumer; only the functions in this header may
 * manipulate them.  Errors are reported via the TkStatus enum and a thread‑local
 * error string that can be queried with tk_get_last_error().
 *
 * Dependencies:
 *   - stdint.h   (fixed‑width integer types)
 *   - stddef.h   (size_t, NULL)
 *   - stdbool.h  (bool)
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#ifndef TK_FFI_API_H
#define TK_FFI_API_H

/*==========================================================================*/
/*  Compiler / Platform Detection                                           */
/*==========================================================================*/

#if defined(_MSC_VER)
#   define TK_INLINE __inline
#   define TK_FORCE_INLINE __forceinline
#   define TK_EXPORT __declspec(dllexport)
#   define TK_THREAD_LOCAL __declspec(thread)
#elif defined(__GNUC__) || defined(__clang__)
#   define TK_INLINE static inline
#   define TK_FORCE_INLINE __attribute__((always_inline)) static inline
#   define TK_EXPORT __attribute__((visibility("default")))
#   define TK_THREAD_LOCAL __thread
#else
#   error "Unsupported compiler"
#endif

/*==========================================================================*/
/*  Basic Types & Alignment Helpers                                        */
/*==========================================================================*/

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* Align a value up to the nearest multiple of ALIGN (which must be a power of two) */
#define TK_ALIGN_UP(value, ALIGN) \
    (((value) + ((ALIGN) - 1)) & ~((ALIGN) - 1))

/* Minimum alignment required for SIMD (AVX‑256) */
#define TK_SIMD_ALIGNMENT 32U

/*==========================================================================*/
/*  Opaque Handles                                                          */
/*==========================================================================*/

/**
 * @brief Opaque handle to the main Trackie context.
 *
 * The internal representation is hidden from the consumer.  All access must be
 * performed through the functions exported by this API.
 */
typedef struct TkContext TkContext;

/**
 * @brief Opaque handle to a generic multi‑dimensional tensor.
 *
 * The tensor can store float32, int32 or uint8 data.  Its layout is row‑major
 * and tightly packed.  The handle owns the memory for shape information and
 * the data buffer.
 */
typedef struct TkTensor TkTensor;

/**
 * @brief Opaque handle to an audio stream buffer.
 *
 * The buffer is a circular ring that can be written to by producers and read
 * by consumers.  It is reference‑counted internally.
 */
typedef struct TkAudioStream TkAudioStream;

/**
 * @brief Opaque handle to a vision frame.
 *
 * The frame contains a planar YUV420 image and optional metadata.
 */
typedef struct TkVisionFrame TkVisionFrame;

/*==========================================================================*/
/*  Enumerations – Status, Modules, Data Types, etc.                        */
/*==========================================================================*/

/**
 * @brief Return status for all FFI calls.
 *
 * Positive values are reserved for future extensions.  Zero indicates success.
 */
typedef enum {
    TK_STATUS_OK                         = 0,   /**< Success */
    TK_STATUS_ERROR_NULL_POINTER         = -1,  /**< One of the required pointers was NULL */
    TK_STATUS_ERROR_INVALID_ARGUMENT     = -2,  /**< Argument out of range or malformed */
    TK_STATUS_ERROR_ALLOCATION_FAILED    = -3,  /**< malloc/realloc failed */
    TK_STATUS_ERROR_INVALID_HANDLE       = -4,  /**< Opaque handle does not belong to this context */
    TK_STATUS_ERROR_MODULE_NOT_INITIALIZED = -5,/**< Requested module has not been initialized */
    TK_STATUS_ERROR_OPERATION_FAILED    = -6,  /**< Generic operation failure */
    TK_STATUS_ERROR_UNSUPPORTED_FEATURE  = -7,  /**< Feature not compiled in */
    TK_STATUS_ERROR_DEADLOCK_DETECTED    = -8,  /**< Potential deadlock detected */
    TK_STATUS_ERROR_TIMEOUT              = -9,  /**< Operation timed out */
    TK_STATUS_ERROR_UNKNOWN              = -100 /**< Catch‑all for unexpected errors */
} TkStatus;

/**
 * @brief Identifies a core module.
 *
 * The enum values are deliberately sparse to allow insertion of new modules
 * without breaking binary compatibility.
 */
typedef enum {
    TK_MODULE_CORTEX      = 0,
    TK_MODULE_VISION      = 10,
    TK_MODULE_AUDIO       = 20,
    TK_MODULE_SENSORS     = 30,
    TK_MODULE_NAVIGATION  = 40,
    TK_MODULE_NETWORKING  = 50,
    TK_MODULE_CUSTOM_BASE = 1000   /**< Base for user‑defined modules */
} TkModuleType;

/**
 * @brief Data types supported by TkTensor.
 */
typedef enum {
    TK_DATA_TYPE_FLOAT32 = 0,
    TK_DATA_TYPE_INT32   = 1,
    TK_DATA_TYPE_UINT8   = 2
} TkDataType;

/**
 * @brief Audio sample formats.
 */
typedef enum {
    TK_AUDIO_FMT_S16LE = 0,   /**< Signed 16‑bit little‑endian */
    TK_AUDIO_FMT_S24LE = 1,   /**< Signed 24‑bit little‑endian */
    TK_AUDIO_FMT_F32   = 2    /**< 32‑bit float */
} TkAudioFormat;

/**
 * @brief Vision pixel formats.
 */
typedef enum {
    TK_VISION_FMT_YUV420 = 0,
    TK_VISION_FMT_RGB24  = 1,
    TK_VISION_FMT_BGR24  = 2
} TkVisionFormat;

/*==========================================================================*/
/*  Thread‑Local Error Handling                                            */
/*==========================================================================*/

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Retrieve the last error message for the calling thread.
 *
 * The returned string is owned by the library and must **not** be freed by the
 * caller.  It is valid until the next FFI call on the same thread.
 *
 * @return Null‑terminated UTF‑8 string describing the last error, or an empty
 *         string if no error has occurred.
 */
TK_EXPORT const char* tk_get_last_error(void);

/*==========================================================================*/
/*  Core Context Management                                                */
/*==========================================================================*/

/**
 * @brief Create a new Trackie context.
 *
 * The context aggregates all module instances and holds global configuration.
 *
 * @param[out] out_context Pointer that receives the newly allocated context.
 * @return TK_STATUS_OK on success, otherwise an error code.
 *
 * @note The caller is responsible for destroying the context with
 *       tk_context_destroy().
 */
TK_EXPORT TkStatus tk_context_create(TkContext **out_context);

/**
 * @brief Destroy a Trackie context.
 *
 * All modules attached to the context are shut down in the reverse order of
 * initialization.  The function is safe to call multiple times; subsequent calls
 * become no‑ops.
 *
 * @param[in,out] context Pointer to the context pointer. After successful
 *                       destruction the pointer is set to NULL.
 * @return TK_STATUS_OK on success, or TK_STATUS_ERROR_NULL_POINTER.
 */
TK_EXPORT TkStatus tk_context_destroy(TkContext **context);

/**
 * @brief Retrieve a module handle from a context.
 *
 * This function does **not** increase the module's reference count; the
 * returned pointer is only valid while the context lives.
 *
 * @param[in]  context   The context created with tk_context_create().
 * @param[in]  module    The module type to retrieve.
 * @param[out] out_handle Pointer that receives the module handle.
 * @return TK_STATUS_OK on success, or an error code.
 *
 * @note The returned handle is opaque and must be passed to other API calls.
 */
TK_EXPORT TkStatus tk_context_get_module(TkContext *context,
                                         TkModuleType module,
                                         void **out_handle);

/*==========================================================================*/
/*  Tensor Management (TkTensor)                                           */
/*==========================================================================*/

/**
 * @brief Create a tensor.
 *
 * The function copies the shape array and, if @p data is non‑NULL, copies the
 * data buffer.  The tensor owns its memory and will free it on destruction.
 *
 * @param[out] out_tensor   Pointer that receives the new tensor handle.
 * @param[in]  data_type    Data type of the tensor elements.
 * @param[in]  shape        Array of dimension sizes (row‑major order).
 * @param[in]  shape_len    Number of dimensions.
 * @param[in]  data         Optional pointer to initial data.  May be NULL.
 * @return TK_STATUS_OK on success, or an error code.
 *
 * @warning The shape array must contain only positive values.
 */
TK_EXPORT TkStatus tk_tensor_create(TkTensor **out_tensor,
                                    TkDataType data_type,
                                    const int64_t *shape,
                                    size_t shape_len,
                                    const void *data);

/**
 * @brief Destroy a tensor.
 *
 * All memory owned by the tensor (shape array, data buffer) is released.
 *
 * @param[in,out] tensor Pointer to the tensor handle. Set to NULL on success.
 * @return TK_STATUS_OK on success, or TK_STATUS_ERROR_NULL_POINTER.
 */
TK_EXPORT TkStatus tk_tensor_destroy(TkTensor **tensor);

/**
 * @brief Retrieve a read‑only pointer to the tensor data.
 *
 * The returned pointer is valid until the tensor is destroyed or its data is
 * re‑allocated via tk_tensor_reshape().
 *
 * @param[in]  tensor   Tensor handle.
 * @param[out] out_data Pointer that receives the data address.
 * @return TK_STATUS_OK on success, or an error code.
 */
TK_EXPORT TkStatus tk_tensor_get_data(const TkTensor *tensor,
                                      const void **out_data);

/**
 * @brief Retrieve a mutable pointer to the tensor data.
 *
 * The caller may modify the contents directly.  Modifications must respect the
 * tensor's data type and element count.
 *
 * @param[in]  tensor   Tensor handle.
 * @param[out] out_data Pointer that receives the mutable data address.
 * @return TK_STATUS_OK on success, or an error code.
 */
TK_EXPORT TkStatus tk_tensor_get_mutable_data(TkTensor *tensor,
                                              void **out_data);

/**
 * @brief Get the tensor shape.
 *
 * The caller provides a buffer large enough to hold @p shape_len elements.
 * On entry, @p in_out_shape_len contains the buffer capacity; on exit it
 * contains the actual number of dimensions.
 *
 * @param[in]  tensor            Tensor handle.
 * @param[out] out_shape         Buffer that receives the shape.
 * @param[in,out] in_out_shape_len  Capacity on input, actual length on output.
 * @return TK_STATUS_OK on success, or an error code.
 */
TK_EXPORT TkStatus tk_tensor_get_shape(const TkTensor *tensor,
                                      int64_t *out_shape,
                                      size_t *in_out_shape_len);

/**
 * @brief Reshape an existing tensor.
 *
 * The total number of elements must remain unchanged.  The data buffer is kept
 * intact; only the shape metadata is updated.
 *
 * @param[in,out] tensor   Tensor handle.
 * @param[in]     shape    New shape array.
 * @param[in]     shape_len Number of dimensions.
 * @return TK_STATUS_OK on success, or an error code.
 */
TK_EXPORT TkStatus tk_tensor_reshape(TkTensor *tensor,
                                     const int64_t *shape,
                                     size_t shape_len);

/**
 * @brief Fill a tensor with a constant value.
 *
 * The value is interpreted according to the tensor's data type.
 *
 * @param[in] tensor Tensor handle.
 * @param[in] value  Pointer to the constant value (must match data type).
 * @return TK_STATUS_OK on success, or an error code.
 */
TK_EXPORT TkStatus tk_tensor_fill(TkTensor *tensor,
                                  const void *value);

/**
 * @brief Perform element‑wise addition of two tensors.
 *
 * The tensors must have identical shape and data type.  The result is stored
 * in @p out_result, which may be the same as @p a or @p b (in‑place addition).
 *
 * @param[in]  a          First operand.
 * @param[in]  b          Second operand.
 * @param[out] out_result Result tensor.
 * @return TK_STATUS_OK on success, or an error code.
 */
TK_EXPORT TkStatus tk_tensor_add(const TkTensor *a,
                                 const TkTensor *b,
                                 TkTensor *out_result);

/**
 * @brief Perform matrix multiplication (2‑D tensors only).
 *
 * @p a must be of shape (M, K) and @p b of shape (K, N).  The result tensor
 * must be pre‑allocated with shape (M, N) and the same data type.
 *
 * @param[in]  a          Left matrix.
 * @param[in]  b          Right matrix.
 * @param[out] out_result Result matrix.
 * @return TK_STATUS_OK on success, or an error code.
 *
 * @note This implementation uses a cache‑friendly blocking algorithm
 *       (see tk_tensor_matmul_blocked()).
 */
TK_EXPORT TkStatus tk_tensor_matmul(const TkTensor *a,
                                    const TkTensor *b,
                                    TkTensor *out_result);

/**
 * @brief Blocked matrix multiplication (internal helper).
 *
 * This function is exposed for advanced users who need fine‑grained control
 * over the block size.  The block size must be a power of two and not exceed
 * 256.
 *
 * @param[in]  a          Left matrix.
 * @param[in]  b          Right matrix.
 * @param[out] out_result Result matrix.
 * @param[in]  block_size Block dimension (e.g., 64).
 * @return TK_STATUS_OK on success, or an error code.
 */
TK_EXPORT TkStatus tk_tensor_matmul_blocked(const TkTensor *a,
                                            const TkTensor *b,
                                            TkTensor *out_result,
                                            size_t block_size);

/*==========================================================================*/
/*  Audio Stream API                                                       */
/*==========================================================================*/

/**
 * @brief Create an audio stream buffer.
 *
 * The buffer size is rounded up to the nearest multiple of TK_SIMD_ALIGNMENT
 * to allow SIMD‑accelerated processing.
 *
 * @param[out] out_stream   Pointer that receives the new stream handle.
 * @param[in]  sample_rate  Samples per second (e.g., 48000).
 * @param[in]  channels     Number of interleaved channels.
 * @param[in]  format       Sample format (see TkAudioFormat).
 * @param[in]  capacity     Desired capacity in frames (rounded up).
 * @return TK_STATUS_OK on success, or an error code.
 */
TK_EXPORT TkStatus tk_audio_stream_create(TkAudioStream **out_stream,
                                         uint32_t sample_rate,
                                         uint16_t channels,
                                         TkAudioFormat format,
                                         size_t capacity);

/**
 * @brief Destroy an audio stream.
 *
 * All pending data is discarded.  The function is thread‑safe.
 *
 * @param[in,out] stream Pointer to the stream handle. Set to NULL on success.
 * @return TK_STATUS_OK on success, or TK_STATUS_ERROR_NULL_POINTER.
 */
TK_EXPORT TkStatus tk_audio_stream_destroy(TkAudioStream **stream);

/**
 * @brief Write raw audio frames into the stream.
 *
 * The function copies @p frames * @p channels samples into the circular buffer.
 * If the buffer does not have enough free space, the call fails with
 * TK_STATUS_ERROR_OPERATION_FAILED.
 *
 * @param[in]  stream   Stream handle.
 * @param[in]  frames   Number of audio frames to write.
 * @param[in]  data     Pointer to interleaved audio samples.
 * @return TK_STATUS_OK on success, or an error code.
 */
TK_EXPORT TkStatus tk_audio_stream_write(TkAudioStream *stream,
                                        size_t frames,
                                        const void *data);

/**
 * @brief Read raw audio frames from the stream.
 *
 * The function copies up to @p frames samples into @p out_data.  If fewer
 * frames are available, the function returns the number of frames actually
 * read via @p out_frames_read.
 *
 * @param[in]  stream          Stream handle.
 * @param[in]  frames_requested Number of frames to read.
 * @param[out] out_data        Destination buffer (must be large enough).
 * @param[out] out_frames_read Number of frames actually read.
 * @return TK_STATUS_OK on success, or an error code.
 */
TK_EXPORT TkStatus tk_audio_stream_read(TkAudioStream *stream,
                                       size_t frames_requested,
                                       void *out_data,
                                       size_t *out_frames_read);

/**
 * @brief Reset the stream to an empty state.
 *
 * All buffered data is discarded but the internal allocation remains.
 *
 * @param[in] stream Stream handle.
 * @return TK_STATUS_OK on success, or TK_STATUS_ERROR_NULL_POINTER.
 */
TK_EXPORT TkStatus tk_audio_stream_reset(TkAudioStream *stream);

/*==========================================================================*/
/*  Vision Frame API                                                       */
/*==========================================================================*/

/**
 * @brief Allocate a new vision frame.
 *
 * The frame is allocated with the requested pixel format and dimensions.
 * Memory is aligned to TK_SIMD_ALIGNMENT for SIMD processing.
 *
 * @param[out] out_frame   Pointer that receives the new frame handle.
 * @param[in]  width       Frame width in pixels.
 * @param[in]  height      Frame height in pixels.
 * @param[in]  format      Pixel format (see TkVisionFormat).
 * @return TK_STATUS_OK on success, or an error code.
 */
TK_EXPORT TkStatus tk_vision_frame_create(TkVisionFrame **out_frame,
                                          uint32_t width,
                                          uint32_t height,
                                          TkVisionFormat format);

/**
 * @brief Destroy a vision frame.
 *
 * @param[in,out] frame Pointer to the frame handle. Set to NULL on success.
 * @return TK_STATUS_OK on success, or TK_STATUS_ERROR_NULL_POINTER.
 */
TK_EXPORT TkStatus tk_vision_frame_destroy(TkVisionFrame **frame);

/**
 * @brief Get a read‑only pointer to the raw pixel buffer.
 *
 * The layout depends on the pixel format:
 *   - YUV420: planar Y, U, V planes (contiguous).
 *   - RGB24/BGR24: interleaved 3‑byte pixels.
 *
 * @param[in]  frame   Frame handle.
 * @param[out] out_ptr Pointer that receives the buffer address.
 * @return TK_STATUS_OK on success, or an error code.
 */
TK_EXPORT TkStatus tk_vision_frame_get_data(const TkVisionFrame *frame,
                                            const void **out_ptr);

/**
 * @brief Get mutable access to the pixel buffer.
 *
 * @param[in]  frame   Frame handle.
 * @param[out] out_ptr Pointer that receives the mutable buffer address.
 * @return TK_STATUS_OK on success, or an error code.
 */
TK_EXPORT TkStatus tk_vision_frame_get_mutable_data(TkVisionFrame *frame,
                                                   void **out_ptr);

/**
 * @brief Retrieve frame metadata (width, height, format).
 *
 * @param[in]  frame   Frame handle.
 * @param[out] out_width  Width in pixels.
 * @param[out] out_height Height in pixels.
 * @param[out] out_format Pixel format.
 * @return TK_STATUS_OK on success, or an error code.
 */
TK_EXPORT TkStatus tk_vision_frame_get_info(const TkVisionFrame *frame,
                                            uint32_t *out_width,
                                            uint32_t *out_height,
                                            TkVisionFormat *out_format);

/*==========================================================================*/
/*  Generic Callback / Asynchronous Execution                              */
/*==========================================================================*/

/**
 * @brief Generic callback type used by asynchronous module commands.
 *
 * The callback is invoked exactly once.  The @p result pointer is opaque and
 * depends on the command; it may be NULL.
 *
 * @param status   Result status of the operation.
 * @param result   Opaque pointer to command‑specific result data.
 * @param user_data User‑supplied context pointer passed to the original call.
 */
typedef void (*TkCallback)(TkStatus status,
                           void *result,
                           void *user_data);

/**
 * @brief Execute a command on a specific module.
 *
 * The function can operate synchronously (callback == NULL) or asynchronously.
 * For asynchronous execution the library internally schedules the work on a
 * thread‑pool and returns immediately.
 *
 * @param[in]  context       Main Trackie context.
 * @param[in]  module        Target module.
 * @param[in]  command_name  Null‑terminated command identifier.
 * @param[in]  input         Opaque input handle (module‑specific).
 * @param[in]  callback      Optional callback for async completion.
 * @param[in]  user_data     Optional user data passed to the callback.
 * @return TK_STATUS_OK if the command was successfully queued.
 *
 * @note The caller must ensure that @p input remains valid until the callback
 *       is invoked (or the call returns for synchronous execution).
 */
TK_EXPORT TkStatus tk_module_execute_command(TkContext *context,
                                            TkModuleType module,
                                            const char *command_name,
                                            void *input,
                                            TkCallback callback,
                                            void *user_data);

/*==========================================================================*/
/*  Utility Functions – Alignment, SIMD, Debug, etc.                        */
/*==========================================================================*/

/**
 * @brief Allocate memory aligned to TK_SIMD_ALIGNMENT.
 *
 * The returned pointer must be freed with tk_aligned_free().
 *
 * @param[out] out_ptr Pointer that receives the allocated memory.
 * @param[in]  size    Allocation size in bytes.
 * @return TK_STATUS_OK on success, or TK_STATUS_ERROR_ALLOCATION_FAILED.
 */
TK_EXPORT TkStatus tk_aligned_alloc(void **out_ptr, size_t size);

/**
 * @brief Free memory allocated with tk_aligned_alloc().
 *
 * @param[in] ptr Pointer returned by tk_aligned_alloc().
 */
TK_EXPORT void tk_aligned_free(void *ptr);

/**
 * @brief Zero‑fill a memory region in a way that the compiler cannot optimise
 *        the store away (useful for secret data).
 *
 * @param[in] ptr   Pointer to the region.
 * @param[in] size  Size of the region in bytes.
 */
TK_EXPORT void tk_secure_zero(void *ptr, size_t size);

/**
 * @brief Perform a constant‑time memory comparison.
 *
 * The function returns true (1) if the buffers are equal, false (0) otherwise.
 * Execution time does not depend on the contents of the buffers.
 *
 * @param[in] a    First buffer.
 * @param[in] b    Second buffer.
 * @param[in] len  Length of both buffers (must be equal).
 * @return 1 if equal, 0 otherwise.
 */
TK_EXPORT int tk_memcmp_const_time(const void *a,
                                   const void *b,
                                   size_t len);

/**
 * @brief Log a formatted message (debug level).
 *
 * The implementation forwards to the platform‑specific logger (syslog on Linux,
 * OutputDebugString on Windows).  The function is thread‑safe.
 *
 * @param[in] fmt  printf‑style format string.
 * @param[in] ...  Arguments.
 */
TK_EXPORT void tk_log_debug(const char *fmt, ...);

/**
 * @brief Log an error message (error level).
 *
 * @param[in] fmt  printf‑style format string.
 * @param[in] ...  Arguments.
 */
TK_EXPORT void tk_log_error(const char *fmt, ...);

/*==========================================================================*/
/*  Version Information                                                    */
/*==========================================================================*/

/**
 * @brief Library version string (semantic versioning).
 *
 * The string is static and does not need to be freed.
 */
TK_EXPORT const char* tk_version_string(void);

/**
 * @brief Library version as three separate integers.
 *
 * @param[out] major Major version number.
 * @param[out] minor Minor version number.
 * @param[out] patch Patch version number.
 */
TK_EXPORT void tk_version_numbers(uint32_t *major,
                                  uint32_t *minor,
                                  uint32_t *patch);

/*==========================================================================*/
/*  Internal Helper Macros (not part of the public API)                    */
/*==========================================================================*/

#define TK_CHECK_STATUS(expr)                     \
    do {                                          \
        TkStatus _st = (expr);                    \
        if (_st != TK_STATUS_OK) {               \
            return _st;                           \
        }                                         \
    } while (0)

/*==========================================================================*/
/*  End of Header                                                          */
/*==========================================================================*/

#ifdef __cplusplus
}
#endif

#endif /* TK_FFI_API_H */
