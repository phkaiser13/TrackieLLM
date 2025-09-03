/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_metal_dispatch.h
 *
 * This header defines the public API for the Metal Dispatcher, the high‑level
 * orchestration engine for the Apple‑metal GPU backend.  The dispatcher is the
 * sole entry point for all other TrackieLLM components that require GPU
 * acceleration on macOS / iOS devices.  Its design mirrors the CUDA dispatcher
 * (tk_cuda_dispatch.h) but exploits the unique characteristics of Metal:
 *
 *   • Opaque handles for GPU resources (buffers, textures, events) to prevent
 *     accidental misuse.
 *   • Asynchronous‑by‑default execution using MTLCommandQueue / MTLCommandBuffer.
 *   • Explicit synchronization via event objects or command‑buffer completion
 *     callbacks.
 *   • High‑level workflow functions that encapsulate complete image‑processing
 *     or tensor‑processing pipelines.
 *   • Tight integration with Apple frameworks (Core ML, Metal Performance
 *     Shaders) and unified memory modes to minimise copies.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#ifndef TRACKIELLM_GPU_METAL_TK_METAL_DISPATCH_H
#define TRACKIELLM_GPU_METAL_TK_METAL_DISPATCH_H

#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "vision/tk_vision_pipeline.h"          /* tk_video_frame_t               */
#include "gpu/metal/tk_metal_kernels.h"         /* kernel‑parameter structs       */
#include <CoreVideo/CoreVideo.h>                /* CVPixelBufferRef for Core ML   */

/* -------------------------------------------------------------------------
 * Opaque Types – the public API never exposes the internal Objective‑C /
 * Metal objects.  This guarantees binary compatibility and prevents resource
 * leaks caused by direct pointer manipulation.
 * ------------------------------------------------------------------------- */
typedef struct tk_metal_dispatcher_s tk_metal_dispatcher_t;

/* Handles for GPU resources.  The concrete structs are defined in the .c file
 * and contain MTLBuffer / MTLTexture / MTLSharedEvent pointers. */
typedef struct tk_gpu_buffer_s*   tk_gpu_buffer_t;   /* Device memory buffer   */
typedef struct tk_gpu_texture_s* tk_gpu_texture_t;  /* 2‑D image texture    */
typedef struct tk_gpu_event_s*   tk_gpu_event_t;    /* Synchronisation event */

/* -------------------------------------------------------------------------
 * Configuration structure – passed to tk_metal_dispatch_create().
 * ------------------------------------------------------------------------- */
typedef struct {
    /** The Metal device to use.  Pass NULL to select the default system device. */
    id<MTLDevice> device;
    /** Preferred storage mode for buffers that will be accessed by both CPU and GPU.
     *  One of MTLStorageModeShared, MTLStorageModePrivate or MTLStorageModeManaged.
     *  The default is MTLStorageModePrivate for maximum GPU throughput. */
    MTLStorageMode preferred_storage_mode;
    /** Maximum number of concurrent command buffers that may be in flight.
     *  The dispatcher internally creates a pool of MTLCommandQueue objects.
     *  Reasonable defaults are 4–8. */
    uint32_t max_in_flight;
} tk_metal_dispatcher_config_t;

/* -------------------------------------------------------------------------
 * C++ compatibility
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Dispatcher Lifecycle
 * ------------------------------------------------------------------------- */

/**
 * @brief Create and initialise a Metal dispatcher.
 *
 * The function selects a suitable MTLDevice (or uses the one supplied in the
 * config), creates a command‑queue pool, and initialises internal event pools.
 *
 * @param[out] out_dispatcher   Pointer that receives the newly allocated dispatcher.
 * @param[in]  config           Configuration describing the device and memory mode.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_GPU_DEVICE_NOT_FOUND if no compatible Metal device exists.
 * @return TK_ERROR_OUT_OF_MEMORY on host allocation failure.
 */
TK_NODISCARD tk_error_code_t
tk_metal_dispatch_create(tk_metal_dispatcher_t **out_dispatcher,
                         const tk_metal_dispatcher_config_t *config);

/**
 * @brief Destroy a Metal dispatcher and free all associated GPU resources.
 *
 * All outstanding command buffers are flushed, event objects are released and
 * the underlying MTLDevice reference is released.  After this call the
 * dispatcher pointer is set to NULL.
 *
 * @param[in,out] dispatcher   Pointer to the dispatcher to destroy.
 */
void
tk_metal_dispatch_destroy(tk_metal_dispatcher_t **dispatcher);

/* -------------------------------------------------------------------------
 * GPU Memory Management
 * ------------------------------------------------------------------------- */

/**
 * @brief Allocate a GPU buffer.
 *
 * The allocation respects the storage mode selected in the dispatcher config.
 *
 * @param[in]  dispatcher   Dispatcher instance.
 * @param[out] out_buffer   Handle that receives the allocated buffer.
 * @param[in]  size_bytes   Size of the allocation in bytes.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_GPU_MEMORY if the device cannot satisfy the request.
 */
TK_NODISCARD tk_error_code_t
tk_metal_dispatch_malloc(tk_metal_dispatcher_t *dispatcher,
                         tk_gpu_buffer_t *out_buffer,
                         size_t size_bytes);

/**
 * @brief Allocate a 2‑D texture.
 *
 * @param[in]  dispatcher   Dispatcher instance.
 * @param[out] out_texture  Handle that receives the allocated texture.
 * @param[in]  width        Texture width in pixels.
 * @param[in]  height       Texture height in pixels.
 * @param[in]  pixel_format Metal pixel format (e.g. MTLPixelFormatRGBA8Unorm).
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_GPU_MEMORY if allocation fails.
 */
TK_NODISCARD tk_error_code_t
tk_metal_dispatch_alloc_texture(tk_metal_dispatcher_t *dispatcher,
                               tk_gpu_texture_t *out_texture,
                               uint32_t width,
                               uint32_t height,
                               MTLPixelFormat pixel_format);

/**
 * @brief Free a previously allocated GPU buffer.
 *
 * The handle is invalidated after the call.
 *
 * @param[in] dispatcher   Dispatcher instance.
 * @param[in,out] buffer   Buffer handle to free.
 */
void
tk_metal_dispatch_free(tk_metal_dispatcher_t *dispatcher,
                       tk_gpu_buffer_t *buffer);

/**
 * @brief Free a previously allocated GPU texture.
 *
 * @param[in] dispatcher   Dispatcher instance.
 * @param[in,out] texture Texture handle to free.
 */
void
tk_metal_dispatch_free_texture(tk_metal_dispatcher_t *dispatcher,
                               tk_gpu_texture_t *texture);

/**
 * @brief Asynchronously upload data from host memory to a GPU buffer.
 *
 * The operation is enqueued on the dispatcher’s default command queue and
 * returns immediately.  The caller may optionally provide an event that will be
 * signalled when the copy completes.
 *
 * @param[in] dispatcher   Dispatcher instance.
 * @param[in] dst_buffer   Destination GPU buffer.
 * @param[in] src_host_ptr Pointer to source data in host memory.
 * @param[in] size_bytes   Number of bytes to copy.
 * @param[out] out_event   Optional event that will be signalled on completion
 *                         (may be NULL).
 *
 * @return TK_SUCCESS if the copy was successfully queued.
 */
TK_NODISCARD tk_error_code_t
tk_metal_dispatch_upload_async(tk_metal_dispatcher_t *dispatcher,
                               tk_gpu_buffer_t dst_buffer,
                               const void *src_host_ptr,
                               size_t size_bytes,
                               tk_gpu_event_t *out_event);

/**
 * @brief Asynchronously download data from a GPU buffer to host memory.
 *
 * @param[in] dispatcher   Dispatcher instance.
 * @param[out] dst_host_ptr Destination pointer in host memory.
 * @param[in] src_buffer   Source GPU buffer.
 * @param[in] size_bytes   Number of bytes to copy.
 * @param[out] out_event   Optional event signalled on completion (may be NULL).
 *
 * @return TK_SUCCESS if the copy was successfully queued.
 */
TK_NODISCARD tk_error_code_t
tk_metal_dispatch_download_async(tk_metal_dispatcher_t *dispatcher,
                                 void *dst_host_ptr,
                                 tk_gpu_buffer_t src_buffer,
                                 size_t size_bytes,
                                 tk_gpu_event_t *out_event);

/* -------------------------------------------------------------------------
 * Synchronisation Primitives
 * ------------------------------------------------------------------------- */

/**
 * @brief Create a GPU event object.
 *
 * Events are thin wrappers around MTLSharedEvent (or MTLFence on older macOS).
 *
 * @param[in]  dispatcher   Dispatcher instance.
 * @param[out] out_event    Receives the newly created event handle.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_OUT_OF_MEMORY if the event cannot be allocated.
 */
TK_NODISCARD tk_error_code_t
tk_metal_dispatch_create_event(tk_metal_dispatcher_t *dispatcher,
                               tk_gpu_event_t *out_event);

/**
 * @brief Destroy a GPU event object.
 *
 * @param[in] dispatcher   Dispatcher instance.
 * @param[in,out] event   Event handle to destroy.
 */
void
tk_metal_dispatch_destroy_event(tk_metal_dispatcher_t *dispatcher,
                                tk_gpu_event_t *event);

/**
 * @brief Wait for a previously signalled event.
 *
 * This blocks the calling CPU thread until the GPU has signalled the event.
 *
 * @param[in] dispatcher   Dispatcher instance.
 * @param[in] event        Event to wait for.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_TIMEOUT if the wait exceeds an internal timeout.
 */
TK_NODISCARD tk_error_code_t
tk_metal_dispatch_wait_for_event(tk_metal_dispatcher_t *dispatcher,
                                 tk_gpu_event_t event);

/**
 * @brief Block until all work submitted to the default command queue has
 *        completed.  Prefer event‑based waiting for finer granularity.
 *
 * @param[in] dispatcher   Dispatcher instance.
 *
 * @return TK_SUCCESS on success.
 */
TK_NODISCARD tk_error_code_t
tk_metal_dispatch_synchronize(tk_metal_dispatcher_t *dispatcher);

/* -------------------------------------------------------------------------
 * High‑Level Workflow Dispatch
 * ------------------------------------------------------------------------- */

/**
 * @brief Dispatch the full vision‑pre‑processing pipeline.
 *
 * The workflow consists of:
 *   1. Asynchronous upload of the source video frame to a GPU texture.
 *   2. Execution of a Metal compute kernel that performs colour‑space conversion,
 *      resizing, and optional Gaussian blur (implemented via MPSImageGaussianBlur).
 *   3. Optional post‑processing (e.g. histogram equalisation) using MPS kernels.
 *
 * The @p params structure contains opaque buffer/texture handles that must have
 * been allocated with the dispatcher prior to the call.
 *
 * @param[in] dispatcher   Dispatcher instance.
 * @param[in] params       Parameters for the preprocessing kernel.
 *
 * @return TK_SUCCESS if the workflow was successfully queued.
 */
TK_NODISCARD tk_error_code_t
tk_metal_dispatch_preprocess_image(tk_metal_dispatcher_t *dispatcher,
                                   const tk_preprocess_params_t *params);

/**
 * @brief Dispatch the depth‑to‑point‑cloud conversion workflow.
 *
 * This function launches a Metal compute kernel (or an MPS kernel if available)
 * that reads a depth map texture, applies intrinsics, and writes XYZ points into
 * a GPU buffer.  The resulting buffer can be consumed directly by the navigation
 * module without an extra copy thanks to unified memory.
 *
 * @param[in] dispatcher   Dispatcher instance.
 * @param[in] params       Parameters for the conversion kernel.
 *
 * @return TK_SUCCESS if the workflow was successfully queued.
 */
TK_NODISCARD tk_error_code_t
tk_metal_dispatch_depth_to_point_cloud(tk_metal_dispatcher_t *dispatcher,
                                       const tk_depth_to_points_params_t *params);

/* -------------------------------------------------------------------------
 * Apple‑Specific Extensions
 * ------------------------------------------------------------------------- */

/**
 * @brief Convert a CVPixelBufferRef (used by Core ML and AVFoundation) into a
 *        Metal texture that can be consumed by the dispatcher.
 *
 * The function creates a texture view of the pixel buffer without copying data.
 *
 * @param[in]  dispatcher   Dispatcher instance.
 * @param[in]  pixel_buffer CVPixelBufferRef containing the image.
 * @param[out] out_texture  Receives a texture handle that references the pixel buffer.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_INVALID_ARGUMENT if the pixel buffer format is unsupported.
 */
TK_NODISCARD tk_error_code_t
tk_metal_dispatch_pixelbuffer_to_texture(tk_metal_dispatcher_t *dispatcher,
                                         CVPixelBufferRef pixel_buffer,
                                         tk_gpu_texture_t *out_texture);

/**
 * @brief Export a Metal texture back to a CVPixelBufferRef.
 *
 * Useful when the result of a Core ML model (or a custom Metal kernel) must be
 * handed back to AVFoundation for display or further processing.
 *
 * @param[in]  dispatcher   Dispatcher instance.
 * @param[in]  texture      Texture containing the image data.
 * @param[out] out_pixel_buffer  Receives a newly created CVPixelBufferRef.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_GPU_MEMORY if the pixel buffer cannot be allocated.
 */
TK_NODISCARD tk_error_code_t
tk_metal_dispatch_texture_to_pixelbuffer(tk_metal_dispatcher_t *dispatcher,
                                         tk_gpu_texture_t texture,
                                         CVPixelBufferRef *out_pixel_buffer);

/**
 * @brief Execute a Core ML model directly on a Metal texture.
 *
 * The function creates an MLCustomLayer that wraps the supplied Core ML model
 * (MLModel) and runs it on the provided texture.  The output is written to a
 * destination texture supplied by the caller.
 *
 * @param[in] dispatcher   Dispatcher instance.
 * @param[in] model        Pre‑loaded Core ML model (MLModel *).
 * @param[in] src_texture  Input texture.
 * @param[in] dst_texture  Destination texture for the model output.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_INVALID_ARGUMENT if the model or textures are incompatible.
 */
TK_NODISCARD tk_error_code_t
tk_metal_dispatch_coreml_inference(tk_metal_dispatcher_t *dispatcher,
                                   void *model,               /* MLModel * (opaque) */
                                   tk_gpu_texture_t src_texture,
                                   tk_gpu_texture_t dst_texture);

/**
 * @brief Compile a Metal Shading Language (MSL) kernel from source at runtime.
 *
 * This enables on‑the‑fly generation of specialised compute kernels (e.g.
 * custom activation functions).  The compiled pipeline state is returned as an
 * opaque handle that can be used with tk_metal_dispatch_launch_custom_kernel().
 *
 * @param[in]  dispatcher   Dispatcher instance.
 * @param[in]  msl_source   Null‑terminated string containing the MSL source.
 * @param[in]  entry_point  Name of the kernel function within the source.
 * @param[out] out_pipeline_handle  Receives the compiled pipeline handle.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_COMPILATION_FAILED if the shader fails to compile.
 */
TK_NODISCARD tk_error_code_t
tk_metal_dispatch_compile_kernel(tk_metal_dispatcher_t *dispatcher,
                                 const char *msl_source,
                                 const char *entry_point,
                                 void **out_pipeline_handle);

/**
 * @brief Launch a previously compiled custom kernel.
 *
 * The caller provides a list of buffer and texture handles that will be bound
 * to the kernel according to the layout described in the MSL source.
 *
 * @param[in] dispatcher   Dispatcher instance.
 * @param[in] pipeline     Opaque handle returned by tk_metal_dispatch_compile_kernel().
 * @param[in] grid_x       Number of threadgroups in the X dimension.
 * @param[in] grid_y       Number of threadgroups in the Y dimension.
 * @param[in] grid_z       Number of threadgroups in the Z dimension.
 * @param[in] args         Pointer to a struct containing kernel arguments
 *                          (device pointers, scalars, etc.).  The struct layout
 *                          must match the kernel’s argument list.
 *
 * @return TK_SUCCESS if the kernel launch was successfully queued.
 */
TK_NODISCARD tk_error_code_t
tk_metal_dispatch_launch_custom_kernel(tk_metal_dispatcher_t *dispatcher,
                                       void *pipeline,
                                       uint32_t grid_x,
                                       uint32_t grid_y,
                                       uint32_t grid_z,
                                       const void *args);

/* -------------------------------------------------------------------------
 * Utility Functions – optional helpers for debugging and profiling
 * ------------------------------------------------------------------------- */

/**
 * @brief Retrieve the underlying MTLDevice pointer for advanced use‑cases.
 *
 * The returned object is retained by the dispatcher; callers must not release it.
 *
 * @param[in] dispatcher   Dispatcher instance.
 * @return id<MTLDevice>   The Metal device used by the dispatcher.
 */
id<MTLDevice>
tk_metal_dispatch_get_device(tk_metal_dispatcher_t *dispatcher);

/**
 * @brief Enable Metal Performance Shaders (MPS) profiling for the next N frames.
 *
 * This is useful during development to collect GPU kernel execution times.
 *
 * @param[in] dispatcher   Dispatcher instance.
 * @param[in] enable       true to enable profiling, false to disable.
 * @param[in] frame_count  Number of frames to profile (0 = unlimited).
 *
 * @return TK_SUCCESS on success.
 */
TK_NODISCARD tk_error_code_t
tk_metal_dispatch_set_mps_profiling(tk_metal_dispatcher_t *dispatcher,
                                    bool enable,
                                    uint32_t frame_count);

/* -------------------------------------------------------------------------
 * End of API
 * ------------------------------------------------------------------------- */

#ifdef __cplusplus
}
#endif

#endif /* TRACKIELLM_GPU_METAL_TK_METAL_DISPATCH_H */
