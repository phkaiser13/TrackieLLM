/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_metal_image_ops.mm
 *
 * Implementation of the Metal‑backed image‑processing API declared in
 * tk_metal_image_ops.h.  The code is deliberately verbose and heavily
 * commented to serve both as production‑ready code and as a teaching
 * reference for future contributors.  All heavy lifting is delegated to the
 * Metal dispatcher (tk_metal_dispatch) which hides the raw MTLDevice,
 * MTLCommandQueue and pipeline‑state management.
 *
 * Dependencies:
 *   - tk_metal_image_ops.h
 *   - tk_metal_dispatch.h
 *   - Metal/Metal.h
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "tk_metal_image_ops.h"
#import "tk_metal_dispatch.h"

/* -------------------------------------------------------------------------
 *  Helper macros – they make the code easier to read and guarantee that we
 *  always return a proper tk_status_t on error.
 * ------------------------------------------------------------------------- */
#define RETURN_IF_NULL(ptr, err)               \
    do {                                       \
        if ((ptr) == NULL) return (err);       \
    } while (0)

#define RETURN_IF_FALSE(cond, err)             \
    do {                                       \
        if (!(cond)) return (err);             \
    } while (0)

/* -------------------------------------------------------------------------
 *  Internal utility: fetch a compiled compute pipeline from the dispatcher.
 *  The dispatcher owns the MTLComputePipelineState objects; we only retain
 *  them for the duration of the command encoding.
 * ------------------------------------------------------------------------- */
static tk_status_t
tk_metal_get_pipeline(const char *kernel_name,
                      id<MTLComputePipelineState> *outPipeline)
{
    RETURN_IF_NULL(kernel_name, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(outPipeline, TK_STATUS_INVALID_ARGUMENT);

    tk_status_t status = tk_metal_dispatch_get_compute_pipeline(kernel_name,
                                                               outPipeline);
    if (status != TK_STATUS_OK) {
        return TK_STATUS_SHADER_NOT_FOUND;
    }
    return TK_STATUS_OK;
}

/* -------------------------------------------------------------------------
 *  Internal utility: encode a generic compute pass.  The caller supplies the
 *  pipeline, a list of GPU buffers (including the image descriptors) and the
 *  thread‑group dimensions.  This function hides the repetitive boiler‑plate
 *  required for every kernel launch.
 * ------------------------------------------------------------------------- */
static tk_status_t
tk_metal_encode_and_dispatch(id<MTLComputePipelineState> pipeline,
                             const tk_image_descriptor_t *src,
                             const tk_image_descriptor_t *dst,
                             const void *extra_params,
                             size_t extra_params_size,
                             MTLSize threadgroupSize,
                             MTLSize threadgroupCount)
{
    id<MTLCommandQueue> queue = nil;
    tk_status_t status = tk_metal_dispatch_get_command_queue(&queue);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);
    RETURN_IF_NULL(queue, TK_STATUS_INTERNAL_ERROR);

    id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
    if (!cmdBuffer) {
        return TK_STATUS_DISPATCH_FAILURE;
    }

    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    if (!encoder) {
        return TK_STATUS_DISPATCH_FAILURE;
    }

    [encoder setComputePipelineState:pipeline];

    /* Bind the source and destination GPU buffers.  The dispatcher guarantees
     * that tk_gpu_buffer_t contains a valid MTLBuffer* in its `handle` field.
     */
    id<MTLBuffer> srcBuf = (id<MTLBuffer>)src->buffer.handle;
    id<MTLBuffer> dstBuf = (id<MTLBuffer>)dst->buffer.handle;
    [encoder setBuffer:srcBuf offset:0 atIndex:0];
    [encoder setBuffer:dstBuf offset:0 atIndex:1];

    /* If the kernel needs extra parameters (e.g. convolution kernel,
     * transformation matrix, …) we upload them into a temporary GPU buffer.
     * The dispatcher provides a cheap “scratch” allocation for exactly this
     * purpose.
     */
    if (extra_params && extra_params_size > 0) {
        tk_gpu_buffer_t tmp;
        status = tk_metal_dispatch_allocate_scratch_buffer(extra_params_size, &tmp);
        RETURN_IF_FALSE(status == TK_STATUS_OK, status);
        memcpy(tmp.handle.contents, extra_params, extra_params_size);
        [encoder setBuffer:(id<MTLBuffer>)tmp.handle offset:0 atIndex:2];
        // The scratch buffer will be released automatically when the command
        // buffer completes (dispatcher owns its lifetime).
    }

    /* Dispatch – the threadgroup size is chosen by the kernel author; we
     * simply forward the values supplied by the caller.
     */
    [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];

    if (cmdBuffer.error) {
        return TK_STATUS_DISPATCH_FAILURE;
    }
    return TK_STATUS_OK;
}

/* -------------------------------------------------------------------------
 *  PUBLIC FUNCTIONS – each follows the same pattern:
 *    1. Validate arguments.
 *    2. Acquire the appropriate compute pipeline.
 *    3. Prepare any kernel‑specific parameter structures.
 *    4. Call tk_metal_encode_and_dispatch().
 * ------------------------------------------------------------------------- */

tk_status_t
tk_metal_image_separable_convolution(const tk_image_descriptor_t *src,
                                     const tk_image_descriptor_t *dst,
                                     const tk_convolution_params_t *horiz,
                                     const tk_convolution_params_t *vert)
{
    /* -----------------------------------------------------------------
     *  Argument validation – we keep the checks lightweight because the
     *  higher‑level code already performs extensive validation.  Still,
     *  we guard against NULL pointers and mismatched image formats.
     * ----------------------------------------------------------------- */
    RETURN_IF_NULL(src,  TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(dst,  TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(horiz, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(vert,  TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(src->width  == dst->width &&
                    src->height == dst->height,
                    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(src->format == dst->format,
                    TK_STATUS_INVALID_ARGUMENT);

    /* -----------------------------------------------------------------
     *  Load the two kernels: one for the horizontal pass and one for the
     *  vertical pass.  The shader source lives in tk_metal_kernels.metal and
     *  is compiled into a metallib that the dispatcher knows about.
     * ----------------------------------------------------------------- */
    id<MTLComputePipelineState> horizPipeline = nil;
    id<MTLComputePipelineState> vertPipeline  = nil;
    tk_status_t status = tk_metal_get_pipeline("separable_convolution_horiz",
                                               &horizPipeline);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);
    status = tk_metal_get_pipeline("separable_convolution_vert",
                                   &vertPipeline);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    /* -----------------------------------------------------------------
     *  Temporary intermediate buffer – the horizontal pass writes into a
     *  temporary GPU buffer of the same size as the source image.  The
     *  dispatcher provides a fast allocation routine that re‑uses a pool.
     * ----------------------------------------------------------------- */
    tk_gpu_buffer_t tmp;
    size_t tmpSize = src->data_size_bytes;
    status = tk_metal_dispatch_allocate_buffer(tmpSize, &tmp);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    /* -----------------------------------------------------------------
     *  Encode horizontal pass.
     * ----------------------------------------------------------------- */
    {
        MTLSize tgSize = MTLSizeMake(16, 16, 1);   // chosen by kernel author
        MTLSize tgCount = MTLSizeMake(
            (src->width  + tgSize.width  - 1) / tgSize.width,
            (src->height + tgSize.height - 1) / tgSize.height,
            1);

        /* Pack the convolution parameters into a struct that matches the
         * Metal shader layout.  The struct is defined in the .metal file;
         * we duplicate it here for clarity.
         */
        struct {
            uint32_t kernelWidth;
            uint32_t kernelHeight;
            uint32_t padding;          // keep 16‑byte alignment
            float    divisor;
            float    offset;
            float    kernel[64];        // static upper bound (adjust as needed)
        } horizParams = {0};

        horizParams.kernelWidth  = horiz->kernel_width;
        horizParams.kernelHeight = horiz->kernel_height;
        horizParams.divisor      = horiz->divisor;
        horizParams.offset       = horiz->offset;
        memcpy(horizParams.kernel, horiz->kernel,
               sizeof(float) * horiz->kernel_width * horiz->kernel_height);

        /* Encode the pass – note that we bind the temporary buffer as the
         * destination (index 1) and the source image as the input (index 0).
         */
        id<MTLBuffer> tmpBuf = (id<MTLBuffer>)tmp.handle;
        id<MTLCommandQueue> queue = nil;
        status = tk_metal_dispatch_get_command_queue(&queue);
        RETURN_IF_FALSE(status == TK_STATUS_OK, status);

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:horizPipeline];
        [enc setBuffer:(id<MTLBuffer>)src->buffer.handle offset:0 atIndex:0];
        [enc setBuffer:tmpBuf offset:0 atIndex:1];
        [enc setBytes:&horizParams length:sizeof(horizParams) atIndex:2];
        [enc dispatchThreadgroups:tgCount threadsPerThreadgroup:tgSize];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) {
            tk_metal_dispatch_release_buffer(tmp);
            return TK_STATUS_DISPATCH_FAILURE;
        }
    }

    /* -----------------------------------------------------------------
     *  Encode vertical pass – now the temporary buffer is the source and the
     *  final destination buffer is `dst`.
     * ----------------------------------------------------------------- */
    {
        MTLSize tgSize = MTLSizeMake(16, 16, 1);
        MTLSize tgCount = MTLSizeMake(
            (dst->width  + tgSize.width  - 1) / tgSize.width,
            (dst->height + tgSize.height - 1) / tgSize.height,
            1);

        struct {
            uint32_t kernelWidth;
            uint32_t kernelHeight;
            uint32_t padding;
            float    divisor;
            float    offset;
            float    kernel[64];
        } vertParams = {0};

        vertParams.kernelWidth  = vert->kernel_width;
        vertParams.kernelHeight = vert->kernel_height;
        vertParams.divisor      = vert->divisor;
        vertParams.offset       = vert->offset;
        memcpy(vertParams.kernel, vert->kernel,
               sizeof(float) * vert->kernel_width * vert->kernel_height);

        id<MTLCommandQueue> queue = nil;
        status = tk_metal_dispatch_get_command_queue(&queue);
        RETURN_IF_FALSE(status == TK_STATUS_OK, status);

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:vertPipeline];
        [enc setBuffer:(id<MTLBuffer>)tmp.handle offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)dst->buffer.handle offset:0 atIndex:1];
        [enc setBytes:&vertParams length:sizeof(vertParams) atIndex:2];
        [enc dispatchThreadgroups:tgCount threadsPerThreadgroup:tgSize];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) {
            tk_metal_dispatch_release_buffer(tmp);
            return TK_STATUS_DISPATCH_FAILURE;
        }
    }

    /* -----------------------------------------------------------------
     *  Clean‑up temporary buffer.
     * ----------------------------------------------------------------- */
    tk_metal_dispatch_release_buffer(tmp);
    return TK_STATUS_OK;
}

/* -------------------------------------------------------------------------
 *  Sobel edge detection – a single kernel that reads the source image and
 *  writes the gradient magnitude.  The kernel name in the metallib is
 *  "sobel_edge_detection".
 * ------------------------------------------------------------------------- */
tk_status_t
tk_metal_image_sobel_edge_detection(const tk_image_descriptor_t *src,
                                    const tk_image_descriptor_t *dst)
{
    RETURN_IF_NULL(src, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(dst, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(src->width  == dst->width &&
                    src->height == dst->height,
                    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(src->format == TK_IMAGE_FORMAT_GRAYSCALE_F32 &&
                    dst->format == TK_IMAGE_FORMAT_GRAYSCALE_F32,
                    TK_STATUS_INVALID_ARGUMENT);

    id<MTLComputePipelineState> pipeline = nil;
    tk_status_t status = tk_metal_get_pipeline("sobel_edge_detection",
                                               &pipeline);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    MTLSize tgSize = MTLSizeMake(16, 16, 1);
    MTLSize tgCount = MTLSizeMake(
        (src->width  + tgSize.width  - 1) / tgSize.width,
        (src->height + tgSize.height - 1) / tgSize.height,
        1);

    return tk_metal_encode_and_dispatch(pipeline,
                                        src,
                                        dst,
                                        NULL,
                                        0,
                                        tgSize,
                                        tgCount);
}

/* -------------------------------------------------------------------------
 *  Bilateral filter – the kernel expects two floats packed into a small
 *  constant buffer: spatialSigma and intensitySigma.
 * ------------------------------------------------------------------------- */
tk_status_t
tk_metal_image_bilateral_filter(const tk_image_descriptor_t *src,
                                const tk_image_descriptor_t *dst,
                                const tk_convolution_params_t *params)
{
    RETURN_IF_NULL(src,    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(dst,    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(params, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(src->width  == dst->width &&
                    src->height == dst->height,
                    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(src->format == dst->format,
                    TK_STATUS_INVALID_ARGUMENT);

    id<MTLComputePipelineState> pipeline = nil;
    tk_status_t status = tk_metal_get_pipeline("bilateral_filter",
                                               &pipeline);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    struct {
        float spatialSigma;
        float intensitySigma;
    } bilateralParams = {0};

    bilateralParams.spatialSigma   = params->kernel[0];
    bilateralParams.intensitySigma = params->kernel[1];

    MTLSize tgSize = MTLSizeMake(16, 16, 1);
    MTLSize tgCount = MTLSizeMake(
        (src->width  + tgSize.width  - 1) / tgSize.width,
        (src->height + tgSize.height - 1) / tgSize.height,
        1);

    return tk_metal_encode_and_dispatch(pipeline,
                                        src,
                                        dst,
                                        &bilateralParams,
                                        sizeof(bilateralParams),
                                        tgSize,
                                        tgCount);
}

/* -------------------------------------------------------------------------
 *  Morphology operation – the kernel receives a struct with the operation
 *  type and radius.  The Metal shader name is "morphology_op".
 * ------------------------------------------------------------------------- */
tk_status_t
tk_metal_image_morphology_operation(const tk_image_descriptor_t *src,
                                    const tk_image_descriptor_t *dst,
                                    const tk_morphology_params_t *params)
{
    RETURN_IF_NULL(src,    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(dst,    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(params, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(src->width  == dst->width &&
                    src->height == dst->height,
                    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(src->format == dst->format,
                    TK_STATUS_INVALID_ARGUMENT);

    id<MTLComputePipelineState> pipeline = nil;
    tk_status_t status = tk_metal_get_pipeline("morphology_op",
                                               &pipeline);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    struct {
        uint32_t op;      // matches tk_morphology_op_t
        uint32_t radius;
    } morphParams = { (uint32_t)params->op, params->radius };

    MTLSize tgSize = MTLSizeMake(16, 16, 1);
    MTLSize tgCount = MTLSizeMake(
        (src->width  + tgSize.width  - 1) / tgSize.width,
        (src->height + tgSize.height - 1) / tgSize.height,
        1);

    return tk_metal_encode_and_dispatch(pipeline,
                                        src,
                                        dst,
                                        &morphParams,
                                        sizeof(morphParams),
                                        tgSize,
                                        tgCount);
}

/* -------------------------------------------------------------------------
 *  Color‑space conversion – a single kernel that switches on the conversion
 *  enum.  The kernel name is "color_space_convert".
 * ------------------------------------------------------------------------- */
tk_status_t
tk_metal_image_color_space_conversion(const tk_image_descriptor_t *src,
                                      const tk_image_descriptor_t *dst,
                                      tk_color_conversion_t conversion)
{
    RETURN_IF_NULL(src, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(dst, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(src->width  == dst->width &&
                    src->height == dst->height,
                    TK_STATUS_INVALID_ARGUMENT);

    id<MTLComputePipelineState> pipeline = nil;
    tk_status_t status = tk_metal_get_pipeline("color_space_convert",
                                               &pipeline);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    uint32_t conv = (uint32_t)conversion;

    MTLSize tgSize = MTLSizeMake(16, 16, 1);
    MTLSize tgCount = MTLSizeMake(
        (src->width  + tgSize.width  - 1) / tgSize.width,
        (src->height + tgSize.height - 1) / tgSize.height,
        1);

    return tk_metal_encode_and_dispatch(pipeline,
                                        src,
                                        dst,
                                        &conv,
                                        sizeof(conv),
                                        tgSize,
                                        tgCount);
}

/* -------------------------------------------------------------------------
 *  Compute histogram – the kernel writes a 32‑bit bin count for each bin.
 *  The destination buffer must be a GPU buffer (not an image descriptor)
 *  because the histogram is a 1‑D array.
 * ------------------------------------------------------------------------- */
tk_status_t
tk_metal_image_compute_histogram(const tk_image_descriptor_t *src,
                                 tk_gpu_buffer_t histogram_buffer,
                                 uint32_t num_bins)
{
    RETURN_IF_NULL(src, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(num_bins > 0, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(histogram_buffer.handle, TK_STATUS_INVALID_ARGUMENT);

    id<MTLComputePipelineState> pipeline = nil;
    tk_status_t status = tk_metal_get_pipeline("compute_histogram",
                                               &pipeline);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    struct {
        uint32_t numBins;
    } histParams = { num_bins };

    /* The kernel expects three buffers:
     *   0 – source image data,
     *   1 – destination histogram buffer,
     *   2 – constant parameters.
     */
    id<MTLCommandQueue> queue = nil;
    status = tk_metal_dispatch_get_command_queue(&queue);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:(id<MTLBuffer>)src->buffer.handle offset:0 atIndex:0];
    [enc setBuffer:(id<MTLBuffer>)histogram_buffer.handle offset:0 atIndex:1];
    [enc setBytes:&histParams length:sizeof(histParams) atIndex:2];

    MTLSize tgSize = MTLSizeMake(256, 1, 1);   // 256 threads per group
    MTLSize tgCount = MTLSizeMake(
        (src->width * src->height + tgSize.width - 1) / tgSize.width,
        1,
        1);

    [enc dispatchThreadgroups:tgCount threadsPerThreadgroup:tgSize];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.error) {
        return TK_STATUS_DISPATCH_FAILURE;
    }
    return TK_STATUS_OK;
}

/* -------------------------------------------------------------------------
 *  Histogram equalization – reads the pre‑computed histogram and writes a
 *  remapped image.  The kernel name is "histogram_equalization".
 * ------------------------------------------------------------------------- */
tk_status_t
tk_metal_image_histogram_equalization(const tk_image_descriptor_t *src,
                                      const tk_image_descriptor_t *dst,
                                      tk_gpu_buffer_t histogram_buffer,
                                      uint32_t num_bins)
{
    RETURN_IF_NULL(src, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(dst, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(histogram_buffer.handle, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(num_bins > 0, TK_STATUS_INVALID_ARGUMENT);

    id<MTLComputePipelineState> pipeline = nil;
    tk_status_t status = tk_metal_get_pipeline("histogram_equalization",
                                               &pipeline);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    struct {
        uint32_t numBins;
    } params = { num_bins };

    MTLSize tgSize = MTLSizeMake(16, 16, 1);
    MTLSize tgCount = MTLSizeMake(
        (src->width  + tgSize.width  - 1) / tgSize.width,
        (src->height + tgSize.height - 1) / tgSize.height,
        1);

    /* Encode – three buffers: src image, dst image, histogram. */
    id<MTLCommandQueue> queue = nil;
    status = tk_metal_dispatch_get_command_queue(&queue);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:(id<MTLBuffer>)src->buffer.handle offset:0 atIndex:0];
    [enc setBuffer:(id<MTLBuffer>)dst->buffer.handle offset:0 atIndex:1];
    [enc setBuffer:(id<MTLBuffer>)histogram_buffer.handle offset:0 atIndex:2];
    [enc setBytes:&params length:sizeof(params) atIndex:3];
    [enc dispatchThreadgroups:tgCount threadsPerThreadgroup:tgSize];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.error) {
        return TK_STATUS_DISPATCH_FAILURE;
    }
    return TK_STATUS_OK;
}

/* -------------------------------------------------------------------------
 *  Geometric transform – the kernel receives a 3×3 affine matrix plus a few
 *  integer flags.  The kernel name is "geometric_transform".
 * ------------------------------------------------------------------------- */
tk_status_t
tk_metal_image_geometric_transform(const tk_image_descriptor_t *src,
                                   const tk_image_descriptor_t *dst,
                                   const tk_geometric_transform_t *transform)
{
    RETURN_IF_NULL(src, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(dst, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(transform, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(src->format == dst->format,
                    TK_STATUS_INVALID_ARGUMENT);

    id<MTLComputePipelineState> pipeline = nil;
    tk_status_t status = tk_metal_get_pipeline("geometric_transform",
                                               &pipeline);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    /* The Metal shader expects the matrix as a flat array of 9 floats,
     * followed by two 32‑bit integers (interpolation, border mode).
     */
    struct {
        float matrix[9];
        int32_t interpolation;
        int32_t borderMode;
    } params = {0};

    memcpy(params.matrix, transform->matrix, sizeof(float) * 9);
    params.interpolation = transform->interpolation;
    params.borderMode    = transform->border_mode;

    MTLSize tgSize = MTLSizeMake(16, 16, 1);
    MTLSize tgCount = MTLSizeMake(
        (dst->width  + tgSize.width  - 1) / tgSize.width,
        (dst->height + tgSize.height - 1) / tgSize.height,
        1);

    return tk_metal_encode_and_dispatch(pipeline,
                                        src,
                                        dst,
                                        &params,
                                        sizeof(params),
                                        tgSize,
                                        tgCount);
}

/* -------------------------------------------------------------------------
 *  Harris corner detection – the kernel computes the Harris response map.
 *  The kernel name is "harris_corner_detection".
 * ------------------------------------------------------------------------- */
tk_status_t
tk_metal_image_harris_corner_detection(const tk_image_descriptor_t *src,
                                       const tk_image_descriptor_t *dst,
                                       const tk_harris_params_t *params)
{
    RETURN_IF_NULL(src,    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(dst,    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(params, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(src->width  == dst->width &&
                    src->height == dst->height,
                    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(src->format == TK_IMAGE_FORMAT_GRAYSCALE_F32 &&
                    dst->format == TK_IMAGE_FORMAT_GRAYSCALE_F32,
                    TK_STATUS_INVALID_ARGUMENT);

    id<MTLComputePipelineState> pipeline = nil;
    tk_status_t status = tk_metal_get_pipeline("harris_corner_detection",
                                               &pipeline);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    struct {
        float k;
        float threshold;
        uint32_t blockSize;
    } harrisParams = {
        params->k,
        params->threshold,
        params->block_size
    };

    MTLSize tgSize = MTLSizeMake(16, 16, 1);
    MTLSize tgCount = MTLSizeMake(
        (src->width  + tgSize.width  - 1) / tgSize.width,
        (src->height + tgSize.height - 1) / tgSize.height,
        1);

    return tk_metal_encode_and_dispatch(pipeline,
                                        src,
                                        dst,
                                        &harrisParams,
                                        sizeof(harrisParams),
                                        tgSize,
                                        tgCount);
}

/* -------------------------------------------------------------------------
 *  End of file.
 * ------------------------------------------------------------------------- */
