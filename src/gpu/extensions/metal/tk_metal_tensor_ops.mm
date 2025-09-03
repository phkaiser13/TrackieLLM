/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_metal_tensor_ops.mm
 *
 * Implementation of the Metal‑backed tensor primitives declared in
 * tk_metal_tensor_ops.h.  The code follows the same design philosophy as the
 * CUDA/ROCm back‑ends: a thin, well‑documented C‑style API that forwards the
 * heavy lifting to Metal compute kernels (or to Metal Performance Shaders for
 * matrix multiplication).  All interaction with the GPU is performed through
 * the Metal dispatcher (tk_metal_dispatch) which abstracts the MTLDevice,
 * MTLCommandQueue and pipeline‑state management.
 *
 * Dependencies:
 *   - tk_metal_tensor_ops.h
 *   - tk_metal_dispatch.h
 *   - Metal/Metal.h
 *   - MetalPerformanceShaders/MetalPerformanceShaders.h
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#import "tk_metal_tensor_ops.h"
#import "tk_metal_dispatch.h"

/* -------------------------------------------------------------------------
 *  Helper macros – they keep the code readable and guarantee that every
 *  public entry point returns a well‑defined tk_status_t.
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
 *  The dispatcher owns the pipeline objects; we only retain them for the
 *  duration of the command encoding.
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
 *  pipeline, the source/destination tensor descriptors and an optional block
 *  of constant parameters (packed into a struct that matches the Metal
 *  shader layout).  This function hides the repetitive boiler‑plate required
 *  for every kernel launch.
 * ------------------------------------------------------------------------- */
static tk_status_t
tk_metal_encode_and_dispatch(id<MTLComputePipelineState> pipeline,
                             const tk_tensor_descriptor_t *src0,
                             const tk_tensor_descriptor_t *src1,
                             const tk_tensor_descriptor_t *dst,
                             const void *extra_params,
                             size_t extra_params_size,
                             MTLSize threadgroupSize,
                             MTLSize threadgroupCount)
{
    /* -----------------------------------------------------------------
     *  1️⃣  Acquire a command queue from the dispatcher.
     * ----------------------------------------------------------------- */
    id<MTLCommandQueue> queue = nil;
    tk_status_t status = tk_metal_dispatch_get_command_queue(&queue);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);
    RETURN_IF_NULL(queue, TK_STATUS_INTERNAL_ERROR);

    /* -----------------------------------------------------------------
     *  2️⃣  Create a command buffer and a compute encoder.
     * ----------------------------------------------------------------- */
    id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
    if (!cmdBuffer) {
        return TK_STATUS_DISPATCH_FAILURE;
    }

    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    if (!encoder) {
        return TK_STATUS_DISPATCH_FAILURE;
    }

    [encoder setComputePipelineState:pipeline];

    /* -----------------------------------------------------------------
     *  3️⃣  Bind the buffers.  The Metal kernels expect the buffers at
     *      deterministic argument indices:
     *        0 – first tensor (or nullptr if not used)
     *        1 – second tensor (or nullptr)
     *        2 – destination tensor
     *        3 – constant parameters (if any)
     * ----------------------------------------------------------------- */
    uint32_t bindIdx = 0;
    if (src0) {
        id<MTLBuffer> buf0 = (id<MTLBuffer>)src0->buffer.handle;
        [encoder setBuffer:buf0 offset:0 atIndex:bindIdx++];
    }
    if (src1) {
        id<MTLBuffer> buf1 = (id<MTLBuffer>)src1->buffer.handle;
        [encoder setBuffer:buf1 offset:0 atIndex:bindIdx++];
    }
    if (dst) {
        id<MTLBuffer> outBuf = (id<MTLBuffer>)dst->buffer.handle;
        [encoder setBuffer:outBuf offset:0 atIndex:bindIdx++];
    }

    if (extra_params && extra_params_size > 0) {
        /* -----------------------------------------------------------------
         *  For small constant buffers we can use `setBytes:` which copies the
         *  data directly into the argument table – no extra GPU allocation
         *  required.
         * ----------------------------------------------------------------- */
        [encoder setBytes:extra_params length:extra_params_size atIndex:bindIdx];
    }

    /* -----------------------------------------------------------------
     *  4️⃣  Dispatch the compute grid.
     * ----------------------------------------------------------------- */
    [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    /* -----------------------------------------------------------------
     *  5️⃣  Submit and wait for completion.  In a production system we would
     *      use callbacks, but for the sake of a simple, deterministic API we
     *      block here.
     * ----------------------------------------------------------------- */
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];

    if (cmdBuffer.error) {
        return TK_STATUS_DISPATCH_FAILURE;
    }
    return TK_STATUS_OK;
}

/* -------------------------------------------------------------------------
 *  ELEMENT‑WISE ARITHMETIC (FLOAT32 ONLY)
 * ------------------------------------------------------------------------- */
tk_status_t
tk_metal_tensor_add(const tk_tensor_descriptor_t *a,
                    const tk_tensor_descriptor_t *b,
                    const tk_tensor_descriptor_t *out)
{
    RETURN_IF_NULL(a,   TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(b,   TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(out, TK_STATUS_INVALID_ARGUMENT);

    /* Shapes, data type and element count must match. */
    RETURN_IF_FALSE(a->num_dimensions == b->num_dimensions &&
                    a->num_dimensions == out->num_dimensions,
                    TK_STATUS_INVALID_ARGUMENT);
    for (uint32_t i = 0; i < a->num_dimensions; ++i) {
        RETURN_IF_FALSE(a->dimensions[i] == b->dimensions[i] &&
                        a->dimensions[i] == out->dimensions[i],
                        TK_STATUS_INVALID_ARGUMENT);
    }
    RETURN_IF_FALSE(a->data_type == TK_DATA_TYPE_FLOAT32 &&
                    b->data_type == TK_DATA_TYPE_FLOAT32 &&
                    out->data_type == TK_DATA_TYPE_FLOAT32,
                    TK_STATUS_INVALID_ARGUMENT);

    id<MTLComputePipelineState> pipeline = nil;
    tk_status_t status = tk_metal_get_pipeline("tensor_add", &pipeline);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    /* The kernel works on a flat 1‑D grid of elements. */
    const uint64_t elementCount = out->data_size_bytes / sizeof(float);
    const uint32_t threadsPerGroup = 256;
    MTLSize tgSize = MTLSizeMake(threadsPerGroup, 1, 1);
    MTLSize tgCount = MTLSizeMake((elementCount + threadsPerGroup - 1) / threadsPerGroup,
                                  1, 1);

    return tk_metal_encode_and_dispatch(pipeline,
                                        a, b, out,
                                        NULL, 0,
                                        tgSize, tgCount);
}

/* Subtraction – identical validation, different kernel name. */
tk_status_t
tk_metal_tensor_subtract(const tk_tensor_descriptor_t *a,
                         const tk_tensor_descriptor_t *b,
                         const tk_tensor_descriptor_t *out)
{
    RETURN_IF_NULL(a,   TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(b,   TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(out, TK_STATUS_INVALID_ARGUMENT);

    RETURN_IF_FALSE(a->num_dimensions == b->num_dimensions &&
                    a->num_dimensions == out->num_dimensions,
                    TK_STATUS_INVALID_ARGUMENT);
    for (uint32_t i = 0; i < a->num_dimensions; ++i) {
        RETURN_IF_FALSE(a->dimensions[i] == b->dimensions[i] &&
                        a->dimensions[i] == out->dimensions[i],
                        TK_STATUS_INVALID_ARGUMENT);
    }
    RETURN_IF_FALSE(a->data_type == TK_DATA_TYPE_FLOAT32 &&
                    b->data_type == TK_DATA_TYPE_FLOAT32 &&
                    out->data_type == TK_DATA_TYPE_FLOAT32,
                    TK_STATUS_INVALID_ARGUMENT);

    id<MTLComputePipelineState> pipeline = nil;
    tk_status_t status = tk_metal_get_pipeline("tensor_subtract", &pipeline);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    const uint64_t elementCount = out->data_size_bytes / sizeof(float);
    const uint32_t threadsPerGroup = 256;
    MTLSize tgSize = MTLSizeMake(threadsPerGroup, 1, 1);
    MTLSize tgCount = MTLSizeMake((elementCount + threadsPerGroup - 1) / threadsPerGroup,
                                  1, 1);

    return tk_metal_encode_and_dispatch(pipeline,
                                        a, b, out,
                                        NULL, 0,
                                        tgSize, tgCount);
}

/* Multiplication – element‑wise. */
tk_status_t
tk_metal_tensor_multiply(const tk_tensor_descriptor_t *a,
                         const tk_tensor_descriptor_t *b,
                         const tk_tensor_descriptor_t *out)
{
    RETURN_IF_NULL(a,   TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(b,   TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(out, TK_STATUS_INVALID_ARGUMENT);

    RETURN_IF_FALSE(a->num_dimensions == b->num_dimensions &&
                    a->num_dimensions == out->num_dimensions,
                    TK_STATUS_INVALID_ARGUMENT);
    for (uint32_t i = 0; i < a->num_dimensions; ++i) {
        RETURN_IF_FALSE(a->dimensions[i] == b->dimensions[i] &&
                        a->dimensions[i] == out->dimensions[i],
                        TK_STATUS_INVALID_ARGUMENT);
    }
    RETURN_IF_FALSE(a->data_type == TK_DATA_TYPE_FLOAT32 &&
                    b->data_type == TK_DATA_TYPE_FLOAT32 &&
                    out->data_type == TK_DATA_TYPE_FLOAT32,
                    TK_STATUS_INVALID_ARGUMENT);

    id<MTLComputePipelineState> pipeline = nil;
    tk_status_t status = tk_metal_get_pipeline("tensor_multiply", &pipeline);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    const uint64_t elementCount = out->data_size_bytes / sizeof(float);
    const uint32_t threadsPerGroup = 256;
    MTLSize tgSize = MTLSizeMake(threadsPerGroup, 1, 1);
    MTLSize tgCount = MTLSizeMake((elementCount + threadsPerGroup - 1) / threadsPerGroup,
                                  1, 1);

    return tk_metal_encode_and_dispatch(pipeline,
                                        a, b, out,
                                        NULL, 0,
                                        tgSize, tgCount);
}

/* Division – element‑wise. */
tk_status_t
tk_metal_tensor_divide(const tk_tensor_descriptor_t *a,
                       const tk_tensor_descriptor_t *b,
                       const tk_tensor_descriptor_t *out)
{
    RETURN_IF_NULL(a,   TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(b,   TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(out, TK_STATUS_INVALID_ARGUMENT);

    RETURN_IF_FALSE(a->num_dimensions == b->num_dimensions &&
                    a->num_dimensions == out->num_dimensions,
                    TK_STATUS_INVALID_ARGUMENT);
    for (uint32_t i = 0; i < a->num_dimensions; ++i) {
        RETURN_IF_FALSE(a->dimensions[i] == b->dimensions[i] &&
                        a->dimensions[i] == out->dimensions[i],
                        TK_STATUS_INVALID_ARGUMENT);
    }
    RETURN_IF_FALSE(a->data_type == TK_DATA_TYPE_FLOAT32 &&
                    b->data_type == TK_DATA_TYPE_FLOAT32 &&
                    out->data_type == TK_DATA_TYPE_FLOAT32,
                    TK_STATUS_INVALID_ARGUMENT);

    id<MTLComputePipelineState> pipeline = nil;
    tk_status_t status = tk_metal_get_pipeline("tensor_divide", &pipeline);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    const uint64_t elementCount = out->data_size_bytes / sizeof(float);
    const uint32_t threadsPerGroup = 256;
    MTLSize tgSize = MTLSizeMake(threadsPerGroup, 1, 1);
    MTLSize tgCount = MTLSizeMake((elementCount + threadsPerGroup - 1) / threadsPerGroup,
                                  1, 1);

    return tk_metal_encode_and_dispatch(pipeline,
                                        a, b, out,
                                        NULL, 0,
                                        tgSize, tgCount);
}

/* -------------------------------------------------------------------------
 *  SCALAR OPERATIONS
 * ------------------------------------------------------------------------- */
tk_status_t
tk_metal_tensor_add_scalar(const tk_tensor_descriptor_t *src,
                           const tk_scalar_f32_t *scalar,
                           const tk_tensor_descriptor_t *out)
{
    RETURN_IF_NULL(src,    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(scalar, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(out,    TK_STATUS_INVALID_ARGUMENT);

    RETURN_IF_FALSE(src->num_dimensions == out->num_dimensions,
                    TK_STATUS_INVALID_ARGUMENT);
    for (uint32_t i = 0; i < src->num_dimensions; ++i) {
        RETURN_IF_FALSE(src->dimensions[i] == out->dimensions[i],
                        TK_STATUS_INVALID_ARGUMENT);
    }
    RETURN_IF_FALSE(src->data_type == TK_DATA_TYPE_FLOAT32 &&
                    out->data_type == TK_DATA_TYPE_FLOAT32,
                    TK_STATUS_INVALID_ARGUMENT);

    id<MTLComputePipelineState> pipeline = nil;
    tk_status_t status = tk_metal_get_pipeline("tensor_add_scalar", &pipeline);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    const uint64_t elementCount = out->data_size_bytes / sizeof(float);
    const uint32_t threadsPerGroup = 256;
    MTLSize tgSize = MTLSizeMake(threadsPerGroup, 1, 1);
    MTLSize tgCount = MTLSizeMake((elementCount + threadsPerGroup - 1) / threadsPerGroup,
                                  1, 1);

    return tk_metal_encode_and_dispatch(pipeline,
                                        src, NULL, out,
                                        scalar, sizeof(*scalar),
                                        tgSize, tgCount);
}

tk_status_t
tk_metal_tensor_mul_scalar(const tk_tensor_descriptor_t *src,
                           const tk_scalar_f32_t *scalar,
                           const tk_tensor_descriptor_t *out)
{
    RETURN_IF_NULL(src,    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(scalar, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(out,    TK_STATUS_INVALID_ARGUMENT);

    RETURN_IF_FALSE(src->num_dimensions == out->num_dimensions,
                    TK_STATUS_INVALID_ARGUMENT);
    for (uint32_t i = 0; i < src->num_dimensions; ++i) {
        RETURN_IF_FALSE(src->dimensions[i] == out->dimensions[i],
                        TK_STATUS_INVALID_ARGUMENT);
    }
    RETURN_IF_FALSE(src->data_type == TK_DATA_TYPE_FLOAT32 &&
                    out->data_type == TK_DATA_TYPE_FLOAT32,
                    TK_STATUS_INVALID_ARGUMENT);

    id<MTLComputePipelineState> pipeline = nil;
    tk_status_t status = tk_metal_get_pipeline("tensor_mul_scalar", &pipeline);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    const uint64_t elementCount = out->data_size_bytes / sizeof(float);
    const uint32_t threadsPerGroup = 256;
    MTLSize tgSize = MTLSizeMake(threadsPerGroup, 1, 1);
    MTLSize tgCount = MTLSizeMake((elementCount + threadsPerGroup - 1) / threadsPerGroup,
                                  1, 1);

    return tk_metal_encode_and_dispatch(pipeline,
                                        src, NULL, out,
                                        scalar, sizeof(*scalar),
                                        tgSize, tgCount);
}

/* -------------------------------------------------------------------------
 *  MATRIX MULTIPLICATION – the performance‑critical primitive.
 *
 *  We delegate the heavy lifting to Metal Performance Shaders (MPS) because
 *  MPS implements a highly tuned GEMM that automatically selects the best
 *  vector width, threadgroup size and uses the Apple‑metal matrix cores when
 *  available.  The dispatcher supplies a MTLDevice; we create an MPSMatrixMultiplication
 *  object on‑the‑fly, encode it into the command buffer and let MPS handle the
 *  rest.
 * ------------------------------------------------------------------------- */
tk_status_t
tk_metal_tensor_matmul(const tk_tensor_descriptor_t *a,
                       const tk_tensor_descriptor_t *b,
                       const tk_tensor_descriptor_t *out)
{
    /* -----------------------------------------------------------------
     *  Validation – we only support float32 matrices interpreted as
     *  row‑major 2‑D tensors.
     * ----------------------------------------------------------------- */
    RETURN_IF_NULL(a,   TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(b,   TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(out, TK_STATUS_INVALID_ARGUMENT);

    RETURN_IF_FALSE(a->num_dimensions == 2 && b->num_dimensions == 2 &&
                    out->num_dimensions == 2,
                    TK_STATUS_INVALID_ARGUMENT);

    RETURN_IF_FALSE(a->data_type == TK_DATA_TYPE_FLOAT32 &&
                    b->data_type == TK_DATA_TYPE_FLOAT32 &&
                    out->data_type == TK_DATA_TYPE_FLOAT32,
                    TK_STATUS_INVALID_ARGUMENT);

    uint32_t M = a->dimensions[0];          // rows of A
    uint32_t K = a->dimensions[1];          // cols of A / rows of B
    uint32_t N = b->dimensions[1];          // cols of B

    RETURN_IF_FALSE(b->dimensions[0] == K,
                    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(out->dimensions[0] == M && out->dimensions[1] == N,
                    TK_STATUS_INVALID_ARGUMENT);

    /* -----------------------------------------------------------------
     *  1️⃣  Obtain a command buffer from the dispatcher.
     * ----------------------------------------------------------------- */
    id<MTLCommandQueue> queue = nil;
    tk_status_t status = tk_metal_dispatch_get_command_queue(&queue);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);
    RETURN_IF_NULL(queue, TK_STATUS_INTERNAL_ERROR);

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) {
        return TK_STATUS_DISPATCH_FAILURE;
    }

    /* -----------------------------------------------------------------
     *  2️⃣  Wrap the raw GPU buffers into MPSMatrix objects.  MPS expects
     *      a descriptor that tells it the stride (rowBytes) and the matrix
     *      dimensions.
     * ----------------------------------------------------------------- */
    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                        columns:K
                                                                      rowBytes:K * sizeof(float)
                                                                    dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                                                        columns:N
                                                                      rowBytes:N * sizeof(float)
                                                                    dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                        columns:N
                                                                      rowBytes:N * sizeof(float)
                                                                    dataType:MPSDataTypeFloat32];

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:(id<MTLBuffer>)a->buffer.handle
                                            descriptor:descA];
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:(id<MTLBuffer>)b->buffer.handle
                                            descriptor:descB];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:(id<MTLBuffer>)out->buffer.handle
                                            descriptor:descC];

    /* -----------------------------------------------------------------
     *  3️⃣  Create the MPSMatrixMultiplication kernel.  The alpha and beta
     *      scalars implement the classic GEMM equation:
     *          C = alpha * A * B + beta * C
     *      For a pure matmul we use alpha = 1.0, beta = 0.0.
     * ----------------------------------------------------------------- */
    MPSMatrixMultiplication *gemm = [[MPSMatrixMultiplication alloc]
        initWithDevice:queue.device
        transposeLeft:false
        transposeRight:false
        resultRows:M
        resultColumns:N
        interiorColumns:K
        alpha:1.0
        beta:0.0];

    /* -----------------------------------------------------------------
     *  4️⃣  Encode the operation.
     * ----------------------------------------------------------------- */
    [gemm encodeToCommandBuffer:cmd
                     leftMatrix:matA
                    rightMatrix:matB
                   resultMatrix:matC];

    /* -----------------------------------------------------------------
     *  5️⃣  Submit and wait.
     * ----------------------------------------------------------------- */
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.error) {
        return TK_STATUS_DISPATCH_FAILURE;
    }

    /* -----------------------------------------------------------------
     *  Cleanup – MPS objects are ARC‑managed; we simply release the local
     *  references (no explicit GPU deallocation required).
     * ----------------------------------------------------------------- */
    return TK_STATUS_OK;
}

/* -------------------------------------------------------------------------
 *  TRANSPOSE – swaps the two innermost dimensions of a 2‑D tensor.
 * ------------------------------------------------------------------------- */
tk_status_t
tk_metal_tensor_transpose(const tk_tensor_descriptor_t *src,
                          const tk_tensor_descriptor_t *out)
{
    RETURN_IF_NULL(src, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(out, TK_STATUS_INVALID_ARGUMENT);

    RETURN_IF_FALSE(src->num_dimensions == 2 && out->num_dimensions == 2,
                    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(src->dimensions[0] == out->dimensions[1] &&
                    src->dimensions[1] == out->dimensions[0],
                    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(src->data_type == TK_DATA_TYPE_FLOAT32 &&
                    out->data_type == TK_DATA_TYPE_FLOAT32,
                    TK_STATUS_INVALID_ARGUMENT);

    id<MTLComputePipelineState> pipeline = nil;
    tk_status_t status = tk_metal_get_pipeline("tensor_transpose", &pipeline);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    const uint32_t rows = src->dimensions[0];
    const uint32_t cols = src->dimensions[1];
    const uint32_t threadsPerGroup = 16;   // 16×16 threadgroup is a common choice
    MTLSize tgSize = MTLSizeMake(threadsPerGroup, threadsPerGroup, 1);
    MTLSize tgCount = MTLSizeMake((cols + threadsPerGroup - 1) / threadsPerGroup,
                                  (rows + threadsPerGroup - 1) / threadsPerGroup,
                                  1);

    return tk_metal_encode_and_dispatch(pipeline,
                                        src, NULL, out,
                                        NULL, 0,
                                        tgSize, tgCount);
}

/* -------------------------------------------------------------------------
 *  ACTIVATION FUNCTIONS (ELEMENT‑WISE)
 * ------------------------------------------------------------------------- */
tk_status_t
tk_metal_tensor_relu(const tk_tensor_descriptor_t *src,
                     const tk_tensor_descriptor_t *out)
{
    RETURN_IF_NULL(src, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(out, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(src->num_dimensions == out->num_dimensions,
                    TK_STATUS_INVALID_ARGUMENT);
    for (uint32_t i = 0; i < src->num_dimensions; ++i) {
        RETURN_IF_FALSE(src->dimensions[i] == out->dimensions[i],
                        TK_STATUS_INVALID_ARGUMENT);
    }
    RETURN_IF_FALSE(src->data_type == TK_DATA_TYPE_FLOAT32 &&
                    out->data_type == TK_DATA_TYPE_FLOAT32,
                    TK_STATUS_INVALID_ARGUMENT);

    id<MTLComputePipelineState> pipeline = nil;
    tk_status_t status = tk_metal_get_pipeline("tensor_relu", &pipeline);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    const uint64_t elementCount = out->data_size_bytes / sizeof(float);
    const uint32_t threadsPerGroup = 256;
    MTLSize tgSize = MTLSizeMake(threadsPerGroup, 1, 1);
    MTLSize tgCount = MTLSizeMake((elementCount + threadsPerGroup - 1) / threadsPerGroup,
                                  1, 1);

    return tk_metal_encode_and_dispatch(pipeline,
                                        src, NULL, out,
                                        NULL, 0,
                                        tgSize, tgCount);
}

tk_status_t
tk_metal_tensor_sigmoid(const tk_tensor_descriptor_t *src,
                        const tk_tensor_descriptor_t *out)
{
    RETURN_IF_NULL(src, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(out, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(src->num_dimensions == out->num_dimensions,
                    TK_STATUS_INVALID_ARGUMENT);
    for (uint32_t i = 0; i < src->num_dimensions; ++i) {
        RETURN_IF_FALSE(src->dimensions[i] == out->dimensions[i],
                        TK_STATUS_INVALID_ARGUMENT);
    }
    RETURN_IF_FALSE(src->data_type == TK_DATA_TYPE_FLOAT32 &&
                    out->data_type == TK_DATA_TYPE_FLOAT32,
                    TK_STATUS_INVALID_ARGUMENT);

    id<MTLComputePipelineState> pipeline = nil;
    tk_status_t status = tk_metal_get_pipeline("tensor_sigmoid", &pipeline);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    const uint64_t elementCount = out->data_size_bytes / sizeof(float);
    const uint32_t threadsPerGroup = 256;
    MTLSize tgSize = MTLSizeMake(threadsPerGroup, 1, 1);
    MTLSize tgCount = MTLSizeMake((elementCount + threadsPerGroup - 1) / threadsPerGroup,
                                  1, 1);

    return tk_metal_encode_and_dispatch(pipeline,
                                        src, NULL, out,
                                        NULL, 0,
                                        tgSize, tgCount);
}

tk_status_t
tk_metal_tensor_tanh(const tk_tensor_descriptor_t *src,
                     const tk_tensor_descriptor_t *out)
{
    RETURN_IF_NULL(src, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(out, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(src->num_dimensions == out->num_dimensions,
                    TK_STATUS_INVALID_ARGUMENT);
    for (uint32_t i = 0; i < src->num_dimensions; ++i) {
        RETURN_IF_FALSE(src->dimensions[i] == out->dimensions[i],
                        TK_STATUS_INVALID_ARGUMENT);
    }
    RETURN_IF_FALSE(src->data_type == TK_DATA_TYPE_FLOAT32 &&
                    out->data_type == TK_DATA_TYPE_FLOAT32,
                    TK_STATUS_INVALID_ARGUMENT);

    id<MTLComputePipelineState> pipeline = nil;
    tk_status_t status = tk_metal_get_pipeline("tensor_tanh", &pipeline);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    const uint64_t elementCount = out->data_size_bytes / sizeof(float);
    const uint32_t threadsPerGroup = 256;
    MTLSize tgSize = MTLSizeMake(threadsPerGroup, 1, 1);
    MTLSize tgCount = MTLSizeMake((elementCount + threadsPerGroup - 1) / threadsPerGroup,
                                  1, 1);

    return tk_metal_encode_and_dispatch(pipeline,
                                        src, NULL, out,
                                        NULL, 0,
                                        tgSize, tgCount);
}

/* Softmax is a two‑step operation (max reduction + exponentiation + sum
 * reduction + normalization).  For simplicity we implement it as a single
 * kernel that internally performs the reductions in shared memory.  The
 * kernel name is "tensor_softmax".
 */
tk_status_t
tk_metal_tensor_softmax(const tk_tensor_descriptor_t *src,
                        const tk_tensor_descriptor_t *out)
{
    RETURN_IF_NULL(src, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(out, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(src->num_dimensions == out->num_dimensions,
                    TK_STATUS_INVALID_ARGUMENT);
    for (uint32_t i = 0; i < src->num_dimensions; ++i) {
        RETURN_IF_FALSE(src->dimensions[i] == out->dimensions[i],
                        TK_STATUS_INVALID_ARGUMENT);
    }
    RETURN_IF_FALSE(src->data_type == TK_DATA_TYPE_FLOAT32 &&
                    out->data_type == TK_DATA_TYPE_FLOAT32,
                    TK_STATUS_INVALID_ARGUMENT);

    id<MTLComputePipelineState> pipeline = nil;
    tk_status_t status = tk_metal_get_pipeline("tensor_softmax", &pipeline);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    const uint64_t elementCount = out->data_size_bytes / sizeof(float);
    const uint32_t threadsPerGroup = 256;
    MTLSize tgSize = MTLSizeMake(threadsPerGroup, 1, 1);
    MTLSize tgCount = MTLSizeMake((elementCount + threadsPerGroup - 1) / threadsPerGroup,
                                  1, 1);

    return tk_metal_encode_and_dispatch(pipeline,
                                        src, NULL, out,
                                        NULL, 0,
                                        tgSize, tgCount);
}

/* -------------------------------------------------------------------------
 *  POOLING OPERATIONS – we support 2‑D max and average pooling.
 * ------------------------------------------------------------------------- */
tk_status_t
tk_metal_tensor_max_pool(const tk_tensor_descriptor_t *src,
                         const tk_pooling_params_t *params,
                         const tk_tensor_descriptor_t *out)
{
    RETURN_IF_NULL(src,    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(params, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(out,    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(src->num_dimensions == 4 && out->num_dimensions == 4,
                    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(src->data_type == TK_DATA_TYPE_FLOAT32 &&
                    out->data_type == TK_DATA_TYPE_FLOAT32,
                    TK_STATUS_INVALID_ARGUMENT);

    id<MTLComputePipelineState> pipeline = nil;
    tk_status_t status = tk_metal_get_pipeline("tensor_max_pool", &pipeline);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    /* The kernel expects the pooling parameters packed into a struct that
     * mirrors the Metal shader layout. */
    struct {
        uint32_t kernelW;
        uint32_t kernelH;
        uint32_t strideX;
        uint32_t strideY;
        uint32_t padX;
        uint32_t padY;
    } poolParams = {
        params->kernel_width,
        params->kernel_height,
        params->stride_x,
        params->stride_y,
        params->padding_x,
        params->padding_y
    };

    /* Compute threadgroup size – we process one output pixel per thread. */
    const uint32_t outW = out->dimensions[3];
    const uint32_t outH = out->dimensions[2];
    const uint32_t threadsPerGroup = 16;
    MTLSize tgSize = MTLSizeMake(threadsPerGroup, threadsPerGroup, 1);
    MTLSize tgCount = MTLSizeMake((outW + threadsPerGroup - 1) / threadsPerGroup,
                                  (outH + threadsPerGroup - 1) / threadsPerGroup,
                                  out->dimensions[0] * out->dimensions[1]); /* batch * channels */

    return tk_metal_encode_and_dispatch(pipeline,
                                        src, NULL, out,
                                        &poolParams, sizeof(poolParams),
                                        tgSize, tgCount);
}

tk_status_t
tk_metal_tensor_avg_pool(const tk_tensor_descriptor_t *src,
                         const tk_pooling_params_t *params,
                         const tk_tensor_descriptor_t *out)
{
    RETURN_IF_NULL(src,    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(params, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(out,    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(src->num_dimensions == 4 && out->num_dimensions == 4,
                    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(src->data_type == TK_DATA_TYPE_FLOAT32 &&
                    out->data_type == TK_DATA_TYPE_FLOAT32,
                    TK_STATUS_INVALID_ARGUMENT);

    id<MTLComputePipelineState> pipeline = nil;
    tk_status_t status = tk_metal_get_pipeline("tensor_avg_pool", &pipeline);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    struct {
        uint32_t kernelW;
        uint32_t kernelH;
        uint32_t strideX;
        uint32_t strideY;
        uint32_t padX;
        uint32_t padY;
    } poolParams = {
        params->kernel_width,
        params->kernel_height,
        params->stride_x,
        params->stride_y,
        params->padding_x,
        params->padding_y
    };

    const uint32_t outW = out->dimensions[3];
    const uint32_t outH = out->dimensions[2];
    const uint32_t threadsPerGroup = 16;
    MTLSize tgSize = MTLSizeMake(threadsPerGroup, threadsPerGroup, 1);
    MTLSize tgCount = MTLSizeMake((outW + threadsPerGroup - 1) / threadsPerGroup,
                                  (outH + threadsPerGroup - 1) / threadsPerGroup,
                                  out->dimensions[0] * out->dimensions[1]);

    return tk_metal_encode_and_dispatch(pipeline,
                                        src, NULL, out,
                                        &poolParams, sizeof(poolParams),
                                        tgSize, tgCount);
}

/* -------------------------------------------------------------------------
 *  LAYER NORMALIZATION – normalizes across the channel dimension.
 * ------------------------------------------------------------------------- */
tk_status_t
tk_metal_tensor_layer_norm(const tk_tensor_descriptor_t *src,
                           const tk_layer_norm_params_t *params,
                           const tk_tensor_descriptor_t *out)
{
    RETURN_IF_NULL(src,    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(params, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(out,    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(src->num_dimensions == 4 && out->num_dimensions == 4,
                    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(src->data_type == TK_DATA_TYPE_FLOAT32 &&
                    out->data_type == TK_DATA_TYPE_FLOAT32,
                    TK_STATUS_INVALID_ARGUMENT);

    id<MTLComputePipelineState> pipeline = nil;
    tk_status_t status = tk_metal_get_pipeline("tensor_layer_norm", &pipeline);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    struct {
        float epsilon;
    } lnParams = { params->epsilon };

    const uint32_t batch   = src->dimensions[0];
    const uint32_t channels = src->dimensions[1];
    const uint32_t height  = src->dimensions[2];
    const uint32_t width   = src->dimensions[3];
    const uint64_t elementCount = (uint64_t)batch * channels * height * width;

    const uint32_t threadsPerGroup = 256;
    MTLSize tgSize = MTLSizeMake(threadsPerGroup, 1, 1);
    MTLSize tgCount = MTLSizeMake((elementCount + threadsPerGroup - 1) / threadsPerGroup,
                                  1, 1);

    return tk_metal_encode_and_dispatch(pipeline,
                                        src, NULL, out,
                                        &lnParams, sizeof(lnParams),
                                        tgSize, tgCount);
}

/* -------------------------------------------------------------------------
 *  RESHAPE – logical view change only.  No GPU work is required; we simply
 *  verify that the total element count matches and copy the descriptor.
 * ------------------------------------------------------------------------- */
tk_status_t
tk_metal_tensor_reshape(const tk_tensor_descriptor_t *src,
                        const tk_tensor_descriptor_t *out)
{
    RETURN_IF_NULL(src, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(out, TK_STATUS_INVALID_ARGUMENT);

    /* Compute total element count for both tensors. */
    uint64_t srcCount = 1;
    for (uint32_t i = 0; i < src->num_dimensions; ++i) {
        srcCount *= src->dimensions[i];
    }

    uint64_t outCount = 1;
    for (uint32_t i = 0; i < out->num_dimensions; ++i) {
        outCount *= out->dimensions[i];
    }

    RETURN_IF_FALSE(srcCount == outCount,
                    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(src->data_type == out->data_type,
                    TK_STATUS_INVALID_ARGUMENT);
    /* No GPU kernel – the buffers are identical, only the descriptor changes. */
    return TK_STATUS_OK;
}

/* -------------------------------------------------------------------------
 *  CONCATENATION – joins a list of tensors along a user‑specified axis.
 *  The implementation copies each input tensor into the appropriate slice of
 *  the output buffer using a simple copy kernel.  For brevity we implement a
 *  generic “tensor_concat” kernel that receives:
 *      - src buffer pointer
 *      - dst buffer pointer
 *      - src offset (in elements)
 *      - copy length (in elements)
 *  The host loops over the input tensors and dispatches the kernel for each
 *  slice.
 * ------------------------------------------------------------------------- */
tk_status_t
tk_metal_tensor_concat(const tk_tensor_descriptor_t * const *tensors,
                       uint32_t tensor_count,
                       const tk_concat_params_t *params,
                       const tk_tensor_descriptor_t *out)
{
    RETURN_IF_NULL(tensors,    TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(params,     TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_NULL(out,        TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(tensor_count > 0, TK_STATUS_INVALID_ARGUMENT);
    RETURN_IF_FALSE(params->axis < out->num_dimensions,
                    TK_STATUS_INVALID_ARGUMENT);

    /* Verify that all tensors share the same shape except along the concat
     * axis and that they all have the same data type. */
    tk_data_type_t dtype = tensors[0]->data_type;
    uint32_t expectedDims[4];
    for (uint32_t d = 0; d < out->num_dimensions; ++d) {
        expectedDims[d] = out->dimensions[d];
    }

    uint32_t concatDimSize = 0;
    for (uint32_t i = 0; i < tensor_count; ++i) {
        const tk_tensor_descriptor_t *t = tensors[i];
        RETURN_IF_FALSE(t->num_dimensions == out->num_dimensions,
                        TK_STATUS_INVALID_ARGUMENT);
        RETURN_IF_FALSE(t->data_type == dtype,
                        TK_STATUS_INVALID_ARGUMENT);
        for (uint32_t d = 0; d < out->num_dimensions; ++d) {
            if (d == params->axis) continue;
            RETURN_IF_FALSE(t->dimensions[d] == expectedDims[d],
                            TK_STATUS_INVALID_ARGUMENT);
        }
        concatDimSize += t->dimensions[params->axis];
    }
    RETURN_IF_FALSE(concatDimSize == out->dimensions[params->axis],
                    TK_STATUS_INVALID_ARGUMENT);

    /* Load the generic concat kernel. */
    id<MTLComputePipelineState> pipeline = nil;
    tk_status_t status = tk_metal_get_pipeline("tensor_concat", &pipeline);
    RETURN_IF_FALSE(status == TK_STATUS_OK, status);

    /* Helper to compute the linear offset of a multi‑dimensional index. */
    auto linear_offset = [&](const tk_tensor_descriptor_t *desc,
                             uint32_t axisPos) -> uint64_t {
        uint64_t offset = 0;
        uint64_t stride = 1;
        for (int32_t d = (int32_t)desc->num_dimensions - 1; d >= 0; --d) {
            uint32_t dim = desc->dimensions[d];
            uint32_t idx = (d == (int32_t)params->axis) ? axisPos : 0;
            offset += (uint64_t)idx * stride;
            stride *= dim;
        }
        return offset;
    };

    uint64_t dstOffsetElements = 0;   /* Offset inside the destination buffer. */

    for (uint32_t i = 0; i < tensor_count; ++i) {
        const tk_tensor_descriptor_t *src = tensors[i];
        uint64_t srcElements = src->data_size_bytes / sizeof(float);
        uint64_t dstElements = srcElements;   /* Same number of elements are copied. */

        /* Pack kernel arguments – we use a small struct that mirrors the Metal
         * shader layout. */
        struct {
            uint64_t srcOffset;   // in elements
            uint64_t dstOffset;   // in elements
            uint64_t length;      // number of elements to copy
        } copyParams = {
            .srcOffset = 0,
            .dstOffset = dstOffsetElements,
            .length    = srcElements
        };

        /* Dispatch a 1‑D grid where each thread copies one element. */
        const uint32_t threadsPerGroup = 256;
        MTLSize tgSize = MTLSizeMake(threadsPerGroup, 1, 1);
        MTLSize tgCount = MTLSizeMake((srcElements + threadsPerGroup - 1) / threadsPerGroup,
                                      1, 1);

        tk_status_t rc = tk_metal_encode_and_dispatch(pipeline,
                                                      src, NULL, out,
                                                      &copyParams, sizeof(copyParams),
                                                      tgSize, tgCount);
        if (rc != TK_STATUS_OK) {
            return rc;
        }

        dstOffsetElements += srcElements;
    }

    return TK_STATUS_OK;
}

/* -------------------------------------------------------------------------
 *  END OF FILE
 * ------------------------------------------------------------------------- */
