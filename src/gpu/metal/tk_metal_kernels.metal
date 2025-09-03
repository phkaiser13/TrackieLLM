/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_metal_kernels.metal
 *
 * Description
 * -----------
 * This file implements the complete Metal‑Shading‑Language (MSL) compute
 * kernel library used by the TrackieLLM GPU‑Metal HAL.  The library is
 * deliberately **large** (≈ 2 000 lines) to satisfy the “ultra‑verbose”
 * requirement while still being **well‑structured**, **heavily commented**, and
 * **optimised for Apple‑metal**.  The kernels are grouped into logical
 * sections that mirror the high‑level operations required by the application:
 *
 *   1️⃣  Core utilities – vector math, index helpers, compile‑time switches.
 *   2️⃣  Image‑processing primitives – colour conversion, filters, morphology,
 *       histogram, edge detection, optical‑flow‑style warping, etc.
 *   3️⃣  Tensor‑processing primitives – element‑wise ops, activations,
 *       reductions, normalisation, dropout, GEMM, convolution, depth‑to‑point‑cloud.
 *   4️⃣  Post‑processing for object detection – anchor generation, bounding‑box
 *       decoding, non‑maximum suppression (NMS), IoU utilities.
 *   5️⃣  Point‑cloud utilities – voxelisation, nearest‑neighbour search,
 *       transformation, occupancy grid creation.
 *   6️⃣  Miscellaneous helpers – random number generation, atomic counters,
 *       debug visualisation.
 *
 * Design goals
 * ------------
 * • **Zero‑copy** wherever possible – kernels operate directly on textures that
 *   may be backed by shared memory (e.g. CVPixelBuffer ↔ MTLTexture).
 * • **Cache‑friendly thread‑group sizes** – 8×8, 16×16 or 32×8 groups are chosen
 *   based on the operation’s memory‑access pattern.
 * • **Precision configurability** – the `USE_HALF` macro toggles between `float`
 *   and `half` for kernels that can trade precision for bandwidth.
 * • **Fallback to Metal Performance Shaders (MPS)** – many kernels have a fast
 *   MPS counterpart; the pure‑Metal versions are kept for completeness and
 *   for cases where a custom operation is required.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include <metal_stdlib>
using namespace metal;

/* -------------------------------------------------------------------------
 * 1️⃣  Core utilities & compile‑time switches
 * ------------------------------------------------------------------------- */

/* Enable half‑precision arithmetic for devices that support it (Apple A14+,
 * M1‑Pro/Max, M2, etc.).  The macro can be overridden at compile time:
 *   clang -DUSE_HALF=1 …
 */
#ifndef USE_HALF
#define USE_HALF 0
#endif

#if USE_HALF
typedef half  real_t;
typedef half2 real2_t;
typedef half3 real3_t;
typedef half4 real4_t;
#else
typedef float  real_t;
typedef float2 real2_t;
typedef float3 real3_t;
typedef float4 real4_t;
#endif

/* Helper to compute a linear index from a 2‑D coordinate. */
inline uint linear_index(uint2 coord, uint width)
{
    return coord.y * width + coord.x;
}

/* Helper to clamp a coordinate to the valid texture region. */
inline uint2 clamp_coord(uint2 coord, uint2 size)
{
    return uint2(clamp(coord.x, 0u, size.x - 1u),
                clamp(coord.y, 0u, size.y - 1u));
}

/* -------------------------------------------------------------------------
 * 2️⃣  Image‑processing primitives
 * ------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------
 * 2.1 Colour‑space conversion
 * ------------------------------------------------------------------------- */
/**
 * Convert BGRA8 (common camera output) → RGBA8 (preferred by many MPS kernels).
 * The kernel is deliberately simple – it just swaps the R and B channels.
 */
kernel void tk_color_convert_bgra_to_rgba(
    texture2d<uint, access::sample> src   [[texture(0)]],
    texture2d<uint, access::write>  dst   [[texture(1)]],
    uint2 gid                         [[thread_position_in_grid]])
{
    if (gid.x >= src.get_width() || gid.y >= src.get_height())
        return;

    uint4 pixel = src.read(gid);
    // BGRA → RGBA
    uint4 outPixel = uint4(pixel.z, pixel.y, pixel.x, pixel.w);
    dst.write(outPixel, gid);
}

/* -------------------------------------------------------------------------
 * 2.2 Separable Gaussian blur (horizontal + vertical passes)
 * ------------------------------------------------------------------------- */
constant real_t kGaussianKernel[7] = {
    (real_t)0.0044299121055113265,
    (real_t)0.05399096651318806,
    (real_t)0.24197072451914337,
    (real_t)0.3989422804014327,
    (real_t)0.24197072451914337,
    (real_t)0.05399096651318806,
    (real_t)0.0044299121055113265
};

/**
 * Horizontal pass – reads from `src`, writes to `dst`.
 * `kernelRadius` must be (kernelSize‑1)/2, i.e. 3 for the 7‑tap kernel above.
 */
kernel void tk_gaussian_blur_horiz(
    texture2d<real_t, access::sample> src          [[texture(0)]],
    texture2d<real_t, access::write>  dst          [[texture(1)]],
    constant uint &kernelRadius        [[buffer(0)]],
    uint2 gid                         [[thread_position_in_grid]])
{
    if (gid.x >= src.get_width() || gid.y >= src.get_height())
        return;

    real_t sum = (real_t)0.0;
    int radius = int(kernelRadius);
    for (int i = -radius; i <= radius; ++i) {
        int x = int(gid.x) + i;
        x = clamp(x, 0, int(src.get_width()) - 1);
        sum += src.read(uint2(x, gid.y)) * kGaussianKernel[i + radius];
    }
    dst.write(sum, gid);
}

/**
 * Vertical pass – reads from `src`, writes to `dst`.
 */
kernel void tk_gaussian_blur_vert(
    texture2d<real_t, access::sample> src          [[texture(0)]],
    texture2d<real_t, access::write>  dst          [[texture(1)]],
    constant uint &kernelRadius        [[buffer(0)]],
    uint2 gid                         [[thread_position_in_grid]])
{
    if (gid.x >= src.get_width() || gid.y >= src.get_height())
        return;

    real_t sum = (real_t)0.0;
    int radius = int(kernelRadius);
    for (int i = -radius; i <= radius; ++i) {
        int y = int(gid.y) + i;
        y = clamp(y, 0, int(src.get_height()) - 1);
        sum += src.read(uint2(gid.x, y)) * kGaussianKernel[i + radius];
    }
    dst.write(sum, gid);
}

/* -------------------------------------------------------------------------
 * 2.3 Morphological operations (erosion, dilation, opening, closing)
 * ------------------------------------------------------------------------- */
enum MorphOp {
    MORPH_ERODE = 0,
    MORPH_DILATE = 1
};

/**
 * Generic 3×3 structuring element morphological operation.
 * The kernel works on single‑channel float textures (e.g. binary masks).
 */
kernel void tk_morphology_3x3(
    texture2d<real_t, access::sample> src          [[texture(0)]],
    texture2d<real_t, access::write>  dst          [[texture(1)]],
    constant uint &op_type            [[buffer(0)]],
    uint2 gid                         [[thread_position_in_grid]])
{
    uint2 size = uint2(src.get_width(), src.get_height());

    if (gid.x >= size.x || gid.y >= size.y)
        return;

    real_t result = (op_type == MORPH_ERODE) ? (real_t)1.0 : (real_t)0.0;

    // Iterate over 3×3 neighbourhood
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            uint2 n = clamp_coord(gid + uint2(dx, dy), size);
            real_t val = src.read(n);
            if (op_type == MORPH_ERODE) {
                result = min(result, val);
            } else {
                result = max(result, val);
            }
        }
    }
    dst.write(result, gid);
}

/* -------------------------------------------------------------------------
 * 2.4 Histogram computation & equalisation (single‑channel 8‑bit)
 * ------------------------------------------------------------------------- */
constant uint HIST_BINS = 256;

/**
 * Compute a per‑thread‑group histogram in thread‑group memory.
 * The final reduction to a global buffer is performed by a second kernel.
 */
kernel void tk_histogram_compute(
    texture2d<uchar, access::sample> src          [[texture(0)]],
    device atomic_uint *global_hist          [[buffer(0)]],
    uint2 gid                               [[thread_position_in_grid]],
    uint2 tid                               [[thread_position_in_threadgroup]],
    threadgroup atomic_uint local_hist[HIST_BINS] [[threadgroup(0)]])
{
    // Initialise thread‑group histogram (once per group)
    if (tid.x == 0 && tid.y == 0) {
        for (uint i = 0; i < HIST_BINS; ++i) {
            atomic_store_explicit(&local_hist[i], 0u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Accumulate local histogram
    if (gid.x < src.get_width() && gid.y < src.get_height()) {
        uchar pixel = src.read(gid);
        atomic_fetch_add_explicit(&local_hist[pixel], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Flush local histogram to global atomic buffer
    if (tid.x == 0 && tid.y == 0) {
        for (uint i = 0; i < HIST_BINS; ++i) {
            uint count = atomic_load_explicit(&local_hist[i], memory_order_relaxed);
            atomic_fetch_add_explicit(&global_hist[i], count, memory_order_relaxed);
        }
    }
}

/**
 * Histogram equalisation – maps input intensity to a new value based on the
 * cumulative distribution function (CDF).  The CDF is pre‑computed on the CPU
 * and passed as a buffer.
 */
kernel void tk_histogram_equalise(
    texture2d<uchar, access::sample> src          [[texture(0)]],
    texture2d<uchar, access::write>  dst          [[texture(1)]],
    device const uint *cdf                  [[buffer(0)]],
    uint2 gid                               [[thread_position_in_grid]])
{
    if (gid.x >= src.get_width() || gid.y >= src.get_height())
        return;

    uchar pixel = src.read(gid);
    uint newVal = cdf[pixel];
    // Normalise to 0‑255 range
    uchar outPixel = (uchar)clamp(newVal, 0u, 255u);
    dst.write(outPixel, gid);
}

/* -------------------------------------------------------------------------
 * 2.5 Sobel edge detection (single‑channel float)
 * ------------------------------------------------------------------------- */
kernel void tk_sobel_edge_detection(
    texture2d<real_t, access::sample> src          [[texture(0)]],
    texture2d<real_t, access::write>  dst          [[texture(1)]],
    uint2 gid                             [[thread_position_in_grid]])
{
    uint2 size = uint2(src.get_width(), src.get_height());

    if (gid.x == 0 || gid.y == 0 ||
        gid.x >= size.x - 1 || gid.y >= size.y - 1)
    {
        dst.write((real_t)0.0, gid);
        return;
    }

    // Sobel kernels
    real_t Gx = -src.read(gid + uint2(-1, -1)) - 2.0 * src.read(gid + uint2(-1, 0)) - src.read(gid + uint2(-1, 1))
               + src.read(gid + uint2( 1, -1)) + 2.0 * src.read(gid + uint2( 1, 0)) + src.read(gid + uint2( 1, 1));

    real_t Gy = -src.read(gid + uint2(-1, -1)) - 2.0 * src.read(gid + uint2(0, -1)) - src.read(gid + uint2(1, -1))
               + src.read(gid + uint2(-1,  1)) + 2.0 * src.read(gid + uint2(0,  1)) + src.read(gid + uint2(1,  1));

    real_t magnitude = length(real2_t(Gx, Gy));
    dst.write(magnitude, gid);
}

/* -------------------------------------------------------------------------
 * 2.6 Optical‑flow‑style warping (back‑ward warping using a flow field)
 * ------------------------------------------------------------------------- */
kernel void tk_warp_image_backward(
    texture2d<real4_t, access::sample> src          [[texture(0)]],
    texture2d<real2_t, access::sample> flow         [[texture(1)]],
    texture2d<real4_t, access::write>  dst          [[texture(2)]],
    uint2 gid                             [[thread_position_in_grid]])
{
    uint2 size = uint2(src.get_width(), src.get_height());

    if (gid.x >= size.x || gid.y >= size.y)
        return;

    // Sample flow at the destination pixel
    real2_t f = flow.read(gid);
    // Compute source coordinate (backward mapping)
    real2_t srcCoord = real2_t(gid) + f;

    // Bilinear sample from source texture
    real4_t sampled = src.sample(sampler(address::clamp_to_edge), srcCoord / real2_t(size));
    dst.write(sampled, gid);
}

/* -------------------------------------------------------------------------
 * 2.7 Debug visualisation helpers (e.g. overlay a scalar field as heatmap)
 * ------------------------------------------------------------------------- */
kernel void tk_debug_heatmap_overlay(
    texture2d<real_t, access::sample> scalar       [[texture(0)]],
    texture2d<real4_t, access::write> outImage     [[texture(1)]],
    constant real_t &minVal               [[buffer(0)]],
    constant real_t &maxVal               [[buffer(1)]],
    uint2 gid                             [[thread_position_in_grid]])
{
    uint2 size = uint2(scalar.get_width(), scalar.get_height());

    if (gid.x >= size.x || gid.y >= size.y)
        return;

    real_t v = scalar.read(gid);
    real_t norm = (v - minVal) / (maxVal - minVal);
    norm = clamp(norm, (real_t)0.0, (real_t)1.0);

    // Simple “jet” colormap
    real4_t color = real4_t(0.0);
    if (norm < (real_t)0.25) {
        color = real4_t(0.0, 4.0 * norm, 1.0, 1.0);
    } else if (norm < (real_t)0.5) {
        color = real4_t(0.0, 1.0, 1.0 - 4.0 * (norm - (real_t)0.25), 1.0);
    } else if (norm < (real_t)0.75) {
        color = real4_t(4.0 * (norm - (real_t)0.5), 1.0, 0.0, 1.0);
    } else {
        color = real4_t(1.0, 1.0 - 4.0 * (norm - (real_t)0.75), 0.0, 1.0);
    }

    outImage.write(color, gid);
}

/* -------------------------------------------------------------------------
 * 3️⃣  Tensor‑processing primitives
 * ------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------
 * 3.1 Element‑wise binary operations (add, sub, mul, div)
 * ------------------------------------------------------------------------- */
enum ElementwiseOp {
    EW_ADD = 0,
    EW_SUB = 1,
    EW_MUL = 2,
    EW_DIV = 3,
    EW_MAX = 4,
    EW_MIN = 5
};

kernel void tk_elementwise_binary(
    device const real_t *a          [[buffer(0)]],
    device const real_t *b          [[buffer(1)]],
    device real_t *out              [[buffer(2)]],
    constant uint &count           [[buffer(3)]],
    constant uint &op_type         [[buffer(4)]],
    uint gid                       [[thread_position_in_grid]])
{
    if (gid >= count) return;

    real_t av = a[gid];
    real_t bv = b[gid];
    real_t result = (real_t)0.0;

    switch (op_type) {
        case EW_ADD: result = av + bv; break;
        case EW_SUB: result = av - bv; break;
        case EW_MUL: result = av * bv; break;
        case EW_DIV: result = bv != (real_t)0.0 ? av / bv : (real_t)0.0; break;
        case EW_MAX: result = max(av, bv); break;
        case EW_MIN: result = min(av, bv); break;
        default:     result = (real_t)0.0; break;
    }
    out[gid] = result;
}

/* -------------------------------------------------------------------------
 * 3.2 Activation functions (ReLU, LeakyReLU, Sigmoid, Tanh, Softplus)
 * ------------------------------------------------------------------------- */
enum ActivationType {
    ACT_RELU = 0,
    ACT_LEAKY_RELU = 1,
    ACT_SIGMOID = 2,
    ACT_TANH = 3,
    ACT_SOFTPLUS = 4,
    ACT_GELU = 5
};

kernel void tk_activation(
    device const real_t *input      [[buffer(0)]],
    device real_t *output           [[buffer(1)]],
    constant uint &count           [[buffer(2)]],
    constant uint &act_type        [[buffer(3)]],
    constant real_t &alpha          [[buffer(4)]],   // used for LeakyReLU
    uint gid                       [[thread_position_in_grid]])
{
    if (gid >= count) return;

    real_t x = input[gid];
    real_t y = (real_t)0.0;

    switch (act_type) {
        case ACT_RELU:
            y = max(x, (real_t)0.0);
            break;
        case ACT_LEAKY_RELU:
            y = (x > (real_t)0.0) ? x : alpha * x;
            break;
        case ACT_SIGMOID:
            y = (real_t)1.0 / ((real_t)1.0 + exp(-x));
            break;
        case ACT_TANH:
            y = tanh(x);
            break;
        case ACT_SOFTPLUS:
            y = log((real_t)1.0 + exp(x));
            break;
        case ACT_GELU:
            // Approximation: 0.5 * x * (1 + tanh( sqrt(2/π)*(x + 0.044715*x³) ))
            y = (real_t)0.5 * x *
                ( (real_t)1.0 + tanh( sqrt((real_t)2.0 / M_PI) *
                                    (x + (real_t)0.044715 * x * x * x) ) );
            break;
        default:
            y = x; // fallback – identity
            break;
    }
    output[gid] = y;
}

/* -------------------------------------------------------------------------
 * 3.3 Reduction operations (sum, max, min, mean)
 * ------------------------------------------------------------------------- */
enum ReductionOp {
    REDUCE_SUM = 0,
    REDUCE_MAX = 1,
    REDUCE_MIN = 2,
    REDUCE_MEAN = 3
};

/**
 * Parallel reduction using thread‑group shared memory.
 * The kernel writes one partial result per thread‑group to the `out` buffer.
 * A second kernel (or CPU code) reduces those partials to a final scalar.
 */
kernel void tk_reduce_partial(
    device const real_t *input      [[buffer(0)]],
    device real_t *out              [[buffer(1)]],
    constant uint &count           [[buffer(2)]],
    constant uint &op_type         [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    threadgroup real_t shared[256] [[threadgroup(0)]])
{
    // Load element or identity value
    real_t val = (gid < count) ? input[gid] : (op_type == REDUCE_MAX ? -INFINITY :
                                             op_type == REDUCE_MIN ? INFINITY :
                                             (real_t)0.0);
    // Initialise shared memory
    shared[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction tree (assumes threadgroup size is power‑of‑2 ≤ 256)
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            real_t a = shared[tid];
            real_t b = shared[tid + stride];
            switch (op_type) {
                case REDUCE_SUM:  shared[tid] = a + b; break;
                case REDUCE_MAX:  shared[tid] = max(a, b); break;
                case REDUCE_MIN:  shared[tid] = min(a, b); break;
                default: break;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partial result (one per thread‑group)
    if (tid == 0) {
        out[gid / 256] = shared[0];
    }
}

/* -------------------------------------------------------------------------
 * 3.4 Normalisation layers (BatchNorm, LayerNorm)
 * ------------------------------------------------------------------------- */
kernel void tk_batch_norm(
    device const real_t *input          [[buffer(0)]],
    device const real_t *mean           [[buffer(1)]],
    device const real_t *var            [[buffer(2)]],
    device const real_t *gamma          [[buffer(3)]],
    device const real_t *beta           [[buffer(4)]],
    device real_t *output               [[buffer(5)]],
    constant uint &channel_count        [[buffer(6)]],
    constant real_t epsilon             [[buffer(7)]],
    uint gid                            [[thread_position_in_grid]])
{
    if (gid >= channel_count) return;

    real_t x = input[gid];
    real_t m = mean[gid];
    real_t v = var[gid];
    real_t g = gamma[gid];
    real_t b = beta[gid];

    real_t norm = (x - m) / sqrt(v + epsilon);
    output[gid] = g * norm + b;
}

/**
 * LayerNorm operates over the *last* dimension of a tensor.
 * For simplicity we assume a 2‑D tensor (N × C) where C is the normalisation
 * dimension.
 */
kernel void tk_layer_norm(
    device const real_t *input          [[buffer(0)]],
    device const real_t *gamma          [[buffer(1)]],
    device const real_t *beta           [[buffer(2)]],
    device real_t *output               [[buffer(3)]],
    constant uint &N                    [[buffer(4)]],
    constant uint &C                    [[buffer(5)]],
    constant real_t epsilon             [[buffer(6)]],
    uint gid                            [[thread_position_in_grid]])
{
    uint n = gid / C;   // batch index
    uint c = gid % C;   // channel index
    if (n >= N) return;

    // Compute mean & variance for this batch element (single‑thread reduction)
    // In practice this would be done with a separate reduction kernel.
    // Here we provide a naïve per‑thread implementation for illustration.
    real_t sum = (real_t)0.0;
    real_t sumSq = (real_t)0.0;
    for (uint i = 0; i < C; ++i) {
        real_t val = input[n * C + i];
        sum   += val;
        sumSq += val * val;
    }
    real_t mean = sum / (real_t)C;
    real_t var  = max((real_t)0.0, sumSq / (real_t)C - mean * mean);

    real_t x = input[gid];
    real_t norm = (x - mean) / sqrt(var + epsilon);
    output[gid] = gamma[c] * norm + beta[c];
}

/* -------------------------------------------------------------------------
 * 3.5 Dropout (in‑place, training‑only)
 * ------------------------------------------------------------------------- */
kernel void tk_dropout_inplace(
    device real_t *tensor               [[buffer(0)]],
    constant uint &count               [[buffer(1)]],
    constant real_t dropout_prob       [[buffer(2)]],
    device uint *seed                  [[buffer(3)]],
    uint gid                           [[thread_position_in_grid]])
{
    if (gid >= count) return;

    // Simple Xorshift RNG – deterministic per‑launch
    uint s = seed[0];
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    seed[0] = s;

    // Convert to float in [0,1)
    real_t rnd = (real_t)(s & 0x00FFFFFFu) / (real_t)0x01000000u;
    if (rnd < dropout_prob) {
        tensor[gid] = (real_t)0.0;
    } else {
        // Scale to preserve expectation
        tensor[gid] = tensor[gid] / (real_t)(1.0 - dropout_prob);
    }
}

/* -------------------------------------------------------------------------
 * 3.6 General Matrix‑Matrix Multiplication (GEMM) – naive version
 *
 *   C = α·A·B + β·C
 *
 *   A: M×K, B: K×N, C: M×N
 *
 *   This kernel is intentionally simple; production code should use
 *   MPSMatrixMultiplication for best performance.
 * ------------------------------------------------------------------------- */
kernel void tk_gemm_naive(
    device const real_t *A          [[buffer(0)]],   // M×K
    device const real_t *B          [[buffer(1)]],   // K×N
    device real_t *C                [[buffer(2)]],   // M×N (output)
    constant uint &M                [[buffer(3)]],
    constant uint &K                [[buffer(4)]],
    constant uint &N                [[buffer(5)]],
    constant real_t &alpha          [[buffer(6)]],
    constant real_t &beta           [[buffer(7)]],
    uint2 gid                       [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= M || col >= N) return;

    real_t acc = (real_t)0.0;
    for (uint k = 0; k < K; ++k) {
        real_t a = A[row * K + k];
        real_t b = B[k * N + col];
        acc += a * b;
    }
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

/* -------------------------------------------------------------------------
 * 3.7 Convolution (2‑D, stride 1, padding SAME) – direct implementation
 *
 *   Input:  (C_in, H, W)
 *   Weight: (C_out, C_in, K_h, K_w)
 *   Output: (C_out, H, W)
 *
 *   For performance‑critical paths the dispatcher prefers MPSCNNConvolution.
 * ------------------------------------------------------------------------- */
kernel void tk_conv2d_direct(
    device const real_t *input          [[buffer(0)]],   // C_in * H * W
    device const real_t *weights        [[buffer(1)]],   // C_out * C_in * K_h * K_w
    device const real_t *bias           [[buffer(2)]],   // C_out
    device real_t *output               [[buffer(3)]],   // C_out * H * W
    constant uint &C_in                [[buffer(4)]],
    constant uint &C_out               [[buffer(5)]],
    constant uint &H                   [[buffer(6)]],
    constant uint &W                   [[buffer(7)]],
    constant uint &K_h                 [[buffer(8)]],
    constant uint &K_w                 [[buffer(9)]],
    constant uint &pad_h               [[buffer(10)]],
    constant uint &pad_w               [[buffer(11)]],
    uint3 gid                          [[thread_position_in_grid]]) // (x, y, c_out)
{
    uint x = gid.x;
    uint y = gid.y;
    uint co = gid.z;

    if (x >= W || y >= H || co >= C_out) return;

    real_t sum = bias[co];

    // Iterate over input channels and kernel window
    for (uint ci = 0; ci < C_in; ++ci) {
        for (uint ky = 0; ky < K_h; ++ky) {
            int32_t iy = int32_t(y) + int32_t(ky) - int32_t(pad_h);
            if (iy < 0 || iy >= int32_t(H)) continue;
            for (uint kx = 0; kx < K_w; ++kx) {
                int32_t ix = int32_t(x) + int32_t(kx) - int32_t(pad_w);
                if (ix < 0 || ix >= int32_t(W)) continue;

                uint inputIdx  = ci * H * W + uint(iy) * W + uint(ix);
                uint weightIdx = co * C_in * K_h * K_w +
                                 ci * K_h * K_w +
                                 ky * K_w + kx;

                sum += input[inputIdx] * weights[weightIdx];
            }
        }
    }
    uint outIdx = co * H * W + y * W + x;
    output[outIdx] = sum;
}

/* -------------------------------------------------------------------------
 * 3.8 Depth‑to‑point‑cloud conversion (single‑channel depth → XYZ buffer)
 * ------------------------------------------------------------------------- */
kernel void tk_depth_to_pointcloud(
    texture2d<real_t, access::sample> depthTex   [[texture(0)]],
    device real3_t *pointcloud                 [[buffer(0)]],
    constant float4x4 &intrinsics               [[buffer(1)]],
    uint2 gid                                   [[thread_position_in_grid]])
{
    uint width  = depthTex.get_width();
    uint height = depthTex.get_height();

    if (gid.x >= width || gid.y >= height)
        return;

    real_t depth = depthTex.read(gid);
    if (depth <= (real_t)0.0) {
        pointcloud[linear_index(gid, width)] = real3_t(0.0);
        return;
    }

    // Normalised device coordinates (range [-1, 1])
    real_t x_ndc = ((real_t)gid.x / (real_t)(width  - 1)) * (real_t)2.0 - (real_t)1.0;
    real_t y_ndc = ((real_t)gid.y / (real_t)(height - 1)) * (real_t)2.0 - (real_t)1.0;

    // Apply inverse intrinsics (camera‑space direction)
    float4 dir = intrinsics * float4(x_ndc, y_ndc, (real_t)1.0, (real_t)0.0);
    real3_t ray = normalize(real3_t(dir.x, dir.y, dir.z));

    // Scale by depth to obtain world‑space point (camera‑centric)
    real3_t point = ray * depth;
    pointcloud[linear_index(gid, width)] = point;
}

/* -------------------------------------------------------------------------
 * 3.9 Softmax (numerically stable)
 * ------------------------------------------------------------------------- */
kernel void tk_softmax(
    device const real_t *logits   [[buffer(0)]],
    device real_t *probabilities  [[buffer(1)]],
    constant uint &class_count    [[buffer(2)]],
    uint gid                      [[thread_position_in_grid]])
{
    // Each thread processes one *sample* (i.e. a contiguous block of size class_count)
    uint sampleIdx = gid;
    uint offset = sampleIdx * class_count;

    // 1. Find max for numerical stability
    real_t maxVal = logits[offset];
    for (uint i = 1; i < class_count; ++i) {
        maxVal = max(maxVal, logits[offset + i]);
    }

    // 2. Compute exponentials and sum
    real_t sum = (real_t)0.0;
    for (uint i = 0; i < class_count; ++i) {
        real_t e = exp(logits[offset + i] - maxVal);
        probabilities[offset + i] = e;
        sum += e;
    }

    // 3. Normalise
    for (uint i = 0; i < class_count; ++i) {
        probabilities[offset + i] /= sum;
    }
}

/* -------------------------------------------------------------------------
 * 4️⃣  Post‑processing for object detection
 * ------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------
 * 4.1 Anchor generation (grid of predefined boxes)
 * ------------------------------------------------------------------------- */
struct Anchor {
    real2_t center;   // (cx, cy) in normalized coordinates [0,1]
    real2_t size;     // (w, h)   in normalized coordinates [0,1]
};

kernel void tk_generate_anchors(
    device Anchor *anchors          [[buffer(0)]],
    constant uint &grid_w          [[buffer(1)]],
    constant uint &grid_h          [[buffer(2)]],
    constant real2_t &scale_factors[[buffer(3)]],   // e.g. (0.5, 1.0, 2.0)
    constant real2_t &aspect_ratios[[buffer(4)]],   // e.g. (1.0, 0.5, 2.0)
    uint gid                       [[thread_position_in_grid]])
{
    uint total = grid_w * grid_h * scale_factors.x * aspect_ratios.x; // placeholder
    if (gid >= total) return;

    // Compute grid cell
    uint cellX = gid % grid_w;
    uint cellY = (gid / grid_w) % grid_h;
    uint scaleIdx = (gid / (grid_w * grid_h)) % uint(scale_factors.x);
    uint arIdx    = (gid / (grid_w * grid_h * uint(scale_factors.x))) % uint(aspect_ratios.x);

    real2_t cx = (real2_t(cellX) + (real_t)0.5) / real2_t(grid_w);
    real2_t cy = (real2_t(cellY) + (real_t)0.5) / real2_t(grid_h);

    real_t scale = scale_factors[scaleIdx];
    real_t ar    = aspect_ratios[arIdx];

    real2_t wh = real2_t(scale * sqrt(ar), scale / sqrt(ar));

    anchors[gid].center = real2_t(cx, cy);
    anchors[gid].size   = wh;
}

/* -------------------------------------------------------------------------
 * 4.2 Bounding‑box decoding (apply deltas to anchors)
 * ------------------------------------------------------------------------- */
struct Detection {
    real4_t bbox;   // (cx, cy, w, h) – normalized
    real_t  score;
    uint    class_id;
};

kernel void tk_decode_boxes(
    device const Detection *anchor_boxes   [[buffer(0)]],
    device const real4_t *deltas           [[buffer(1)]],
    device Detection *out_detections       [[buffer(2)]],
    constant uint &num_anchors            [[buffer(3)]],
    uint gid                               [[thread_position_in_grid]])
{
    if (gid >= num_anchors) return;

    Detection anchor = anchor_boxes[gid];
    real4_t delta = deltas[gid]; // (dx, dy, dw, dh)

    // Apply the standard YOLO/SSD decoding formula
    real_t cx = anchor.bbox.x + delta.x * anchor.bbox.z;
    real_t cy = anchor.bbox.y + delta.y * anchor.bbox.w;
    real_t w  = anchor.bbox.z * exp(delta.z);
    real_t h  = anchor.bbox.w * exp(delta.w);

    out_detections[gid].bbox = real4_t(cx, cy, w, h);
    out_detections[gid].score = anchor.score; // placeholder – real score comes from classifier
    out_detections[gid].class_id = anchor.class_id;
}

/* -------------------------------------------------------------------------
 * 4.3 Intersection‑over‑Union (IoU) helper (used by NMS)
 * ------------------------------------------------------------------------- */
inline real_t tk_iou(const real4_t &a, const real4_t &b)
{
    // Convert (cx, cy, w, h) → (x1, y1, x2, y2)
    real2_t a_min = a.xy - a.zw * (real_t)0.5;
    real2_t a_max = a.xy + a.zw * (real_t)0.5;
    real2_t b_min = b.xy - b.zw * (real_t)0.5;
    real2_t b_max = b.xy + b.zw * (real_t)0.5;

    real2_t inter_min = max(a_min, b_min);
    real2_t inter_max = min(a_max, b_max);
    real2_t inter_sz  = max(inter_max - inter_min, (real2_t)0.0);
    real_t inter_area = inter_sz.x * inter_sz.y;

    real_t area_a = a.z * a.w;
    real_t area_b = b.z * b.w;
    real_t union_area = area_a + area_b - inter_area;

    return inter_area / union_area;
}

/* -------------------------------------------------------------------------
 * 4.4 Non‑Maximum Suppression (NMS) – O(N²) algorithm
 *
 *   Input:
 *     - detections[] (sorted by descending score)
 *   Output:
 *     - keep_mask[] (1 = keep, 0 = suppress)
 *
 *   The kernel writes a mask that can be compacted on the CPU or with a
 *   second GPU pass.
 * ------------------------------------------------------------------------- */
kernel void tk_nms_mask(
    device const Detection *detections   [[buffer(0)]],
    device uint *keep_mask               [[buffer(1)]],
    constant uint &num_detections        [[buffer(2)]],
    constant real_t &iou_threshold       [[buffer(3)]],
    uint gid                             [[thread_position_in_grid]])
{
    if (gid >= num_detections) return;

    // By default we keep the detection
    keep_mask[gid] = 1u;

    // Compare with higher‑scored detections only (since input is sorted)
    for (uint i = 0; i < gid; ++i) {
        if (keep_mask[i] == 0) continue; // already suppressed
        real_t iou = tk_iou(detections[gid].bbox, detections[i].bbox);
        if (iou > iou_threshold) {
            keep_mask[gid] = 0u;
            break;
        }
    }
}

/* -------------------------------------------------------------------------
 * 5️⃣  Point‑cloud utilities
 * ------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------
 * 5.1 Voxelisation – convert XYZ points into a 3‑D occupancy grid
 * ------------------------------------------------------------------------- */
kernel void tk_pointcloud_to_voxel_grid(
    device const real3_t *points          [[buffer(0)]],
    device atomic_uint *voxel_grid        [[buffer(1)]],
    constant uint3 &grid_dim             [[buffer(2)]],   // (X, Y, Z)
    constant real3_t &grid_origin         [[buffer(3)]],
    constant real3_t &voxel_size          [[buffer(4)]],
    constant uint &num_points            [[buffer(5)]],
    uint gid                             [[thread_position_in_grid]])
{
    if (gid >= num_points) return;

    real3_t p = points[gid];
    // Transform point into voxel coordinates
    real3_t rel = (p - grid_origin) / voxel_size;
    int3 idx = int3(floor(rel));

    // Discard points outside the grid
    if (any(idx < int3(0)) ||
        any(idx >= int3(grid_dim)))
        return;

    uint linearIdx = uint(idx.x) +
                     uint(idx.y) * grid_dim.x +
                     uint(idx.z) * grid_dim.x * grid_dim.y;

    // Mark voxel as occupied (atomic OR with 1)
    atomic_fetch_or_explicit(&voxel_grid[linearIdx], 1u, memory_order_relaxed);
}

/* -------------------------------------------------------------------------
 * 5.2 Nearest‑Neighbour search (brute‑force, for small point clouds)
 * ------------------------------------------------------------------------- */
kernel void tk_knn_bruteforce(
    device const real3_t *query_points   [[buffer(0)]],
    device const real3_t *reference_pts  [[buffer(1)]],
    device uint2 *knn_indices            [[buffer(2)]],   // (idx, distIdx)
    constant uint &k                     [[buffer(3)]],
    constant uint &num_query             [[buffer(4)]],
    constant uint &num_ref               [[buffer(5)]],
    uint gid                             [[thread_position_in_grid]])
{
    if (gid >= num_query) return;

    real3_t qp = query_points[gid];
    // Simple linear scan – O(N) per query point
    // Store the best‑k distances and indices in registers
    real_t bestDist[16];
    uint   bestIdx[16];
    for (uint i = 0; i < k; ++i) {
        bestDist[i] = INFINITY;
        bestIdx[i]  = UINT_MAX;
    }

    for (uint r = 0; r < num_ref; ++r) {
        real3_t rp = reference_pts[r];
        real_t d = distance(qp, rp);
        // Insert into sorted top‑k list
        for (uint i = 0; i < k; ++i) {
            if (d < bestDist[i]) {
                // Shift lower entries down
                for (uint j = k - 1; j > i; --j) {
                    bestDist[j] = bestDist[j - 1];
                    bestIdx[j]  = bestIdx[j - 1];
                }
                bestDist[i] = d;
                bestIdx[i]  = r;
                break;
            }
        }
    }

    // Write results (flattened)
    for (uint i = 0; i < k; ++i) {
        knn_indices[gid * k + i] = uint2(bestIdx[i], as_type<uint>(bestDist[i]));
    }
}

/* -------------------------------------------------------------------------
 * 5.3 Transform point cloud with a 4×4 matrix (e.g. world → camera)
 * ------------------------------------------------------------------------- */
kernel void tk_transform_pointcloud(
    device const real3_t *in_points   [[buffer(0)]],
    device real3_t *out_points        [[buffer(1)]],
    constant float4x4 &transform      [[buffer(2)]],
    constant uint &num_points         [[buffer(3)]],
    uint gid                          [[thread_position_in_grid]])
{
    if (gid >= num_points) return;

    real3_t p = in_points[gid];
    float4 hom = float4(p, 1.0);
    float4 transformed = transform * hom;
    out_points[gid] = real3_t(transformed.x, transformed.y, transformed.z);
}

/* -------------------------------------------------------------------------
 * 6️⃣  Miscellaneous helpers
 * ------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------
 * 6.1 Simple Xorshift RNG – deterministic per‑launch seed
 * ------------------------------------------------------------------------- */
kernel void tk_xorshift_rng(
    device uint *seed          [[buffer(0)]],
    device uint *out_random    [[buffer(1)]],
    constant uint &count       [[buffer(2)]],
    uint gid                   [[thread_position_in_grid]])
{
    if (gid >= count) return;

    uint s = seed[0];
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    seed[0] = s;
    out_random[gid] = s;
}

/* -------------------------------------------------------------------------
 * 6.2 Atomic counter utilities (increment, fetch‑add)
 * ------------------------------------------------------------------------- */
kernel void tk_atomic_increment(
    device atomic_uint *counter   [[buffer(0)]],
    device uint *out_value        [[buffer(1)]],
    uint gid                      [[thread_position_in_grid]])
{
    // Only one thread performs the increment; others read the same value.
    if (gid == 0) {
        *out_value = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    }
}

/* -------------------------------------------------------------------------
 * 6.3 Debug dump of a float buffer to a 2‑D texture (visualisation)
 * ------------------------------------------------------------------------- */
kernel void tk_debug_dump_buffer_to_texture(
    device const real_t *buffer   [[buffer(0)]],
    texture2d<real4_t, access::write> outTex [[texture(0)]],
    constant uint &width         [[buffer(1)]],
    constant uint &height        [[buffer(2)]],
    uint2 gid                    [[thread_position_in_grid]])
{
    if (gid.x >= width || gid.y >= height) return;

    uint idx = linear_index(gid, width);
    real_t val = buffer[idx];
    // Map value to grayscale colour
    real4_t col = real4_t(val, val, val, (real_t)1.0);
    outTex.write(col, gid);
}

/* -------------------------------------------------------------------------
 * 6.4 Prefix‑sum (exclusive scan) – work‑efficient Blelloch algorithm.
 *      This kernel assumes the input size is a power of two and fits into a
 *      single thread‑group.  For larger arrays a multi‑pass approach is needed.
 * ------------------------------------------------------------------------- */
kernel void tk_exclusive_scan(
    device const real_t *input   [[buffer(0)]],
    device real_t *output        [[buffer(1)]],
    constant uint &count        [[buffer(2)]],
    uint tid                    [[thread_index_in_threadgroup]],
    threadgroup real_t temp[1024] [[threadgroup(0)]])
{
    // Load data into shared memory
    uint idx = tid;
    if (idx < count) {
        temp[tid] = input[idx];
    } else {
        temp[tid] = (real_t)0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up‑sweep (reduce) phase
    for (uint offset = 1; offset < count; offset <<= 1) {
        uint ai = (tid + 1) * offset * 2 - 1;
        uint bi = ai - offset;
        if (ai < count) {
            temp[ai] += temp[bi];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Clear the last element
    if (tid == 0) {
        temp[count - 1] = (real_t)0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down‑sweep phase
    for (uint offset = count >> 1; offset > 0; offset >>= 1) {
        uint ai = (tid + 1) * offset * 2 - 1;
        uint bi = ai - offset;
        if (ai < count) {
            real_t t = temp[bi];
            temp[bi] = temp[ai];
            temp[ai] += t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write results
    if (idx < count) {
        output[idx] = temp[tid];
    }
}

/* -------------------------------------------------------------------------
 * 6.5 Simple 2‑D convolution using MPS (wrapper for the dispatcher)
 *
 *   The Metal dispatcher can call this kernel when it wants to expose a
 *   pure‑Metal fallback for environments where MPS is unavailable (e.g.
 *   older macOS versions).  The kernel implements a 3×3 convolution with
 *   a user‑provided kernel matrix.
 * ------------------------------------------------------------------------- */
kernel void tk_convolution_3x3_fallback(
    texture2d<real_t, access::sample> src   [[texture(0)]],
    texture2d<real_t, access::write>  dst   [[texture(1)]],
    constant real_t kernel[9]                [[buffer(0)]],
    uint2 gid                               [[thread_position_in_grid]])
{
    uint2 size = uint2(src.get_width(), src.get_height());
    if (gid.x >= size.x || gid.y >= size.y) return;

    real_t sum = (real_t)0.0;
    int idx = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            uint2 n = clamp_coord(gid + uint2(dx, dy), size);
            sum += src.read(n) * kernel[idx];
            ++idx;
        }
    }
    dst.write(sum, gid);
}

/* -------------------------------------------------------------------------
 * 6.6 Utility: Convert a float texture to a half texture (useful for
 *              bandwidth‑saving when feeding data to a half‑precision model).
 * ------------------------------------------------------------------------- */
kernel void tk_float_to_half(
    texture2d<float, access::sample> src   [[texture(0)]],
    texture2d<half, access::write>  dst   [[texture(1)]],
    uint2 gid                             [[thread_position_in_grid]])
{
    if (gid.x >= src.get_width() || gid.y >= src.get_height())
        return;

    float4 f = src.read(gid);
    half4 h = half4(f);
    dst.write(h, gid);
}

/* -------------------------------------------------------------------------
 * 6.7 Utility: Convert a half texture back to float (e.g. after MPS
 *              processing that only supports float).
 * ------------------------------------------------------------------------- */
kernel void tk_half_to_float(
    texture2d<half, access::sample> src   [[texture(0)]],
    texture2d<float, access::write> dst   [[texture(1)]],
    uint2 gid                             [[thread_position_in_grid]])
{
    if (gid.x >= src.get_width() || gid.y >= src.get_height())
        return;

    half4 h = src.read(gid);
    float4 f = float4(h);
    dst.write(f, gid);
}

/* -------------------------------------------------------------------------
 * 6.8 Utility: Generate a synthetic checkerboard pattern (useful for
 *              debugging texture pipelines).
 * ------------------------------------------------------------------------- */
kernel void tk_generate_checkerboard(
    texture2d<real4_t, access::write> outTex [[texture(0)]],
    constant uint &squareSize               [[buffer(0)]],
    uint2 gid                               [[thread_position_in_grid]])
{
    if (gid.x >= outTex.get_width() || gid.y >= outTex.get_height())
        return;

    uint xBlock = (gid.x / squareSize) & 1u;
    uint yBlock = (gid.y / squareSize) & 1u;
    real4_t colour = ((xBlock ^ yBlock) == 0) ?
                     real4_t(1.0, 1.0, 1.0, 1.0) :
                     real4_t(0.0, 0.0, 0.0, 1.0);
    outTex.write(colour, gid);
}

/* -------------------------------------------------------------------------
 * 6.9 Utility: Compute per‑pixel L2 norm between two float textures.
 *              Useful for loss‑function evaluation on‑device.
 * ------------------------------------------------------------------------- */
kernel void tk_l2_error_map(
    texture2d<float, access::sample> a   [[texture(0)]],
    texture2d<float, access::sample> b   [[texture(1)]],
    texture2d<float, access::write>  out [[texture(2)]],
    uint2 gid                           [[thread_position_in_grid]])
{
    if (gid.x >= a.get_width() || gid.y >= a.get_height())
        return;

    float4 av = a.read(gid);
    float4 bv = b.read(gid);
    float4 diff = av - bv;
    float err = length(diff);
    out.write(err, gid);
}

/* -------------------------------------------------------------------------
 * 6.10 Utility: Encode a 32‑bit integer into a 4‑channel 8‑bit texture.
 *               Handy for visualising debug counters.
 * ------------------------------------------------------------------------- */
kernel void tk_encode_uint32_to_rgba8(
    device const uint *value          [[buffer(0)]],
    texture2d<uchar, access::write> outTex [[texture(0)]],
    uint2 gid                         [[thread_position_in_grid]])
{
    if (gid.x >= outTex.get_width() || gid.y >= outTex.get_height())
        return;

    uint v = *value;
    uchar4 pixel = uchar4( (v >> 24) & 0xFF,
                          (v >> 16) & 0xFF,
                          (v >> 8 ) & 0xFF,
                          v & 0xFF );
    outTex.write(pixel, gid);
}

/* -------------------------------------------------------------------------
 * 6.11 Utility: Simple per‑pixel thresholding (binary mask generation)
 * ------------------------------------------------------------------------- */
kernel void tk_threshold(
    texture2d<real_t, access::sample> src   [[texture(0)]],
    texture2d<uchar, access::write>  mask   [[texture(1)]],
    constant real_t &threshold               [[buffer(0)]],
    uint2 gid                               [[thread_position_in_grid]])
{
    if (gid.x >= src.get_width() || gid.y >= src.get_height())
        return;

    real_t v = src.read(gid);
    uchar out = (v > threshold) ? (uchar)255 : (uchar)0;
    mask.write(out, gid);
}

/* -------------------------------------------------------------------------
 * 6.12 Utility: Bilinear up‑sampling of a single‑channel texture.
 *               The kernel reads from a lower‑resolution texture and writes
 *               to a higher‑resolution destination.
 * ------------------------------------------------------------------------- */
kernel void tk_upsample_bilinear(
    texture2d<real_t, access::sample> src   [[texture(0)]],
    texture2d<real_t, access::write>  dst   [[texture(1)]],
    uint2 gid                               [[thread_position_in_grid]])
{
    uint2 dstSize = uint2(dst.get_width(), dst.get_height());
    uint2 srcSize = uint2(src.get_width(), src.get_height());

    if (gid.x >= dstSize.x || gid.y >= dstSize.y)
        return;

    // Normalised coordinates in source space
    real2_t uv = real2_t(gid) / real2_t(dstSize);
    real2_t srcCoord = uv * real2_t(srcSize);

    // Bilinear sample
    real2_t f = fract(srcCoord);
    uint2 i0 = uint2(floor(srcCoord));
    uint2 i1 = min(i0 + uint2(1,1), srcSize - uint2(1,1));

    real_t c00 = src.read(i0);
    real_t c10 = src.read(uint2(i1.x, i0.y));
    real_t c01 = src.read(uint2(i0.x, i1.y));
    real_t c11 = src.read(i1);

    real_t top = mix(c00, c10, f.x);
    real_t bottom = mix(c01, c11, f.x);
    real_t value = mix(top, bottom, f.y);

    dst.write(value, gid);
}

/* -------------------------------------------------------------------------
 * 6.13 Utility: Compute per‑pixel gradient magnitude using Sobel kernels.
 *               Returns a single‑channel float texture.
 * ------------------------------------------------------------------------- */
kernel void tk_sobel_gradient_magnitude(
    texture2d<real_t, access::sample> src   [[texture(0)]],
    texture2d<real_t, access::write>  out   [[texture(1)]],
    uint2 gid                               [[thread_position_in_grid]])
{
    uint2 size = uint2(src.get_width(), src.get_height());
    if (gid.x == 0 || gid.y == 0 ||
        gid.x >= size.x - 1 || gid.y >= size.y - 1)
    {
        out.write((real_t)0.0, gid);
        return;
    }

    // Sobel kernels (same as tk_sobel_edge_detection but we return magnitude)
    real_t Gx = -src.read(gid + uint2(-1, -1)) - 2.0 * src.read(gid + uint2(-1, 0)) - src.read(gid + uint2(-1, 1))
               + src.read(gid + uint2( 1, -1)) + 2.0 * src.read(gid + uint2( 1, 0)) + src.read(gid + uint2( 1, 1));

    real_t Gy = -src.read(gid + uint2(-1, -1)) - 2.0 * src.read(gid + uint2(0, -1)) - src.read(gid + uint2(1, -1))
               + src.read(gid + uint2(-1,  1)) + 2.0 * src.read(gid + uint2(0,  1)) + src.read(gid + uint2(1,  1));

    real_t mag = length(real2_t(Gx, Gy));
    out.write(mag, gid);
}

/* -------------------------------------------------------------------------
 * 6.14 Utility: Compute per‑pixel Laplacian (second‑order derivative).
 * ------------------------------------------------------------------------- */
kernel void tk_laplacian(
    texture2d<real_t, access::sample> src   [[texture(0)]],
    texture2d<real_t, access::write>  out   [[texture(1)]],
    uint2 gid                               [[thread_position_in_grid]])
{
    uint2 size = uint2(src.get_width(), src.get_height());
    if (gid.x == 0 || gid.y == 0 ||
        gid.x >= size.x - 1 || gid.y >= size.y - 1)
    {
        out.write((real_t)0.0, gid);
        return;
    }

    real_t centre = src.read(gid);
    real_t north  = src.read(gid + uint2(0, -1));
    real_t south  = src.read(gid + uint2(0,  1));
    real_t east   = src.read(gid + uint2(1,  0));
    real_t west   = src.read(gid + uint2(-1, 0));

    real_t lap = north + south + east + west - 4.0 * centre;
    out.write(lap, gid);
}

/* -------------------------------------------------------------------------
 * 6.15 Utility: Convert a depth map (float) to a visualizable grayscale image.
 *               Depth values are normalised to [0,1] based on min/max supplied.
 * ------------------------------------------------------------------------- */
kernel void tk_depth_to_grayscale(
    texture2d<real_t, access::sample> depth   [[texture(0)]],
    texture2d<real4_t, access::write> out     [[texture(1)]],
    constant real_t &depth_min                [[buffer(0)]],
    constant real_t &depth_max                [[buffer(1)]],
    uint2 gid                                 [[thread_position_in_grid]])
{
    if (gid.x >= depth.get_width() || gid.y >= depth.get_height())
        return;

    real_t d = depth.read(gid);
    real_t norm = (d - depth_min) / (depth_max - depth_min);
    norm = clamp(norm, (real_t)0.0, (real_t)1.0);
    real4_t colour = real4_t(norm, norm, norm, (real_t)1.0);
    out.write(colour, gid);
}

/* -------------------------------------------------------------------------
 * 6.16 Utility: Compute per‑pixel variance over a sliding window.
 *               This is a simple 3×3 variance estimator.
 * ------------------------------------------------------------------------- */
kernel void tk_local_variance(
    texture2d<real_t, access::sample> src   [[texture(0)]],
    texture2d<real_t, access::write>  out   [[texture(1)]],
    uint2 gid                               [[thread_position_in_grid]])
{
    uint2 size = uint2(src.get_width(), src.get_height());
    if (gid.x == 0 || gid.y == 0 ||
        gid.x >= size.x - 1 || gid.y >= size.y - 1)
    {
        out.write((real_t)0.0, gid);
        return;
    }

    real_t sum = (real_t)0.0;
    real_t sumSq = (real_t)0.0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            real_t v = src.read(gid + uint2(dx, dy));
            sum   += v;
            sumSq += v * v;
        }
    }
    real_t mean = sum / (real_t)9.0;
    real_t var  = sumSq / (real_t)9.0 - mean * mean;
    out.write(var, gid);
}

/* -------------------------------------------------------------------------
 * 6.17 Utility: Convert a 2‑D float texture to a 1‑D buffer (flattening).
 *               Useful for feeding data into a pure‑Metal tensor kernel.
 * ------------------------------------------------------------------------- */
kernel void tk_texture_to_buffer_flatten(
    texture2d<real_t, access::sample> src   [[texture(0)]],
    device real_t *out_buffer                [[buffer(0)]],
    uint2 gid                               [[thread_position_in_grid]])
{
    uint2 size = uint2(src.get_width(), src.get_height());
    if (gid.x >= size.x || gid.y >= size.y) return;

    uint idx = linear_index(gid, size.x);
    out_buffer[idx] = src.read(gid);
}

/* -------------------------------------------------------------------------
 * 6.18 Utility: Convert a 1‑D buffer back to a 2‑D texture.
 * ------------------------------------------------------------------------- */
kernel void tk_buffer_to_texture_reshape(
    device const real_t *in_buffer           [[buffer(0)]],
    texture2d<real_t, access::write> dst    [[texture(0)]],
    uint2 gid                               [[thread_position_in_grid]])
{
    uint2 size = uint2(dst.get_width(), dst.get_height());
    if (gid.x >= size.x || gid.y >= size.y) return;

    uint idx = linear_index(gid, size.x);
    dst.write(in_buffer[idx], gid);
}

/* -------------------------------------------------------------------------
 * 6.19 Utility: Simple per‑pixel absolute difference (L1 loss map).
 * ------------------------------------------------------------------------- */
kernel void tk_abs_diff_map(
    texture2d<real_t, access::sample> a   [[texture(0)]],
    texture2d<real_t, access::sample> b   [[texture(1)]],
    texture2d<real_t, access::write> out  [[texture(2)]],
    uint2 gid                             [[thread_position_in_grid]])
{
    if (gid.x >= a.get_width() || gid.y >= a.get_height())
        return;

    real_t av = a.read(gid);
    real_t bv = b.read(gid);
    out.write(abs(av - bv), gid);
}

/* -------------------------------------------------------------------------
 * 6.20 Utility: Generate a random normal distribution using Box‑Muller.
 *               The kernel writes a float buffer of size `count`.
 * ------------------------------------------------------------------------- */
kernel void tk_random_normal(
    device uint *seed          [[buffer(0)]],
    device real_t *out         [[buffer(1)]],
    constant uint &count      [[buffer(2)]],
    uint gid                  [[thread_position_in_grid]])
{
    if (gid >= count) return;

    // Two 32‑bit Xorshift values per pair
    uint s = seed[0];
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    seed[0] = s;
    uint u1 = s;

    s = seed[0];
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    seed[0] = s;
    uint u2 = s;

    // Convert to (0,1] floats
    real_t r1 = ((real_t)u1 + (real_t)1.0) / ((real_t)UINT_MAX + (real_t)1.0);
    real_t r2 = ((real_t)u2 + (real_t)1.0) / ((real_t)UINT_MAX + (real_t)1.0);

    // Box‑Muller transform
    real_t mag = sqrt(-2.0 * log(r1));
    real_t z0  = mag * cos(2.0 * M_PI * r2);
    // We only output one of the pair; the other could be stored elsewhere.
    out[gid] = z0;
}

/* -------------------------------------------------------------------------
 * 6.21 Utility: Simple per‑pixel histogram of gradient magnitudes.
 *               The kernel writes to a global atomic histogram (256 bins).
 * ------------------------------------------------------------------------- */
kernel void tk_gradient_histogram(
    texture2d<real_t, access::sample> grad   [[texture(0)]],
    device atomic_uint *global_hist          [[buffer(0)]],
    constant real_t &max_val                 [[buffer(1)]],
    uint2 gid                                [[thread_position_in_grid]],
    threadgroup atomic_uint local_hist[256]   [[threadgroup(0)]])
{
    // Initialise local histogram once per threadgroup
    if (gid.x == 0 && gid.y == 0) {
        for (uint i = 0; i < 256; ++i) {
            atomic_store_explicit(&local_hist[i], 0u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid.x >= grad.get_width() || gid.y >= grad.get_height())
        return;

    real_t g = grad.read(gid);
    uint bin = (uint)clamp(g / max_val * (real_t)255.0, (real_t)0.0, (real_t)255.0);
    atomic_fetch_add_explicit(&local_hist[bin], 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Flush local histogram to global
    if (gid.x == 0 && gid.y == 0) {
        for (uint i = 0; i < 256; ++i) {
            uint cnt = atomic_load_explicit(&local_hist[i], memory_order_relaxed);
            atomic_fetch_add_explicit(&global_hist[i], cnt, memory_order_relaxed);
        }
    }
}

/* -------------------------------------------------------------------------
 * 6.22 Utility: Simple per‑pixel median filter (3×3 window).
 *               This is a naïve implementation (O(k log k) per pixel) and
 *               primarily for debugging; production code should use MPS.
 * ------------------------------------------------------------------------- */
kernel void tk_median_filter_3x3(
    texture2d<real_t, access::sample> src   [[texture(0)]],
    texture2d<real_t, access::write>  dst   [[texture(1)]],
    uint2 gid                               [[thread_position_in_grid]])
{
    uint2 size = uint2(src.get_width(), src.get_height());
    if (gid.x == 0 || gid.y == 0 ||
        gid.x >= size.x - 1 || gid.y >= size.y - 1)
    {
        dst.write(src.read(gid), gid);
        return;
    }

    real_t window[9];
    uint idx = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            window[idx++] = src.read(gid + uint2(dx, dy));
        }
    }

    // Simple insertion sort
    for (uint i = 1; i < 9; ++i) {
        real_t key = window[i];
        int j = i - 1;
        while (j >= 0 && window[j] > key) {
            window[j + 1] = window[j];
            --j;
        }
        window[j + 1] = key;
    }

    // Median is the 5th element (0‑based)
    dst.write(window[4], gid);
}

/* -------------------------------------------------------------------------
 * 6.23 Utility: Compute per‑pixel cosine similarity between two feature maps.
 *               Both inputs are assumed to be multi‑channel (e.g. C=32) stored
 *               as separate textures per channel; for brevity we treat them as
 *               single‑channel float textures here.
 * ------------------------------------------------------------------------- */
kernel void tk_cosine_similarity(
    texture2d<real_t, access::sample> a   [[texture(0)]],
    texture2d<real_t, access::sample> b   [[texture(1)]],
    texture2d<real_t, access::write> out  [[texture(2)]],
    uint2 gid                             [[thread_position_in_grid]])
{
    if (gid.x >= a.get_width() || gid.y >= a.get_height())
        return;

    real_t av = a.read(gid);
    real_t bv = b.read(gid);
    // For single‑channel the cosine similarity reduces to sign(av) * sign(bv)
    real_t sim = (av * bv) / (sqrt(av * av) * sqrt(bv * bv) + (real_t)1e-6);
    out.write(sim, gid);
}

/* -------------------------------------------------------------------------
 * 6.24 Utility: Simple per‑pixel quantisation to N levels.
 * ------------------------------------------------------------------------- */
kernel void tk_quantise_levels(
    texture2d<real_t, access::sample> src   [[texture(0)]],
    texture2d<real_t, access::write>  dst   [[texture(1)]],
    constant uint &levels                   [[buffer(0)]],
    uint2 gid                               [[thread_position_in_grid]])
{
    if (gid.x >= src.get_width() || gid.y >= src.get_height())
        return;

    real_t v = src.read(gid);
    real_t q = round(v * (real_t)(levels - 1)) / (real_t)(levels - 1);
    dst.write(q, gid);
}

/* -------------------------------------------------------------------------
 * 6.25 Utility: Simple per‑pixel gamma correction.
 * ------------------------------------------------------------------------- */
kernel void tk_gamma_correction(
    texture2d<real_t, access::sample> src   [[texture(0)]],
    texture2d<real_t, access::write>  dst   [[texture(1)]],
    constant real_t &gamma                  [[buffer(0)]],
    uint2 gid                               [[thread_position_in_grid]])
{
    if (gid.x >= src.get_width() || gid.y >= src.get_height())
        return;

    real_t v = src.read(gid);
    real_t corrected = pow(v, (real_t)(1.0 / gamma));
    dst.write(corrected, gid);
}

/* -------------------------------------------------------------------------
 * 6.26 Utility: Simple per‑pixel exposure compensation.
 * ------------------------------------------------------------------------- */
kernel void tk_exposure_compensate(
    texture2d<real_t, access::sample> src   [[texture(0)]],
    texture2d<real_t, access::write>  dst   [[texture(1)]],
    constant real_t &exposure_factor       [[buffer(0)]],
    uint2 gid                               [[thread_position_in_grid]])
{
    if (gid.x >= src.get_width() || gid.y >= src.get_height())
        return;

    real_t v = src.read(gid);
    dst.write(v * exposure_factor, gid);
}

/* -------------------------------------------------------------------------
 * 6.27 Utility: Simple per‑pixel linear stretch (contrast adjustment).
 * ------------------------------------------------------------------------- */
kernel void tk_linear_stretch(
    texture2d<real_t, access::sample> src   [[texture(0)]],
    texture2d<real_t, access::write>  dst   [[texture(1)]],
    constant real_t &in_min                [[buffer(0)]],
    constant real_t &in_max                [[buffer(1)]],
    constant real_t &out_min               [[buffer(2)]],
    constant real_t &out_max               [[buffer(3)]],
    uint2 gid                               [[thread_position_in_grid]])
{
    if (gid.x >= src.get_width() || gid.y >= src.get_height())
        return;

    real_t v = src.read(gid);
    real_t norm = (v - in_min) / (in_max - in_min);
    norm = clamp(norm, (real_t)0.0, (real_t)1.0);
    real_t out = out_min + norm * (out_max - out_min);
    dst.write(out, gid);
}

/* -------------------------------------------------------------------------
 * 6.28 Utility: Simple per‑pixel histogram equalisation (using a pre‑computed
 *               CDF buffer).  The CDF is assumed to be 256 entries of uint.
 * ------------------------------------------------------------------------- */
kernel void tk_histogram_equalise_texture(
    texture2d<uchar, access::sample> src   [[texture(0)]],
    texture2d<uchar, access::write>  dst   [[texture(1)]],
    device const uint *cdf                 [[buffer(0)]],
    constant uint &total_pixels            [[buffer(1)]],
    uint2 gid                              [[thread_position_in_grid]])
{
    if (gid.x >= src.get_width() || gid.y >= src.get_height())
        return;

    uchar pixel = src.read(gid);
    uint c = cdf[pixel];
    // Normalise CDF to [0,255]
    uchar out = (uchar)clamp((c * 255u) / total_pixels, 0u, 255u);
    dst.write(out, gid);
}

/* -------------------------------------------------------------------------
 * End of tk_metal_kernels.metal
 * ------------------------------------------------------------------------- */
