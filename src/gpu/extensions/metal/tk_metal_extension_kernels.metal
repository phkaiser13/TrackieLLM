/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_metal_extension_kernels.metal
 *
 * Metal‑shading‑language (MSL) implementation of every compute kernel used by
 * the TrackieLLM GPU abstraction layer.  The kernels are deliberately verbose,
 * heavily commented and written with a “research‑grade” style so that they can
 * serve both as production code and as a learning reference for future
 * contributors.  All kernels are written for float‑32 data (the only type
 * currently required by the Tensor and Image back‑ends).  When a kernel needs
 * additional constant data (e.g. convolution coefficients, pooling parameters,
 * etc.) a small POD struct is passed as a device buffer; the host side packs the
 * struct in the same layout as described in the comments.
 *
 * The file is split into two logical sections:
 *   1) Tensor operations – element‑wise arithmetic, activations, pooling,
 *      layer‑norm, etc.
 *   2) Image operations – colour‑space conversion, 2‑D convolution,
 *      Sobel edge detection, bilateral filter, morphology, etc.
 *
 * Every kernel follows the same pattern:
 *   • The first argument(s) are device buffers or textures.
 *   • The last argument is the thread identifier (`[[thread_position_in_grid]]`
 *     for 1‑D work‑items or `[[thread_position_in_grid]]` with a `uint2`
 *     for 2‑D work‑items.
 *   • Bounds checking (`gid < elementCount` or `gid.x < width && …`) is
 *     performed explicitly to avoid out‑of‑bounds memory accesses.
 *
 * NOTE: Matrix multiplication (`matmul`) is *not* implemented here.  The host
 * implementation in `tk_metal_tensor_ops.mm` forwards the operation to
 * Metal Performance Shaders (MPSMatrixMultiplication), which provides a
 * highly‑optimised GEMM implementation that automatically exploits Apple‑metal
 * matrix cores.
 *
 * SPDX‑License‑Identifier: AGPL‑3.0
 */

#include <metal_stdlib>
using namespace metal;

/*==========================================================================*/
/*  SECTION 1 – TENSOR OPERATIONS (device buffers, 1‑D work‑items)        */
/*==========================================================================*/

/* -------------------------------------------------------------------------
 *  Helper: compute the total number of elements in a tensor descriptor.
 *  The host side passes this value as a kernel constant to avoid recomputing
 *  it on the GPU for every thread.
 * ------------------------------------------------------------------------- */
struct TensorSize {
    uint64_t elementCount;   // total number of float elements
};

/* -------------------------------------------------------------------------
 *  Element‑wise binary arithmetic kernels (add, sub, mul, div).
 * ------------------------------------------------------------------------- */
kernel void kernel_tensor_add(
    device float *out               [[buffer(0)]],
    const device float *in_a        [[buffer(1)]],
    const device float *in_b        [[buffer(2)]],
    constant TensorSize &size       [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= size.elementCount) return;
    out[gid] = in_a[gid] + in_b[gid];
}

kernel void kernel_tensor_subtract(
    device float *out               [[buffer(0)]],
    const device float *in_a        [[buffer(1)]],
    const device float *in_b        [[buffer(2)]],
    constant TensorSize &size       [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= size.elementCount) return;
    out[gid] = in_a[gid] - in_b[gid];
}

kernel void kernel_tensor_multiply(
    device float *out               [[buffer(0)]],
    const device float *in_a        [[buffer(1)]],
    const device float *in_b        [[buffer(2)]],
    constant TensorSize &size       [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= size.elementCount) return;
    out[gid] = in_a[gid] * in_b[gid];
}

kernel void kernel_tensor_divide(
    device float *out               [[buffer(0)]],
    const device float *in_a        [[buffer(1)]],
    const device float *in_b        [[buffer(2)]],
    constant TensorSize &size       [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= size.elementCount) return;
    // Guard against division by zero – the host guarantees non‑zero divisor,
    // but we keep a tiny epsilon for safety.
    const float epsilon = 1e-7f;
    float divisor = in_b[gid];
    divisor = (fabs(divisor) < epsilon) ? epsilon : divisor;
    out[gid] = in_a[gid] / divisor;
}

/* -------------------------------------------------------------------------
 *  Scalar kernels – the scalar value is passed as a constant buffer of size
 *  one float (struct tk_scalar_f32_t on the host side).  The layout is:
 *      struct { float value; };
 * ------------------------------------------------------------------------- */
kernel void kernel_tensor_add_scalar(
    device float *out               [[buffer(0)]],
    const device float *in          [[buffer(1)]],
    constant float &scalar          [[buffer(2)]],
    constant TensorSize &size       [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= size.elementCount) return;
    out[gid] = in[gid] + scalar;
}

kernel void kernel_tensor_mul_scalar(
    device float *out               [[buffer(0)]],
    const device float *in          [[buffer(1)]],
    constant float &scalar          [[buffer(2)]],
    constant TensorSize &size       [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= size.elementCount) return;
    out[gid] = in[gid] * scalar;
}

/* -------------------------------------------------------------------------
 *  Activation kernels – ReLU, Sigmoid, Tanh.
 * ------------------------------------------------------------------------- */
kernel void kernel_tensor_relu(
    device float *out               [[buffer(0)]],
    const device float *in          [[buffer(1)]],
    constant TensorSize &size       [[buffer(2)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= size.elementCount) return;
    out[gid] = max(in[gid], 0.0f);
}

kernel void kernel_tensor_sigmoid(
    device float *out               [[buffer(0)]],
    const device float *in          [[buffer(1)]],
    constant TensorSize &size       [[buffer(2)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= size.elementCount) return;
    // Numerically stable sigmoid: clamp the exponent to avoid overflow.
    float x = in[gid];
    if (x >= 0.0f) {
        out[gid] = 1.0f / (1.0f + exp(-x));
    } else {
        float e = exp(x);
        out[gid] = e / (1.0f + e);
    }
}

kernel void kernel_tensor_tanh(
    device float *out               [[buffer(0)]],
    const device float *in          [[buffer(1)]],
    constant TensorSize &size       [[buffer(2)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= size.elementCount) return;
    out[gid] = tanh(in[gid]);
}

/* -------------------------------------------------------------------------
 *  Softmax – a two‑pass algorithm that works on a 2‑D tensor where each row
 *  represents a separate probability distribution (common in classification).
 *
 *  Pass 1 (max reduction per row) writes the per‑row maximum into a temporary
 *  buffer `row_max`.  Pass 2 (exp & sum) reads `row_max`, computes the
 *  exponentials, accumulates the per‑row sum into `row_sum`, and finally writes
 *  the normalised probabilities to the output buffer.
 *
 *  The host side allocates two auxiliary buffers (`row_max` and `row_sum`) of
 *  size `rows * sizeof(float)`.  The kernels below assume those buffers are
 *  bound at indices 3 and 4 respectively.
 * ------------------------------------------------------------------------- */

/* Pass 1 – compute max per row */
kernel void kernel_softmax_row_max(
    const device float *in          [[buffer(0)]],
    device float *row_max           [[buffer(1)]],
    constant uint &rows             [[buffer(2)]],
    constant uint &cols             [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]])
{
    // gid encodes the row index; each thread processes one row.
    uint row = gid;
    if (row >= rows) return;

    float maxVal = -FLT_MAX;
    uint base = row * cols;
    for (uint c = 0; c < cols; ++c) {
        float v = in[base + c];
        maxVal = max(maxVal, v);
    }
    row_max[row] = maxVal;
}

/* Pass 2 – exponentiate, accumulate sum, write normalized output */
kernel void kernel_softmax_exp_and_normalize(
    const device float *in          [[buffer(0)]],
    device float *out               [[buffer(1)]],
    const device float *row_max     [[buffer(2)]],
    device float *row_sum           [[buffer(3)]],
    constant uint &rows             [[buffer(4)]],
    constant uint &cols             [[buffer(5)]],
    uint gid                        [[thread_position_in_grid]])
{
    // gid encodes a linear index over the entire tensor.
    uint idx = gid;
    uint total = rows * cols;
    if (idx >= total) return;

    uint row = idx / cols;
    uint col = idx % cols;

    float shifted = in[idx] - row_max[row];
    float expVal  = exp(shifted);
    out[idx] = expVal;               // store temporary exponent

    // Atomic add to accumulate the row sum.
    // Metal guarantees 32‑bit atomic add on device buffers.
    atomic_fetch_add_explicit(
        (device atomic_uint *)&row_sum[row],
        as_type<uint>(expVal),
        memory_order_relaxed);
}

/* Pass 3 – final division by the row sum (performed after the host has
 *          converted the atomic uint sums back to float). */
kernel void kernel_softmax_normalize(
    device float *out               [[buffer(0)]],
    const device float *row_sum     [[buffer(1)]],
    constant uint &rows             [[buffer(2)]],
    constant uint &cols             [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]])
{
    uint idx = gid;
    uint total = rows * cols;
    if (idx >= total) return;

    uint row = idx / cols;
    float sum = row_sum[row];
    out[idx] = out[idx] / sum;
}

/* -------------------------------------------------------------------------
 *  Transpose – swaps the two innermost dimensions of a 2‑D tensor.
 *  The kernel works on a flat buffer; the host supplies the source and
 *  destination dimensions (rows, cols).  The thread grid is sized to the
 *  destination matrix.
 * ------------------------------------------------------------------------- */
kernel void kernel_tensor_transpose(
    const device float *src          [[buffer(0)]],
    device float *dst               [[buffer(1)]],
    constant uint &srcRows          [[buffer(2)]],
    constant uint &srcCols          [[buffer(3)]],
    uint2 gid                       [[thread_position_in_grid]])
{
    uint dstRow = gid.y;   // destination row
    uint dstCol = gid.x;   // destination column
    if (dstRow >= srcCols || dstCol >= srcRows) return;

    // Compute linear indices.
    uint srcIdx = dstCol * srcCols + dstRow; // note the swap
    uint dstIdx = dstRow * srcRows + dstCol;

    dst[dstIdx] = src[srcIdx];
}

/* -------------------------------------------------------------------------
 *  Pooling – generic max and average pooling for 4‑D tensors
 *  (N, C, H, W).  The kernel receives a packed struct with all parameters.
 * ------------------------------------------------------------------------- */
struct PoolingParams {
    uint kernelWidth;
    uint kernelHeight;
    uint strideX;
    uint strideY;
    uint padX;
    uint padY;
    uint inputWidth;   // W
    uint inputHeight;  // H
    uint channels;     // C
    uint batch;        // N
};

kernel void kernel_tensor_max_pool(
    const device float *src          [[buffer(0)]],
    device float *dst               [[buffer(1)]],
    constant PoolingParams &p       [[buffer(2)]],
    constant uint &outputWidth      [[buffer(3)]],
    constant uint &outputHeight     [[buffer(4)]],
    uint3 gid                       [[thread_position_in_grid]])
{
    // gid.x -> output column, gid.y -> output row,
    // gid.z -> channel * batch (flattened)
    uint outX = gid.x;
    uint outY = gid.y;
    uint cb   = gid.z;               // combined batch*channel index

    if (outX >= outputWidth || outY >= outputHeight) return;

    uint batchIdx   = cb / p.channels;
    uint channelIdx = cb % p.channels;

    // Compute the start of the receptive field in the input tensor.
    int inXOrigin = int(outX) * int(p.strideX) - int(p.padX);
    int inYOrigin = int(outY) * int(p.strideY) - int(p.padY);

    float maxVal = -FLT_MAX;
    for (uint ky = 0; ky < p.kernelHeight; ++ky) {
        int inY = inYOrigin + int(ky);
        if (inY < 0 || inY >= int(p.inputHeight)) continue;
        for (uint kx = 0; kx < p.kernelWidth; ++kx) {
            int inX = inXOrigin + int(kx);
            if (inX < 0 || inX >= int(p.inputWidth)) continue;

            // Linear index into the source buffer:
            // ((batch * C + channel) * H + inY) * W + inX
            uint srcIdx = ((batchIdx * p.channels + channelIdx) *
                           p.inputHeight + uint(inY)) * p.inputWidth + uint(inX);
            maxVal = max(maxVal, src[srcIdx]);
        }
    }

    // Destination index follows the same layout as the source.
    uint dstIdx = ((batchIdx * p.channels + channelIdx) *
                   outputHeight + outY) * outputWidth + outX;
    dst[dstIdx] = maxVal;
}

kernel void kernel_tensor_avg_pool(
    const device float *src          [[buffer(0)]],
    device float *dst               [[buffer(1)]],
    constant PoolingParams &p       [[buffer(2)]],
    constant uint &outputWidth      [[buffer(3)]],
    constant uint &outputHeight     [[buffer(4)]],
    uint3 gid                       [[thread_position_in_grid]])
{
    uint outX = gid.x;
    uint outY = gid.y;
    uint cb   = gid.z;

    if (outX >= outputWidth || outY >= outputHeight) return;

    uint batchIdx   = cb / p.channels;
    uint channelIdx = cb % p.channels;

    int inXOrigin = int(outX) * int(p.strideX) - int(p.padX);
    int inYOrigin = int(outY) * int(p.strideY) - int(p.padY);

    float sum = 0.0f;
    uint   count = 0;

    for (uint ky = 0; ky < p.kernelHeight; ++ky) {
        int inY = inYOrigin + int(ky);
        if (inY < 0 || inY >= int(p.inputHeight)) continue;
        for (uint kx = 0; kx < p.kernelWidth; ++kx) {
            int inX = inXOrigin + int(kx);
            if (inX < 0 || inX >= int(p.inputWidth)) continue;

            uint srcIdx = ((batchIdx * p.channels + channelIdx) *
                           p.inputHeight + uint(inY)) * p.inputWidth + uint(inX);
            sum += src[srcIdx];
            ++count;
        }
    }

    float avg = (count > 0) ? (sum / float(count)) : 0.0f;
    uint dstIdx = ((batchIdx * p.channels + channelIdx) *
                   outputHeight + outY) * outputWidth + outX;
    dst[dstIdx] = avg;
}

/* -------------------------------------------------------------------------
 *  Layer Normalization – normalizes each channel independently across the
 *  spatial dimensions (H × W).  The kernel receives epsilon as a constant.
 * ------------------------------------------------------------------------- */
struct LayerNormParams {
    float epsilon;
    uint  channels;
    uint  height;
    uint  width;
};

kernel void kernel_tensor_layer_norm(
    const device float *src          [[buffer(0)]],
    device float *dst               [[buffer(1)]],
    constant LayerNormParams &p     [[buffer(2)]],
    uint gid                        [[thread_position_in_grid]])
{
    // gid encodes a linear index over the entire tensor.
    uint totalElements = p.channels * p.height * p.width;
    if (gid >= totalElements) return;

    // Compute channel index.
    uint channel = (gid / (p.height * p.width)) % p.channels;

    // Compute the start of the channel slice.
    uint channelBase = channel * p.height * p.width;

    // First pass – compute mean.
    // (We could compute mean and variance in a single pass using Welford's
    //  algorithm, but the two‑pass version is easier to read and still fast.)
    threadgroup float sharedMean;
    threadgroup float sharedVar;

    // Reduce within the threadgroup.
    float sum = 0.0f;
    for (uint i = gid; i < totalElements; i += get_thread_execution_width()) {
        sum += src[i];
    }
    // Atomic add to a global accumulator (implemented via a separate kernel
    // in production; here we use a simple reduction for clarity).
    // For brevity we assume the tensor fits into a single threadgroup.
    sharedMean = sum / float(p.height * p.width);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Second pass – compute variance.
    float varSum = 0.0f;
    for (uint i = gid; i < totalElements; i += get_thread_execution_width()) {
        float diff = src[i] - sharedMean;
        varSum += diff * diff;
    }
    sharedVar = varSum / float(p.height * p.width);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final pass – normalize.
    float invStd = rsqrt(sharedVar + p.epsilon);
    dst[gid] = (src[gid] - sharedMean) * invStd;
}

/*==========================================================================*/
/*  SECTION 2 – IMAGE OPERATIONS (textures, 2‑D work‑items)                */
/*==========================================================================*/

/* -------------------------------------------------------------------------
 *  Colour‑space conversion: RGB → Grayscale.
 *  The input texture is assumed to be `float4` (RGBA) with values in [0,1].
 * ------------------------------------------------------------------------- */
kernel void kernel_color_space_conversion_rgb_to_gray(
    texture2d<float, access::read>  inTex  [[texture(0)]],
    texture2d<float, access::write> outTex [[texture(1)]],
    uint2 gid                         [[thread_position_in_grid]])
{
    if (gid.x >= inTex.get_width() || gid.y >= inTex.get_height()) return;

    float4 rgb = inTex.read(gid);
    // Luminance coefficients per ITU‑BT.601.
    float gray = dot(rgb.rgb, float3(0.299, 0.587, 0.114));
    outTex.write(float4(gray, gray, gray, 1.0), gid);
}

/* -------------------------------------------------------------------------
 *  2‑D convolution kernel – generic separable or full kernel.
 *  The convolution coefficients are stored in a linear device buffer.
 * ------------------------------------------------------------------------- */
struct Conv2DParams {
    uint kernelWidth;
    uint kernelHeight;
    uint strideX;
    uint strideY;
    uint padX;
    uint padY;
    uint inputWidth;
    uint inputHeight;
    uint outputWidth;
    uint outputHeight;
};

kernel void kernel_image_convolution_2d(
    texture2d<float, access::read>  inTex   [[texture(0)]],
    texture2d<float, access::write> outTex  [[texture(1)]],
    const device float *kernel       [[buffer(0)]],
    constant Conv2DParams &p         [[buffer(1)]],
    uint2 gid                        [[thread_position_in_grid]])
{
    if (gid.x >= p.outputWidth || gid.y >= p.outputHeight) return;

    // Compute the top‑left corner of the receptive field in the input.
    int inXOrigin = int(gid.x) * int(p.strideX) - int(p.padX);
    int inYOrigin = int(gid.y) * int(p.strideY) - int(p.padY);

    float accum = 0.0f;
    for (uint ky = 0; ky < p.kernelHeight; ++ky) {
        int inY = inYOrigin + int(ky);
        if (inY < 0 || inY >= int(p.inputHeight)) continue;
        for (uint kx = 0; kx < p.kernelWidth; ++kx) {
            int inX = inXOrigin + int(kx);
            if (inX < 0 || inX >= int(p.inputWidth)) continue;

            float4 pixel = inTex.read(uint2(inX, inY));
            // Assume a single channel (grayscale) – use .r component.
            // For RGB kernels you could multiply each component separately.
            float coeff = kernel[ky * p.kernelWidth + kx];
            accum += pixel.r * coeff;
        }
    }

    // Write the result as a single‑channel grayscale texture.
    outTex.write(float4(accum, accum, accum, 1.0), gid);
}

/* -------------------------------------------------------------------------
 *  Sobel edge detection – computes gradient magnitude using the classic
 *  3×3 Sobel kernels Gx and Gy.
 * ------------------------------------------------------------------------- */
kernel void kernel_image_sobel_edge_detection(
    texture2d<float, access::read>  inTex   [[texture(0)]],
    texture2d<float, access::write> outTex  [[texture(1)]],
    uint2 gid                        [[thread_position_in_grid]])
{
    uint width  = inTex.get_width();
    uint height = inTex.get_height();

    if (gid.x >= width || gid.y >= height) return;

    // Sobel kernels (horizontal and vertical).
    const int Gx[3][3] = { { -1, 0, 1 },
                           { -2, 0, 2 },
                           { -1, 0, 1 } };
    const int Gy[3][3] = { { -1, -2, -1 },
                           {  0,  0,  0 },
                           {  1,  2,  1 } };

    float gradX = 0.0f;
    float gradY = 0.0f;

    // Iterate over the 3×3 neighbourhood.
    for (int dy = -1; dy <= 1; ++dy) {
        int y = int(gid.y) + dy;
        y = clamp(y, 0, int(height) - 1);
        for (int dx = -1; dx <= 1; ++dx) {
            int x = int(gid.x) + dx;
            x = clamp(x, 0, int(width) - 1);
            float sample = inTex.read(uint2(x, y)).r; // assume grayscale input
            gradX += sample * float(Gx[dy + 1][dx + 1]);
            gradY += sample * float(Gy[dy + 1][dx + 1]);
        }
    }

    float magnitude = sqrt(gradX * gradX + gradY * gradY);
    outTex.write(float4(magnitude, magnitude, magnitude, 1.0), gid);
}

/* -------------------------------------------------------------------------
 *  Bilateral filter – edge‑preserving smoothing.
 *  Parameters are packed in a small struct; the kernel uses a fixed radius
 *  of 3 (7×7 window) for simplicity.  The spatial sigma and intensity sigma
 *  are supplied by the host.
 * ------------------------------------------------------------------------- */
struct BilateralParams {
    float spatialSigma;   // controls spatial decay
    float intensitySigma; // controls range decay
    uint  radius;         // half‑window size (e.g. 3 → 7×7)
    uint  width;
    uint  height;
};

kernel void kernel_image_bilateral_filter(
    texture2d<float, access::read>  inTex   [[texture(0)]],
    texture2d<float, access::write> outTex  [[texture(1)]],
    constant BilateralParams &p      [[buffer(0)]],
    uint2 gid                        [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) return;

    float center = inTex.read(gid).r;
    float sum = 0.0f;
    float weightSum = 0.0f;

    // Pre‑compute constants.
    float twoSpatial2 = 2.0f * p.spatialSigma * p.spatialSigma;
    float twoIntensity2 = 2.0f * p.intensitySigma * p.intensitySigma;

    for (int dy = -int(p.radius); dy <= int(p.radius); ++dy) {
        int y = int(gid.y) + dy;
        if (y < 0 || y >= int(p.height)) continue;
        for (int dx = -int(p.radius); dx <= int(p.radius); ++dx) {
            int x = int(gid.x) + dx;
            if (x < 0 || x >= int(p.width)) continue;

            float sample = inTex.read(uint2(x, y)).r;
            float spatialDist = float(dx * dx + dy * dy);
            float intensityDist = (sample - center) * (sample - center);

            float w = exp(-spatialDist / twoSpatial2) *
                      exp(-intensityDist / twoIntensity2);
            sum += w * sample;
            weightSum += w;
        }
    }

    float result = (weightSum > 0.0f) ? (sum / weightSum) : center;
    outTex.write(float4(result, result, result, 1.0), gid);
}

/* -------------------------------------------------------------------------
 *  Morphology operation – generic erode / dilate with a square structuring
 *  element.  The operation type is encoded as an integer:
 *      0 = erode, 1 = dilate.
 * ------------------------------------------------------------------------- */
struct MorphologyParams {
    uint operation;   // 0 = erode, 1 = dilate
    uint radius;      // half‑size of the square kernel
    uint width;
    uint height;
};

kernel void kernel_image_morphology_operation(
    texture2d<float, access::read>  inTex   [[texture(0)]],
    texture2d<float, access::write> outTex  [[texture(1)]],
    constant MorphologyParams &p      [[buffer(0)]],
    uint2 gid                        [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) return;

    float result = (p.operation == 0) ? FLT_MAX : -FLT_MAX;

    for (int dy = -int(p.radius); dy <= int(p.radius); ++dy) {
        int y = int(gid.y) + dy;
        if (y < 0 || y >= int(p.height)) continue;
        for (int dx = -int(p.radius); dx <= int(p.radius); ++dx) {
            int x = int(gid.x) + dx;
            if (x < 0 || x >= int(p.width)) continue;

            float sample = inTex.read(uint2(x, y)).r;
            if (p.operation == 0) { // erode
                result = min(result, sample);
            } else {                // dilate
                result = max(result, sample);
            }
        }
    }

    outTex.write(float4(result, result, result, 1.0), gid);
}

/* -------------------------------------------------------------------------
 *  Geometric transform – affine warp (rotation, scaling, translation).
 *  The 3×3 matrix is supplied in row‑major order.  The kernel performs a
 *  backward mapping (sample from source using the inverse of the matrix) to
 *  avoid holes.  Bilinear interpolation is used when `interpolation == 1`.
 * ------------------------------------------------------------------------- */
struct GeometricTransformParams {
    float matrix[9];      // row‑major 3×3 affine matrix
    uint  interpolation; // 0 = nearest, 1 = bilinear
    uint  borderMode;    // 0 = clamp, 1 = repeat, 2 = constant
    float constantValue; // used when borderMode == 2
    uint  srcWidth;
    uint  srcHeight;
    uint  dstWidth;
    uint  dstHeight;
};

float2 apply_affine(const float2 coord, constant float *matrix) {
    // matrix = [ m00 m01 m02
    //            m10 m11 m12
    //            0   0   1   ]  (last row is implicit)
    float x = matrix[0] * coord.x + matrix[1] * coord.y + matrix[2];
    float y = matrix[3] * coord.x + matrix[4] * coord.y + matrix[5];
    return float2(x, y);
}

/* Nearest‑neighbor sampling helper */
float sample_nearest(texture2d<float, access::read> tex,
                    float2 uv,
                    uint width,
                    uint height,
                    uint borderMode,
                    float constantValue)
{
    int2 coord = int2(round(uv.x), round(uv.y));

    if (borderMode == 0) { // clamp
        coord.x = clamp(coord.x, 0, int(width) - 1);
        coord.y = clamp(coord.y, 0, int(height) - 1);
        return tex.read(uint2(coord)).r;
    } else if (borderMode == 1) { // repeat
        coord.x = (coord.x % int(width) + int(width)) % int(width);
        coord.y = (coord.y % int(height) + int(height)) % int(height);
        return tex.read(uint2(coord)).r;
    } else { // constant
        if (coord.x < 0 || coord.x >= int(width) ||
            coord.y < 0 || coord.y >= int(height)) {
            return constantValue;
        }
        return tex.read(uint2(coord)).r;
    }
}

/* Bilinear interpolation helper */
float sample_bilinear(texture2d<float, access::read> tex,
                     float2 uv,
                     uint width,
                     uint height,
                     uint borderMode,
                     float constantValue)
{
    float2 f = floor(uv);
    float2 frac = uv - f;

    float v00 = sample_nearest(tex, f,               width, height, borderMode, constantValue);
    float v10 = sample_nearest(tex, f + float2(1,0), width, height, borderMode, constantValue);
    float v01 = sample_nearest(tex, f + float2(0,1), width, height, borderMode, constantValue);
    float v11 = sample_nearest(tex, f + float2(1,1), width, height, borderMode, constantValue);

    return mix(mix(v00, v10, frac.x), mix(v01, v11, frac.x), frac.y);
}

kernel void kernel_image_geometric_transform(
    texture2d<float, access::read>  srcTex   [[texture(0)]],
    texture2d<float, access::write> dstTex   [[texture(1)]],
    constant GeometricTransformParams &p [[buffer(0)]],
    uint2 gid                           [[thread_position_in_grid]])
{
    if (gid.x >= p.dstWidth || gid.y >= p.dstHeight) return;

    // Destination pixel centre in homogeneous coordinates.
    float2 dstCoord = float2(gid.x, gid.y);

    // Compute the inverse mapping (source coordinate that maps to dstCoord).
    // For an affine matrix the inverse can be computed analytically.
    // Here we compute the inverse on the CPU side and pass it in `p.matrix`.
    // The host guarantees that `p.matrix` already contains the inverse.
    float2 srcCoord = apply_affine(dstCoord, p.matrix);

    float value;
    if (p.interpolation == 0) {
        value = sample_nearest(srcTex, srcCoord,
                               p.srcWidth, p.srcHeight,
                               p.borderMode, p.constantValue);
    } else {
        value = sample_bilinear(srcTex, srcCoord,
                                p.srcWidth, p.srcHeight,
                                p.borderMode, p.constantValue);
    }

    dstTex.write(float4(value, value, value, 1.0), gid);
}

