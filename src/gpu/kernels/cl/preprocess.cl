/*
* OpenCL kernel for image preprocessing.
* Performs scaling, normalization, and layout conversion.
*/

// Define a simple float3 type if not built-in
#ifndef __OPENCL_C_VERSION__
    typedef struct { float x, y, z; } float3;
#endif

__kernel void preprocess_image(
    __global const uchar* input_image,
    __global float* output_tensor,
    const uint input_width,
    const uint input_height,
    const uint output_width,
    const uint output_height,
    const float3 mean,
    const float3 std_dev)
{
    // Get the global ID for the output pixel
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= output_width || y >= output_height) {
        return;
    }

    // --- Nearest Neighbor Scaling (for simplicity) ---
    float2 scale = (float2)((float)input_width / (float)output_width, (float)input_height / (float)output_height);
    int2 read_coord = convert_int2(round((float2)(x, y) + 0.5f) * scale - 0.5f);
    read_coord = clamp(read_coord, (int2)(0), (int2)(input_width - 1, input_height - 1));

    int read_idx = (read_coord.y * input_width + read_coord.x) * 3;

    // --- Type Conversion and Normalization ---
    float3 pixel_in = (float3)(
        (float)input_image[read_idx + 0] / 255.0f,
        (float)input_image[read_idx + 1] / 255.0f,
        (float)input_image[read_idx + 2] / 255.0f
    );

    float3 pixel_normalized = (pixel_in - mean) / std_dev;

    // --- Planar (NCHW) Output ---
    uint c_stride = output_width * output_height;
    uint out_idx_r = c_stride * 0 + y * output_width + x;
    uint out_idx_g = c_stride * 1 + y * output_width + x;
    uint out_idx_b = c_stride * 2 + y * output_width + x;

    output_tensor[out_idx_r] = pixel_normalized.x;
    output_tensor[out_idx_g] = pixel_normalized.y;
    output_tensor[out_idx_b] = pixel_normalized.z;
}
