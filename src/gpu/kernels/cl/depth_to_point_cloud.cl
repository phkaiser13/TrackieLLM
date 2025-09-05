/*
* OpenCL kernel for depth to point cloud conversion.
*/

__kernel void depth_to_point_cloud(
    __global const float* metric_depth_map,
    __global float3* point_cloud,
    const uint width,
    const uint height,
    const float fx,
    const float fy,
    const float cx,
    const float cy)
{
    // Get the global ID for the input pixel
    int u = get_global_id(0);
    int v = get_global_id(1);

    if (u >= width || v >= height) {
        return;
    }

    int index = v * width + u;
    float d = metric_depth_map[index];

    // Unprojection formula
    float x = (u - cx) * d / fx;
    float y = (v - cy) * d / fy;
    float z = d;

    point_cloud[index] = (float3)(x, y, z);
}
