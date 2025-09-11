#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cassert>
#include <iomanip>

#include "gpu/cuda/tk_cuda_kernels.h"
#include "gpu/tk_gpu_helper.h"
#include <cuda_runtime.h>

// Helper macro for checking CUDA calls
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error in " << __FILE__ << " at line " << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// CPU implementation of softmax for validation
void softmax_cpu(const std::vector<float>& input, std::vector<float>& output, int rows, int cols) {
    assert(input.size() == rows * cols);
    output.resize(rows * cols);

    for (int i = 0; i < rows; ++i) {
        const float* row_input = &input[i * cols];
        float* row_output = &output[i * cols];

        // Find max for numerical stability
        float max_val = row_input[0];
        for (int j = 1; j < cols; ++j) {
            if (row_input[j] > max_val) {
                max_val = row_input[j];
            }
        }

        // Calculate sum of exps
        float sum_exp = 0.0f;
        for (int j = 0; j < cols; ++j) {
            sum_exp += std::exp(row_input[j] - max_val);
        }

        // Calculate softmax
        for (int j = 0; j < cols; ++j) {
            row_output[j] = std::exp(row_input[j] - max_val) / sum_exp;
        }
    }
}

bool compare_results(const std::vector<float>& a, const std::vector<float>& b, float tolerance = 1e-6f) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    // Test parameters
    const int NUM_ROWS = 4;
    const int NUM_COLS = 256; // Must be <= 1024 for this kernel
    const size_t tensor_size = NUM_ROWS * NUM_COLS;
    const size_t tensor_bytes = tensor_size * sizeof(float);

    std::cout << "Initializing test for softmax kernel..." << std::endl;
    std::cout << "Tensor dimensions: " << NUM_ROWS << "x" << NUM_COLS << std::endl;

    // 1. Setup Phase
    std::vector<float> h_input(tensor_size);
    std::vector<float> h_gpu_output(tensor_size);
    std::vector<float> h_cpu_ref(tensor_size);

    // Initialize input data with some values
    for (size_t i = 0; i < tensor_size; ++i) {
        h_input[i] = static_cast<float>(i % NUM_COLS);
    }

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, tensor_bytes));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), tensor_bytes, cudaMemcpyHostToDevice));

    // 2. Execution Phase
    tk_softmax_params_t params;
    params.d_input_tensor = d_input;
    params.d_output_tensor = d_output;
    params.num_rows = NUM_ROWS;
    params.num_cols = NUM_COLS;

    std::cout << "Launching softmax kernel..." << std::endl;
    tk_error_code_t err = tk_kernels_softmax(&params, 0); // Use default stream
    if (err != TK_SUCCESS) {
        std::cerr << "Kernel launch failed with error code: " << err << std::endl;
        exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Kernel execution finished." << std::endl;

    CUDA_CHECK(cudaMemcpy(h_gpu_output.data(), d_output, tensor_bytes, cudaMemcpyDeviceToHost));

    // 3. Validation Phase
    std::cout << "Computing reference result on CPU..." << std::endl;
    softmax_cpu(h_input, h_cpu_ref, NUM_ROWS, NUM_COLS);

    std::cout << "Comparing GPU and CPU results..." << std::endl;
    bool success = compare_results(h_gpu_output, h_cpu_ref);

    // 4. Teardown Phase
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    if (success) {
        std::cout << "SUCCESS: Softmax kernel validation passed!" << std::endl;
        return 0;
    } else {
        std::cout << "FAILURE: Softmax kernel validation failed!" << std::endl;
        return 1;
    }
}
