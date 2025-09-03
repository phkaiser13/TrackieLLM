<!-- This documentation was written by Jules - Google labs bot. -->

# GPU Module

## 1. Overview

The GPU module provides a hardware abstraction layer for accelerating compute-intensive tasks, particularly the neural network inference required by the Vision and Audio modules. It dispatches operations to the appropriate backend based on the available hardware, ensuring optimal performance on a variety of platforms.

## 2. Core Responsibilities

-   **Hardware Abstraction:** Provides a unified interface for GPU operations, hiding the complexities of the underlying graphics APIs (CUDA, ROCm, Metal).
-   **Kernel Dispatch:** Dispatches custom compute kernels to the GPU for parallel processing.
-   **Tensor Operations:** Implements highly optimized tensor and matrix operations, which are the building blocks of neural network inference.
-   **Image Processing:** Provides functions for accelerating image manipulation tasks, such as resizing and color space conversion.

## 3. Architecture and Backends

The module is designed with a backend-based architecture. A generic frontend dispatches calls to the specific backend that is available and compiled for the target system.

### 3.1. CUDA (NVIDIA)

-   **Description:** The backend for NVIDIA GPUs, using the CUDA platform. It is the most mature and feature-rich backend, offering the highest performance on NVIDIA hardware.
-   **Key Files:**
    -   `tk_cuda_dispatch.cu`: Handles the dispatching of CUDA kernels.
    -   `tk_cuda_kernels.cu`: Contains the custom CUDA kernels for tensor and image operations.
    -   `tk_cuda_math_helpers.cu`: Provides math utility functions for CUDA.
    -   `extensions/cuda/tk_cuda_tensor_ops.cu`: Implements high-level tensor operations.

### 3.2. ROCm (AMD)

-   **Description:** The backend for AMD GPUs, using the ROCm (Radeon Open Compute) platform. It provides a programming model similar to CUDA for AMD hardware.
-   **Key Files:**
    -   `tk_rocm_dispatch.cpp`: Handles the dispatching of ROCm kernels.
    -   `tk_rocm_kernels.cpp`: Contains the custom HIP kernels (the C++ dialect for ROCm).
    -   `extensions/rocm/tk_rocm_tensor_ops.cpp`: Implements high-level tensor operations.

### 3.3. Metal (Apple metal)

-   **Description:** The backend for Apple's M-series SoCs (System on a Chip), using the Metal API. It is highly optimized for the unified memory architecture of Apple metal, allowing for very efficient data sharing between the CPU and GPU.
-   **Key Files:**
    -   `tk_metal_dispatch.mm`: Handles the dispatching of Metal compute shaders (kernels). The `.mm` extension indicates Objective-C++.
    -   `tk_metal_kernels.metal`: Contains the compute shaders written in the Metal Shading Language (MSL).
    -   `tk_metal_helpers.mm`: Provides helper functions for setting up Metal command queues, buffers, and pipelines.
    -   `extensions/metal/tk_metal_tensor_ops.mm`: Implements high-level tensor operations using Metal.

## 4. Integration

-   **Vision and Audio Modules:** These modules are the primary clients of the GPU module. They use it to run their respective neural network models (YOLO, MiDaS, Whisper, etc.) on the GPU.
-   **Build System (`CMakeLists.txt`):** The build system detects the available toolchains (CUDA toolkit, ROCm toolchain, Xcode) and compiles the appropriate backend. Build flags like `TRACKIE_ENABLE_CUDA` are used to control which backends are enabled.
