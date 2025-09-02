<!--
This documentation was written by Jules - Google labs bot.
Original code by phkaiser13.
-->

# TrackieLLM GPU Inference Architecture

## 1. Overview & Core Philosophy

The TrackieLLM platform is engineered for high-performance, real-time multimodal processing on a diverse range of hardware, from powerful edge servers to resource-constrained embedded systems. A cornerstone of this capability is its sophisticated GPU acceleration architecture. When a compatible GPU is available, the system offloads computationally intensive tasks—primarily computer vision model inference—to achieve the low latency required for real-time user assistance.

This document provides a deep dive into the design principles, platform-specific backends, and operational flow of the GPU inference pipeline. It is intended for developers contributing to the core system, hardware integrators, and anyone seeking to understand how TrackieLLM leverages hardware acceleration.

**Emphasis:** The use of a GPU is a critical *performance enhancement*, not a baseline requirement. In the absence of a compatible GPU, the system gracefully falls back to CPU-based inference for all models, ensuring maximum hardware compatibility.

Our architecture is built on several key engineering principles:

  * **Abstraction is Paramount:** The core logic of the system (the "Cortex") is completely decoupled from the specifics of any single GPU API. It interacts with a generic Hardware Abstraction Layer (HAL) that presents a unified interface for dispatching computational work.
  * **Asynchronous-by-Default:** All GPU operations, including memory transfers and kernel launches, are non-blocking by design. This allows the CPU to continue with other tasks, such as audio processing or contextual reasoning, while the GPU is busy, maximizing system throughput.
  * **Explicit Synchronization:** We avoid implicit, blocking synchronization points. Control is managed through an explicit, event-based system, giving the dispatcher fine-grained control over the execution flow and dependency management between CPU and GPU tasks.
  * **Opaque Resource Handles:** All GPU resources (memory buffers, textures, events) are managed via opaque handles. This is a critical security and stability feature that prevents other modules from directly manipulating GPU memory or leaking resources.
  * **Workflow-Oriented API:** The HAL exposes high-level functions that represent complete computational workflows (e.g., `preprocess_image_for_onnx`) rather than low-level primitives (`launch_kernel_X`). This simplifies integration and reduces the surface area for errors.

## 2. The GPU Hardware Abstraction Layer (HAL)

To support multiple platforms, TrackieLLM implements a HAL that provides a consistent interface to the underlying GPU hardware. The `tk_vision_pipeline` and other high-level modules interact with this HAL, which in turn routes commands to the appropriate platform-specific backend.

This layered design is crucial for portability and maintainability.

```mermaid
graph TD
    subgraph Cortex & High-Level Logic
        A[tk_cortex_main] --> B{tk_vision_pipeline};
    end

    subgraph GPU Hardware Abstraction Layer (HAL)
        B --> C{GPU Dispatcher};
        C --> D{Select Backend};
    end

    subgraph Platform-Specific Backends
        D --> E[CUDA Dispatcher (tk_cuda_dispatch)];
        D --> F[Metal Dispatcher (tk_metal_dispatch)];
        D --> G[ROCm Dispatcher (Experimental)];
        D --> H[Android NDK/Vulkan (via ONNX Runtime)];
    end

    subgraph GPU Kernels & Drivers
        E --> I[CUDA Kernels (.cu)];
        F --> J[Metal Shaders (.metal)];
        G --> K[HIP Kernels];
        H --> L[Vendor Drivers];
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#ccf,stroke:#333,stroke-width:2px
    style C fill:#cfc,stroke:#333,stroke-width:2px
```

The `CMakeLists.txt` file contains the logic for detecting the host system and enabling the appropriate backend during compilation using options like `TK_ENABLE_CUDA` and `TK_ENABLE_METAL`.

## 3. NVIDIA CUDA Backend (Primary HPC Backend)

For NVIDIA GPUs, found in desktop systems and high-performance embedded boards like the Orange Pi with CUDA or Jetson series, the CUDA backend is the primary choice. It offers the highest performance and the most mature toolchain.

**Key Components:**

  * **`tk_cuda_dispatch.h`:** This is the high-level orchestrator for the CUDA backend. It manages the device, CUDA streams, and abstracts memory operations (`tk_cuda_dispatch_malloc`, `tk_cuda_dispatch_upload_async`) and workflow execution. It is the sole entry point for the rest of the system into the CUDA ecosystem.
  * **`tk_cuda_kernels.h`:** This header defines the strict C-callable API for the CUDA kernels. It uses parameter structs (e.g., `tk_preprocess_params_t`) to provide a stable interface between the C/C++ dispatch layer and the CUDA C++ (`.cu`) implementation. This separation of concerns is a core design principle.
  * **`tk_cuda_math_helpers.h`:** A crucial "bilingual" header compatible with both standard C++ and NVCC. It provides fundamental, `__host__ __device__` qualified math types (vectors, matrices) and inlined functions, eliminating code duplication between the CPU and GPU.

**Execution Flow (Example: YOLOv5nu Pre-processing):**

1.  The `tk_vision_pipeline` receives a new video frame.
2.  It requests a pre-processing operation from the GPU Dispatcher.
3.  The dispatcher, configured for CUDA, calls `tk_cuda_dispatch_preprocess_image`.
4.  The CUDA dispatcher:
    a.  Initiates an asynchronous memory copy to upload the video frame from CPU RAM to a `tk_gpu_buffer_t` on the GPU VRAM.
    b.  Populates a `tk_preprocess_params_t` struct with pointers to the GPU source buffer and a destination tensor buffer.
    c.  Launches the `tk_kernels_preprocess_image` CUDA kernel onto a CUDA stream. This kernel performs resizing, data type conversion (uint8 to float32), layout conversion (interleaved to planar NCHW), and normalization in a single, highly parallel pass.
5.  Control returns immediately to the `tk_vision_pipeline`, which can proceed to enqueue the ONNX Runtime inference task. ONNX Runtime, also configured with a CUDA provider, will then consume the pre-processed tensor directly from GPU memory without needing a round-trip to the CPU.

## 4. Apple Metal Backend (macOS & iOS)

For Apple platforms, Metal is the native, high-performance graphics and compute API. TrackieLLM leverages Metal for optimal performance on iPhones, iPads, and macOS devices. This is a high-priority, fully supported backend.

**Key Components:**

  * **`tk_metal_helpers.h`:** This header serves as a C-callable facade over the Objective-C Metal API. It abstracts the verbosity of creating devices, command queues, pipeline state objects (PSOs), and resources like buffers and textures. It also provides a crucial error translation function (`tk_metal_translate_error`) to map `NSError` objects into the project's standard `tk_error_code_t` enum.
  * **`tk_metal_dispatch.mm` (Implementation):** This Objective-C++ file contains the concrete implementation of the dispatch logic, using the helpers to interact with the Metal framework.
  * **Metal Shaders (`.metal` files):** These files contain the kernel code written in the Metal Shading Language (MSL).

**Execution Flow:**

The flow is conceptually similar to CUDA but uses Metal's paradigms:

1.  On startup, `tk_metal_get_default_device` is called to get a reference to the GPU. A `MTLCommandQueue` is created.
2.  The `tk_vision_pipeline` requests a pre-processing operation.
3.  The Metal dispatcher creates a `MTLCommandBuffer`.
4.  It uses a `MTLBlitCommandEncoder` to efficiently copy the `tk_video_frame_t` data into a `MTLTexture`.
5.  It then creates a `MTLComputeCommandEncoder` and sets the appropriate compute pipeline state object (PSO), which was pre-compiled from a `.metal` shader function.
6.  It binds the input texture and output `MTLBuffer` and dispatches the compute kernel.
7.  The command buffer is committed, and control returns to the caller. The ONNX Runtime, configured with a CoreML (which uses Metal) provider, can then run inference on the GPU-resident data.

## 5. AMD ROCm Backend (Experimental)

Support for AMD GPUs is provided through ROCm, which is primarily targeted at Linux-based desktop and server environments.

  * **Approach:** For this project, the ROCm support is designed as a lightweight wrapper. The `CMakeLists.txt` indicates that ROCm is an optional, experimental backend. The primary strategy involves using the **HIP (Heterogeneous-compute Interface for Portability)** API. HIP provides a C++ dialect and runtime that is very similar to CUDA, allowing developers to write code that can be compiled to run on both AMD and NVIDIA hardware with minimal changes.
  * **Implementation:** The ROCm backend consists of wrappers that map the high-level dispatch calls to their HIP equivalents. Given the similarities, much of the logic from the CUDA backend can be reused. For example, `cudaMalloc` becomes `hipMalloc`. This makes ROCm support primarily a matter of build system configuration and API mapping rather than a complete rewrite of the GPU logic.

## 6. Android Accelerators (Vulkan & Others)

Android presents a more fragmented ecosystem with a variety of GPU vendors (Qualcomm Adreno, ARM Mali, etc.). TrackieLLM's strategy for Android focuses on leveraging standardized APIs and the capabilities of the ONNX Runtime.

  * **Primary API: Vulkan:** Vulkan is the modern, low-level, cross-platform graphics and compute API that has succeeded OpenGL ES on Android. It provides direct, high-performance access to the GPU.
  * **Implementation Strategy:** Instead of writing a bespoke Vulkan dispatcher within TrackieLLM, we leverage the highly optimized backends already present in the **ONNX Runtime**. When building the TrackieLLM Android application, the core logic is compiled as a native library (`.so`) using the NDK. This native library is bundled with a version of ONNX Runtime that has its Vulkan or OpenGL ES execution providers enabled.
  * **Execution Flow:**
    1.  The `tk_vision_pipeline` running in the native C/C++ layer on Android receives a video frame.
    2.  It performs any necessary pre-processing on the CPU (or via specialized NDK libraries if available).
    3.  It passes the prepared tensor to the ONNX Runtime C++ API.
    4.  The ONNX Runtime, configured to prioritize the **Vulkan Execution Provider**, takes over. It manages the data transfer to the GPU, the execution of the model (YOLO, MiDaS), and the retrieval of the results.

This approach abstracts the complexity of dealing with different Android GPU drivers and vendor-specifics, delegating the task to the specialized ONNX Runtime team.

## 7. Model Execution and Further Reading

The primary beneficiaries of this GPU architecture are the computer vision models, which are typically formatted as ONNX files.

  * **YOLOv5nu:** Used for object detection.
  * **DPT-SwinV2-Tiny (MiDaS):** Used for monocular depth estimation.

These models are loaded by the `tk_vision_pipeline`, and their inference sessions are configured to use the appropriate GPU execution provider (CUDA, CoreML/Metal, Vulkan) if the build configuration and runtime environment support it.

For more detailed information on the specific models used, their formats (ONNX, GGUF), and their training data, please refer to the official **TrackieAssets** repository:

> [https://github.com/phkaiser13/TrackieAssets.git](https://github.com/phkaiser13/TrackieAssets.git)