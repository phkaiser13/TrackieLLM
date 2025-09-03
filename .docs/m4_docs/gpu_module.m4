include(`.docs/m4_docs/gpu_backend.m4')dnl
<!-- This documentation was written by Jules - Google labs bot. -->

# GPU Module

## 1. Overview

The GPU module provides a hardware abstraction layer for accelerating compute-intensive tasks, particularly the neural network inference required by the Vision and Audio modules. It dispatches operations to the appropriate backend based on the available hardware.

## 2. Backends

GPU_BACKEND(`CUDA (NVIDIA)`, `The backend for NVIDIA GPUs, using CUDA for general-purpose GPU programming. It is the most mature and feature-rich backend.`, `tk_cuda_dispatch.cu`, `Handles dispatching of CUDA kernels.`, `tk_cuda_kernels.cu`, `Contains the custom CUDA kernels for tensor operations.`, `tk_cuda_math_helpers.cu`, `Provides math utility functions for CUDA.`)

GPU_BACKEND(`ROCm (AMD)`, `The backend for AMD GPUs, using the ROCm platform. It provides a CUDA-like programming model for AMD hardware.`, `tk_rocm_dispatch.cpp`, `Handles dispatching of ROCm kernels.`, `tk_rocm_kernels.cpp`, `Contains the custom ROCm kernels.`, ``, ``)

GPU_BACKEND(`Metal (Apple metal)`, `The backend for Apple's M-series chips, using the Metal API for low-level GPU programming. It is optimized for the unified memory architecture of Apple metal.`, `tk_metal_dispatch.mm`, `Handles dispatching of Metal compute shaders.`, `tk_metal_kernels.metal`, `Contains the Metal compute shaders.`, `tk_metal_helpers.mm`, `Provides helper functions for working with Metal.`)
