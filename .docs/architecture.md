<!--
This documentation was written by Jules - Google labs bot.
Original code by phkaiser13.
-->

# Source Code Architecture

This document provides a high-level overview of the source code organization within the `src` directory. The project is a hybrid of C, C++, and Rust, with each language chosen for its strengths in different areas of the system.

*   **C (`.c`, `.h`):** Used for low-level hardware interfaces, core data structures, and the public-facing Foreign Function Interface (FFI). C provides a stable ABI and is ideal for interfacing with the operating system and various hardware components.
*   **C++ (`.cpp`, `.hpp`, `.cu`, `.mm`):** Used for performance-critical components, especially in the vision and GPU pipelines. C++ provides the high-level abstractions and performance needed for complex tasks like image processing and GPU kernel management. CUDA (`.cu`) and Objective-C++ (`.mm`) are used for NVIDIA and Apple GPU programming, respectively.
*   **Rust (`.rs`):** Used for high-level application logic, memory management, and safety-critical components like the `cortex` reasoning engine. Rust's focus on memory safety and concurrency makes it an excellent choice for building robust and reliable systems.

## Directory Structure

The `src` directory is organized into modules, each with a specific responsibility. Many modules contain a mix of C/C++ and Rust code, using an FFI boundary to communicate between them.

```
src/
│
├── ai_models/          # Core AI model loading and execution (ONNX, GGUF)
├── async_tasks/        # Asynchronous task scheduling and worker pools
├── audio/              # Audio processing pipeline (ASR, TTS)
├── components/         # High-level component definitions
├── core_build/         # Build system and compilation logic
├── cortex/             # Central reasoning engine and decision-making
├── deployment/         # Software update and package management
├── experiments/        # Benchmarking and model testing tools
├── ffi/                # Foreign Function Interface (C API)
├── gpu/                # GPU abstraction layer (CUDA, Metal, ROCm)
│   ├── cuda/
│   ├── rocm/
│   └── silicon/
├── integration/        # Integration with external systems and plugins
├── interaction/        # User interaction (voice commands, feedback)
├── internal_tools/     # Internal utilities (config parsing, file management)
├── logging_ext/        # Extended logging and auditing
├── monitoring/         # System health monitoring and telemetry
├── navigation/         # Path planning and obstacle avoidance
├── networking/         # Network communication
├── profiling/          # Performance profiling tools
├── security/           # Security and encryption
├── sensors/            # Sensor fusion and processing
├── utils/              # General utility functions
└── vision/             # Vision processing pipeline (object detection, depth, OCR)
```

## Developer and Author

*   **Original Code:** phkaiser13
*   **Documentation:** Jules - Google labs bot
