#
# Copyright (C) 2025 Pedro Henrique / phdev13
#
# File: CMakeLists.txt
#
# This is the root CMake script for the TrackieLLM project. It is responsible for
# setting up the project-wide configurations, detecting necessary tools and
# libraries, defining build options, and including the subdirectories that
# contain the actual source code and build targets.
#
# SPDX-License-Identifier: AGPL-3.0 license
#

#--------------------------------------------------------------------------------
# Project Definition and Configuration
#--------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(TrackieLLM
    VERSION 1.0.0
    DESCRIPTION "High-performance, multi-modal AI system for visually impaired users."
    LANGUAGES C CXX
)

# Set a consistent version variable for use throughout the build scripts
set(TRACKIELLM_VERSION ${PROJECT_VERSION})
message(STATUS "Configuring TrackieLLM version ${TRACKIELLM_VERSION}")

#--------------------------------------------------------------------------------
# Standard and Compiler Feature Setup
#--------------------------------------------------------------------------------
# Enforce modern C and C++ standards. C++17 is required for features like
# std::filesystem and std::optional. C11 is a modern baseline for C.
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# For Apple platforms, Objective-C++ requires a standard as well.
if(APPLE)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
endif()

# Set the position-independent code flag, which is required for shared libraries
# and often good practice for executables for security (ASLR).
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#--------------------------------------------------------------------------------
# Build Options and Customizations
#--------------------------------------------------------------------------------
# These options allow developers to tailor the build to their specific hardware
# and development needs, such as enabling or disabling specific GPU backends.
option(TK_ENABLE_CUDA "Enable NVIDIA CUDA support for GPU acceleration" ON)
option(TK_ENABLE_METAL "Enable Apple Metal support for GPU acceleration (macOS/iOS)" ON)
option(TK_ENABLE_ROCM "Enable AMD ROCm support for GPU acceleration" OFF)
option(TK_BUILD_TESTS "Build unit and integration tests" ON)
option(TK_BUILD_BENCHMARKS "Build performance benchmarks" ON)

#--------------------------------------------------------------------------------
# Compiler Flags and Warnings
#--------------------------------------------------------------------------------
# A strict warning policy is crucial for catching potential bugs early.
# These flags are common across GCC and Clang.
set(COMMON_WARNING_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wformat=2
    -Wconversion
    -Wsign-conversion
    -Wnull-dereference
    -Wdouble-promotion
    -Wimplicit-fallthrough
)

# MSVC has a different set of warning flags.
if(MSVC)
    set(COMMON_WARNING_FLAGS /W4 /WX)
else()
    # Treat warnings as errors in non-Debug builds to enforce clean code.
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND COMMON_WARNING_FLAGS -Werror)
    endif()
endif()

foreach(lang C CXX)
    add_compile_options(${COMMON_WARNING_FLAGS})
endforeach()

#--------------------------------------------------------------------------------
# GPU Backend Configuration
#--------------------------------------------------------------------------------
if(TK_ENABLE_CUDA)
    enable_language(CUDA)
    # Find the CUDA toolkit. The `find_package` command for CUDA is robust and
    # sets up all necessary variables and targets.
    find_package(CUDA 11.0 REQUIRED)
    if(CUDA_FOUND)
        message(STATUS "CUDA support enabled. Found CUDA toolkit version ${CUDA_VERSION}.")
        # Propagate CUDA support to subdirectories
        add_definitions(-DTK_HAS_CUDA)
    else()
        message(WARNING "CUDA support was enabled (TK_ENABLE_CUDA=ON) but the CUDA toolkit was not found. Disabling CUDA.")
        set(TK_ENABLE_CUDA OFF)
    endif()
endif()

if(APPLE AND TK_ENABLE_METAL)
    # Metal support is primarily handled by checking for the availability of the
    # Metal and Foundation frameworks.
    find_library(METAL_LIBRARY Metal)
    find_library(FOUNDATION_LIBRARY Foundation)
    if(METAL_LIBRARY AND FOUNDATION_LIBRARY)
        message(STATUS "Apple Metal support enabled.")
        add_definitions(-DTK_HAS_METAL)
    else()
        message(WARNING "Metal support was enabled (TK_ENABLE_METAL=ON) but required frameworks were not found. Disabling Metal.")
        set(TK_ENABLE_METAL OFF)
    endif()
endif()

# Placeholder for ROCm configuration
if(TK_ENABLE_ROCM)
    message(STATUS "ROCm support is enabled (experimental).")
    # Add find_package logic for HIP/ROCm here when implemented.
    add_definitions(-DTK_HAS_ROCM)
endif()

# Vulkan Backend Configuration
option(TK_ENABLE_VULKAN "Enable Vulkan support for GPU acceleration" ON)
if(TK_ENABLE_VULKAN)
    find_package(Vulkan REQUIRED)
    if(Vulkan_FOUND)
        message(STATUS "Vulkan support enabled. Found Vulkan version ${Vulkan_VERSION}.")
        add_definitions(-DTK_HAS_VULKAN)
    else()
        message(WARNING "Vulkan support was enabled (TK_ENABLE_VULKAN=ON) but the Vulkan SDK was not found. Disabling Vulkan.")
        set(TK_ENABLE_VULKAN OFF)
    endif()
endif()

# OpenGL ES Backend Configuration
option(TK_ENABLE_GLES "Enable OpenGL ES support for GPU acceleration" ON)
if(TK_ENABLE_GLES)
    find_package(EGL REQUIRED)
    find_package(GLESv3 REQUIRED)
    if(EGL_FOUND AND GLESv3_FOUND)
        message(STATUS "OpenGL ES support enabled.")
        add_definitions(-DTK_HAS_GLES)
    else()
        message(WARNING "GLES support was enabled (TK_ENABLE_GLES=ON) but EGL or GLESv3 libraries were not found. Disabling GLES.")
        set(TK_ENABLE_GLES OFF)
    endif()
endif()

# OpenCL Backend Configuration
option(TK_ENABLE_OPENCL "Enable OpenCL support for GPU acceleration" ON)
if(TK_ENABLE_OPENCL)
    find_package(OpenCL REQUIRED)
    if(OpenCL_FOUND)
        message(STATUS "OpenCL support enabled.")
        add_definitions(-DTK_HAS_OPENCL)
    else()
        message(WARNING "OpenCL support was enabled (TK_ENABLE_OPENCL=ON) but OpenCL was not found. Disabling OpenCL.")
        set(TK_ENABLE_OPENCL OFF)
    endif()
endif()

# Android NNAPI Backend Configuration
option(TK_ENABLE_NNAPI "Enable Android NNAPI support for model acceleration" ON)
if(ANDROID AND TK_ENABLE_NNAPI)
    # NNAPI is part of the Android NDK, found with the toolchain.
    # We just need to link against the `neuralnetworks` library.
    find_library(NEURALNETWORKS_LIBRARY neuralnetworks)
    if(NEURALNETWORKS_LIBRARY)
         message(STATUS "Android NNAPI support enabled.")
         add_definitions(-DTK_HAS_NNAPI)
    else()
        message(WARNING "NNAPI support was enabled but the neuralnetworks library was not found. Disabling NNAPI.")
        set(TK_ENABLE_NNAPI OFF)
    endif()
endif()

#--------------------------------------------------------------------------------
# Rust/Cargo Integration
#--------------------------------------------------------------------------------
# We use Corrosion to integrate Rust crates seamlessly into the CMake build process.
# It handles the `cargo build` calls and creates imported targets for CMake.
# find_package(Corrosion REQUIRED)

#--------------------------------------------------------------------------------
# Subdirectory Processing
#--------------------------------------------------------------------------------
# Add the main source directory. All targets will be defined in the CMakeLists.txt
# file within this directory and its children.
add_subdirectory(src)

# Add tests and benchmarks if enabled
if(TK_BUILD_TESTS)
    add_subdirectory(tests)
    message(STATUS "Test builds are enabled.")
endif()

if(TK_BUILD_BENCHMARKS)
    # add_subdirectory(benchmarks)
    message(STATUS "Benchmark builds are enabled.")
endif()

message(STATUS "TrackieLLM configuration complete.")