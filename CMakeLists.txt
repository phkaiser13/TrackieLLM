#
# Copyright (C) 2025 Pedro Henrique / phdev13
#
# File: CMakeLists.txt
#
# This is the root CMake file for the TrackieLLM project. It is responsible for:
#   - Defining the project name, version, and supported languages.
#   - Setting up modern C and C++ standards with strict compiler warnings.
#   - Configuring build types (Debug, Release, etc.).
#   - Detecting and configuring hardware acceleration backends (CUDA, Metal, ROCm).
#   - Integrating with the vcpkg dependency manager.
#   - Adding the main source, test, example, and benchmark subdirectories.
#   - Defining installation rules for the library and public headers.
#
# SPDX-License-Identifier:
#

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(TrackieLLM
    VERSION 0.1.0
    DESCRIPTION "Real-time, on-device multimodal AI assistant for the visually impaired."
    LANGUAGES C CXX CUDA OBJC OBJCXX
)

#-------------------------------------------------------------------------------
# Project-wide Options and Configurations
#-------------------------------------------------------------------------------

# Set a consistent C standard (C17)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Set a consistent C++ standard (C++20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Configure output directories for a clean project structure
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable Position Independent Code, essential for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#-------------------------------------------------------------------------------
# Compiler Warnings and Static Analysis
#-------------------------------------------------------------------------------

if(MSVC)
    # Add strict warnings for MSVC
    add_compile_options(/W4 /WX)
else()
    # Add strict warnings for GCC/Clang
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wformat=2 -Wno-unused-parameter)
endif()

# Option to treat warnings as errors, enabled by default in CI
option(TRACKIELLM_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
if(TRACKIELLM_WARNINGS_AS_ERRORS OR (DEFINED ENV{CI} AND "$ENV{CI}" STREQUAL "true"))
    message(STATUS "Treating warnings as errors.")
    if(MSVC)
        add_compile_options(/WX)
    else()
        add_compile_options(-Werror)
    endif()
endif()

#-------------------------------------------------------------------------------
# Dependency Management (vcpkg)
#-------------------------------------------------------------------------------

# This project uses vcpkg in manifest mode. The user must configure it via the
# CMAKE_TOOLCHAIN_FILE variable when running CMake for the first time.
# Example: cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=[path-to-vcpkg]/scripts/buildsystems/vcpkg.cmake
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    message(WARNING "CMAKE_TOOLCHAIN_FILE is not set. vcpkg dependencies may not be found. "
                    "Please configure your environment to use the vcpkg toolchain.")
endif()

#-------------------------------------------------------------------------------
# Hardware Acceleration Backend Configuration
#-------------------------------------------------------------------------------

option(TRACKIELLM_ENABLE_CUDA "Enable CUDA backend for NVIDIA GPUs" ON)
option(TRACKIELLM_ENABLE_METAL "Enable Metal backend for Apple Silicon" ON)
option(TRACKIELLM_ENABLE_ROCM "Enable ROCm/HIP backend for AMD GPUs" OFF)
option(TRACKIELLM_ENABLE_CPU "Enable CPU backend (fallback)" ON)

if(TRACKIELLM_ENABLE_CUDA)
    find_package(CUDA 11.4)
    if(CUDA_FOUND)
        message(STATUS "CUDA backend enabled. Version: ${CUDA_VERSION_STRING}")
        add_compile_definitions(TRACKIELLM_HAS_CUDA)
    else()
        message(WARNING "CUDA backend was requested, but CUDA toolkit was not found. Disabling.")
        set(TRACKIELLM_ENABLE_CUDA OFF)
    endif()
endif()

if(TRACKIELLM_ENABLE_METAL AND APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
        message(STATUS "Metal backend enabled for Apple platform.")
        add_compile_definitions(TRACKIELLM_HAS_METAL)
    else()
        message(WARNING "Metal backend was requested on Apple platform, but frameworks were not found. Disabling.")
        set(TRACKIELLM_ENABLE_METAL OFF)
    endif()
elseif(TRACKIELLM_ENABLE_METAL AND NOT APPLE)
    message(STATUS "Metal backend can only be enabled on Apple platforms. Disabling.")
    set(TRACKIELLM_ENABLE_METAL OFF)
endif()

if(TRACKIELLM_ENABLE_ROCM)
    # ROCm/HIP configuration would go here. Typically involves finding the hipcc compiler.
    # This is left as a placeholder until the ROCm toolchain is fully integrated.
    message(STATUS "ROCm backend support is under development.")
    # find_package(ROCM) ...
    # add_compile_definitions(TRACKIELLM_HAS_ROCM)
endif()

if(TRACKIELLM_ENABLE_CPU)
    message(STATUS "CPU backend enabled.")
    add_compile_definitions(TRACKIELLM_HAS_CPU)
endif()

#-------------------------------------------------------------------------------
# Subdirectory Integration
#-------------------------------------------------------------------------------

# Add the core source code directory
add_subdirectory(src)

# Add tests if enabled
option(TRACKIELLM_BUILD_TESTS "Build project unit and integration tests" ON)
if(TRACKIELLM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Add examples if enabled
option(TRACKIELLM_BUILD_EXAMPLES "Build example usage applications" ON)
if(TRACKIELLM_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Add benchmarks if enabled
option(TRACKIELLM_BUILD_BENCHMARKS "Build performance benchmarks" ON)
if(TRACKIELLM_BUILD_BENCHMARKS)
    add_subdirectory(tools/benchmarks)
endif()

#-------------------------------------------------------------------------------
# Installation Rules
#-------------------------------------------------------------------------------

# Install the public headers from src/include
install(DIRECTORY src/include/
    DESTINATION include
)

# Installation for the main trackiellm library target will be defined in src/CMakeLists.txt
# and will be executed from here.

message(STATUS "TrackieLLM configuration complete.")
message(STATUS "Build with: cmake --build build")
message(STATUS "Run tests with: ctest --test-dir build")