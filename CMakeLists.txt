#
# Copyright (C) 2025 Pedro Henrique / phdev13
#
# File: CMakeLists.txt
#
# This is the root CMake build script for the TrackieLLM project.
# It is responsible for top-level project configuration, dependency management,
# toolchain detection (including GPU backends), and orchestrating the build
# of all sub-modules (C, C++, Rust).
#
# SPDX-License-Identifier:
#

# --- 1. Project Definition and Boilerplate ---
# Require a modern version of CMake for features like FetchContent and robust
# language support.
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(TrackieLLM
    VERSION 1.0.0
    DESCRIPTION "Real-time, on-device multimodal AI assistant for the visually impaired."
    LANGUAGES C CXX CUDA OBJCXX
)

# --- 2. Global Configuration and Options ---
# Provide a standard option to control the build of the test suite.
option(TK_BUILD_TESTS "Build unit and integration tests" ON)

# Set project-wide compiler standards.
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Enforce standard-required flags for portability.
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set a global output directory for binaries to keep the build tree clean.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- 3. GPU Acceleration Backend Detection ---
# This section detects available GPU toolkits and sets up compile-time flags.
# Users can override detection by passing -D<OPTION>=ON/OFF to CMake.

# CUDA (NVIDIA)
option(TK_ENABLE_CUDA "Enable CUDA backend for NVIDIA GPUs" ON)
if(TK_ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(TK_CUDA_FOUND TRUE)
        message(STATUS "CUDA Toolkit found (version ${CUDAToolkit_VERSION}). CUDA backend enabled.")
    else()
        set(TK_CUDA_FOUND FALSE)
        message(STATUS "CUDA Toolkit not found. CUDA backend will be disabled.")
    endif()
endif()

# Metal (Apple Silicon)
option(TK_ENABLE_METAL "Enable Metal backend for Apple Silicon" ON)
if(TK_ENABLE_METAL AND CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(TK_METAL_FOUND TRUE)
    message(STATUS "Apple platform detected. Metal backend enabled.")
else()
    set(TK_METAL_FOUND FALSE)
    if(TK_ENABLE_METAL)
        message(STATUS "Metal backend is only available on Apple platforms. Disabling.")
    endif()
endif()

# ROCm (AMD)
option(TK_ENABLE_ROCM "Enable ROCm backend for AMD GPUs" ON)
if(TK_ENABLE_ROCM)
    # ROCm detection can be complex. We check for a key environment variable or a standard path.
    # A more robust check might be needed for production.
    if(DEFINED ENV{ROCM_PATH} OR EXISTS "/opt/rocm")
        set(TK_ROCM_FOUND TRUE)
        message(STATUS "ROCm installation detected. ROCm backend enabled.")
    else()
        set(TK_ROCM_FOUND FALSE)
        message(STATUS "ROCm installation not found. ROCm backend will be disabled.")
    endif()
endif()

# --- 4. Dependency Management (FetchContent) ---
# We use FetchContent to manage external dependencies. This avoids git submodules
# and makes the build self-contained and reproducible.
include(FetchContent)

# Corrosion: The bridge between CMake and Rust's Cargo build system.
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.4 # Use a specific, stable tag
)
FetchContent_MakeAvailable(Corrosion)

# GoogleTest: The primary framework for C/C++ unit and integration tests.
if(TK_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
endif()


# --- 5. Rust Workspace Integration ---
# Import the entire Rust workspace defined in Cargo.toml as a single CMake target.
# Corrosion will orchestrate the `cargo build` process and make the resulting
# libraries available to CMake targets.
corrosion_import_crate(MANIFEST_PATH ${CMAKE_SOURCE_DIR}/Cargo.toml)


# --- 6. Main Build Logic ---
# Delegate the definition of libraries and executables to the subdirectories.
# This keeps the root CMakeLists.txt clean and focused on configuration.
add_subdirectory(src)

# Conditionally add the test suite to the build.
if(TK_BUILD_TESTS)
    enable_testing()
    # The tests directory will contain tests that span multiple modules.
    # add_subdirectory(tests) # This will be enabled once the tests directory is created.
endif()


# --- 7. Final Build Summary ---
# Provide a clear summary of the configured build for the user.
message(STATUS "--------------------------------------------------")
message(STATUS "TrackieLLM Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "")
message(STATUS "Enabled GPU Backends:")
message(STATUS "  CUDA (NVIDIA):   ${TK_CUDA_FOUND}")
message(STATUS "  Metal (Apple):     ${TK_METAL_FOUND}")
message(STATUS "  ROCm (AMD):        ${TK_ROCM_FOUND}")
message(STATUS "")
message(STATUS "Build Options:")
message(STATUS "  Build Tests:     ${TK_BUILD_TESTS}")
message(STATUS "--------------------------------------------------")