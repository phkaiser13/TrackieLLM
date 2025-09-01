#
# Copyright (C) 2025 Pedro Henrique / phdev13
#
# File: CMakeLists.txt
#
# This is the root CMake file for the TrackieLLM project. It is responsible for
# top-level project configuration, dependency management integration (vcpkg),
# setting global compiler flags, defining build options, and including the
# various subdirectories that comprise the system (e.g., src, tests, examples).
#
# SPDX-License-Identifier:
#

# --- Project Definition and Configuration ---
# Require a modern version of CMake to ensure access to robust features for
# multi-language projects, CUDA support, and dependency management.
cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

# Define the project. The version number should be updated according to
# semantic versioning principles. Languages specified here enable support
# throughout the project. CUDA is included for GPU acceleration.
project(TrackieLLM
        VERSION 0.1.0
        DESCRIPTION "Real-time, multimodal AI companion for the visually impaired."
        LANGUAGES C CXX CUDA
)

# --- Dependency Management (vcpkg) ---
# We use vcpkg for robust dependency management. The build will fail if the
# vcpkg toolchain is not specified, ensuring a reproducible build environment.
# Usage: cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=[path-to-vcpkg]/scripts/buildsystems/vcpkg.cmake
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    message(FATAL_ERROR "vcpkg toolchain not found. Please configure the project with -DCMAKE_TOOLCHAIN_FILE.")
else()
    message(STATUS "Using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
endif()

# --- Global Build Options ---
# Provide users with options to customize the build process.
option(BUILD_SHARED_LIBS "Build shared libraries (.so/.dll/.dylib)" ON)
option(TRACKIELLM_ENABLE_TESTING "Enable building of unit and integration tests" ON)
option(TRACKIELLM_ENABLE_BENCHMARKS "Enable building of performance benchmarks" OFF)
option(TRACKIELLM_ENABLE_FUZZING "Enable fuzz testing targets" OFF)
option(TRACKIELLM_ENABLE_CLANG_TIDY "Enable static analysis with clang-tidy" ON)
option(TRACKIELLM_ENABLE_ASAN "Enable AddressSanitizer for memory error detection" OFF)
option(TRACKIELLM_ENABLE_TSAN "Enable ThreadSanitizer for data race detection" OFF)

# --- Compiler and Linker Settings ---
# Set modern language standards for C and C++.
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard and enable relocatable device code for building libraries.
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(POLICY CMP0105 IS NEW)
    cmake_policy(SET CMP0105 NEW) # Let CMAKE_CUDA_ARCHITECTURES handle -arch flags
endif()
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86") # Target common NVIDIA architectures (Turing, Ampere)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Enforce Position Independent Code (PIC) for all targets, which is required
# for shared libraries and good practice for executables for security (ASLR).
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Global Include Directories ---
# The top-level include directory is made available to all targets in the project.
include_directories(src/include)

# --- Static Analysis and Sanitizers ---
# Configure static analysis and sanitizers if they were enabled in the options.
if(TRACKIELLM_ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE clang-tidy)
    if(CLANG_TIDY_EXE)
        message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-extra-arg=-Wno-unknown-pragmas")
        set(CMAKE_C_CLANG_TIDY "${CLANG_TIDY_EXE};-extra-arg=-Wno-unknown-pragmas")
    else()
        message(WARNING "clang-tidy requested but not found. Static analysis will be skipped.")
    endif()
endif()

if(TRACKIELLM_ENABLE_ASAN)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif()

if(TRACKIELLM_ENABLE_TSAN)
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
endif()

# --- Subdirectory Processing ---
# Add the main source, tests, examples, and other directories to the build.
# These directories must contain their own CMakeLists.txt files.
add_subdirectory(src)

if(TRACKIELLM_ENABLE_TESTING)
    enable_testing() # Enables CTest support
    add_subdirectory(tests)
endif()

if(TRACKIELLM_ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# The examples directory is included by default if not cross-compiling for a
# target where they cannot be run easily.
if(NOT CMAKE_CROSSCOMPILING)
    add_subdirectory(examples)
endif()

# --- Packaging (CPack) ---
# Configure CPack to generate source and binary packages (e.g., .zip, .tar.gz, .deb).
# This provides a standardized way to distribute the project.
include(CPack)

set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_VENDOR "Pedro Henrique / phdev13")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PROJECT_DESCRIPTION})
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/phdev13/TrackieLLM") # Placeholder URL
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

# Configure source archive generator
set(CPACK_SOURCE_GENERATOR "TGZ")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "${PROJECT_NAME}-${PROJECT_VERSION}-Source")

# Configure binary package generator (can be extended for DEB, RPM, etc.)
set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-${PROJECT_VERSION}-${CMAKE_SYSTEM_NAME}")

# This command must be last, after all targets and install rules are defined.
include(CPack)