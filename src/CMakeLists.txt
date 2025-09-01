#
# Copyright (C) 2025 Pedro Henrique / phdev13
#
# File: src/CMakeLists.txt
#
# This CMake file orchestrates the build of the main `trackiellm` library.
# Its responsibilities include:
#   - Discovering all source files for C, C++, CUDA, and Metal backends.
#   - Conditionally compiling source files based on enabled hardware backends.
#   - Defining the `trackiellm` shared/static library target.
#   - Integrating the Rust FFI crate build via ExternalProject.
#   - Linking all necessary dependencies (GPU runtimes, third-party libs).
#   - Setting public include directories for the library's API.
#   - Defining installation rules for the final library artifact.
#
# SPDX-License-Identifier:
#

#-------------------------------------------------------------------------------
# Source File Discovery
#-------------------------------------------------------------------------------

# Use file globbing to automatically discover source files. This simplifies
# maintenance as new files are added to the source tree.

# Core C sources (platform-agnostic logic)
file(GLOB_RECURSE CORE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/core/*.c"
)

# C++ sources (AI model adapters and high-level logic)
file(GLOB_RECURSE CPP_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/cortex/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/vision/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/audio/*.cpp"
)

# HAL (Hardware Abstraction Layer) sources are added conditionally
set(HAL_SOURCES "")
set(HAL_LINK_LIBS "")
set(HAL_INCLUDE_DIRS "")

#-------------------------------------------------------------------------------
# Conditional Backend Source Compilation
#-------------------------------------------------------------------------------

if(TRACKIELLM_ENABLE_CUDA AND CUDA_FOUND)
    file(GLOB_RECURSE CUDA_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/hal/cuda/*.cu"
    )
    list(APPEND HAL_SOURCES ${CUDA_SOURCES})
    list(APPEND HAL_LINK_LIBS CUDA::cudart)
    list(APPEND HAL_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
    # Enable CUDA language support for the target
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    set(CMAKE_CUDA_STANDARD 20)
endif()

if(TRACKIELLM_ENABLE_METAL AND APPLE)
    file(GLOB_RECURSE METAL_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/hal/silicon/*.m"
        "${CMAKE_CURRENT_SOURCE_DIR}/hal/silicon/*.metal"
    )
    list(APPEND HAL_SOURCES ${METAL_SOURCES})
    list(APPEND HAL_LINK_LIBS "-framework Foundation" "-framework Metal")
endif()

if(TRACKIELLM_ENABLE_ROCM)
    # Placeholder for ROCm/HIP source discovery and linking
    # file(GLOB_RECURSE ROCM_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/hal/rocm/*.hip.cpp")
    # list(APPEND HAL_SOURCES ${ROCM_SOURCES})
    # list(APPEND HAL_LINK_LIBS hip::hipruntime)
endif()

if(TRACKIELLM_ENABLE_CPU)
    file(GLOB_RECURSE CPU_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/hal/cpu/*.c"
    )
    list(APPEND HAL_SOURCES ${CPU_SOURCES})
endif()

#-------------------------------------------------------------------------------
# Rust FFI Crate Integration (via ExternalProject)
#-------------------------------------------------------------------------------

# We build the Rust FFI crate as a static library using ExternalProject, which
# calls `cargo` to perform the build.

include(ExternalProject)

set(RUST_FFI_CRATE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ffi/rust/trackiellm-ffi)
set(RUST_FFI_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/rust_build)
set(RUST_FFI_LIB_NAME trackiellm_ffi)
set(RUST_FFI_ARTIFACT ${RUST_FFI_BUILD_DIR}/release/lib${RUST_FFI_LIB_NAME}.a)

ExternalProject_Add(trackiellm_ffi_build
    SOURCE_DIR          ${RUST_FFI_CRATE_DIR}
    BINARY_DIR          ${RUST_FFI_BUILD_DIR}
    # Always configure, never download or update
    CONFIGURE_COMMAND   ""
    UPDATE_COMMAND      ""
    # Use cargo to build in release mode
    BUILD_COMMAND       cargo build --release --target-dir <BINARY_DIR>
    INSTALL_COMMAND     ""
    # Define the output artifact
    LOG_BUILD           ON
)

# Create an IMPORTED library target to represent the compiled Rust artifact.
add_library(${RUST_FFI_LIB_NAME} STATIC IMPORTED)
set_target_properties(${RUST_FFI_LIB_NAME} PROPERTIES
    IMPORTED_LOCATION ${RUST_FFI_ARTIFACT}
)
# Ensure the Rust project is built before any target that links to it.
add_dependencies(${RUST_FFI_LIB_NAME} trackiellm_ffi_build)

#-------------------------------------------------------------------------------
# Main Library Target Definition
#-------------------------------------------------------------------------------

add_library(trackiellm ${CORE_SOURCES} ${CPP_SOURCES} ${HAL_SOURCES})
add_dependencies(trackiellm ${RUST_FFI_LIB_NAME})

# Set target properties, like C++ standard
set_target_properties(trackiellm PROPERTIES
    CXX_STANDARD ${CMAKE_CXX_STANDARD}
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    C_STANDARD ${CMAKE_C_STANDARD}
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
    VERSION ${PROJECT_VERSION}
)

# Make public headers available to consumers of this library
target_include_directories(trackiellm
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HAL_INCLUDE_DIRS}
)

# Link against all required libraries
target_link_libraries(trackiellm
    PUBLIC
        # Link the compiled Rust FFI library
        ${RUST_FFI_LIB_NAME}
        # Link HAL-specific libraries
        ${HAL_LINK_LIBS}
    PRIVATE
        # Dependencies from vcpkg will be linked here.
        # These are placeholders until vcpkg.json is created.
        # Example:
        #   onnxruntime
        #   Tesseract::tesseract
        #   Threads::Threads
)

#-------------------------------------------------------------------------------
# Installation
#-------------------------------------------------------------------------------

install(TARGETS trackiellm
    EXPORT TrackieLLMTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(EXPORT TrackieLLMTargets
    FILE TrackieLLMTargets.cmake
    NAMESPACE TrackieLLM::
    DESTINATION lib/cmake/TrackieLLM
)