#
# Copyright (C) 2025 Pedro Henrique / phdev13
#
# File: src/CMakeLists.txt
#
# This CMake file orchestrates the build of the entire TrackieLLM source tree.
# It follows a modular design pattern where each subdirectory (core, hal, cortex,
# etc.) is responsible for building its own static library. This file then
# collects these modular libraries and links them into the final primary
# artifact: the 'trackiellm' library (either shared or static, depending on
# the BUILD_SHARED_LIBS option). It also defines the public API exposure and
# installation rules for the library and its headers.
#
# SPDX-License-Identifier:
#

# --- Modular Subdirectories ---
# Each of the following subdirectories contains a functional domain of the
# TrackieLLM system. They are compiled as separate static libraries to enforce
# modularity, improve build parallelism, and clarify dependencies.
add_subdirectory(core)
add_subdirectory(hal)
add_subdirectory(cortex)
add_subdirectory(vision)
add_subdirectory(audio)
add_subdirectory(ffi)

# The 'tools' directory contains executables (like a CLI) that link against
# the main trackiellm library. It is added after the library is defined.
add_subdirectory(tools)

# --- Main Library Target Definition ---
# Define the primary library target for the project. By not specifying STATIC or
# SHARED, we allow the global BUILD_SHARED_LIBS option (defined in the root
# CMakeLists.txt) to control the final library type.
add_library(trackiellm)

# --- Linking Stage ---
# Link the main 'trackiellm' library against all the internal static libraries
# built by the subdirectories. The PUBLIC keyword ensures that any usage
# requirements (like include directories or linked dependencies) from the
# sub-libraries are propagated to consumers of 'trackiellm'.
target_link_libraries(trackiellm
    PUBLIC
        tk_core
        tk_hal
        tk_cortex
        tk_vision
        tk_audio
        tk_ffi
)

# --- Public API and Versioning ---
# Define the public include directory for consumers of this library.
# We use generator expressions to specify different paths for the build tree
# and the final installation, which is a modern CMake best practice.
target_include_directories(trackiellm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Set version properties for the library. This is crucial for managing ABI
# compatibility on platforms that support it (e.g., Linux, macOS).
# SOVERSION is typically the major version number.
set_target_properties(trackiellm PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/trackiellm.h"
)

# --- Installation Rules ---
# Include standard GNU installation directory variables (e.g., CMAKE_INSTALL_LIBDIR).
include(GNUInstallDirs)

# Define installation rules for the library target and its public headers.
# This allows the project to be properly installed on a system or packaged.
install(TARGETS trackiellm
    EXPORT TrackieLLMTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/trackiellm
)

# Install the entire public API header directory.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# --- CMake Package Configuration ---
# Generate and install a CMake package configuration file. This allows other
# CMake projects to easily find and use our installed library using
# `find_package(TrackieLLM)`.
install(EXPORT TrackieLLMTargets
    FILE TrackieLLMTargets.cmake
    NAMESPACE TrackieLLM::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TrackieLLM
)

# Generate the version configuration file for find_package version checking.
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/TrackieLLMConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/TrackieLLMConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TrackieLLM
)