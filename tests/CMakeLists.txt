#
# Copyright (C) 2025 Pedro Henrique / phdev13
#
# File: tests/CMakeLists.txt
#
# This script builds the unit and integration tests for TrackieLLM.
#
# SPDX-License-Identifier: AGPL-3.0 license
#

# Enable CTest for this directory
enable_testing()

# Set the C++ standard for the test files, as they might not inherit it
set(CMAKE_CXX_STANDARD 17)

# --- Cortex Test ---
add_executable(cortex_test tk_cortex_test.cpp)
# The main library target is `trackiellm_core` as defined in src/core_build/CMakeLists.txt
target_link_libraries(cortex_test PRIVATE trackiellm_core)
add_test(NAME CortexTest COMMAND cortex_test)

# --- Core Test ---
add_executable(core_test tk_core_test.c)
target_link_libraries(core_test PRIVATE trackiellm_core)
add_test(NAME CoreTest COMMAND core_test)

# --- HAL Test ---
add_executable(hal_test tk_hal_test.cpp)
target_link_libraries(hal_test PRIVATE trackiellm_core)
add_test(NAME HalTest COMMAND hal_test)

message(STATUS "Test executables configured.")
