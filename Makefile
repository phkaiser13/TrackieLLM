# ==============================================================================
# Makefile for TrackieLLM
# ==============================================================================
#
# This Makefile is a convenience wrapper around the CMake build system.
# It simplifies common development tasks like building, running, and cleaning.
#
# Variables:
#   BUILD_TYPE  - The build type for CMake (Debug, Release, RelWithDebInfo).
#                 Default: Release
#   BUILD_DIR   - The directory where all build artifacts will be sxtored.
#                 Default: build/
#

# --- Configuration ---

# Default build type if not specified by the user.
# Use `make BUILD_TYPE=Debug` to override.
BUILD_TYPE ?= Release

# The directory for all out-of-source build files.
BUILD_DIR ?= build

# The name of the final executable defined in CMake.
TARGET_EXEC = trackiellm

# Use a shell command to get the number of CPU cores for parallel builds.
# On Linux/macOS, `nproc` or `sysctl` works. On Windows (with Git Bash/MSYS), `nproc` is often available.
# Fallback to 4 if the command fails.
CPU_CORES ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# --- Phony Targets (Targets that are not files) ---

.PHONY: all build configure run test clean install help

# --- Main Targets ---

all: build
	@echo "Build complete. Type 'make run' to execute."

# Configure the project using CMake.
# This step only needs to be run once, or when CMakeLists.txt files change.
configure:
	@echo "--- Configuring project with CMake (Build type: $(BUILD_TYPE)) ---"
	@mkdir -p $(BUILD_DIR)
	@cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

# Build the project.
# Depends on `configure` to ensure the project is configured first.
build: configure
	@echo "--- Building project with $(CPU_CORES) cores ---"
	@cmake --build $(BUILD_DIR) --parallel $(CPU_CORES)

# Run the main executable.
# Depends on `build` to ensure the project is compiled first.
run: build
	@echo "--- Running $(TARGET_EXEC) ---"
	@cd $(BUILD_DIR) && ./bin/$(TARGET_EXEC)

# Run tests (if any are defined in CMake with CTest).
test: build
	@echo "--- Running tests ---"
	@cd $(BUILD_DIR) && ctest --output-on-failure

# Clean the build directory.
# This removes all compiled files and CMake cache.
clean:
	@echo "--- Cleaning build directory ---"
	@rm -rf $(BUILD_DIR)

# Install the application.
# This will copy the executable, assets, and configs to the location
# specified by CMAKE_INSTALL_PREFIX (defaults to /usr/local).
# Use `make install PREFIX=./deploy` to install to a local directory.
install: build
	@echo "--- Installing application ---"
	@cmake --install $(BUILD_DIR) $(if $(PREFIX),--prefix $(PREFIX))

# --- Utility Targets ---

# Display help information.
help:
	@echo "TrackieLLM Makefile"
	@echo "-------------------"
	@echo "Usage: make [TARGET]"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Configure and build the project (default)."
	@echo "  build       - Compile the project."
	@echo "  configure   - Run CMake to configure the project."
	@echo "  run         - Build and run the main executable."
	@echo "  test        - Build and run tests."
	@echo "  clean       - Remove all build artifacts."
	@echo "  install     - Install the application (use with PREFIX=./path)."
	@echo "  help        - Display this help message."
	@echo ""
	@echo "Options:"
	@echo "  BUILD_TYPE=Debug|Release   - Set the CMake build type (default: Release)."
	@echo "  PREFIX=<path>              - Set the installation path for 'make install'."
