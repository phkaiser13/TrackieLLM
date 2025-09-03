#!/bin/bash

#
# Copyright (C) 2025 Pedro Henrique / phdev13
#
# File: ci/build.sh
#
# This script handles the compilation of the project in a controlled and
# optimized CI environment. It goes beyond a simple 'cargo build' by
# incorporating best practices for performance, reproducibility, and analysis
# of the final artifact. It is designed to be both robust and informative.
#
# Dependencies: bash, coreutils, rustc, cargo, time
#
# SPDX-License-Identifier: AGPL-3.0 license
#

# ---
# Script Configuration and Safety
# ---
# -e: Exit immediately if a command exits with a non-zero status.
# -u: Treat unset variables as an error when substituting.
# -o pipefail: The return value of a pipeline is the status of the last
#              command to exit with a non-zero status, or zero if no
#              command exited with a non-zero status.
set -euo pipefail

# ---
# Logging Utilities
# ---
# Provides structured and colored output for better readability in CI logs.
log_info() {
    printf "\033[34m[INFO]\033[0m %s\n" "$1"
}

log_success() {
    printf "\033[32m[SUCCESS]\033[0m %s\n" "$1"
}

log_warn() {
    printf "\033[33m[WARN]\033[0m %s\n" "$1"
}

log_error() {
    printf "\033[31m[ERROR]\033[0m %s\n" "$1" >&2
}

# ---
# Main Build Logic
# ---
main() {
    log_info "Starting Production Build Process..."

    # ---
    # 1. Environment Verification
    # ---
    # Before starting, we must ensure that the necessary toolchain is available.
    # This prevents cryptic errors later in the process.
    log_info "Verifying build environment..."
    if ! command -v cargo &> /dev/null || ! command -v rustc &> /dev/null; then
        log_error "Rust toolchain (cargo, rustc) not found. Please install Rust."
        exit 1
    fi
    log_info "Rust toolchain found: $(rustc --version)"

    # ---
    # 2. Configuration
    # ---
    # We define our build profile and optimization flags here.
    # Using environment variables allows for flexible configuration.
    # RUSTFLAGS are crucial for performance, instructing the compiler to use
    # advanced optimizations.
    # -C target-cpu=native: Optimizes for the specific CPU architecture of the build machine.
    # -C link-arg=...: Passes flags to the linker for further optimization of the binary.
    local build_profile="release"
    export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native -C link-arg=-Wl,-O1,--sort-common,--as-needed"
    
    log_info "Build Profile: ${build_profile}"
    log_info "Compiler Flags (RUSTFLAGS): ${RUSTFLAGS}"

    # ---
    # 3. Workspace Cleaning
    # ---
    # To ensure a completely clean and reproducible build, we purge all previous
    # artifacts. This prevents stale dependencies or build cache issues.
    log_info "Cleaning workspace to ensure a fresh build..."
    cargo clean
    log_info "Workspace cleaned."

    # ---
    # 4. Compilation
    # ---
    # This is the core build step. We use 'time' to measure the duration,
    # which is critical for monitoring CI performance. The '--verbose' flag
    # provides detailed output from the compiler, aiding in debugging.
    log_info "Compiling project in '${build_profile}' mode..."
    time cargo build --profile "${build_profile}" --verbose

    # ---
    # 5. Post-Build Analysis
    # ---
    # A compiled binary is not the end of the story. We must analyze the artifact
    # to ensure it meets our expectations.
    log_info "Analyzing final binary..."
    
    # Locate the binary artifact path
    local artifact_path
    artifact_path=$(cargo metadata --format-version=1 --no-deps | jq -r ".target_directory")
    local package_name
    package_name=$(cargo metadata --format-version=1 --no-deps | jq -r ".packages[0].name")
    local binary_path="${artifact_path}/${build_profile}/${package_name}"

    if [[ ! -f "${binary_path}" ]]; then
        log_error "Build artifact not found at expected path: ${binary_path}"
        exit 1
    fi

    # Check binary size. Bloated binaries can be a performance concern.
    log_info "Binary size:"
    ls -lh "${binary_path}"

    # Check dynamic library dependencies. For high-performance applications,
    # we aim for minimal dynamic linking.
    log_info "Dynamic library dependencies:"
    ldd "${binary_path}" || log_warn "Could not run 'ldd'. This may be a non-Linux OS."

    log_success "Build process completed successfully. Artifact is ready at: ${binary_path}"
}

# ---
# Script Execution
# ---
# This ensures the main logic is called only when the script is executed directly.
main "$@"
