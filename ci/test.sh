#!/bin/bash

#
# Copyright (C) 2025 Pedro Henrique / phdev13
#
# File: ci/test.sh
#
# This script orchestrates the execution of the project's full test suite.
# It ensures that all layers of testing (unit, integration, and documentation)
# are executed, providing a comprehensive quality gate. Running tests under
# different profiles (e.g., debug and release) helps catch bugs that may only
# appear under compiler optimizations.
#
# Dependencies: bash, coreutils, rustc, cargo
#
# SPDX-License-Identifier: AGPL-3.0 license
#

# ---
# Script Configuration and Safety
# ---
set -euo pipefail

# ---
# Logging Utilities
# ---
log_info() {
    printf "\033[34m[INFO]\033[0m %s\n" "$1"
}

log_success() {
    printf "\033[32m[SUCCESS]\033[0m %s\n" "$1"
}

log_error() {
    printf "\033[31m[ERROR]\033[0m %s\n" "$1" >&2
}

# ---
# Main Test Logic
# ---
main() {
    log_info "Starting Comprehensive Test Suite..."

    # ---
    # 1. Environment Verification
    # ---
    log_info "Verifying test environment..."
    if ! command -v cargo &> /dev/null; then
        log_error "Rust toolchain (cargo) not found. Please install Rust."
        exit 1
    fi
    log_info "Rust toolchain found."

    # ---
    # 2. Build Test Harnesses
    # ---
    # We first compile all tests without running them. This separates build
    # failures from test failures, making it easier to diagnose CI issues.
    log_info "Building all test harnesses..."
    cargo test --no-run --quiet
    log_info "Test harnesses built successfully."

    # ---
    # 3. Execute Unit and Integration Tests (Debug Profile)
    # ---
    # This is the standard test run. We use '-- --nocapture' to ensure that
    # any output from tests (e.g., println!) is displayed in the CI logs,
    # which is invaluable for debugging.
    log_info "Running unit and integration tests (Debug Profile)..."
    cargo test -- --nocapture
    log_info "Debug profile tests passed."

    # ---
    # 4. Execute Unit and Integration Tests (Release Profile)
    # ---
    # Running tests in release mode is a critical step. Compiler optimizations
    # can sometimes expose subtle bugs (like race conditions or undefined
    # behavior) that do not appear in debug builds.
    log_info "Running unit and integration tests (Release Profile)..."
    cargo test --release -- --nocapture
    log_info "Release profile tests passed."

    # ---
    # 5. Execute Documentation Tests
    # ---
    # This command specifically runs the code examples embedded in the
    # documentation comments (///). It is crucial for ensuring that our
    # documentation is always accurate and the examples are compilable.
    log_info "Running documentation tests..."
    cargo test --doc
    log_info "Documentation tests passed."

    log_success "All tests passed successfully across all profiles."
}

# ---
# Script Execution
# ---
main "$@"
