#!/bin/bash

#
# Copyright (C) 2025 Pedro Henrique / phdev13
#
# File: ci/lint.sh
#
# This script enforces strict code quality and style standards. It serves as an
# automated code reviewer, ensuring that all code adheres to the established
# conventions of the Rust community and the project's own high standards.
# Using 'cargo fmt' guarantees stylistic consistency, while 'cargo clippy' with
# a '-D warnings' policy ensures that no potential issues are ignored.
#
# Dependencies: bash, coreutils, rustc, cargo, cargo-fmt, cargo-clippy
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
# Main Linting Logic
# ---
main() {
    log_info "Starting Code Quality and Linting Checks..."

    # ---
    # 1. Environment Verification
    # ---
    log_info "Verifying linting environment..."
    if ! command -v cargo &> /dev/null; then
        log_error "Rust toolchain (cargo) not found. Please install Rust."
        exit 1
    fi
    # The 'fmt' and 'clippy' components are installed by default with 'rustup'.
    log_info "Rust toolchain and components found."

    # ---
    # 2. Code Formatting Check
    # ---
    # 'cargo fmt -- --check' runs the formatter in a 'dry-run' mode. It will
    # exit with a non-zero status if any file is not correctly formatted.
    # This is crucial for CI, as it enforces style without modifying files.
    log_info "Checking code formatting with 'cargo fmt'..."
    cargo fmt -- --check
    log_info "Code formatting is consistent."

    # ---
    # 3. Static Analysis with Clippy
    # ---
    # 'cargo clippy' is a powerful linter that catches common mistakes and
    # improves code. We run it with a very strict policy:
    # --all-targets: Ensures that examples, tests, and benchmarks are also linted.
    # -D warnings: Deny all warnings. This elevates every single warning to a
    #              build-breaking error, enforcing the highest possible code quality.
    # -D clippy::pedantic: Enables an additional set of very strict, often opinionated,
    #                      lints for developers who want to be extremely thorough.
    log_info "Running static analysis with 'cargo clippy' (strict policy)..."
    cargo clippy --all-targets -- -D warnings -D clippy::pedantic
    log_info "Clippy analysis passed with no issues."

    log_success "All code quality and linting checks have passed."
}

# ---
# Script Execution
# ---
main "$@"
