#!/bin/bash

#
# Copyright (C) 2025 Pedro Henrique / phdev13
#
# File: ci/security_audit.sh
#
# This script performs a security audit of the project's dependencies. In modern
# software development, a significant portion of the code comes from third-party
# libraries, each a potential vector for vulnerabilities. This script automates
# the process of checking against known security advisories and also identifies
# outdated dependencies, promoting a secure and up-to-date codebase.
#
# Dependencies: bash, coreutils, rustc, cargo, cargo-audit, cargo-outdated
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

log_warn() {
    printf "\033[33m[WARN]\033[0m %s\n" "$1"
}

log_error() {
    printf "\033[31m[ERROR]\033[0m %s\n" "$1" >&2
}

# ---
# Helper Functions
# ---

# Ensures a given cargo tool is installed, installing it if necessary.
# This makes the CI environment more self-sufficient.
ensure_cargo_tool() {
    local tool="$1"
    log_info "Checking for '${tool}'..."
    if ! cargo --list | grep -q "${tool}"; then
        log_warn "'${tool}' not found in cargo command list. Attempting to install..."
        if ! cargo install "${tool}"; then
            log_error "Failed to install '${tool}'. Please install it manually or check CI permissions."
            exit 1
        fi
        log_info "'${tool}' installed successfully."
    else
        log_info "'${tool}' is already installed."
    fi
}

# ---
# Main Security Logic
# ---
main() {
    log_info "Starting Security and Dependency Audit..."

    # ---
    # 1. Environment and Tool Verification
    # ---
    ensure_cargo_tool "cargo-audit"
    ensure_cargo_tool "cargo-outdated"

    # ---
    # 2. Security Vulnerability Scan
    # ---
    # 'cargo audit' scans the 'Cargo.lock' file and compares the exact versions
    # of all dependencies against the RustSec Advisory Database. This is a
    # critical step to prevent known vulnerabilities from entering the codebase.
    log_info "Scanning for known security vulnerabilities with 'cargo audit'..."
    cargo audit
    log_info "No security vulnerabilities found in dependencies."

    # ---
    # 3. Stale Dependency Check
    # ---
    # 'cargo outdated' checks for dependencies that have newer versions available.
    # While not a direct security threat, outdated packages can miss important
    # bugfixes, performance improvements, and security patches.
    # We run this as an informational step. For a stricter policy, one could
    # fail the build if any dependency is outdated.
    log_info "Checking for outdated dependencies with 'cargo outdated'..."
    # The command exits with a non-zero code if outdated packages are found.
    # We add '|| true' to prevent this from failing the build, treating it as
    # an informational step. For a strict policy, remove '|| true'.
    if ! cargo outdated --exit-code 0; then
        log_warn "Outdated dependencies found. Consider running 'cargo update' locally to review and update them."
    else
        log_info "All dependencies are up-to-date."
    fi


    log_success "Security and dependency audit completed successfully."
}

# ---
# Script Execution
# ---
main "$@"
