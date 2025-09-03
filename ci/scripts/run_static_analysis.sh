#!/bin/bash

#
# Copyright (C) 2025 Pedro Henrique / phdev13
#
# File: ci/scripts/run_static_analysis.sh
#
# This script executes a comprehensive suite of static analysis tools. It acts
# as a primary quality gate, checking for formatting consistency, code quality
# issues, and known security vulnerabilities before any functional tests are run.
# This "fail-fast" approach saves CI resources and provides rapid feedback.
#
# Dependencies: bash, cargo, cargo-fmt, cargo-clippy, cargo-audit
#
# SPDX-License-Identifier: AGPL-3.0 license
#

set -euo pipefail

# ---
# Logging Utilities
# ---
log_info() {
    printf "\n\033[34m--- %s ---\033[0m\n" "$1"
}

log_success() {
    printf "\033[32m[SUCCESS]\033[0m %s\n" "$1"
}

log_error() {
    printf "\033[31m[ERROR]\033[0m %s\n" "$1" >&2
}

ensure_cargo_tool() {
    local tool="$1"
    if ! cargo --list | grep -q "${tool}"; then
        echo "Tool '${tool}' not found, installing..."
        if ! cargo install "${tool}"; then
            log_error "Failed to install '${tool}'."
            exit 1
        fi
    fi
}

# ---
# Main Analysis Logic
# ---
main() {
    log_info "Starting Comprehensive Static Analysis"

    # ---
    # 1. Code Formatting Check
    # ---
    log_info "Checking code formatting with 'cargo fmt'"
    cargo fmt -- --check
    log_success "Code formatting is consistent."

    # ---
    # 2. Linter and Code Quality Check
    # ---
    # We use a strict Clippy configuration to enforce high code quality.
    # -D warnings: Turns all warnings into hard errors.
    # -D clippy::pedantic: Enables extra-strict, opinionated lints.
    log_info "Running linter with 'cargo clippy' (Strict Policy)"
    cargo clippy --all-targets -- -D warnings -D clippy::pedantic
    log_success "Clippy analysis passed."

    # ---
    # 3. Security Vulnerability Audit
    # ---
    # Ensure cargo-audit is installed before running it.
    ensure_cargo_tool "cargo-audit"
    log_info "Auditing dependencies for security vulnerabilities"
    cargo audit
    log_success "No known security vulnerabilities found."

    log_success "All static analysis checks passed."
}

main "$@"
