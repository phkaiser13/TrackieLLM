#!/bin/bash

#
# Copyright (C) 2025 Pedro Henrique / phdev13
#
# File: ci/scripts/run_fuzzers.sh
#
# This script executes fuzz tests for the project. Fuzzing is an automated
# software testing technique that involves providing invalid, unexpected, or
# random data as inputs to a computer program. The goal is to find software
# defects and security vulnerabilities that might be missed by traditional
# testing methods. We use 'cargo-fuzz' to manage and run our fuzz targets.
#
# Dependencies: bash, cargo, cargo-fuzz
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
        # cargo-fuzz requires a nightly toolchain to install
        if ! cargo +nightly install "${tool}"; then
            log_error "Failed to install '${tool}'. Ensure nightly toolchain is available."
            exit 1
        fi
    fi
}

# ---
# Main Fuzzing Logic
# ---
main() {
    log_info "Starting Fuzz Testing"

    # ---
    # 1. Ensure cargo-fuzz is installed
    # ---
    # cargo-fuzz is a sensitive tool and often requires the nightly toolchain.
    ensure_cargo_tool "cargo-fuzz"

    # ---
    # 2. List available fuzz targets
    # ---
    # This helps in debugging and gives a clear view of what will be tested.
    log_info "Listing available fuzz targets..."
    # The '|| true' is to prevent the script from failing if no fuzz targets
    # are defined yet.
    local targets
    targets=$(cargo +nightly fuzz list || true)
    if [[ -z "${targets}" ]]; then
        log_success "No fuzz targets found. Skipping fuzzing."
        exit 0
    fi
    echo "${targets}"

    # ---
    # 3. Run each fuzz target for a limited time
    # ---
    # Fuzzing can run indefinitely. For a CI environment, we must time-box it.
    # We run each target for a short duration (e.g., 60 seconds). If any
    # crash or bug is found, 'cargo fuzz' will exit with a non-zero status.
    local fuzz_duration=60
    log_info "Running each fuzz target for ${fuzz_duration} seconds..."
    
    for target in ${targets}; do
        log_info "Fuzzing target: '${target}'"
        # We use the nightly toolchain as required by cargo-fuzz.
        # The '-s' flag specifies the sanitizer to use.
        # The last argument is the maximum time to run.
        cargo +nightly fuzz run "${target}" -- -max_total_time="${fuzz_duration}"
        log_success "Fuzzing for '${target}' completed without finding crashes."
    done

    log_success "All fuzz targets ran successfully without finding any crashes."
}

main "$@"
