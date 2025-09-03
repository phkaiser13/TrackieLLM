#!/bin/bash

#
# Copyright (C) 2025 Pedro Henrique / phdev13
#
# File: ci/scripts/generate_sbom.sh
#
# This script generates a Software Bill of Materials (SBOM) for the project.
# An SBOM is a formal, machine-readable inventory of software components and
# dependencies, which is critical for modern cybersecurity and supply chain
# risk management. We use 'cargo-sbom' to generate the list in SPDX format.
#
# Dependencies: bash, cargo, cargo-sbom
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
# Main SBOM Generation Logic
# ---
main() {
    log_info "Generating Software Bill of Materials (SBOM)"

    # ---
    # 1. Ensure cargo-sbom is installed
    # ---
    ensure_cargo_tool "cargo-sbom"

    # ---
    # 2. Generate SBOM
    # ---
    local project_name
    project_name=$(cargo metadata --format-version=1 --no-deps | jq -r ".packages[0].name")
    local output_dir="artifacts/sbom"
    local output_file="${output_dir}/${project_name}.spdx.json"

    log_info "Creating output directory at '${output_dir}'"
    mkdir -p "${output_dir}"

    log_info "Generating SPDX-formatted SBOM..."
    # We generate the SBOM from the dependency graph of the main binary.
    # The output is specified as SPDX JSON, a widely used standard.
    cargo sbom --output-format spdx-json --output-file "${output_file}"

    log_success "SBOM generated successfully at '${output_file}'"
    
    log_info "Verifying generated SBOM..."
    # A simple verification to ensure the file is not empty and is valid JSON.
    if [[ -s "${output_file}" ]] && jq . "${output_file}" > /dev/null; then
        log_success "SBOM file is valid JSON and not empty."
    else
        log_error "SBOM generation failed or produced an invalid file."
        exit 1
    fi
}

main "$@"
