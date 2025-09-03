#!/bin/bash

#
# Copyright (C) 2025 Pedro Henrique / phdev13
#
# File: ci/scripts/build_release_artifacts.sh
#
# This script builds the application in release mode and packages the
# resulting binary along with other essential files (LICENSE, README) into
# a compressed tarball. This creates a distributable artifact suitable for
# GitHub releases or other distribution channels.
#
# Dependencies: bash, cargo, coreutils (tar, mkdir, cp)
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

# ---
# Main Artifact Build Logic
# ---
main() {
    log_info "Building Release Artifacts"

    local project_name
    project_name=$(cargo metadata --format-version=1 --no-deps | jq -r ".packages[0].name")
    local project_version
    project_version=$(cargo metadata --format-version=1 --no-deps | jq -r ".packages[0].version")
    local target_arch="x86_64-unknown-linux-gnu" # Assuming x86_64 for this script

    # ---
    # 1. Compile the binary in release mode
    # ---
    log_info "Compiling '${project_name}' in release mode for '${target_arch}'"
    cargo build --release --target "${target_arch}"
    log_success "Compilation finished."

    # ---
    # 2. Prepare packaging directory
    # ---
    local staging_dir="artifacts/${project_name}-${project_version}-${target_arch}"
    log_info "Creating staging directory at '${staging_dir}'"
    mkdir -p "${staging_dir}"

    # ---
    # 3. Copy files to staging directory
    # ---
    log_info "Copying artifacts to staging directory"
    # Copy the main binary
    cp "target/${target_arch}/release/${project_name}" "${staging_dir}/"
    # Copy documentation and license files
    cp README.md "${staging_dir}/"
    # Ensure LICENSE file exists, or create a placeholder
    if [[ -f "LICENSE" ]]; then
        cp LICENSE "${staging_dir}/"
    else
        echo "AGPL-3.0" > "${staging_dir}/LICENSE"
    fi
    log_success "Files copied."
    
    log_info "Staging directory contents:"
    ls -lR "${staging_dir}"

    # ---
    # 4. Create compressed archive
    # ---
    local artifact_name="${project_name}-${project_version}-${target_arch}.tar.gz"
    log_info "Creating tarball '${artifact_name}'"
    # Use 'tar' to create a gzipped archive.
    # The -C flag changes the directory to 'artifacts' so that the tarball
    # doesn't contain the 'artifacts/' prefix in its path structure.
    tar -czf "${artifact_name}" -C "artifacts" "${project_name}-${project_version}-${target_arch}"
    log_success "Artifact tarball created successfully at ./${artifact_name}"
    
    log_info "Final artifact size:"
    ls -lh "${artifact_name}"
}

main "$@"
