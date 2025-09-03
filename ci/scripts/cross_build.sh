#!/bin/bash

#
# Copyright (C) 2025 Pedro Henrique / phdev13
#
# File: ci/scripts/cross_build.sh
#
# This script orchestrates the cross-compilation build process using Docker.
# It abstracts the complexity of building for different architectures by
# selecting the appropriate Dockerfile, running the containerized build, and
# extracting the final binary. This allows the main CI pipeline to build for
# any supported architecture with a single command.
#
# Dependencies: bash, docker, coreutils
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

# ---
# Main Cross-Build Logic
# ---
main() {
    # ---
    # 1. Input Validation
    # ---
    if [[ $# -ne 1 ]]; then
        log_error "Usage: $0 <architecture>"
        log_error "Supported architectures: x86_64, arm64"
        exit 1
    fi

    local arch="$1"
    local dockerfile
    local target_arch_string
    local image_tag

    log_info "Starting cross-build process for architecture: '${arch}'"

    # ---
    # 2. Select Dockerfile based on architecture
    # ---
    case "${arch}" in
        x86_64)
            dockerfile="ci/docker/Dockerfile.linux_x86_64"
            target_arch_string="x86_64-unknown-linux-gnu"
            ;;
        arm64)
            dockerfile="ci/docker/Dockerfile.linux_arm64_cross"
            target_arch_string="aarch64-unknown-linux-gnu"
            ;;
        *)
            log_error "Unsupported architecture: '${arch}'"
            exit 1
            ;;
    esac
    image_tag="project-builder:${arch}"
    log_info "Using Dockerfile: '${dockerfile}' with image tag: '${image_tag}'"

    # ---
    # 3. Build the Docker builder image
    # ---
    log_info "Building Docker image..."
    docker build -f "${dockerfile}" -t "${image_tag}" .
    log_success "Docker image '${image_tag}' built successfully."

    # ---
    # 4. Extract the binary from the image
    # ---
    # We create a temporary, named container from the builder image.
    # The 'docker cp' command then copies the compiled binary from the
    # container's filesystem to the host's filesystem.
    local project_name
    project_name=$(cargo metadata --format-version=1 --no-deps | jq -r ".packages[0].name")
    local container_name="builder-container-${arch}-$$"
    local artifact_source="/app/target/${target_arch_string}/release/${project_name}"
    local artifact_dest="artifacts/${arch}/"

    log_info "Extracting binary from container..."
    mkdir -p "${artifact_dest}"
    
    # Create the container but don't start it.
    docker create --name "${container_name}" "${image_tag}"
    
    # Copy the file out.
    docker cp "${container_name}:${artifact_source}" "${artifact_dest}"
    
    # Clean up the temporary container.
    docker rm "${container_name}"
    
    log_success "Binary extracted to '${artifact_dest}${project_name}'"
    
    log_info "Final binary details:"
    ls -lh "${artifact_dest}${project_name}"
}

main "$@"
