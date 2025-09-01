#
# Copyright (C) 2025 Pedro Henrique / phdev13
#
# File: Brewfile
#
# This file specifies the system-level dependencies required for developing
# TrackieLLM on macOS, managed by the Homebrew package manager. Running
# `brew bundle` in the project root will automatically install, upgrade, or
# verify the presence of all listed dependencies, ensuring a consistent and
# reproducible development environment.
#
# SPDX-License-Identifier:
#

# Tap into third-party repositories if necessary.
# For example, if a specific formula is not in the main Homebrew repository.
# tap "user/repo"

# Core build tools
# ----------------
# cmake is the primary build system orchestrator for the project.
brew "cmake"
# ninja is a high-performance replacement for `make` that works well with CMake.
brew "ninja"
# ccache significantly speeds up recompilation by caching previous compiles.
brew "ccache"
# clang-format is used to enforce C/C++ code style.
brew "clang-format"

# C/C++ Libraries
# ----------------
# tesseract is the OCR engine used in the vision pipeline.
# We install it with all language data for flexibility.
brew "tesseract"
brew "tesseract-lang"

# libsodium is used for cryptographic operations, ensuring security.
brew "libsodium"

# protobuf and grpc can be useful for defining cross-language data structures
# and communication protocols, especially for telemetry or remote control.
brew "protobuf"
brew "grpc"

# Rust toolchain
# ----------------
# While rustup is the preferred way to manage Rust versions (handled by
# rust-toolchain.toml), we ensure rustup-init is available.
# brew "rustup-init" # Often managed manually, but can be included.

# Rust/CMake Integration
# ----------------
# corrosion is the helper tool that integrates Cargo builds into CMake.
# It's a crucial piece of our multi-language build system.
brew "corrosion"

# Python for helper scripts and tooling
# ----------------
# A modern version of Python is often needed for various build scripts,
# code generators, or utility tasks.
brew "python3"

# Documentation Generation
# ----------------
# doxygen is used to generate documentation from C/C++ source code comments.
brew "doxygen"
# graphviz is used by Doxygen to generate diagrams and call graphs.
brew "graphviz"

