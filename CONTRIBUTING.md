Thank you for your interest in contributing to **Trackie / Trackway**. This document explains how to contribute code, models, documentation, hardware designs, and other materials in a way that keeps the project high-quality, secure and maintainable.

> **Short**: open an issue or a PR. If it’s a security or privacy vulnerability, email **[pedro.henrique@vytruve.org](mailto:pedro.henrique@vytruve.org)** (private) or open a private issue. See also `CODE_OF_CONDUCT.md`.

---

## Table of contents

1. [Getting started (developer quickstart)](#getting-started-developer-quickstart)
2. [Contribution workflow & branches](#contribution-workflow--branches)
3. [Pull Request requirements & checklist](#pull-request-requirements--checklist)
4. [Code style, formatting and linters](#code-style-formatting-and-linters)
5. [Build, test and CI](#build-test-and-ci)
6. [Models & assets submission guidelines](#models--assets-submission-guidelines)
7. [Documentation, ADRs and design changes](#documentation-adrs-and-design-changes)
8. [Hardware contributions (Trackway / SpotWay designs)](#hardware-contributions-trackway--spotway-designs)
9. [Security, sensitive data & disclosure](#security-sensitive-data--disclosure)
10. [Licensing and legal](#licensing-and-legal)
11. [Governance, reviews & maintainers](#governance-reviews--maintainers)
12. [Quick links & contact](#quick-links--contact)

---

## Getting started (developer quickstart)

Clone the repository and build the project locally:

```bash
# clone
git clone https://github.com/phkaiser13/TrackieLLM.git
cd trackway

# create a reproducible build directory
cmake -S . -B build -DTRACKIE_BUILD_TESTS=ON
cmake --build build -- -j$(nproc)

# build Rust workspace
cargo build --workspace --release

# run unit tests (C/C++ via CTest, Rust via cargo)
ctest --test-dir build
cargo test --workspace --release
```

Development notes:

* Use an x86\_64 Linux build host or provision a reproducible Docker build container for cross-compiles to aarch64.
* Place model assets for dev under `assets/models/` (see Models & Assets section).
* Use a dedicated system user for deployments (`trackie`) and avoid running build or runtime processes as root.

---

## Contribution workflow & branches

Branching strategy:

* `main` — protected; CI must pass before merges.
* `develop` — optional integration branch (if used by your team).
* Feature branches: `feat/<short-desc>-<issue#>`
* Bugfix branches: `fix/<short-desc>-<issue#>`
* Hotfix branches: `hotfix/<short-desc>`

Naming and commits:

* Use **Conventional Commits**. Examples:

  * `feat(cortex): add reasoning API for contextual memory`
  * `fix(vision): correct depth scaling on MiDaS pipeline`
* Prefer small, focused PRs. If a change is large, split into multiple PRs (API contract first, implementation next).

Pull request flow:

1. Open an issue describing intent (link it).
2. Create a feature branch referencing the issue number.
3. Implement changes, keep commits atomic and meaningful.
4. Run all tests and linters locally.
5. Open a PR against `main` (or `develop`) including description and reviewers.

---

## Pull Request requirements & checklist

Every PR must satisfy the following before it is merged:

* [ ] Linked to an issue (or include a clear motivation in the PR description).
* [ ] Follows Conventional Commit messages (squash or rebase if necessary).
* [ ] Passes CI (builds, unit tests, linters, static analysis).
* [ ] Includes or updates tests to cover new behavior or edge cases.
* [ ] Includes documentation updates (docs or in-code docblocks) when public APIs change.
* [ ] For ABI / kernel / security / GPU / memory-management changes: at least **two** senior approvers required.
* [ ] No secrets, API keys, or credentials included in the code or commit history.
* [ ] If new third-party dependencies are added, include rationale, license, and security considerations.

PR size guidelines:

* Ideal PR: < 500 lines of change.
* Large PRs (> 2000 lines) must be split or accompanied by a high-level design doc / ADR and acceptance tests.

---

## Code style, formatting and linters

We enforce consistent formatting and static checks:

* **C/C++**: `clang-format` and `clang-tidy`
* **Rust**: `rustfmt` and `clippy`
* **General**: `shellcheck` for shell scripts, `yamllint` for YAML metadata

Typical local commands:

```bash
# Rust
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings

# C/C++
# apply clang-format to tracked C/C++ sources
find src -type f \( -name '*.c' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' -o -name '*.mm' \) -print0 \
  | xargs -0 -n1 clang-format -i

# run clang-tidy (example)
clang-tidy src/c/core/*.c -- -Iinclude -std=c11
```

Pre-commit hooks:

* We provide a `.pre-commit-config.yaml` (if present) — install with `pre-commit install`.
* Hooks run formatters and basic linters before commit.

File header rule:

* All generated source files **must** include the project header template at the top exactly as specified in the repository conventions:

  ```
  /*
  * Copyright (C) 2025 Pedro Henrique / phdev13
  *
  * File: Arquivo.
  *
  * Explicação rica e detalhada do arquivo e sua função.
  *
  * SPDX-License-Identifier: AGPL-3.0 license
  */
  ```

  (Inline comments in source files must be in English as per project rules.)

---

## Build, test and CI

CI:

* We use GitHub Actions with matrix builds for Linux (x86\_64, aarch64), macOS (Metal builds) and optional Windows runs.
* CI jobs run: build, unit tests, clippy/clang-tidy, formatting checks, fuzz and benchmark jobs (when scheduled).

Local testing:

```bash
# build the project
cmake -S . -B build
cmake --build build -- -j

# run tests
ctest --test-dir build --output-on-failure

# run Rust tests
cargo test --workspace
```

Fuzzing & benchmarks:

* Critical parsers and FFI surfaces must include fuzz harnesses (libFuzzer / afl++).
* Microbenchmarks use Google Benchmark for C++ and `criterion` for Rust where applicable.

---

## Models & assets submission guidelines

Models are first-class artifacts. Treat them like code: include metadata, provenance and license information.

**Where**: `assets/models/` (or `/opt/trackie/models/` in deployed systems).

**Required metadata file** for each model (`<model-name>.yml`):

```yaml
name: yolov5nu-v1
version: 2025-08-01
format: onnx
sha256: <hex-sha256>
size_bytes: 12345678
license: <license-id-or-url>
source: "https://example.com/origin"
recommended_device: "arm64-cuda, aarch64-neon"
quantization: "int8"
notes: "Trained for industrial AOI; uses NMS postprocessing on CPU"
```

Guidelines:

* Do **not** commit raw proprietary or licensed models if you are not permitted—instead include a small stub and the metadata with an instruction to download from the canonical source.
* If submitting model weights in the repo, add clear license & provenance and confirm redistribution rights.
* Provide inference test vectors (input image + expected output) to validate model behavior in CI.

Model PRs must include:

* Benchmark results (latency, memory usage) and test vectors.
* A short test that runs the model in a containerized environment in CI (optional for very large models, but metadata + download script required).

---

## Documentation, ADRs and design changes

All non-trivial design choices must be captured as an ADR.

* Place ADRs under `docs/design_decisions/`.
* An ADR should include: context, decision, alternatives considered, consequences, and the date/author.

Docs updates:

* Update `docs/` for user-facing or developer-facing changes.
* API changes require updates to `docs/API_reference/` and example snippets.

---

## Hardware contributions (Trackway / SpotWay designs)

Hardware contributions are welcome. Required deliverables for hardware PRs:

* Mechanical drawings and a bill of materials (BOM).
* PCB files (KiCad preferred). Include Gerber exports.
* Photographs and assembly instructions.
* Safety validation data (battery safety, EMC notes, thermal run).
* Firmware for microcontrollers (source + build instructions).
* A hardware README with expected sensors, connectors, and power budgets.

Sensitive safety items:

* For any actuator or mobile robot design, include an explicit failsafe plan and emergency stop behavior in the documentation.

---

## Security, sensitive data & disclosure

If you find a security issue:

* **Do not** open a public issue with exploit details.
* Email **[pedro.henrique@vytruve.org](mailto:pedro.henrique@vytruve.org)** or open a **private** GitHub issue marked `[SECURITY]`.
* Provide reproduction steps, affected versions, and suggested mitigations.

Data and privacy:

* Never commit PII, API keys, credentials or private datasets.
* Use environment variables or OS keyrings for secrets in CI and local dev.
* Telemetry and remote logs are opt-in and must be documented in `SECURITY.md`.

---

## Licensing and legal

* By contributing you agree that your contributions will be licensed under the terms in the repository `LICENSE` file.
* If you cannot or do not want to license your contributions under the project license, do not submit a PR—contact the maintainers to discuss alternatives.

---

## Governance, reviews & maintainers

* Maintainers maintain the repository and approve PRs.
* For changes that affect ABI, GPU kernels, memory management or security, two senior maintainers must sign off before merge.
* The repository uses conventional commit messages and requires CI green builds.

If you’re unsure who to request as reviewer, add `@phkaiser13` and tag the appropriate area maintainers listed in `CODEOWNERS` (if present).

---

## Quick links & contact

* Repo: `https://github.com/phkaiser13/trackway`
* Model assets: `https://github.com/phkaiser13/TrackieAssets`
* Website & downloads: `https://trackway.org`
* Report security issues / private contact: **[pedro.henrique@vytruve.org](mailto:pedro.henrique@vytruve.org)**
* Public issue tracker: `https://github.com/phkaiser13/TrackieLLM/issues`

---

## PR template checklist (copy into PR description)

```
- [ ] Linked issue: #____
- [ ] Commits follow Conventional Commits
- [ ] CI passes (build/test/lint)
- [ ] Tests added/updated
- [ ] Docs updated (if applicable)
- [ ] Model metadata included (if adding models)
- [ ] No secrets in commits
- [ ] Reviewed and approved by appropriate maintainers
```

---

## Final notes

We appreciate every contribution — large or small. If you want to discuss a major feature or hardware design before implementing it, open an issue with the label `proposal` and include a short architecture sketch and acceptance criteria.

**Thank you** for helping build Trackie / Trackway — accessibility-first, community-powered AI.

---

*Last updated: 2025-09-01*
