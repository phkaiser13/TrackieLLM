

# <div align="center">🤖 TrackieLLM</div>

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Flag_of_Brazil.svg" alt="Brazil Flag" width="50"/>
</div>

<p align="center">
  <strong>A High-Complexity, Multimodal AI Platform for Accessibility, Safety, and Autonomy</strong><br>
  <em>Engineered with precision and innovation for the 2025 Industrial Innovation Olympiad</em>
</p>

## 🌐 Overview

TrackieLLM is a **production-grade multimodal AI framework** designed to expand perception, mobility, and safety for people with visual impairment. It fuses low-latency computer vision, robust audio processing and an on-device LLM core to deliver near real-time reasoning and assistive actions across embedded and mobile platforms.

Supported deployment targets:

* Embedded Linux (Orange Pi with CUDA, Raspberry Pi variants)
* Mobile (Android, iOS — Metal optimized)
* Desktop/Edge servers (x86\_64 with CUDA / ROCm)
* Hybrid: device + phone (SpotWay model described below)

---

---
## 🧭 New Product Variants & Form Factors

### Trackway (Hat) — full embedded compute

Trackway is the full-featured, computerized hat platform: camera, IMU, microphone array, edge compute (OrangePi/Jetson/embedded module), and battery. The on-hat system can run TrackieLLM locally (models + reasoning). It is designed for maximum autonomy and offline safety.

### SpotWay — low-cost hybrid hat

SpotWay uses microcontrollers on the hat for sensor aggregation and low-latency pre-processing. Heavy models and reasoning run on the user’s smartphone (Android/iOS) over a secure local link (Bluetooth LE / Wi-Fi Direct). SpotWay is optimized for accessibility and cost-efficiency.

### Trackway Glasses

Compact eyewear form factor (stereo cameras, bone-conduction audio). Same software stack but mechanical and thermal constraints require careful model selection and power budgeting.

---

## 🛠️ Deploying Trackie — High-level Path (one-line summary)

1. Obtain hardware (Orange Pi / supported board) → 2. Prepare OS image and toolchain → 3. Acquire model assets (ONNX / GGUF) → 4. Build native binaries (C/C++/Rust) with CMake + Cargo → 5. Package into service/container → 6. Deploy as systemd service or Docker container → 7. Validate sensors and run safety tests.

---

## 🔧 Detailed Deployment Guide — Embedded (Orange Pi / RPi / ARM)

### 1) Preconditions

* Host machine (build): x86\_64 Linux or containerized cross-build environment.
* Cross-toolchains: `gcc` / `clang` for target arch, `aarch64-linux-gnu-gcc` etc. (or use Docker build images).
* NDK for Android builds, Xcode for iOS builds (macOS host).
* Required runtimes: ONNX Runtime (ARM build), `llama.cpp` (GGUF runtime), `onnxruntime` or custom lightweight runners, libsndfile/pulseaudio or ALSA.
* Model assets placed in `/opt/trackie/models/` (permissions restricted).
* Hardware: camera (UVC or CSI), microphone(s), IMU, battery management, optional LIDAR.

### 2) Clone code + assets

```bash
git clone https://github.com/phkaiser13/trackway.git trackie-src
cd trackie-src
# Acquire model bundles (place into /opt/trackie/models or ./models for dev)
# Example local path:
mkdir -p assets/models
# copy ONNX / GGUF files into assets/models
```

### 3) Build (native or cross)

Prefer reproducible build containers. Example with native CMake + Cargo:

```bash
# create build dir
cmake -S . -B build -DTRACKIE_ENABLE_CUDA=ON -DTRACKIE_BUILD_TESTS=ON
cmake --build build -- -j$(nproc)
# Optionally build Rust crates (if Cargo not wired with CMake)
cargo build --workspace --release
```

Cross-compiling for aarch64 from x86\_64 should use a Dockerfile that installs `aarch64-linux-gnu-gcc`, sets `CMAKE_TOOLCHAIN_FILE`, or uses `dockcross` images.

### 4) Packaging & Service

Recommended: package as a systemd user service + optional Docker image for field updates.

Systemd service example (place as `/etc/systemd/system/trackie.service`):

```ini
[Unit]
Description=TrackieLLM runtime
After=network.target

[Service]
User=trackie
Group=trackie
WorkingDirectory=/opt/trackie
ExecStart=/opt/trackie/bin/trackie-core --config /etc/trackie/config.yaml
Restart=on-failure
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

Alternatively build and publish a Docker image:

```dockerfile
FROM ubuntu:24.04
# install libs, copy binaries, set entrypoint
COPY build/ /opt/trackie/
ENTRYPOINT ["/opt/trackie/bin/trackie-core"]
```

### 5) Model & Runtime Considerations

* Use ONNX Runtime for vision models (TensorOps optimized with CUDA/ROCm where available).
* For LLM (Mistral-7B GGUF), use `llama.cpp` or equivalent optimized runtimes compiled for ARM aarch64 + NEON / VNNI / FP16.
* Place large model files on a high-throughput NVMe or SD card with good I/O.
* For deterministic performance, pin CPU governors, tune `swappiness`, and disable unneeded services.

### 6) Sensors & Safety Calibration

* Calibrate camera intrinsics / extrinsics, IMU alignment, and microphone arrays.
* Run validation suite: obstacle detection, step detection, free-space detection, grasp-point detection.
* Safety modes: audible alarm, haptic/vibration fallback, emergency stop (if robot actuators present).

---

## 🧱 Building your own Trackie Robot — practical blueprint

### A) Mechanical & Power

* Chassis: compact, low center of gravity if wheeled; balanced for wearables (hat).
* Actuators: if moving platform, use differential drive with closed-loop encoders.
* Power: dedicated battery service, battery gauge, low-voltage cutoffs, UPS for critical features (emergency beeps).
* Connectors: standard JST / SMBus for sensors; design for hot-swap battery.

### B) Electronics & Sensors

* Main compute: Orange Pi CM4-class module, or Jetson Orin Nano for higher perf.
* Microcontroller: ESP32 / STM32 for sensor preproc and real-time closed-loop tasks.
* Sensors: stereo/mono camera (UVC or CSI), IMU (6/9DOF), ultrasonic or LIDAR for redundancy, multiple microphones for beamforming.
* Communications: USB for cameras, I2C/SPI for IMU, UART for microcontroller, Wi-Fi + BLE for host-phone comms.

### C) Software Architecture (on-robot)

* `tk_core` (C) — low-level drivers, frame acquisition, actuator safe controllers.
* `cortex` (Rust/C) — reasoning, memory manager, safety manager.
* `vision_pipeline` (C++/Rust) — preprocessing, depth map (MiDaS), detection (YOLOv5nu).
* `audio_pipeline` (C/Rust) — wakeword (Porcupine), VAD (Silero), ASR (whisper.cpp).
* `ffi` — stable C API to expose high-level commands to companion apps or plugin modules.

### D) Integration & CI

* Unit tests, integration tests (hardware-in-loop), and end-to-end validation (simulated obstacles).
* Pre-deployment checklist: sensor calibration, thermal test, power failover, and safe stop latency measured.

---

## 🔁 Mobile & Desktop: TrackieStudio (Trackie companion app)

* **TrackieStudio** is the companion app and configuration tool for Trackie / Trackway devices. It provides firmware updates, model deployment, remote logs, and live telemetry.
* Mobile editions:

  * **iOS** — Metal-accelerated model inference (on-device CoreML/Metal for models that can be converted).
  * **Android** — NDK libraries compiled for armeabi-v7a / arm64-v8a; uses Vulkan/CUDA (where supported) for acceleration.
* Desktop editions provide a simulator, model analysis, and developer tools.

Download landing pages:

* `https://trackway.org/downloads/trackiestudio-ios`  — iOS App Store / TestFlight link
* `https://trackway.org/downloads/trackiestudio-android` — Play Store link
* Releases & installers mirrored at `https://github.com/phkaiser13/trackway/releases`

---

## ⚠️ Security, Privacy & Data Handling (mandatory)

* All raw audio/video processing must, by default, occur locally and be securely wiped from RAM/temporary storage after inference.
* Models and telemetry must be opt-in for cloud uploads; use secure TLS endpoints and token-based auth.
* Keys and secrets: use OS keyrings (libsecret/keyring) or encrypted on-disk vaults; never store plaintext tokens in config files.
* Follow least-privilege: create a dedicated `trackie` system user, restrict file permissions on `/opt/trackie` and model directories.

---

## 🧪 Validation & Certification (safety-critical)

* Provide test harnesses that measure detection latency, false-positive/negative rates, and worst-case CPU/GPU load.
* For devices intended for industrial use, document safety validation: EMC, electrical safety, and fail-safe behaviors.
* Provide an accessible set of test protocols for field technicians (calibration, sensor replacement, software rollback).

---

## ⚙️ Example Commands (developer quickstart)

Clone + build example:

```bash
git clone https://github.com/phkaiser13/trackway.git
cd trackway
cmake -S . -B build -DTRACKIE_ENABLE_CUDA=OFF
cmake --build build -- -j$(nproc)
cargo build --workspace --release
# install to /opt/trackie (requires sudo)
sudo mkdir -p /opt/trackie && sudo cp -r build/bin /opt/trackie/
```

Run locally:

```bash
/opt/trackie/bin/trackie-core --config /etc/trackie/config.yaml
```

Create Docker image (developer template):

```bash
docker build -t trackie/core:latest -f docker/Dockerfile .
docker run --device /dev/video0 --cap-add=SYS_NICE -v /opt/trackie/models:/opt/trackie/models trackie/core:latest
```

---

## 🔗 Related assets & resources

* Project repository: `https://github.com/phkaiser13/trackway`
* Model assets Training & data: `https://github.com/phkaiser13/TrackieAssets` (ONNX, GGUF bundles)
* Official landing & downloads: `https://trackway.org`

---

## 🎯 Product Notes (form-factors & marketing)

* **Trackway Hat** — premium product: integrated compute, highest autonomy, supports full offline LLM runs.
* **SpotWay Hat** — accessible product: sensor hub with lightweight pre-processing; smartphone runs LLM and heavy models.
* **Trackway Glasses** — mid-tier: stereo vision and thermal management; better for indoor/outdoor mixed use cases.

Each form factor includes:

* Device registration flow (paired with TrackieStudio)
* OTA update pipeline (secure signed packages)
* Safety & calibration wizard on first boot

---

## 🤝 Social Impact & Donations
We believe in building solutions that serve communities. Consider supporting:

* [APAE](https://apaebrasil.org.br/) — Association of Parents and Friends of the Exceptional
* [Fundação Dorina Nowill](https://fundacaodorina.org.br/) — Support for visually impaired individuals
* [UNICEF](https://www.unicef.org/)


---

## 🧾 License & Contribution

See `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md` in the repository root for legal, contribution, and vulnerability disclosure policies.

---

## ❤️ Closing — where to go next

* Visit the official site and downloads: `https://trackway.org`
* Clone repo and view hardware blueprints & code: `https://github.com/phkaiser13/trackway`

