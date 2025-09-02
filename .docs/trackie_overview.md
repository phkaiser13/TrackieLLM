<!--
This documentation was written by Jules - Google labs bot.
Original code by phkaiser13.
-->

# TrackieLLM: Project Overview

## 1. Core Concept

**TrackieLLM** is a multimodal assistance platform designed to operate in real-time as a "proactive AI companion." The core of the system is an Artificial Intelligence that acts as a "brain," processing visual and auditory information to analyze context, understand the environment, and provide intelligent support to the user.

The main goal is to expand perception, promote autonomy, and ensure the safety of people with visual impairments, revolutionizing the way they interact with the world around them.

## 2. Technological Pillars

TrackieLLM integrates three technological areas to create a unified perception of the environment:

1.  **Computer Vision:** Uses cameras to "see" and interpret the world, recognizing faces, objects, texts, obstacles, and dangers.
2.  **Audio Processing:** Captures and understands voice commands (STT), environmental sounds (VAD), and provides audio feedback (TTS).
3.  **Artificial Intelligence (LLM):** A Large Language Model acts as the central processing unit, uniting visual and sound information to provide contextualized and useful feedback in real-time.

## 3. Benefits and Differentiators

*   **Expanded Autonomy:** Allows the user to perform daily tasks with more independence and explore new environments with confidence.
*   **Proactive Safety:** Actively detects risks such as obstacles, steps, holes, and smoke.
*   **Natural Interaction:** Communication is done through voice commands, making the user experience fluid and intuitive.
*   **Multimodal Perception:** Integrates data from audio, video, and sensors for a complete understanding of the environment.
*   **Accessibility:** Positions itself as a powerful and low-cost alternative to expensive commercial solutions.

## 4. AI Model Stack

TrackieLLM is built on a set of AI models optimized for offline execution on resource-constrained hardware.

*   **Central AI (LLM):**
    *   **Model:** `Mistral-7B`
    *   **Format:** GGUF (optimized for `llama.cpp`)

*   **Computer Vision:**
    *   **Object Detection:** `YOLOv5nu` (ONNX format)
    *   **Depth Analysis and Navigation:** `DPT-SwinV2-Tiny-256` (MiDaS 3.1, ONNX, INT8) for detecting steps, ramps, free spaces, and grasp points.
    *   **Text Recognition (OCR):** `Tesseract OCR` (via native C++ API)

*   **Audio Processing:**
    *   **Speech Recognition (ASR):** `whisper.cpp tiny.en` (GGML format)
    *   **Voice Activity Detection (VAD):** `Porcupine` and `Silero VAD`
    *   **Text-to-Speech (TTS):** `Piper` (Rhasspy) with pre-trained voices in PT-BR.

## 5. Execution Platforms

### Production Environments (Real Use)

TrackieLLM is designed to run natively and optimized on the following systems:

*   **Embedded Hardware:**
    *   **Orange Pi (8GB RAM + CUDA):** Main platform.
    *   **Orange Pi (RISC-V, 8GB RAM + CUDA):** Secondary high-practicality platform.
    *   **Raspberry Pi / Orange Pi (8-32GB RAM models):** For development and community use.
*   **Mobile Devices (via `TrackWay` app):**
    *   **Android:** Native support.
    *   **iOS:** Native support with high priority, optimized for the **Metal** graphics accelerator.
*   **Desktop (via `TrackWay` terminal):**
    *   **Linux:** Support for CUDA and ROCm.

### Test and Presentation Environments (via `Trackie Studio`)

*   **Windows, macOS, and Linux:** For demonstration, testing, and model training purposes.

## 6. Compilation and Deployment

*   In the **Trackie Studio** (Desktop) and **TrackWay** (Mobile) applications, the TrackieLLM core must be compiled as a dynamic library (`.dll`, `.so`, `.dylib`, etc.).
*   On embedded devices (Orange/Raspberry Pi), the system can run as a direct native executable on the operating system (with a modified kernel) or inside a container for portability.

## 7. Project Mission

> To bring intelligent accessibility to educational, industrial, and daily life environments through cutting-edge AI and accessible hardware.

## Developer and Author

*   **Original Code:** phkaiser13
*   **Documentation:** Jules - Google labs bot
