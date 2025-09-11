<!-- This documentation was written by Jules - Google labs bot. -->

# Audio Module

## 1. Overview

The Audio module serves as the "ears" and "voice" of the TrackieLLM system. It is responsible for all audio-related processing, enabling natural and intuitive interaction between the user and the AI. This includes listening for commands, transcribing speech, providing spoken feedback, and maintaining awareness of the acoustic environment.

Its primary function is to bridge the gap between human language and the system's internal logic, allowing users to control the device and receive information through conversation.

## 2. Core Responsibilities

-   **Wake-Word Detection:** Continuously listens for a specific "wake word" (e.g., "Hey Trackie") in a low-power state to activate full listening mode.
-   **Voice Activity Detection (VAD):** Detects the presence of human speech in the audio stream to identify when the user is speaking.
-   **Automatic Speech Recognition (ASR):** Transcribes the user's spoken commands and questions into text for the Cortex module.
-   **Text-to-Speech (TTS) with Interruption:** Synthesizes natural-sounding speech from text. Features a priority system allowing critical alerts to interrupt and override lower-priority messages.
-   **Ambient Sound Analysis:** Identifies and classifies key environmental sounds (e.g., alarms, sirens) in parallel with wake-word detection, providing the Cortex with crucial environmental context even when the user is not speaking.

## 3. Architecture and Models

The Audio module is a multi-stage, state-driven pipeline designed for low-latency, on-device processing. It uses a combination of highly efficient models and C/Rust components to manage the audio stream.

### 3.1. AI Models

-   **Wake-Word Engine:**
    -   **Model:** `Porcupine`
    -   **Details:** A highly accurate and lightweight wake-word detection engine. It runs continuously in the `AWAITING_WAKE_WORD` state.

-   **Ambient Sound Classifier:**
    -   **Model:** `YAMNet` (or similar ONNX-based classifier)
    -   **Details:** Runs in parallel with Porcupine to detect a wide range of environmental sounds without requiring the wake word.

-   **Voice Activity Detection (VAD):**
    -   **Model:** `Silero VAD`
    -   **Details:** Activated after the wake word is detected to determine the start and end of the user's utterance.

-   **Automatic Speech Recognition (ASR):**
    -   **Model:** `whisper.cpp tiny.en`
    -   **Format:** GGML
    -   **Details:** An efficient implementation of OpenAI's Whisper model, optimized for fast and accurate transcription on edge devices.

-   **Text-to-Speech (TTS):**
    -   **Model:** `Piper` (from Rhasspy)
    -   **Details:** A fast, local neural text-to-speech system that generates high-quality voices.

### 3.2. Pipeline State Machine and Flow

The audio pipeline operates as a state machine to manage power and resources efficiently:

1.  **`TK_PIPELINE_STATE_AWAITING_WAKE_WORD` (Default State):**
    -   The audio stream is processed in parallel by **Porcupine** (for the wake word) and the **Sound Classifier** (for ambient sounds).
    -   If an ambient sound is detected, an `on_ambient_sound_detected` callback is fired immediately to the Cortex.
    -   If the wake word is detected, the state transitions to `LISTENING_FOR_COMMAND`.

2.  **`TK_PIPELINE_STATE_LISTENING_FOR_COMMAND`:**
    -   The pipeline now feeds the audio stream into the **Silero VAD** engine.
    -   A **5-second timer** is active. If the VAD does not detect the start of speech (`TK_VAD_EVENT_SPEECH_STARTED`) within this window, the pipeline times out and automatically returns to the `AWAITING_WAKE_WORD` state to conserve energy.
    -   If speech begins, the audio is buffered for transcription.

3.  **`TK_PIPELINE_STATE_TRANSCRIBING`:**
    -   Once the VAD detects the end of speech, the buffered audio is sent to **`whisper.cpp`** for transcription.
    -   The final text is sent to the Cortex via the `on_transcription` callback.
    -   The pipeline then returns to the `AWAITING_WAKE_WORD` state.

4.  **TTS Processing (Asynchronous):**
    -   The Cortex can queue TTS requests at any time using `tk_audio_pipeline_synthesize_text`.
    -   Requests are added to a priority queue. If a new request has a higher priority (e.g., `CRITICAL`) than the one currently playing, the pipeline calls the `on_tts_interrupt` callback.
    -   The host application is responsible for stopping its audio player when it receives this callback, ensuring high-priority messages are heard immediately.

## 4. Key Components

-   **`tk_audio_pipeline.c`:** Manages the overall audio pipeline state machine, callbacks, and coordination of all models.
-   **`tk_sound_classifier.c`:** The new module for ambient sound classification using ONNX Runtime.
-   **`tk_wake_word_porcupine.c`:** Interface for the Porcupine wake word engine.
-   **`tk_vad_silero.c`:** An interface to the Silero VAD model.
-   **`tk_asr_whisper.c`:** A wrapper for the `whisper.cpp` library.
-   **`tk_tts_piper.c`:** A wrapper for the Piper TTS library.

## 5. Integration with other Modules

-   **Cortex:** The Audio module sends transcribed text and detected ambient sounds to the Cortex. It receives text to be synthesized and relies on the Cortex to implement the `on_tts_interrupt` callback to handle prioritized playback.
-   **Sensors:** Audio data can be correlated with sensor data (e.g., IMU) to suppress noise generated by user movement.
-   **Interaction:** The TTS output is a key part of the feedback provided to the user.
