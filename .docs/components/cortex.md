<!--
This documentation was written by Jules - Google labs bot.
Original code by phkaiser13.
-->

# Cortex Module

## 1. Overview

The Cortex module is the central processing unit of the TrackieLLM system, acting as the "brain" that drives intelligent and proactive behavior. It is responsible for integrating data from various subsystems (vision, audio, sensors), reasoning about the environment, and making decisions to assist the user.

Its primary goal is to transform raw sensor data into actionable insights, enabling the user to navigate and interact with their surroundings safely and efficiently.

## 2. Core Responsibilities

- **Contextual Reasoning:** Fuses multimodal inputs (e.g., detected objects, spoken commands, ambient sounds) to build a comprehensive understanding of the user's current situation.
- **Decision Making:** Evaluates the contextual model to determine the most appropriate action or feedback for the user, such as issuing a warning, providing navigation guidance, or answering a question.
- **Memory Management:** Maintains short-term and long-term memory of the environment, user preferences, and past interactions to provide a personalized and consistent experience.
- **LLM Integration:** Manages the interaction with the on-device Large Language Model (`Mistral-7B`), formatting prompts and interpreting the model's output to generate natural language feedback.

## 3. Architecture

The Cortex module is implemented as a hybrid C and Rust system, leveraging the strengths of each language:

- **C (`tk_cortex_main.c`, `tk_contextual_reasoner.c`, `tk_decision_engine.c`):** Provides a stable, low-level interface for integration with the rest of the C-based TrackieLLM framework. It handles the main event loop and dispatches tasks to the Rust components.
- **Rust (`reasoning.rs`, `memory_manager.rs`):** Implements the core logic for reasoning, memory, and safety-critical computations, taking advantage of Rust's safety and performance features.

This hybrid architecture ensures both high performance and robust, memory-safe operation.

## 4. Key Components

### 4.1. Contextual Reasoner (`tk_contextual_reasoner.c` & `reasoning.rs`)

- **Function:** Aggregates data streams from the Vision, Audio, and Sensor modules.
- **Process:**
    1. Receives inputs like identified objects, transcribed speech, and IMU data.
    2. Constructs a "world model" representing the immediate environment.
    3. Uses rule-based logic and heuristics to perform initial analysis (e.g., "is the object a hazard?").
    4. Prepares a structured prompt for the LLM to perform higher-level reasoning.

### 4.2. Decision Engine (`tk_decision_engine.c`)

- **Function:** Takes the output from the Contextual Reasoner and the LLM to select a course of action.
- **Process:**
    1. Evaluates the LLM's suggestions against a set of safety constraints and operational modes.
    2. Prioritizes critical alerts (e.g., collision warnings) over informational messages.
    3. Translates the final decision into a command for another subsystem (e.g., an audio prompt for the TTS engine, a haptic signal).

### 4.3. Memory Manager (`memory_manager.rs`)

- **Function:** Provides the Cortex with a persistent memory layer.
- **Features:**
    - **Short-Term Memory:** Caches recent objects, locations, and interactions for immediate recall.
    - **Long-Term Memory:** (Future implementation) Will store user preferences, familiar places, and important objects.
    - **Data Association:** Links related information, such as associating a person's name with their face.

## 5. Data Flow

1.  **Input:** The Cortex receives `tk_event_t` structures from other modules via a central event bus.
2.  **Reasoning:** The Contextual Reasoner processes the event and updates its internal world model.
3.  **LLM Query:** If complex reasoning is required, a query is sent to the `Mistral-7B` model.
4.  **Decision:** The Decision Engine uses the reasoner's output and the LLM's response to choose an action.
5.  **Output:** An action is dispatched as a command to the appropriate module (e.g., `audio_pipeline_say()`, `haptics_trigger()`).

## 6. Integration with other Modules

- **Vision:** Receives object detection results, text recognition, and depth information.
- **Audio:** Receives user commands (ASR) and provides text for synthesis (TTS).
- **Sensors:** Receives fused sensor data (e.g., from IMU) to understand user movement and orientation.
- **Interaction:** Sends commands to the feedback manager to communicate with the user (voice, haptics).

## Developer and Author

*   **Original Code:** phkaiser13
*   **Documentation:** Jules - Google labs bot
