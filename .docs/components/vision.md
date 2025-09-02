<!--
This documentation was written by Jules - Google labs bot.
Original code by phkaiser13.
-->

# Vision Module

## 1. Overview

The Vision module is TrackieLLM's primary sensory input for understanding the physical world. It acts as the "eyes" of the system, processing real-time video streams to identify objects, read text, analyze spatial layouts, and detect potential hazards.

Its main purpose is to convert raw pixel data from the camera into structured, meaningful information that the Cortex module can use for reasoning and decision-making.

## 2. Core Responsibilities

- **Object Detection:** Identifies and locates a wide range of objects in the environment, from everyday items like chairs and doors to critical safety elements like staircases and vehicles.
- **Depth Perception:** Estimates the distance to objects and surfaces, enabling navigation, obstacle avoidance, and free-space detection.
- **Text Recognition (OCR):** Extracts and reads text from signs, labels, and documents, providing the user with access to written information.
- **Scene Analysis:** Interprets the overall visual scene to understand the context, such as identifying whether the user is indoors, outdoors, or crossing a street.

## 3. Architecture and Models

The Vision module is a sophisticated pipeline that leverages multiple AI models and computer vision algorithms. It is implemented primarily in C++ for performance, with Rust components for memory-safe data processing.

### 3.1. AI Models

- **Object Detection:**
    - **Model:** `YOLOv5nu`
    - **Format:** ONNX
    - **Details:** A lightweight and fast object detection model optimized for edge devices. It is responsible for identifying the "what" and "where" of objects in the camera's view.

- **Depth Estimation:**
    - **Model:** `DPT-SwinV2-Tiny-256` (MiDaS 3.1)
    - **Format:** ONNX (INT8 quantized)
    - **Details:** A state-of-the-art monocular depth estimation model. It generates a dense depth map from a single camera image, which is crucial for navigation and understanding spatial relationships.

- **Text Recognition (OCR):**
    - **Engine:** `Tesseract OCR`
    - **Integration:** Accessed via its native C++ API.
    - **Details:** A powerful OCR engine used to extract text from detected regions of interest in the video feed.

### 3.2. Pipeline Stages

The vision processing pipeline operates as follows:

1.  **Frame Acquisition:** Captures a new frame from the camera sensor.
2.  **Preprocessing:** The frame is resized, normalized, and prepared for the different AI models.
3.  **Parallel Inference:** The preprocessed frame is sent to the object detection and depth estimation models, which run in parallel to minimize latency.
4.  **Object Analysis (`object_analysis.rs`):** The bounding boxes from YOLO are processed. The corresponding depth values from the MiDaS output are used to determine the distance and size of each detected object.
5.  **Depth Processing (`depth_processing.rs`):** The depth map is analyzed to identify key navigation features:
    - **Free Space:** Detects clear paths for walking.
    - **Obstacles:** Identifies potential hazards.
    - **Grasp Points:** (Future implementation) Suggests points where an object can be picked up.
6.  **Text Recognition:** If text is detected as an object, the corresponding image region is sent to Tesseract for OCR.
7.  **Data Structuring:** The results (detected objects with their properties, recognized text, navigation cues) are packaged into a structured format.
8.  **Event Dispatch:** The structured data is sent to the Cortex module as a `vision_event_t` for high-level reasoning.

## 4. Key Components

- **`tk_vision_pipeline.c`:** The main entry point for the vision pipeline. It orchestrates the flow of data between the different components.
- **`tk_object_detector.c`:** A wrapper for the YOLOv5nu ONNX model, responsible for running inference and outputting bounding boxes.
- **`tk_depth_midas.c`:** A wrapper for the MiDaS ONNX model, responsible for generating the depth map.
- **`tk_text_recognition.cpp`:** The interface to the Tesseract OCR engine.
- **`depth_processing.rs` / `object_analysis.rs`:** Rust components that handle the complex logic of fusing the outputs from the different models in a memory-safe way.

## 5. Integration with other Modules

- **Cortex:** The primary consumer of the Vision module's output. It uses the structured visual data to understand the environment and make decisions.
- **GPU:** The Vision module heavily relies on the GPU for accelerating the ONNX model inference (CUDA, ROCm, or Metal).
- **Sensors:** Vision data is correlated with IMU data in the Cortex to distinguish between user movement and camera movement.

## Developer and Author

*   **Original Code:** phkaiser13
*   **Documentation:** Jules - Google labs bot
