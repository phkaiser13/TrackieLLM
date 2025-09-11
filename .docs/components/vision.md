# Vision Subsystem

The Vision Subsystem is a cornerstone of the TrackieLLM project, responsible for transforming raw video frames into a structured, semantic understanding of the environment. It is designed to be robust, efficient, and intelligent, providing the Cortex with the critical data needed for high-level assistance.

## Core Capabilities

-   **Object Detection**: Utilizes a YOLOv5nu model to identify and locate up to 80 different types of common objects.
-   **Monocular Depth Estimation**: Employs a MiDaS model (specifically, `DPT-SwinV2-Tiny`) to perceive distances from a single camera feed, creating a dense depth map of the scene.
-   **Optical Character Recognition (OCR)**: Integrates the Tesseract engine to perform **targeted OCR**, reading text only within the bounding boxes of relevant objects like signs and books.
-   **Advanced Data Fusion**: A high-performance Rust implementation combines object detection and depth data to calculate real-world distance and size for each object. The fusion logic is robust, using statistical methods (IQR) to reject outliers and provide stable distance estimates. It also detects potential **partial occlusions** by analyzing depth variance.
-   **Navigational Hazard Detection**: The system analyzes the depth map to identify the traversable ground plane and detect potential hazards such as **steps, holes, and ramps**, providing crucial safety information for navigation.

## Data Flow

1.  **Frame Input**: The `Cortex` captures a video frame and passes it to the `tk_vision_pipeline_process_frame` function, along with flags indicating which analyses are required.
2.  **Parallel Analysis**: The pipeline runs object detection and depth estimation.
3.  **Targeted OCR**: If requested, the pipeline iterates through detected objects. For items like books or signs, it runs OCR exclusively on their bounding boxes, associating the recognized text directly with the object.
4.  **Depth Map Analysis**: The depth map is passed to a Rust module that divides the ground into a grid, classifying each cell's traversability and identifying vertical changes (steps, ramps) and hazards (holes).
5.  **Data Fusion**: The object list and depth map are processed by a Rust function that calculates robust distance and size estimates for each object and flags any that appear to be partially occluded.
6.  **Structured Output**: The final result is a `tk_vision_result_t` struct, which contains a comprehensive, multi-layered understanding of the scene. This includes a list of detected objects (with distance, size, occlusion status, and recognized text), and a summary of navigational cues. This rich data is then used by the `Cortex` for higher-level reasoning.

## Current Status

**100% Complete.** The Vision Subsystem now meets all planned requirements, providing a robust and intelligent understanding of the visual environment. All foundational features have been implemented and validated through integration testing.
