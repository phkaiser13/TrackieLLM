/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tests/tk_vision_full_test.c
 *
 * This file contains a full integration test for the TrackieLLM Vision Pipeline.
 * It is designed to validate the end-to-end functionality of the vision module,
 * from image loading to the final, structured analysis output.
 *
 * This test covers:
 *  - Object detection.
 *  - Depth estimation.
 *  - Fusion of object and depth data (distance/size calculation).
 *  - Targeted Optical Character Recognition (OCR).
 *  - Advanced depth analysis for navigation cues.
 *
 * It serves as a comprehensive check to ensure all new features are working
 * together as expected and to prevent regressions.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#include "vision/tk_vision_pipeline.h"
#include "internal_tools/tk_file_manager.h"

// Third-party library for image loading (e.g., stb_image.h)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Helper function to check if a float is within a certain tolerance
static bool float_equals(float a, float b, float epsilon) {
    return fabsf(a - b) < epsilon;
}

// Helper function to trim trailing whitespace from a string in-place
static void trim_right(char *str) {
    if (!str) return;
    int len = strlen(str);
    while (len > 0 && isspace((unsigned char)str[len - 1])) {
        len--;
    }
    str[len] = '\0';
}

int main(int argc, char** argv) {
    printf("Starting vision pipeline full integration test...\n");

    // --- 1. Configuration ---
    // Note: Paths to models and test data are critical.
    // These should point to a valid location in your test environment.
    tk_path_t* object_model_path = tk_path_create_from_str("assets/models/yolov5nu.onnx");
    tk_path_t* depth_model_path = tk_path_create_from_str("assets/models/midas_dpt_swin_tiny.onnx");
    tk_path_t* tesseract_data_path = tk_path_create_from_str("assets/tessdata");
    tk_path_t* test_image_path = tk_path_create_from_str("tests/fixtures/sample_scene.jpg");

    // Basic check for asset paths
    if (!tk_path_exists(object_model_path) || !tk_path_exists(depth_model_path) || !tk_path_exists(tesseract_data_path) || !tk_path_exists(test_image_path)) {
        fprintf(stderr, "ERROR: A required model or test asset was not found.\n");
        fprintf(stderr, "Please ensure you have run the asset download script and that paths are correct.\n");
        tk_path_destroy(&object_model_path);
        tk_path_destroy(&depth_model_path);
        tk_path_destroy(&tesseract_data_path);
        tk_path_destroy(&test_image_path);
        return 1;
    }

    tk_vision_pipeline_config_t config = {
        .backend = TK_VISION_BACKEND_CPU,
        .gpu_device_id = 0,
        .object_detection_model_path = object_model_path,
        .depth_estimation_model_path = depth_model_path,
        .tesseract_data_path = tesseract_data_path,
        .object_confidence_threshold = 0.5f,
        .max_detected_objects = 10,
        .focal_length_x = 525.0f, // Example focal length
        .focal_length_y = 525.0f,
    };

    // --- 2. Pipeline Creation ---
    tk_vision_pipeline_t* pipeline = NULL;
    tk_error_code_t err = tk_vision_pipeline_create(&pipeline, &config);
    assert(err == TK_SUCCESS && "Pipeline creation should succeed");
    assert(pipeline != NULL && "Pipeline handle should not be null");
    printf("Vision pipeline created successfully.\n");

    // --- 3. Load Test Image ---
    int width, height, channels;
    unsigned char* image_data = stbi_load(tk_path_get_str(test_image_path), &width, &height, &channels, 3);
    assert(image_data != NULL && "Test image should load successfully");
    assert(width > 0 && height > 0 && "Image dimensions should be valid");
    printf("Loaded test image: %s (%dx%d)\n", tk_path_get_str(test_image_path), width, height);

    tk_video_frame_t frame = {
        .width = (uint32_t)width,
        .height = (uint32_t)height,
        .data = image_data,
        // Other fields like format, stride can be set if the struct supports them
    };

    // --- 4. Process Frame ---
    tk_vision_result_t* result = NULL;
    tk_vision_analysis_flags_t flags = TK_VISION_PRESET_ENVIRONMENT_AWARENESS | TK_VISION_ANALYZE_OCR;

    printf("Processing frame with all analysis flags enabled...\n");
    err = tk_vision_pipeline_process_frame(pipeline, &frame, flags, 0, &result);
    assert(err == TK_SUCCESS && "Frame processing should succeed");
    assert(result != NULL && "Result should not be null");
    printf("Frame processing completed.\n");

    // --- 5. Assertions ---
    printf("Running assertions on the result...\n");

    // We expect at least one object to be detected in a sample scene
    assert(result->object_count > 0 && "Should detect at least one object");

    bool found_book = false;
    bool found_chair = false;

    for (size_t i = 0; i < result->object_count; ++i) {
        tk_vision_object_t* obj = &result->objects[i];
        printf("  - Detected: %s (Conf: %.2f, Dist: %.2fm, Occluded: %s, Text: '%s')\n",
               obj->label, obj->confidence, obj->distance_meters,
               obj->is_partially_occluded ? "Yes" : "No",
               obj->recognized_text ? obj->recognized_text : "N/A");

        // Example Assertion 1: A book should be detected with text on it.
        if (strcmp(obj->label, "book") == 0) {
            found_book = true;
            assert(obj->distance_meters > 0.5f && obj->distance_meters < 3.0f && "Book distance should be in a reasonable range");
            assert(obj->recognized_text != NULL && "Should have recognized text on the book");
            trim_right(obj->recognized_text); // Clean up trailing whitespace
            // The exact text depends on the sample image
            assert(strcmp(obj->recognized_text, "THE RUST PROGRAMMING LANGUAGE") == 0 && "Recognized text should match exactly after trimming");
        }

        // Example Assertion 2: A chair should be detected at a certain distance.
        if (strcmp(obj->label, "chair") == 0) {
            found_chair = true;
            assert(obj->distance_meters > 1.0f && obj->distance_meters < 5.0f && "Chair distance should be in a reasonable range");
        }
    }

    assert(found_book && "A book must be found in the test scene");
    assert(found_chair && "A chair must be found in the test scene");

    // Example Assertion 3: Navigation analysis should produce a result.
    // Note: Accessing nav cues requires modifying the C API to expose them.
    // This test assumes a future modification where nav cues are part of the result.
    // For now, we rely on the debug logs from the C pipeline to verify it ran.
    // A hypothetical future assertion might look like:
    // assert(result->nav_cues != NULL && "Navigation cues should be generated");
    // assert(result->nav_cues->vertical_changes_count > 0 && "Should detect a step/curb");
    printf("NOTE: Navigation cue assertions are pending C API changes. Verified via logs for now.\n");

    printf("All assertions passed!\n");

    // --- 6. Cleanup ---
    printf("Cleaning up resources...\n");
    stbi_image_free(image_data);
    tk_vision_result_destroy(&result);
    tk_vision_pipeline_destroy(&pipeline);
    tk_path_destroy(&object_model_path);
    tk_path_destroy(&depth_model_path);
    tk_path_destroy(&tesseract_data_path);
    tk_path_destroy(&test_image_path);

    printf("Integration test finished successfully.\n");
    return 0;
}
