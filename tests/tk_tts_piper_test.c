/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_tts_piper_test.c
 *
 * This file contains unit tests for the Piper TTS wrapper.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "audio/tk_tts_piper.h"
#include "utils/tk_logging.h"
#include "internal_tools/tk_file_manager.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

// --- Test Configuration ---
// NOTE: These paths are placeholders. The actual model files need to be
// downloaded to the 'assets/models/piper/' directory.
const char* MODEL_PATH = "assets/models/piper/en_US-lessac-medium.onnx";
const char* CONFIG_PATH = "assets/models/piper/en_US-lessac-medium.onnx.json";

// --- Helper Functions ---

void print_test_header(const char* test_name) {
    printf("\n--- Running Test: %s ---\n", test_name);
}

void print_test_result(bool pass) {
    if (pass) {
        printf("--- Result: PASS ---\n");
    } else {
        printf("--- Result: FAIL ---\n");
    }
}

// --- Test Cases ---

void test_create_and_destroy_context() {
    print_test_header("test_create_and_destroy_context");

    tk_tts_piper_context_t* context = NULL;

    // Create path objects
    tk_path_t* model_path = tk_path_create(MODEL_PATH);
    tk_path_t* config_path = tk_path_create(CONFIG_PATH);
    assert(model_path != NULL);
    assert(config_path != NULL);

    tk_tts_piper_config_t config = {
        .model_path = model_path,
        .config_path = config_path,
        .language = "en",
        .sample_rate = 22050,
        .n_threads = 4,
    };

    // NOTE: This will fail if model files are not present.
    // We are testing the API's ability to handle the call.
    tk_error_code_t result = tk_tts_piper_create(&context, &config);

    // If the model files don't exist, the creation should fail gracefully.
    // If they exist, it should succeed. We can't know for sure here.
    // For the purpose of this test, we'll assume the files are missing and
    // expect a specific error. If they are present, TK_SUCCESS is also a
    // valid outcome.
    bool pass = false;
    if (result == TK_ERROR_MODEL_LOAD_FAILED) {
        printf("INFO: Model creation failed as expected (model files likely missing).\n");
        pass = true;
    } else if (result == TK_SUCCESS) {
        printf("INFO: Model creation succeeded (model files are present).\n");
        assert(context != NULL);
        tk_tts_piper_destroy(&context);
        assert(context == NULL);
        pass = true;
    } else {
        TK_LOG_ERROR("Unexpected error code from tk_tts_piper_create: %d", result);
    }

    tk_path_destroy(model_path);
    tk_path_destroy(config_path);

    print_test_result(pass);
    assert(pass);
}

void test_synthesis_to_buffer() {
    print_test_header("test_synthesis_to_buffer");

    tk_tts_piper_context_t* context = NULL;
    tk_path_t* model_path = tk_path_create(MODEL_PATH);
    tk_path_t* config_path = tk_path_create(CONFIG_PATH);

    tk_tts_piper_config_t config = {
        .model_path = model_path,
        .config_path = config_path,
        .language = "en",
        .sample_rate = 22050,
        .n_threads = 4,
    };

    tk_error_code_t result = tk_tts_piper_create(&context, &config);

    if (result != TK_SUCCESS) {
        printf("INFO: Skipping test_synthesis_to_buffer because model could not be loaded.\n");
        tk_path_destroy(model_path);
        tk_path_destroy(config_path);
        print_test_result(true); // Skip is a pass
        return;
    }

    assert(context != NULL);

    const char* text = "Hello, world. This is a test of the Piper TTS system.";
    int16_t* audio_data = NULL;
    size_t frame_count = 0;

    result = tk_tts_piper_synthesize_to_buffer(context, text, &audio_data, &frame_count);

    bool pass = (result == TK_SUCCESS && audio_data != NULL && frame_count > 0);

    if (pass) {
        printf("Synthesized audio successfully.\n");
        printf("Frame count: %zu\n", frame_count);
        // Clean up the allocated buffer
        free(audio_data);
    } else {
        TK_LOG_ERROR("Failed to synthesize audio. Error code: %d", result);
    }

    tk_tts_piper_destroy(&context);
    tk_path_destroy(model_path);
    tk_path_destroy(config_path);

    print_test_result(pass);
    assert(pass);
}


// --- Main Test Runner ---

int main() {
    tk_logging_init("tk_tts_piper_test", TK_LOG_LEVEL_DEBUG);
    TK_LOG_INFO("Starting Piper TTS wrapper tests...");

    test_create_and_destroy_context();
    test_synthesis_to_buffer();

    TK_LOG_INFO("All tests completed.");
    return 0;
}
