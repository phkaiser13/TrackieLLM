/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: audio_say_test.c
 *
 * This is a simple test executable for the `tk_audio_pipeline_say` function.
 * It calls the function with a sample text and model paths to test the
 * standalone text-to-speech playback functionality.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "audio/tk_audio_pipeline.h"
#include "utils/tk_logging.h"
#include <stdio.h>

// --- Test Configuration ---
// NOTE: These paths are placeholders. The actual model files need to be
// downloaded to the 'assets/models/piper/' directory for this test to work.
const char* MODEL_PATH = "assets/models/piper/en_US-lessac-medium.onnx";
const char* CONFIG_PATH = "assets/models/piper/en_US-lessac-medium.onnx.json";
const char* TEXT_TO_SAY = "Hello, world. This is a test of the Trackie-L-L-M voice system.";

int main() {
    // Initialize logging
    tk_logging_init("audio_say_test", TK_LOG_LEVEL_DEBUG);

    TK_LOG_INFO("Starting tk_audio_pipeline_say test...");
    TK_LOG_INFO("Text to synthesize: '%s'", TEXT_TO_SAY);
    TK_LOG_INFO("Using model: %s", MODEL_PATH);

    // Call the function
    tk_error_code_t result = tk_audio_pipeline_say(
        TEXT_TO_SAY,
        MODEL_PATH,
        CONFIG_PATH
    );

    if (result == TK_SUCCESS) {
        TK_LOG_INFO("tk_audio_pipeline_say executed successfully.");
        printf("\n---> TEST SUCCEEDED (if you heard audio).\n");
    } else {
        TK_LOG_ERROR("tk_audio_pipeline_say failed with error code: %d", result);
        printf("\n---> TEST FAILED.\n");
        return 1;
    }

    return 0;
}
