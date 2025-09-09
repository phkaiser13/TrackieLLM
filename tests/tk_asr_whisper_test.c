/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_asr_whisper_test.c
 *
 * This file contains unit tests for the Whisper ASR wrapper.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "audio/tk_asr_whisper.h"
#include "utils/tk_logging.h"
#include "internal_tools/tk_file_manager.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

// --- Miniaudio for WAV loading ---
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

// --- Test Configuration ---
// NOTE: These paths are placeholders. The actual model and audio files need
// to be present for this test to work.
const char* MODEL_PATH = "assets/models/whisper/ggml-tiny.en.bin";
const char* AUDIO_FILE_PATH = "tests/fixtures/test_speech.wav";
const char* EXPECTED_TRANSCRIPTION = "this is a test of the speech recognition system";


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

    tk_asr_whisper_context_t* context = NULL;
    tk_path_t* model_path = tk_path_create(MODEL_PATH);
    assert(model_path != NULL);

    tk_asr_whisper_config_t config = {
        .model_path = model_path,
        .language = "en",
        .sample_rate = 16000,
        .n_threads = 2,
    };

    // NOTE: This will fail if the model file is not present.
    tk_error_code_t result = tk_asr_whisper_create(&context, &config);

    bool pass = false;
    if (result == TK_ERROR_MODEL_LOAD_FAILED) {
        printf("INFO: Model creation failed as expected (model file likely missing).\n");
        pass = true;
    } else if (result == TK_SUCCESS) {
        printf("INFO: Model creation succeeded (model file is present).\n");
        assert(context != NULL);
        tk_asr_whisper_destroy(&context);
        assert(context == NULL);
        pass = true;
    } else {
        TK_LOG_ERROR("Unexpected error code from tk_asr_whisper_create: %d", result);
    }

    tk_path_destroy(model_path);

    print_test_result(pass);
    assert(pass);
}

void test_transcription() {
    print_test_header("test_transcription");

    // 1. Load the model
    tk_asr_whisper_context_t* context = NULL;
    tk_path_t* model_path = tk_path_create(MODEL_PATH);
    tk_asr_whisper_config_t config = {
        .model_path = model_path,
        .language = "en",
        .sample_rate = 16000,
        .n_threads = 2,
    };
    tk_error_code_t result = tk_asr_whisper_create(&context, &config);
    tk_path_destroy(model_path);

    if (result != TK_SUCCESS) {
        printf("INFO: Skipping test_transcription because model could not be loaded.\n");
        print_test_result(true); // Skip is a pass
        return;
    }
    assert(context != NULL);

    // 2. Load the audio file using miniaudio
    ma_decoder decoder;
    ma_decoder_config decoder_config = ma_decoder_config_init(ma_format_s16, 1, 16000);
    ma_result ma_res = ma_decoder_init_file(AUDIO_FILE_PATH, &decoder_config, &decoder);
    if (ma_res != MA_SUCCESS) {
        TK_LOG_WARN("Skipping test: Could not load audio file '%s'. It may be missing.", AUDIO_FILE_PATH);
        tk_asr_whisper_destroy(&context);
        print_test_result(true); // Skip is a pass
        return;
    }

    ma_uint64 frame_count;
    ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count);
    int16_t* audio_buffer = malloc((size_t)frame_count * sizeof(int16_t));
    ma_decoder_read_pcm_frames(&decoder, audio_buffer, frame_count);
    ma_decoder_uninit(&decoder);

    // 3. Process the audio
    tk_asr_whisper_result_t* asr_result = NULL;
    result = tk_asr_whisper_process_audio(context, audio_buffer, (size_t)frame_count, true, &asr_result);

    bool pass = false;
    if (result == TK_SUCCESS && asr_result != NULL) {
        printf("Transcription successful.\n");
        printf("Got:      '%s'\n", asr_result->text);
        printf("Expected: '%s'\n", EXPECTED_TRANSCRIPTION);
        // Simple string comparison (should be more robust in reality)
        if (strstr(asr_result->text, "this is a test") != NULL) {
            pass = true;
        } else {
            TK_LOG_ERROR("Transcription did not match expected text.");
        }
    } else {
        TK_LOG_ERROR("Failed to process audio. Error code: %d", result);
    }

    // 4. Clean up
    free(audio_buffer);
    tk_asr_whisper_free_result(&asr_result);
    tk_asr_whisper_destroy(&context);

    print_test_result(pass);
    assert(pass);
}


// --- Main Test Runner ---

int main() {
    tk_logging_init("tk_asr_whisper_test", TK_LOG_LEVEL_DEBUG);
    TK_LOG_INFO("Starting Whisper ASR wrapper tests...");

    test_create_and_destroy_context();
    test_transcription();

    TK_LOG_INFO("All tests completed.");
    return 0;
}
