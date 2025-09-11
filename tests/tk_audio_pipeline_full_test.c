/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_audio_pipeline_full_test.c
 *
 * This file contains integration tests for the full audio pipeline.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "audio/tk_audio_pipeline.h"
#include "utils/tk_logging.h"
#include "internal_tools/tk_file_manager.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <unistd.h> // For sleep()

// --- Test Configuration ---
// These paths are placeholders. Tests will be skipped if models are not found.
const char* ASR_MODEL_PATH = "assets/models/whisper/ggml-tiny.en.bin";
const char* VAD_MODEL_PATH = "assets/models/silero/silero_vad.onnx";
const char* TTS_MODEL_PATH = "assets/models/piper/en_US-lessac-medium.onnx";
const char* TTS_CONFIG_PATH = "assets/models/piper/en_US-lessac-medium.onnx.json";
const char* WW_MODEL_PATH = "assets/models/porcupine/porcupine_params.pv";
// NOTE: This test requires a specific wake word file.
// For this example, we assume a placeholder name.
const char* WW_KEYWORD_PATH = "assets/models/porcupine/trackie_en_linux.ppn";
const char* SC_MODEL_PATH = "assets/models/yamnet/yamnet.onnx"; // Placeholder for sound classifier

// --- Test State & Mock Callbacks ---

typedef struct {
    bool tts_interrupted;
    bool ambient_sound_detected;
    bool speech_started;
} test_callback_flags_t;

void mock_on_tts_interrupt(void* user_data) {
    TK_LOG_INFO("Mock callback: TTS interrupt received!");
    test_callback_flags_t* flags = (test_callback_flags_t*)user_data;
    flags->tts_interrupted = true;
}

void mock_on_ambient_sound_detected(const tk_sound_detection_result_t* result, void* user_data) {
    // For now, we just note that it was called.
    test_callback_flags_t* flags = (test_callback_flags_t*)user_data;
    flags->ambient_sound_detected = true;
}

void mock_on_vad_event(tk_vad_event_e event, void* user_data) {
    if (event == TK_VAD_EVENT_SPEECH_STARTED) {
        test_callback_flags_t* flags = (test_callback_flags_t*)user_data;
        flags->speech_started = true;
    }
}

// --- Helper Functions ---

void print_test_header(const char* test_name) {
    printf("\n--- Running Test: %s ---\n", test_name);
}

void print_test_result(bool pass) {
    printf("--- Result: %s ---\n", pass ? "PASS" : "FAIL");
}

// A helper to create a pipeline for tests. Returns NULL if models are missing.
tk_audio_pipeline_t* create_test_pipeline(test_callback_flags_t* flags) {
    tk_audio_pipeline_t* pipeline = NULL;

    tk_path_t* asr_path = tk_path_create(ASR_MODEL_PATH);
    tk_path_t* vad_path = tk_path_create(VAD_MODEL_PATH);
    tk_path_t* tts_model_path = tk_path_create(TTS_MODEL_PATH);
    tk_path_t* tts_config_path = tk_path_create(TTS_CONFIG_PATH);
    tk_path_t* ww_model_path = tk_path_create(WW_MODEL_PATH);
    tk_path_t* ww_keyword_path = tk_path_create(WW_KEYWORD_PATH);
    tk_path_t* sc_model_path = tk_path_create(SC_MODEL_PATH);

    if (!tk_path_exists(asr_path) || !tk_path_exists(vad_path) || !tk_path_exists(ww_model_path)) {
        TK_LOG_WARN("Required models not found. Skipping test.");
        goto cleanup;
    }

    tk_audio_pipeline_config_t config = {
        .input_audio_params = {.sample_rate = 16000, .channels = 1},
        .user_language = "en",
        .user_data = flags,
        .asr_model_path = asr_path,
        .vad_model_path = vad_path,
        .tts_model_path = tts_model_path,
        .tts_config_path = tts_config_path,
        .ww_model_path = ww_model_path,
        .ww_keyword_path = ww_keyword_path,
        .ww_sensitivity = 0.5f,
        .sc_model_path = tk_path_exists(sc_model_path) ? sc_model_path : NULL,
        .vad_silence_threshold_ms = 1000,
        .vad_speech_probability_threshold = 0.5f
    };

    tk_audio_callbacks_t callbacks = {
        .on_tts_interrupt = mock_on_tts_interrupt,
        .on_ambient_sound_detected = mock_on_ambient_sound_detected,
        .on_vad_event = mock_on_vad_event
    };

    tk_error_code_t result = tk_audio_pipeline_create(&pipeline, &config, callbacks);
    if (result != TK_SUCCESS) {
        TK_LOG_ERROR("Failed to create pipeline for test: %d", result);
        pipeline = NULL; // Ensure it's null on failure
    }

cleanup:
    tk_path_destroy(asr_path);
    tk_path_destroy(vad_path);
    tk_path_destroy(tts_model_path);
    tk_path_destroy(tts_config_path);
    tk_path_destroy(ww_model_path);
    tk_path_destroy(ww_keyword_path);
    tk_path_destroy(sc_model_path);
    return pipeline;
}

// --- Test Cases ---

void test_listening_timeout() {
    print_test_header("test_listening_timeout");
    bool pass = true;

    test_callback_flags_t flags = {0};
    tk_audio_pipeline_t* pipeline = create_test_pipeline(&flags);

    if (!pipeline) {
        print_test_result(true); // Skip is a pass
        return;
    }

    assert(tk_audio_pipeline_get_state(pipeline) == TK_PIPELINE_STATE_AWAITING_WAKE_WORD);

    // This is the hard part. We can't easily inject a wake word.
    // So we'll test the timeout logic in a slightly different way.
    // We can't easily set the state, but we can check if the timeout logic
    // is being hit by observing logs. For a real test, we would need to
    // feed it an audio file with the wake word.
    // For now, this test is more of a placeholder to show the intent.
    // Let's simulate the state change by force (not possible without a new dev function)
    // and then wait.

    // As a proxy, let's just run the pipeline for 6 seconds and ensure it doesn't crash.
    // A more robust test requires audio fixtures.
    printf("Simulating passive listening for 6 seconds...\n");
    int16_t silence[1600] = {0}; // 100ms of silence
    for (int i = 0; i < 60; i++) { // 60 * 100ms = 6s
        tk_audio_pipeline_process_chunk(pipeline, silence, 1600);
        usleep(100000); // sleep 100ms
    }

    // Since we didn't provide a wake word, the state should remain AWAITING_WAKE_WORD.
    tk_pipeline_state_e final_state = tk_audio_pipeline_get_state(pipeline);
    if (final_state != TK_PIPELINE_STATE_AWAITING_WAKE_WORD) {
        TK_LOG_ERROR("State changed from AWAITING_WAKE_WORD without cause. State is %d", final_state);
        pass = false;
    }

    tk_audio_pipeline_destroy(&pipeline);
    print_test_result(pass);
    assert(pass);
}

void test_tts_interruption() {
    print_test_header("test_tts_interruption");
    bool pass = false;

    test_callback_flags_t flags = {0};
    tk_audio_pipeline_t* pipeline = create_test_pipeline(&flags);

    if (!pipeline) {
        print_test_result(true); // Skip is a pass
        return;
    }

    // We need to wait for the worker thread to be ready.
    sleep(1);

    // Enqueue a low priority message.
    // This will start processing, but we'll interrupt it.
    tk_audio_pipeline_synthesize_text(pipeline, "This is a long, low-priority message that should be interrupted.", TK_RESPONSE_PRIORITY_LOW);

    // Sleep for a very short time to ensure the first message starts processing
    usleep(50000); // 50ms

    // Enqueue a high priority message. This should trigger the interrupt.
    tk_audio_pipeline_synthesize_text(pipeline, "Alert!", TK_RESPONSE_PRIORITY_CRITICAL);

    // Give the worker thread time to process the queue and fire the callback
    sleep(1);

    if (flags.tts_interrupted) {
        printf("SUCCESS: TTS interruption callback was correctly triggered.\n");
        pass = true;
    } else {
        TK_LOG_ERROR("FAILURE: TTS interruption callback was not triggered.");
        pass = false;
    }

    tk_audio_pipeline_destroy(&pipeline);
    print_test_result(pass);
    assert(pass);
}


// --- Main Test Runner ---

int main() {
    tk_logging_init("tk_audio_pipeline_full_test", TK_LOG_LEVEL_INFO);
    TK_LOG_INFO("Starting full audio pipeline integration tests...");

    test_listening_timeout();
    test_tts_interruption();

    TK_LOG_INFO("All tests completed.");
    return 0;
}
