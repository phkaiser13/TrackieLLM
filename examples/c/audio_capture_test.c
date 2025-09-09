/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: audio_capture_test.c
 *
 * This is a simple test executable to demonstrate audio capture from a
 * microphone and saving it to a WAV file using the miniaudio library.
 *
 * This test fulfills the requirement for Task 2.2.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
#include "utils/tk_logging.h"

#include <stdio.h>
#include <stdlib.h>

#define CAPTURE_DURATION_SECONDS 5
#define SAMPLE_RATE 16000
#define CHANNELS 1
#define FORMAT ma_format_s16
#define OUTPUT_FILENAME "capture_test.wav"

// --- Data for capture callback ---
typedef struct {
    ma_encoder encoder;
    bool       encoder_initialized;
} capture_user_data;

// --- Miniaudio Callbacks ---

void capture_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount)
{
    capture_user_data* pUserData = (capture_user_data*)pDevice->pUserData;
    if (pUserData == NULL || !pUserData->encoder_initialized) {
        return;
    }

    // Write the captured audio frames to the WAV file encoder.
    ma_encoder_write_pcm_frames(&pUserData->encoder, pInput, frameCount, NULL);

    (void)pOutput; // Not using playback.
}

// --- Main Test Function ---

int main() {
    tk_logging_init("audio_capture_test", TK_LOG_LEVEL_DEBUG);
    TK_LOG_INFO("Starting audio capture test...");
    TK_LOG_INFO("This test will capture %d seconds of audio from the default microphone.", CAPTURE_DURATION_SECONDS);
    TK_LOG_INFO("The captured audio will be saved to '%s'.", OUTPUT_FILENAME);

    ma_result result;
    capture_user_data user_data = {0};

    // 1. Initialize the WAV encoder
    ma_encoder_config encoder_config = ma_encoder_config_init(ma_encoding_format_wav, FORMAT, CHANNELS, SAMPLE_RATE);
    result = ma_encoder_init_file(OUTPUT_FILENAME, &encoder_config, &user_data.encoder);
    if (result != MA_SUCCESS) {
        TK_LOG_ERROR("Failed to initialize WAV encoder for file '%s'.", OUTPUT_FILENAME);
        return -1;
    }
    user_data.encoder_initialized = true;

    // 2. Initialize the capture device
    ma_device_config device_config = ma_device_config_init(ma_device_type_capture);
    device_config.capture.format   = FORMAT;
    device_config.capture.channels = CHANNELS;
    device_config.sampleRate       = SAMPLE_RATE;
    device_config.dataCallback     = capture_callback;
    device_config.pUserData        = &user_data;

    ma_device device;
    result = ma_device_init(NULL, &device_config, &device);
    if (result != MA_SUCCESS) {
        TK_LOG_ERROR("Failed to initialize capture device. Error code: %d", result);
        ma_encoder_uninit(&user_data.encoder);
        return -2;
    }

    // 3. Start capturing
    result = ma_device_start(&device);
    if (result != MA_SUCCESS) {
        TK_LOG_ERROR("Failed to start capture device. Error code: %d", result);
        ma_device_uninit(&device);
        ma_encoder_uninit(&user_data.encoder);
        return -3;
    }

    TK_LOG_INFO("Capturing audio... Please speak.");

    // 4. Wait for the specified duration
    ma_sleep(CAPTURE_DURATION_SECONDS * 1000);

    // 5. Stop and clean up
    ma_device_uninit(&device);
    ma_encoder_uninit(&user_data.encoder); // This also finalizes the WAV file

    TK_LOG_INFO("Capture complete. File '%s' has been saved.", OUTPUT_FILENAME);
    printf("\n---> TEST SUCCEEDED (if '%s' was created and contains audio).\n", OUTPUT_FILENAME);

    return 0;
}
