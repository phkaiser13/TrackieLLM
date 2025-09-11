#include "cortex/tk_cortex_main.h"
#include "utils/tk_logging.h"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>

// --- Mock Callbacks & Global State for Validation ---

std::atomic<bool> g_tts_callback_triggered(false);
std::atomic<int> g_state_changes(0);

void mock_on_state_change(tk_system_state_e new_state, void* user_data) {
    std::cout << "[TEST_CALLBACK] State changed to: " << new_state << std::endl;
    g_state_changes++;
}

void mock_on_tts_audio_ready(const int16_t* audio_data, size_t frame_count, uint32_t sample_rate, void* user_data) {
    std::cout << "[TEST_CALLBACK] TTS audio received. Frames: " << frame_count << ", Rate: " << sample_rate << std::endl;
    g_tts_callback_triggered = true;
}

// --- Main Test Function ---

int main() {
    std::cout << "Starting Cortex Integration Test..." << std::endl;

    // 1. Setup Phase
    // Initialize a basic logger for the test
    tk_log_config_t log_config = {};
    log_config.level = TK_LOG_LEVEL_DEBUG;
    log_config.log_to_console = true;
    tk_log_init(&log_config);

    // Define mock paths to models. In a real CI environment, these would
    // point to actual small test models downloaded by a script.
    // For now, we assume they exist at these placeholder paths.
    tk_model_paths_t model_paths = {};
    model_paths.llm_model = "assets/models/mistral-7b-v0.1.Q4_K_M.gguf";
    model_paths.object_detection_model = "assets/models/yolov5n.onnx";
    model_paths.depth_estimation_model = "assets/models/dpt-swin-tiny.onnx";
    model_paths.asr_model = "assets/models/ggml-tiny.en.bin";
    model_paths.tts_model_dir = "assets/models/tts/";
    model_paths.vad_model = "assets/models/silero_vad.onnx";
    model_paths.tesseract_data_dir = "assets/models/tessdata/";

    tk_cortex_config_t config = {};
    config.model_paths = model_paths;
    config.gpu_device_id = -1; // Use CPU for this test to avoid hardware dependency
    config.main_loop_frequency_hz = 10.0f;
    config.user_language = "en-US";
    config.user_data = nullptr;

    tk_cortex_callbacks_t callbacks = {};
    callbacks.on_state_change = mock_on_state_change;
    callbacks.on_tts_audio_ready = mock_on_tts_audio_ready;

    tk_cortex_t* cortex = nullptr;
    tk_error_code_t result = tk_cortex_create(&cortex, &config, callbacks);
    if (result != TK_SUCCESS) {
        std::cerr << "Failed to create Cortex. Error: " << result << std::endl;
        tk_log_shutdown();
        return 1;
    }
    std::cout << "Cortex created successfully." << std::endl;

    // 2. Execution Phase
    std::cout << "Starting Cortex run loop in a background thread..." << std::endl;
    std::thread cortex_thread([&]() {
        tk_cortex_run(cortex);
    });

    // Let the Cortex initialize and enter its idle loop
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Inject a dummy video frame
    std::cout << "Injecting a dummy video frame..." << std::endl;
    std::vector<uint8_t> dummy_video_data(640 * 480 * 3, 128); // Gray frame
    tk_video_frame_t video_frame = {};
    video_frame.width = 640;
    video_frame.height = 480;
    video_frame.stride = 640 * 3;
    video_frame.format = TK_PIXEL_FORMAT_RGB8;
    video_frame.data = dummy_video_data.data();
    tk_cortex_inject_video_frame(cortex, &video_frame);

    // Inject some dummy audio to simulate a user command
    std::cout << "Injecting a dummy audio frame to trigger ASR..." << std::endl;
    std::vector<int16_t> dummy_audio_data(16000 * 2, 0); // 2 seconds of silence
    tk_cortex_inject_audio_frame(cortex, dummy_audio_data.data(), dummy_audio_data.size());

    // Wait for processing to occur
    std::cout << "Waiting for 5 seconds for processing..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(5));

    // 3. Shutdown Phase
    std::cout << "Stopping Cortex..." << std::endl;
    tk_cortex_stop(cortex);
    cortex_thread.join();
    std::cout << "Cortex thread joined." << std::endl;

    tk_cortex_destroy(&cortex);
    std::cout << "Cortex destroyed." << std::endl;

    tk_log_shutdown();

    // 4. Validation Phase
    std::cout << "--- Test Validation ---" << std::endl;
    bool test_passed = true;
    if (g_state_changes > 2) { // Should see at least INITIALIZING -> IDLE -> PROCESSING -> IDLE
        std::cout << "[SUCCESS] State change callback was triggered multiple times." << std::endl;
    } else {
        std::cout << "[FAILURE] State change callback was not triggered as expected." << std::endl;
        test_passed = false;
    }

    // In this simple test, the dummy audio (silence) might not result in a transcription
    // and thus no TTS response. A more advanced test would use a real audio file.
    // So we don't fail the test if TTS wasn't triggered.
    std::cout << "[INFO] TTS callback triggered: " << (g_tts_callback_triggered ? "Yes" : "No") << std::endl;

    if (test_passed) {
        std::cout << "\nIntegration Test Passed!" << std::endl;
        return 0;
    } else {
        std::cout << "\nIntegration Test Failed!" << std::endl;
        return 1;
    }
}
