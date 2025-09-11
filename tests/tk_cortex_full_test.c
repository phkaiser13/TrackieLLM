#include <stdio.h>
#include <string.h>
#include <assert.h>

// Correct include order
#include "cortex/tk_contextual_reasoner.h"
#include "cortex/tk_cortex_main.h"
#include "cortex/tk_decision_engine.h"

// FFI functions from Rust that we need to call
extern void tk_cortex_rust_init_reasoner(tk_contextual_reasoner_t* reasoner_ptr);
extern bool tk_cortex_generate_prompt(char* prompt_buffer, size_t buffer_size, const char* user_query);


void test_stress_scenario_prioritization() {
    printf("Running test: test_stress_scenario_prioritization...\n");

    // 1. Setup
    // Use a proper config and callbacks struct, even if mostly empty for this test
    tk_model_paths_t model_paths = {0}; // All paths are NULL
    tk_cortex_config_t cortex_config = {
        .model_paths = model_paths,
        .gpu_device_id = -1, // CPU
        .main_loop_frequency_hz = 10.0f,
        .user_language = "pt-BR",
        .user_data = NULL
    };
    tk_cortex_callbacks_t callbacks = {0}; // No callbacks needed for this test

    tk_cortex_t* cortex = NULL;
    tk_error_code_t result = tk_cortex_create(&cortex, &cortex_config, callbacks);
    assert(result == TK_SUCCESS && "Cortex creation should succeed");
    assert(cortex != NULL && "Cortex object should not be null");

    // Get the internal reasoner to manipulate its state for the test
    tk_contextual_reasoner_t* reasoner = tk_cortex_get_contextual_reasoner(cortex);
    assert(reasoner != NULL && "Reasoner should not be null");

    // Initialize the rust side with the reasoner pointer
    tk_cortex_rust_init_reasoner(reasoner);

    // 2. Simulate Events
    // A fire alarm is detected
    result = tk_contextual_reasoner_update_ambient_sound(reasoner, TK_AMBIENT_SOUND_FIRE_ALARM, 0.9f);
    assert(result == TK_SUCCESS && "Updating ambient sound should succeed");

    // A step down is detected
    result = tk_contextual_reasoner_update_navigation_cues(reasoner, TK_NAVIGATION_CUE_STEP_DOWN, 1.0f);
    assert(result == TK_SUCCESS && "Updating navigation cues should succeed");

    // 3. Generate the prompt with a user query
    const char* user_query = "Onde está minha garrafa de água?";
    char prompt_buffer[2048];

    bool success = tk_cortex_generate_prompt(prompt_buffer, sizeof(prompt_buffer), user_query);
    assert(success && "Prompt generation should succeed");

    printf("Generated Prompt: %s\n", prompt_buffer);

    // 4. Assertions
    // Check that the prompt prioritizes the urgent information
    assert(strstr(prompt_buffer, "URGENTE") != NULL && "Prompt should contain URGENTE");
    assert(strstr(prompt_buffer, "ALARME DE INCÊNDIO DETECTADO") != NULL && "Prompt should mention the fire alarm");
    assert(strstr(prompt_buffer, "degrau para baixo") != NULL && "Prompt should mention the step down");
    assert(strstr(prompt_buffer, "garrafa de água") != NULL && "Prompt should still contain the user's query");

    // Check that the urgent info comes before the user query
    char* urgent_pos = strstr(prompt_buffer, "URGENTE");
    char* query_pos = strstr(prompt_buffer, "garrafa de água");
    assert(urgent_pos < query_pos && "Urgent information must come before the user query in the prompt");

    // Cleanup
    tk_cortex_destroy(&cortex);
    printf("Test passed!\n");
}

int main() {
    test_stress_scenario_prioritization();
    return 0;
}
