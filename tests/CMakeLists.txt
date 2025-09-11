cmake_minimum_required(VERSION 3.18)

project(TrackieTests C CXX)

find_package(Threads REQUIRED)

enable_testing()

# Add the main Cortex Integration test
# add_executable(tk_cortex_test tk_cortex_test.cpp)
# target_link_libraries(tk_cortex_test PRIVATE cortex vision audio navigation sensors utils internal_tools Threads::Threads)
# add_test(NAME cortex_integration_test COMMAND tk_cortex_test)

# Add the new full Cortex integration test
add_executable(tk_cortex_full_test tk_cortex_full_test.c)
target_link_libraries(tk_cortex_full_test PRIVATE cortex vision audio navigation sensors utils internal_tools Threads::Threads)
add_test(NAME cortex_full_integration_test COMMAND tk_cortex_full_test)

# Add the full Vision Pipeline integration test
# add_executable(tk_vision_full_test tk_vision_full_test.c)
# target_link_libraries(tk_vision_full_test PRIVATE vision utils internal_tools)
# add_test(NAME vision_full_integration_test COMMAND tk_vision_full_test)

# --- ALL OTHER TESTS DISABLED FOR DEBUGGING ---

# # Find CUDA, which is required for GPU tests
# # find_package(CUDA REQUIRED)

# # Add the main executable for the test suite
# add_executable(tk_config_parser_test tk_config_parser_test.c)

# # Link libraries
# target_link_libraries(tk_config_parser_test PRIVATE internal_tools)

# # Include CTest and add a simple test driver
# add_test(NAME config_parser_tests COMMAND tk_config_parser_test)

# # Add the Piper TTS test
# add_executable(tk_tts_piper_test tk_tts_piper_test.c)
# target_link_libraries(tk_tts_piper_test PRIVATE audio internal_tools utils)
# add_test(NAME tts_piper_tests COMMAND tk_tts_piper_test)

# # Add the Whisper ASR test
# add_executable(tk_asr_whisper_test tk_asr_whisper_test.c)
# target_link_libraries(tk_asr_whisper_test PRIVATE audio internal_tools utils)
# add_test(NAME asr_whisper_tests COMMAND tk_asr_whisper_test)

# # Add the GPU Softmax kernel test (DISABLED due to missing CUDA toolkit in environment)
# # add_executable(tk_gpu_softmax_test tk_gpu_softmax_test.cpp)
# # target_link_libraries(tk_gpu_softmax_test PRIVATE gpu utils ${CUDA_LIBRARIES})
# # add_test(NAME gpu_softmax_test COMMAND tk_gpu_softmax_test)

# Add the full Audio Pipeline integration test
add_executable(tk_audio_pipeline_full_test tk_audio_pipeline_full_test.c)
target_link_libraries(tk_audio_pipeline_full_test PRIVATE audio cortex utils internal_tools Threads::Threads)
add_test(NAME audio_full_integration_test COMMAND tk_audio_pipeline_full_test)
