cmake_minimum_required(VERSION 3.10)

project(TrackieTests C)

# Add the main executable for the test suite
add_executable(tk_config_parser_test tk_config_parser_test.c)

# Link libraries
target_link_libraries(tk_config_parser_test PRIVATE internal_tools)

# Include CTest and add a simple test driver
enable_testing()
add_test(NAME config_parser_tests COMMAND tk_config_parser_test)

# Add the Piper TTS test
add_executable(tk_tts_piper_test tk_tts_piper_test.c)
target_link_libraries(tk_tts_piper_test PRIVATE audio internal_tools utils)
add_test(NAME tts_piper_tests COMMAND tk_tts_piper_test)

# Add the Whisper ASR test
add_executable(tk_asr_whisper_test tk_asr_whisper_test.c)
target_link_libraries(tk_asr_whisper_test PRIVATE audio internal_tools utils)
add_test(NAME asr_whisper_tests COMMAND tk_asr_whisper_test)
