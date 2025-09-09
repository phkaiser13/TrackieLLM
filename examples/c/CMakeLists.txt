cmake_minimum_required(VERSION 3.10)

project(TrackieExamplesC C)

# --- Basic Inference Example ---
add_executable(basic_inference basic_inference.c)

# --- Audio Say Test Example ---
add_executable(audio_say_test audio_say_test.c)

# Link libraries for the audio test
# We need 'audio' for the pipeline, 'internal_tools' for tk_path,
# and 'utils' for logging.
target_link_libraries(audio_say_test PRIVATE audio internal_tools utils)

# --- Audio Capture Test Example ---
add_executable(audio_capture_test audio_capture_test.c)
target_link_libraries(audio_capture_test PRIVATE utils)
