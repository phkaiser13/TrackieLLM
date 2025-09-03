cmake_minimum_required(VERSION 3.10)

project(TrackieTests C)

# Add the main executable for the test suite
add_executable(tk_config_parser_test tk_config_parser_test.c)

# Link libraries
target_link_libraries(tk_config_parser_test PRIVATE internal_tools)

# Include CTest and add a simple test driver
enable_testing()
add_test(NAME config_parser_tests COMMAND tk_config_parser_test)
