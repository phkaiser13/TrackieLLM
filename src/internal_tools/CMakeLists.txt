add_library(internal_tools tk_config_parser.c tk_file_manager.c)

target_include_directories(internal_tools PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src)
