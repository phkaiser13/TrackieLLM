add_subdirectory(internal_tools)

if(TK_ENABLE_GPU)
  add_subdirectory(gpu)
endif()
