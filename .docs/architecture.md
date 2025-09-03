# Src ->

Source code of application

        ├── src/
        │   ├── monitoring/
        │   │   ├── tk_system_health.c
        │   │   ├── tk_system_health.h
        │   │   ├── tk_performance_logger.c
        │   │   ├── tk_performance_logger.h
        │   │   └── src/
        │   │       ├── lib.rs
        │   │       ├── metrics_collector.rs
        │   │       └── telemetry.rs
        │   ├── security/
        │   │   ├── tk_auth_manager.c
        │   │   ├── tk_auth_manager.h
        │   │   ├── tk_encryption.c
        │   │   ├── tk_encryption.h
        │   │   └── src/
        │   │       ├── lib.rs
        │   │       ├── key_management.rs
        │   │       └── secure_channels.rs
        │   ├── deployment/
        │   │   ├── tk_updater.c
        │   │   ├── tk_updater.h
        │   │   ├── tk_package_installer.c
        │   │   ├── tk_package_installer.h
        │   │   └── src/
        │   │       ├── lib.rs
        │   │       ├── version_checker.rs
        │   │       └── package_manager.rs
        │   ├── experiments/
        │   │   ├── tk_model_tester.c
        │   │   ├── tk_model_tester.h
        │   │   ├── tk_benchmark_runner.c
        │   │   ├── tk_benchmark_runner.h
        │   │   └── src/
        │   │       ├── lib.rs
        │   │       ├── model_analysis.rs
        │   │       └── metrics_comparator.rs
        │   ├── internal_tools/
        │   │   ├── tk_config_parser.c
        │   │   ├── tk_config_parser.h
        │   │   ├── tk_file_manager.c
        │   │   ├── tk_file_manager.h
        │   │   └── src/
        │   │       ├── lib.rs
        │   │       ├── config_loader.rs
        │   │       └── fs_utils.rs
        │   ├── logging_ext/
        │   │   ├── tk_event_logger.c
        │   │   ├── tk_event_logger.h
        │   │   ├── tk_audit_logger.c
        │   │   ├── tk_audit_logger.h
        │   │   └── src/
        │   │       ├── lib.rs
        │   │       ├── event_formatter.rs
        │   │       └── audit_helpers.rs
        │   ├── memory/
        │   │   ├── tk_memory_pool.c
        │   │   ├── tk_memory_pool.h
        │   │   ├── tk_memory_tracker.c
        │   │   ├── tk_memory_tracker.h
        │   │   └── src/
        │   │       ├── lib.rs
        │   │       ├── allocator.rs
        │   │       └── garbage_collector.rs
        │   ├── ai_models/
        │   │   ├── tk_model_loader.c
        │   │   ├── tk_model_loader.h
        │   │   ├── tk_model_runner.c
        │   │   ├── tk_model_runner.h
        │   │   └── src/
        │   │       ├── lib.rs
        │   │       ├── onnx_runner.rs
        │   │       └── gguf_runner.rs
        │   ├── networking/
        │   │   ├── tk_network_manager.c
        │   │   ├── tk_network_manager.h
        │   │   ├── tk_socket_handler.c
        │   │   ├── tk_socket_handler.h
        │   │   └── src/
        │   │       ├── lib.rs
        │   │       ├── protocol.rs
        │   │       └── connection_pool.rs
        │   ├── async_tasks/
        │   │   ├── tk_task_scheduler.c
        │   │   ├── tk_task_scheduler.h
        │   │   ├── tk_worker_pool.c
        │   │   ├── tk_worker_pool.h
        │   │   └── src/
        │   │       ├── lib.rs
        │   │       ├── task_manager.rs
        │   │       └── async_executor.rs
        │   ├── gpu/extensions/
        │   │   ├── cuda/
        │   │   │   ├── tk_cuda_tensor_ops.cu
        │   │   │   ├── tk_cuda_tensor_ops.h
        │   │   │   ├── tk_cuda_image_ops.cu
        │   │   │   └── tk_cuda_image_ops.h
        │   │   ├── rocm/
        │   │   │   ├── tk_rocm_tensor_ops.cpp
        │   │   │   ├── tk_rocm_tensor_ops.hpp
        │   │   │   ├── tk_rocm_image_ops.cpp
        │   │   │   └── tk_rocm_image_ops.hpp
        │   │   └── metal/
        │   │       ├── tk_metal_tensor_ops.mm
        │   │       ├── tk_metal_tensor_ops.h
        │   │       ├── tk_metal_image_ops.mm
        │   │       └── tk_metal_image_ops.h
        │   ├── integration/
        │   │   ├── tk_external_interface.c
        │   │   ├── tk_external_interface.h
        │   │   └── src/
        │   │       ├── lib.rs
        │   │       ├── bridge.rs
        │   │       └── plugin_manager.rs
        │   └── profiling/
        │       ├── tk_profiler.c
        │       ├── tk_profiler.h
        │       ├── tk_memory_profiler.c
        │       ├── tk_memory_profiler.h
        │       └── src/
        │           ├── lib.rs
        │           ├── profiler_core.rs
        │           └── metrics_collector.rs
        │   ├── cortex/
        │   │   ├── tk_cortex_main.c
        │   │   ├── tk_contextual_reasoner.c
        │   │   ├── tk_decision_engine.c
        │   │   ├── tk_contextual_reasoner.h
        │   │   ├── tk_decision_engine.h
        │   │   └── rust/
        │   │       ├── lib.rs
        │   │       ├── reasoning.rs
        │   │       └── memory_manager.rs
        │   ├── vision/
        │   │   ├── tk_vision_pipeline.c
        │   │   ├── tk_depth_midas.c
        │   │   ├── tk_object_detector.c
        │   │   ├── tk_text_recognition.cpp
        │   │   ├── tk_vision_pipeline.h
        │   │   ├── tk_depth_midas.h
        │   │   ├── tk_object_detector.h
        │   │   └── src/
        │   │       ├── lib.rs
        │   │       ├── depth_processing.rs
        │   │       └── object_analysis.rs
        │   ├── audio/
        │   │   ├── tk_audio_pipeline.c
        │   │   ├── tk_asr_whisper.c
        │   │   ├── tk_tts_piper.c
        │   │   ├── tk_audio_pipeline.h
        │   │   ├── tk_asr_whisper.h
        │   │   ├── tk_tts_piper.h
        │   │   └── src/
        │   │       ├── lib.rs
        │   │       ├── asr_processing.rs
        │   │       └── tts_synthesis.rs
        │   ├── sensors/
        │   │   ├── tk_sensors_fusion.c
        │   │   ├── tk_vad_silero.c
        │   │   ├── tk_sensors_fusion.h
        │   │   ├── tk_vad_silero.h
        │   │   └── src/
        │   │       ├── lib.rs
        │   │       ├── sensor_filters.rs
        │   │       └── sensor_fusion.rs
        │   ├── gpu/
        │   │   ├── cuda/
        │   │   │   ├── tk_cuda_dispatch.cu
        │   │   │   ├── tk_cuda_dispatch.h
        │   │   │   ├── tk_cuda_kernels.cu
        │   │   │   ├── tk_cuda_kernels.h
        │   │   │   ├── tk_cuda_math_helpers.cu
        │   │   │   └── tk_cuda_math_helpers.h
        │   │   ├── rocm/
        │   │   │   ├── tk_rocm_dispatch.cpp
        │   │   │   ├── tk_rocm_dispatch.hpp
        │   │   │   ├── tk_rocm_kernels.cpp
        │   │   │   └── tk_rocm_kernels.hpp
        │   │   └── metal/
        │   │       ├── tk_metal_dispatch.mm
        │   │       ├── tk_metal_kernels.metal
        │   │       ├── tk_metal_helpers.mm
        │   │       └── tk_metal_helpers.h
        │   ├── navigation/
        │   │   ├── tk_path_planner.c
        │   │   ├── tk_free_space_detector.c
        │   │   ├── tk_obstacle_avoider.c
        │   │   ├── tk_path_planner.h
        │   │   ├── tk_free_space_detector.h
        │   │   └── tk_obstacle_avoider.h
        │   ├── interaction/
        │   │   ├── tk_voice_commands.c
        │   │   ├── tk_feedback_manager.c
        │   │   ├── tk_voice_commands.h
        │   │   ├── tk_feedback_manager.h
        │   │   └── src/
        │   │       ├── lib.rs
        │   │       ├── command_parser.rs
        │   │       └── feedback_logic.rs
        │   ├── utils/
        │   │   ├── tk_logging.c
        │   │   ├── tk_error_handling.c
        │   │   ├── tk_logging.h
        │   │   ├── tk_error_handling.h
        │   │   └── src/
        │   │       ├── lib.rs
        │   │       ├── error_utils.rs
        │   │       └── debug_helpers.rs
        │   ├── core_build/
        │   │   ├── CMakeLists.txt
        │   │   ├── tk_build_system.c
        │   │   └── tk_build_system.h
            └── ffi/
                ├── c_api/
                │   ├── tk_ffi_api.c
                │   └── tk_ffi_api.h
                │   ├── tk_ffi_cpp_api.cpp
                │   └── tk_ffi_cpp_api.hpp
                └── src/

                        └── src/
                        ├── lib.rs
                        ├── ffi_bridge.rs
                        └── utils.rs
                ├── Cargo.toml

