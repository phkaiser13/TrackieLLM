/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_cortex_main.c
*
* This file contains the implementation of the TrackieLLM Cortex - the central
* "brain" of the system that orchestrates all sensory inputs, reasoning processes,
* and decision-making capabilities.
*
* The Cortex implements a sophisticated real-time processing loop that integrates:
*   - Multi-modal sensory input (vision, audio, IMU)
*   - Contextual reasoning and memory management
*   - Large Language Model inference and response processing
*   - Decision execution and action coordination
*   - Thread-safe data exchange with external components
*
* This is the executive control system that transforms raw sensory data into
* intelligent, contextually-aware assistance for users with visual impairments.
*
* SPDX-License-Identifier:
*/

#include "cortex/tk_cortex_main.h"
#include "cortex/tk_contextual_reasoner.h"
#include "cortex/tk_decision_engine.h"
#include "vision/tk_vision_pipeline.h"
#include "audio/tk_audio_pipeline.h"
#include "navigation/tk_path_planner.h"
#include "navigation/tk_free_space_detector.h"
#include "navigation/tk_obstacle_avoider.h"
#include "ai_models/tk_model_loader.h"
#include "ai_models/tk_model_runner.h"
#include "utils/tk_logging.h"
#include "utils/tk_error_handling.h"
#include "async_tasks/tk_task_scheduler.h"
#include "sensors/tk_sensors_fusion.h"

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>

//------------------------------------------------------------------------------
// Internal Data Structures
//------------------------------------------------------------------------------

/**
 * @struct tk_cortex_s
 * @brief The main Cortex instance containing all subsystems and state.
 */
struct tk_cortex_s {
    // Configuration
    tk_cortex_config_t config;
    tk_cortex_callbacks_t callbacks;
    
    // Current state
    tk_system_state_e current_state;
    pthread_mutex_t state_mutex;
    
    // Main processing loop control
    volatile bool should_stop;
    pthread_t main_loop_thread;
    bool loop_thread_active;
    
    // Subsystem components
    tk_vision_pipeline_t* vision_pipeline;
    tk_audio_pipeline_t* audio_pipeline;
    tk_navigation_engine_t* navigation_engine;
    tk_free_space_detector_t* free_space_detector;
    tk_obstacle_tracker_t* obstacle_tracker;
    tk_contextual_reasoner_t* contextual_reasoner;
    tk_decision_engine_t* decision_engine;
    tk_sensors_fusion_t* sensor_fusion;
    tk_task_scheduler_t* task_scheduler;
    
    // AI Models
    tk_model_runner_t* llm_runner;
    
    // Input data buffers (thread-safe ring buffers)
    struct {
        tk_video_frame_t* frames;
        size_t frame_count;
        size_t frame_capacity;
        size_t write_index;
        size_t read_index;
        pthread_mutex_t mutex;
        bool has_new_frame;
    } video_buffer;
    
    struct {
        int16_t* samples;
        size_t sample_count;
        size_t sample_capacity;
        size_t write_index;
        size_t read_index;
        pthread_mutex_t mutex;
        bool has_new_audio;
    } audio_buffer;
    
    // Timing and performance
    struct {
        uint64_t loop_iteration_count;
        uint64_t last_loop_time_ns;
        float average_loop_time_ms;
        uint64_t last_vision_process_time_ns;
        uint64_t last_llm_inference_time_ns;
    } performance_stats;
    
    // Emergency state
    bool emergency_stop_requested;
    pthread_mutex_t emergency_mutex;
};

//------------------------------------------------------------------------------
// Internal Function Declarations
//------------------------------------------------------------------------------

static tk_error_code_t cortex_initialize_subsystems(tk_cortex_t* cortex);
static void cortex_cleanup_subsystems(tk_cortex_t* cortex);
static void* cortex_main_loop_thread(void* arg);
static tk_error_code_t cortex_process_single_iteration(tk_cortex_t* cortex);
static tk_error_code_t cortex_process_vision_input(tk_cortex_t* cortex);
static tk_error_code_t cortex_process_navigation_analysis(tk_cortex_t* cortex);
static tk_error_code_t cortex_run_llm_inference(tk_cortex_t* cortex);
static void cortex_update_performance_stats(tk_cortex_t* cortex, uint64_t iteration_time_ns);
static uint64_t get_current_time_ns(void);
static void cortex_change_state(tk_cortex_t* cortex, tk_system_state_e new_state);

// Audio pipeline callbacks
static void on_vad_event(tk_vad_event_e event, void* user_data);
static void on_transcription(const tk_transcription_t* result, void* user_data);
static void on_tts_audio_ready(const int16_t* audio_data, size_t frame_count, uint32_t sample_rate, void* user_data);

// Decision engine callbacks
static void on_action_completed(const tk_action_t* action, void* user_data);
static void on_response_ready(const char* response_text, tk_response_priority_e priority, void* user_data);

//------------------------------------------------------------------------------
// Public API Implementation
//------------------------------------------------------------------------------

tk_error_code_t tk_cortex_create(tk_cortex_t** out_cortex, const tk_cortex_config_t* config, tk_cortex_callbacks_t callbacks) {
    if (!out_cortex || !config) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    tk_log_info("Creating TrackieLLM Cortex instance");
    
    tk_cortex_t* cortex = calloc(1, sizeof(tk_cortex_t));
    if (!cortex) {
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Initialize basic state
    cortex->config = *config;
    cortex->callbacks = callbacks;
    cortex->current_state = TK_STATE_INITIALIZING;
    cortex->should_stop = false;
    cortex->loop_thread_active = false;
    cortex->emergency_stop_requested = false;
    
    // Initialize mutexes
    if (pthread_mutex_init(&cortex->state_mutex, NULL) != 0 ||
        pthread_mutex_init(&cortex->video_buffer.mutex, NULL) != 0 ||
        pthread_mutex_init(&cortex->audio_buffer.mutex, NULL) != 0 ||
        pthread_mutex_init(&cortex->emergency_mutex, NULL) != 0) {
        
        tk_log_error("Failed to initialize mutexes");
        free(cortex);
        return TK_ERROR_SYSTEM_ERROR;
    }
    
    // Initialize input buffers
    const size_t video_buffer_size = 4; // Keep last 4 frames
    const size_t audio_buffer_size = 48000 * 2; // 2 seconds at 48kHz
    
    cortex->video_buffer.frames = calloc(video_buffer_size, sizeof(tk_video_frame_t));
    cortex->video_buffer.frame_capacity = video_buffer_size;
    cortex->audio_buffer.samples = calloc(audio_buffer_size, sizeof(int16_t));
    cortex->audio_buffer.sample_capacity = audio_buffer_size;
    
    if (!cortex->video_buffer.frames || !cortex->audio_buffer.samples) {
        tk_log_error("Failed to allocate input buffers");
        tk_cortex_destroy(&cortex);
        return TK_ERROR_OUT_OF_MEMORY;
    }
    
    // Initialize all subsystems
    tk_error_code_t result = cortex_initialize_subsystems(cortex);
    if (result != TK_SUCCESS) {
        tk_log_error("Failed to initialize Cortex subsystems: %d", result);
        tk_cortex_destroy(&cortex);
        return result;
    }
    
    // Notify state change to initialized
    cortex_change_state(cortex, TK_STATE_IDLE);
    
    *out_cortex = cortex;
    tk_log_info("Cortex created successfully");
    return TK_SUCCESS;
}

void tk_cortex_destroy(tk_cortex_t** cortex) {
    if (!cortex || !*cortex) {
        return;
    }
    
    tk_log_info("Destroying Cortex instance");
    
    tk_cortex_t* c = *cortex;
    
    // Stop the main loop if it's running
    if (c->loop_thread_active) {
        c->should_stop = true;
        pthread_join(c->main_loop_thread, NULL);
    }
    
    // Cleanup subsystems
    cortex_cleanup_subsystems(c);
    
    // Free input buffers
    free(c->video_buffer.frames);
    free(c->audio_buffer.samples);
    
    // Destroy mutexes
    pthread_mutex_destroy(&c->state_mutex);
    pthread_mutex_destroy(&c->video_buffer.mutex);
    pthread_mutex_destroy(&c->audio_buffer.mutex);
    pthread_mutex_destroy(&c->emergency_mutex);
    
    free(c);
    *cortex = NULL;
    
    tk_log_info("Cortex destroyed");
}

tk_error_code_t tk_cortex_run(tk_cortex_t* cortex) {
    if (!cortex) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    if (cortex->current_state != TK_STATE_IDLE) {
        return TK_ERROR_INVALID_STATE;
    }
    
    tk_log_info("Starting Cortex main processing loop");
    
    // Create the main processing thread
    int result = pthread_create(&cortex->main_loop_thread, NULL, cortex_main_loop_thread, cortex);
    if (result != 0) {
        tk_log_error("Failed to create main loop thread: %s", strerror(result));
        return TK_ERROR_SYSTEM_ERROR;
    }
    
    cortex->loop_thread_active = true;
    
    // Wait for the thread to complete
    pthread_join(cortex->main_loop_thread, NULL);
    cortex->loop_thread_active = false;
    
    tk_log_info("Cortex main loop terminated");
    return TK_SUCCESS;
}

tk_error_code_t tk_cortex_stop(tk_cortex_t* cortex) {
    if (!cortex) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    tk_log_info("Requesting Cortex shutdown");
    cortex->should_stop = true;
    
    return TK_SUCCESS;
}

tk_error_code_t tk_cortex_inject_audio_frame(tk_cortex_t* cortex, const int16_t* audio_data, size_t frame_count) {
    if (!cortex || !audio_data || frame_count == 0) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Forward directly to audio pipeline for real-time processing
    return tk_audio_pipeline_process_chunk(cortex->audio_pipeline, audio_data, frame_count);
}

tk_error_code_t tk_cortex_inject_video_frame(tk_cortex_t* cortex, const tk_video_frame_t* frame) {
    if (!cortex || !frame || !frame->data) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // Thread-safe insertion into video buffer
    pthread_mutex_lock(&cortex->video_buffer.mutex);
    
    // Copy frame data to buffer (simple ring buffer implementation)
    size_t write_idx = cortex->video_buffer.write_index;
    tk_video_frame_t* buffer_frame = &cortex->video_buffer.frames[write_idx];
    
    // Simple frame copy (in production, this would need proper pixel data management)
    *buffer_frame = *frame;
    
    // Update write index
    cortex->video_buffer.write_index = (write_idx + 1) % cortex->video_buffer.frame_capacity;
    cortex->video_buffer.has_new_frame = true;
    
    // If buffer is full, advance read index
    if (cortex->video_buffer.frame_count < cortex->video_buffer.frame_capacity) {
        cortex->video_buffer.frame_count++;
    } else {
        cortex->video_buffer.read_index = (cortex->video_buffer.read_index + 1) % cortex->video_buffer.frame_capacity;
    }
    
    pthread_mutex_unlock(&cortex->video_buffer.mutex);
    
    return TK_SUCCESS;
}

tk_error_code_t tk_cortex_get_state(const tk_cortex_t* cortex, tk_system_state_e* out_state) {
    if (!cortex || !out_state) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    pthread_mutex_lock((pthread_mutex_t*)&cortex->state_mutex);
    *out_state = cortex->current_state;
    pthread_mutex_unlock((pthread_mutex_t*)&cortex->state_mutex);
    
    return TK_SUCCESS;
}

//------------------------------------------------------------------------------
// Internal Implementation
//------------------------------------------------------------------------------

static tk_error_code_t cortex_initialize_subsystems(tk_cortex_t* cortex) {
    tk_error_code_t result;
    
    tk_log_info("Initializing Cortex subsystems");
    
    // Initialize sensor fusion first
    tk_sensors_fusion_config_t sensor_config = {
        .imu_sample_rate_hz = 100.0f,
        .fusion_algorithm = TK_FUSION_ALGORITHM_MADGWICK,
        .gyro_bias_estimation = true
    };
    
    result = tk_sensors_fusion_create(&cortex->sensor_fusion, &sensor_config);
    if (result != TK_SUCCESS) {
        tk_log_error("Failed to create sensor fusion: %d", result);
        return result;
    }
    
    // Initialize vision pipeline
    tk_vision_pipeline_config_t vision_config = {
        .backend = (cortex->config.gpu_device_id >= 0) ? TK_VISION_BACKEND_CUDA : TK_VISION_BACKEND_CPU,
        .gpu_device_id = cortex->config.gpu_device_id,
        .object_detection_model_path = NULL, // Would be set from config paths
        .depth_estimation_model_path = NULL,
        .tesseract_data_path = NULL,
        .object_confidence_threshold = 0.5f,
        .max_detected_objects = 20
    };
    
    result = tk_vision_pipeline_create(&cortex->vision_pipeline, &vision_config);
    if (result != TK_SUCCESS) {
        tk_log_error("Failed to create vision pipeline: %d", result);
        return result;
    }
    
    // Initialize navigation engine
    tk_navigation_config_t nav_config = {
        .camera_height_m = 1.5f,
        .default_camera_pitch_deg = -10.0f,
        .step_height_threshold_m = 0.15f,
        .user_clearance_width_m = 0.8f,
        .max_analysis_distance_m = 5.0f
    };
    
    result = tk_navigation_engine_create(&cortex->navigation_engine, &nav_config);
    if (result != TK_SUCCESS) {
        tk_log_error("Failed to create navigation engine: %d", result);
        return result;
    }
    
    // Initialize free space detector
    tk_free_space_config_t space_config = {
        .num_angular_sectors = 7,
        .analysis_fov_deg = 90.0f,
        .user_clearance_width_m = nav_config.user_clearance_width_m
    };
    
    result = tk_free_space_detector_create(&cortex->free_space_detector, &space_config);
    if (result != TK_SUCCESS) {
        tk_log_error("Failed to create free space detector: %d", result);
        return result;
    }
    
    // Initialize obstacle tracker
    tk_obstacle_tracker_config_t obstacle_config = {
        .min_obstacle_area_m2 = 0.1f,
        .max_tracked_obstacles = 10,
        .max_frames_unseen = 5,
        .max_match_distance_m = 1.0f
    };
    
    result = tk_obstacle_tracker_create(&cortex->obstacle_tracker, &obstacle_config);
    if (result != TK_SUCCESS) {
        tk_log_error("Failed to create obstacle tracker: %d", result);
        return result;
    }
    
    // Initialize contextual reasoner
    tk_context_config_t context_config = {
        .max_context_history_items = 100,
        .max_conversation_history_turns = 20,
        .context_relevance_threshold = 0.3f,
        .memory_decay_rate = 0.95f,
        .context_update_interval_ms = 100
    };
    
    result = tk_contextual_reasoner_create(&cortex->contextual_reasoner, &context_config);
    if (result != TK_SUCCESS) {
        tk_log_error("Failed to create contextual reasoner: %d", result);
        return result;
    }
    
    // Initialize decision engine
    tk_decision_config_t decision_config = {
        .action_confidence_threshold = 0.7f,
        .max_concurrent_actions = 3,
        .action_timeout_ms = 5000,
        .enable_safety_constraints = true,
        .response_priority_threshold = 0.5f
    };
    
    tk_decision_callbacks_t decision_callbacks = {
        .on_action_completed = on_action_completed,
        .on_response_ready = on_response_ready,
        .user_data = cortex
    };
    
    result = tk_decision_engine_create(&cortex->decision_engine, &decision_config, decision_callbacks);
    if (result != TK_SUCCESS) {
        tk_log_error("Failed to create decision engine: %d", result);
        return result;
    }
    
    // Initialize audio pipeline
    tk_audio_pipeline_config_t audio_config = {
        .input_audio_params = { .sample_rate = 16000, .channels = 1 },
        .asr_model_path = NULL, // Would be set from config paths
        .vad_model_path = NULL,
        .tts_model_dir_path = NULL,
        .user_language = cortex->config.user_language,
        .user_data = cortex,
        .vad_silence_threshold_ms = 500.0f,
        .vad_speech_probability_threshold = 0.8f
    };
    
    tk_audio_callbacks_t audio_callbacks = {
        .on_vad_event = on_vad_event,
        .on_transcription = on_transcription,
        .on_tts_audio_ready = on_tts_audio_ready
    };
    
    result = tk_audio_pipeline_create(&cortex->audio_pipeline, &audio_config, audio_callbacks);
    if (result != TK_SUCCESS) {
        tk_log_error("Failed to create audio pipeline: %d", result);
        return result;
    }
    
    // Initialize task scheduler
    tk_task_scheduler_config_t task_config = {
        .max_concurrent_tasks = 4,
        .worker_thread_count = 2
    };
    
    result = tk_task_scheduler_create(&cortex->task_scheduler, &task_config);
    if (result != TK_SUCCESS) {
        tk_log_error("Failed to create task scheduler: %d", result);
        return result;
    }
    
    // Initialize LLM runner
    tk_model_runner_config_t llm_config = {
        .model_path = cortex->config.model_paths.llm_model,
        .max_context_length = 4096,
        .max_generation_tokens = 512,
        .temperature = 0.7f,
        .use_gpu = (cortex->config.gpu_device_id >= 0)
    };
    
    result = tk_model_runner_create(&cortex->llm_runner, &llm_config);
    if (result != TK_SUCCESS) {
        tk_log_error("Failed to create LLM runner: %d", result);
        return result;
    }
    
    tk_log_info("All Cortex subsystems initialized successfully");
    return TK_SUCCESS;
}

static void cortex_cleanup_subsystems(tk_cortex_t* cortex) {
    if (!cortex) {
        return;
    }
    
    tk_log_info("Cleaning up Cortex subsystems");
    
    tk_model_runner_destroy(&cortex->llm_runner);
    tk_task_scheduler_destroy(&cortex->task_scheduler);
    tk_audio_pipeline_destroy(&cortex->audio_pipeline);
    tk_decision_engine_destroy(&cortex->decision_engine);
    tk_contextual_reasoner_destroy(&cortex->contextual_reasoner);
    tk_obstacle_tracker_destroy(&cortex->obstacle_tracker);
    tk_free_space_detector_destroy(&cortex->free_space_detector);
    tk_navigation_engine_destroy(&cortex->navigation_engine);
    tk_vision_pipeline_destroy(&cortex->vision_pipeline);
    tk_sensors_fusion_destroy(&cortex->sensor_fusion);
    
    tk_log_info("Cortex subsystems cleanup complete");
}

static void* cortex_main_loop_thread(void* arg) {
    tk_cortex_t* cortex = (tk_cortex_t*)arg;
    
    tk_log_info("Cortex main processing loop started");
    
    // Calculate sleep time based on target frequency
    const uint64_t target_interval_ns = (uint64_t)(1000000000.0f / cortex->config.main_loop_frequency_hz);
    
    while (!cortex->should_stop) {
        uint64_t iteration_start = get_current_time_ns();
        
        // Check for emergency stop
        pthread_mutex_lock(&cortex->emergency_mutex);
        if (cortex->emergency_stop_requested) {
            tk_log_warning("Emergency stop requested - halting main loop");
            pthread_mutex_unlock(&cortex->emergency_mutex);
            break;
        }
        pthread_mutex_unlock(&cortex->emergency_mutex);
        
        // Process one iteration
        tk_error_code_t result = cortex_process_single_iteration(cortex);
        if (result != TK_SUCCESS) {
            tk_log_error("Error in main loop iteration: %d", result);
            
            // On critical errors, change to error state
            if (result == TK_ERROR_CRITICAL_FAILURE) {
                cortex_change_state(cortex, TK_STATE_FATAL_ERROR);
                break;
            }
        }
        
        uint64_t iteration_end = get_current_time_ns();
        uint64_t iteration_time = iteration_end - iteration_start;
        
        // Update performance statistics
        cortex_update_performance_stats(cortex, iteration_time);
        
        // Sleep to maintain target frequency
        if (iteration_time < target_interval_ns) {
            uint64_t sleep_time_ns = target_interval_ns - iteration_time;
            struct timespec sleep_spec = {
                .tv_sec = sleep_time_ns / 1000000000,
                .tv_nsec = sleep_time_ns % 1000000000
            };
            nanosleep(&sleep_spec, NULL);
        }
    }
    
    cortex_change_state(cortex, TK_STATE_SHUTDOWN);
    tk_log_info("Cortex main processing loop terminated");
    
    return NULL;
}

static tk_error_code_t cortex_process_single_iteration(tk_cortex_t* cortex) {
    uint64_t current_time = get_current_time_ns();
    tk_error_code_t result;
    
    // Process pending decision engine actions
    result = tk_decision_engine_process_actions(cortex->decision_engine, current_time);
    if (result != TK_SUCCESS) {
        tk_log_warning("Decision engine processing failed: %d", result);
    }
    
    // Check for new video data and process if available
    if (cortex->video_buffer.has_new_frame) {
        result = cortex_process_vision_input(cortex);
        if (result != TK_SUCCESS) {
            tk_log_warning("Vision processing failed: %d", result);
        }
    }
    
    // Process navigation analysis if vision data was updated
    if (cortex->video_buffer.has_new_frame) {
        result = cortex_process_navigation_analysis(cortex);
        if (result != TK_SUCCESS) {
            tk_log_warning("Navigation analysis failed: %d", result);
        }
    }
    
    // Update contextual reasoner
    result = tk_contextual_reasoner_process_context(cortex->contextual_reasoner, current_time);
    if (result != TK_SUCCESS) {
        tk_log_warning("Context processing failed: %d", result);
    }
    
    // Run LLM inference if context has significantly changed or on schedule
    static uint64_t last_llm_run = 0;
    const uint64_t llm_interval_ns = 2000000000; // 2 seconds
    
    if ((current_time - last_llm_run) > llm_interval_ns) {
        result = cortex_run_llm_inference(cortex);
        if (result == TK_SUCCESS) {
            last_llm_run = current_time;
        }
    }
    
    cortex->performance_stats.loop_iteration_count++;
    return TK_SUCCESS;
}

static tk_error_code_t cortex_process_vision_input(tk_cortex_t* cortex) {
    pthread_mutex_lock(&cortex->video_buffer.mutex);
    
    if (!cortex->video_buffer.has_new_frame || cortex->video_buffer.frame_count == 0) {
        pthread_mutex_unlock(&cortex->video_buffer.mutex);
        return TK_SUCCESS;
    }
    
    // Get the latest frame
    size_t read_idx = (cortex->video_buffer.write_index - 1 + cortex->video