/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_cortex_main.c
 *
 * This file contains the implementation of the TrackieLLM Cortex - the central
 * "brain" of the system that orchestrates all sensory inputs, reasoning processes,
 * and decision-making capabilities.
 *
 * The Cortex implements a sophisticated event-driven processing system that integrates:
 *   - Multi-modal sensory input (vision, audio, IMU)
 *   - Contextual reasoning and memory management
 *   - Large Language Model inference and response processing
 *   - Decision execution and action coordination
 *   - Thread-safe data exchange with external components
 *
 * This is the executive control system that transforms raw sensory data into
 * intelligent, contextually-aware assistance for users with visual impairments.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "tk_cortex_main.h"
#include "tk_contextual_reasoner.h"
#include "tk_decision_engine.h"
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
#include "ffi/c_api/tk_ffi_api.h" // For TkModuleType, TkStatus, etc.

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <stdatomic.h>

// Forward declarations for payload management
static tk_error_code_t event_payload_copy(tk_cortex_event_type_e type, void** dest_payload, const void* src_payload);
static void event_payload_free(tk_cortex_event_type_e type, void* payload);
static tk_error_code_t deep_copy_vision_result(tk_vision_result_t** dest, const tk_vision_result_t* src);
static tk_error_code_t deep_copy_transcription(tk_transcription_t** dest, const tk_transcription_t* src);


//------------------------------------------------------------------------------
// Event System Definitions
//------------------------------------------------------------------------------

/**
 * @enum tk_cortex_event_type_e
 * @brief Types of events that can be processed by the Cortex event loop.
 */
typedef enum {
    CORTEX_EVENT_NEW_VIDEO_FRAME,
    CORTEX_EVENT_USER_SPEECH_FINAL,
    CORTEX_EVENT_VAD_SPEECH_STARTED,
    CORTEX_EVENT_SIGNIFICANT_VISION_CHANGE,
    CORTEX_EVENT_SYSTEM_TIMER,
    CORTEX_EVENT_SHUTDOWN
} tk_cortex_event_type_e;

/**
 * @struct tk_cortex_event_s
 * @brief A single event in the Cortex event queue.
 */
typedef struct {
    tk_cortex_event_type_e type;
    void* payload; // Optional data associated with the event
    uint64_t timestamp_ns;
} tk_cortex_event_t;

/**
 * @struct event_queue_s
 * @brief A thread-safe circular buffer for event queue.
 */
typedef struct {
    tk_cortex_event_t* events;
    size_t capacity;
    atomic_size_t read_index;
    atomic_size_t write_index;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
} event_queue_t;

/**
 * @brief Initializes an event queue.
 * 
 * @param queue Pointer to the event queue to initialize.
 * @param capacity Maximum number of events the queue can hold.
 * @return TK_SUCCESS on success, error code otherwise.
 */
static tk_error_code_t event_queue_init(event_queue_t* queue, size_t capacity) {
    if (!queue || capacity == 0) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    queue->events = calloc(capacity, sizeof(tk_cortex_event_t));
    if (!queue->events) {
        return TK_ERROR_OUT_OF_MEMORY;
    }

    queue->capacity = capacity;
    atomic_init(&queue->read_index, 0);
    atomic_init(&queue->write_index, 0);

    if (pthread_mutex_init(&queue->mutex, NULL) != 0) {
        free(queue->events);
        return TK_ERROR_SYSTEM_ERROR;
    }

    if (pthread_cond_init(&queue->not_empty, NULL) != 0) {
        pthread_mutex_destroy(&queue->mutex);
        free(queue->events);
        return TK_ERROR_SYSTEM_ERROR;
    }

    if (pthread_cond_init(&queue->not_full, NULL) != 0) {
        pthread_cond_destroy(&queue->not_empty);
        pthread_mutex_destroy(&queue->mutex);
        free(queue->events);
        return TK_ERROR_SYSTEM_ERROR;
    }

    return TK_SUCCESS;
}

/**
 * @brief Destroys an event queue.
 * 
 * @param queue Pointer to the event queue to destroy.
 */
static void event_queue_destroy(event_queue_t* queue) {
    if (!queue) {
        return;
    }

    // Free any remaining event payloads
    size_t read_idx = atomic_load(&queue->read_index);
    size_t write_idx = atomic_load(&queue->write_index);
    
    if (queue->events) {
        // Iterate through all remaining events and free payloads
        while (read_idx != write_idx) {
            event_payload_free(queue->events[read_idx].type, queue->events[read_idx].payload);
            read_idx = (read_idx + 1) % queue->capacity;
        }
        free(queue->events);
    }

    pthread_cond_destroy(&queue->not_full);
    pthread_cond_destroy(&queue->not_empty);
    pthread_mutex_destroy(&queue->mutex);
}

/**
 * @brief Enqueues an event.
 * 
 * This function will block if the queue is full.
 * 
 * @param queue Pointer to the event queue.
 * @param event Pointer to the event to enqueue.
 * @return TK_SUCCESS on success, error code otherwise.
 */
static tk_error_code_t event_queue_enqueue(event_queue_t* queue, const tk_cortex_event_t* event) {
    if (!queue || !event) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    pthread_mutex_lock(&queue->mutex);

    // Wait while the queue is full
    while (((atomic_load(&queue->write_index) + 1) % queue->capacity) == atomic_load(&queue->read_index)) {
        pthread_cond_wait(&queue->not_full, &queue->mutex);
    }

    // Copy the event into the queue
    size_t write_idx = atomic_load(&queue->write_index);
    queue->events[write_idx].type = event->type;
    queue->events[write_idx].timestamp_ns = event->timestamp_ns;
    queue->events[write_idx].payload = NULL; // Start with NULL payload

    // If payload is provided, make a deep copy
    if (event->payload) {
        tk_error_code_t copy_result = event_payload_copy(
            event->type,
            &queue->events[write_idx].payload,
            event->payload
        );

        if (copy_result != TK_SUCCESS) {
            pthread_mutex_unlock(&queue->mutex);
            return copy_result;
        }
    }

    atomic_store(&queue->write_index, (write_idx + 1) % queue->capacity);

    pthread_cond_signal(&queue->not_empty);
    pthread_mutex_unlock(&queue->mutex);

    return TK_SUCCESS;
}

/**
 * @brief Dequeues an event.
 * 
 * This function will block if the queue is empty.
 * 
 * @param queue Pointer to the event queue.
 * @param out_event Pointer to store the dequeued event.
 * @return TK_SUCCESS on success, error code otherwise.
 */
static tk_error_code_t event_queue_dequeue(event_queue_t* queue, tk_cortex_event_t* out_event) {
    if (!queue || !out_event) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    pthread_mutex_lock(&queue->mutex);

    // Wait while the queue is empty
    while (atomic_load(&queue->read_index) == atomic_load(&queue->write_index)) {
        pthread_cond_wait(&queue->not_empty, &queue->mutex);
    }

    // Copy the event from the queue
    size_t read_idx = atomic_load(&queue->read_index);
    *out_event = queue->events[read_idx];
    // The payload pointer is transferred to the caller
    queue->events[read_idx].payload = NULL; // Clear our reference

    atomic_store(&queue->read_index, (read_idx + 1) % queue->capacity);

    pthread_cond_signal(&queue->not_full);
    pthread_mutex_unlock(&queue->mutex);

    return TK_SUCCESS;
}

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
    
    // Event system
    event_queue_t event_queue;
    pthread_cond_t event_cond;
    pthread_mutex_t event_mutex;
    
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
    
    // Cache for latest vision results
    tk_vision_result_t* latest_vision_result;
    pthread_mutex_t vision_result_mutex;
};

//------------------------------------------------------------------------------
// C-Side Module Executor
//------------------------------------------------------------------------------

// Define the function pointer type that matches the Rust `ModuleExecutor`
typedef TkStatus (*c_module_executor_fn)(
    void* context, // Note: Using void* here as TkContext is opaque in C
    TkModuleType module_type,
    const char* command_name,
    void* input
);

// Forward-declare the C-side executor
static TkStatus c_module_executor(
    void* context,
    TkModuleType module_type,
    const char* command_name,
    void* input
);

// Forward-declare the registration function from the Rust FFI bridge
extern TkStatus tk_module_register(TkModuleType module_type, c_module_executor_fn executor);

/**
 * @brief Executor function for all C-side modules.
 *
 * This function is registered with the FFI bridge for each C-based module.
 * It acts as a dispatcher, forwarding commands to the appropriate C subsystem
 * based on the module type.
 */
static TkStatus c_module_executor(
    void* context,
    TkModuleType module_type,
    const char* command_name,
    void* input)
{
    tk_cortex_t* cortex = (tk_cortex_t*)context;
    if (!cortex) {
        return TK_STATUS_ERROR_NULL_POINTER;
    }

    tk_log_debug("C executor called for module %d with command '%s'", module_type, command_name);

    // This is where you would dispatch to the actual C implementation
    switch (module_type) {
        case TK_MODULE_VISION:
            // e.g., handle_vision_command(cortex->vision_pipeline, command_name, input);
            break;
        case TK_MODULE_AUDIO:
            // e.g., handle_audio_command(cortex->audio_pipeline, command_name, input);
            break;
        case TK_MODULE_NAVIGATION:
             // e.g., handle_nav_command(cortex->navigation_engine, command_name, input);
            break;
        default:
            tk_log_error("C executor called for unhandled module type: %d", module_type);
            return TK_STATUS_ERROR_UNSUPPORTED_FEATURE;
    }

    // For this example, we just return success without doing anything.
    // A real implementation would have detailed command handling logic.
    return TK_STATUS_OK;
}


//------------------------------------------------------------------------------
// Internal Function Declarations
//------------------------------------------------------------------------------

static tk_error_code_t cortex_initialize_subsystems(tk_cortex_t* cortex);
static void cortex_cleanup_subsystems(tk_cortex_t* cortex);
static void* cortex_main_loop_thread(void* arg);
static tk_error_code_t cortex_handle_event(tk_cortex_t* cortex, const tk_cortex_event_t* event);
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
    cortex->latest_vision_result = NULL;
    
    // Initialize mutexes
    if (pthread_mutex_init(&cortex->state_mutex, NULL) != 0 ||
        pthread_mutex_init(&cortex->video_buffer.mutex, NULL) != 0 ||
        pthread_mutex_init(&cortex->audio_buffer.mutex, NULL) != 0 ||
        pthread_mutex_init(&cortex->emergency_mutex, NULL) != 0 ||
        pthread_mutex_init(&cortex->vision_result_mutex, NULL) != 0 ||
        pthread_mutex_init(&cortex->event_mutex, NULL) != 0) {
        
        tk_log_error("Failed to initialize mutexes");
        free(cortex);
        return TK_ERROR_SYSTEM_ERROR;
    }
    
    // Initialize condition variables
    if (pthread_cond_init(&cortex->event_cond, NULL) != 0) {
        tk_log_error("Failed to initialize condition variables");
        pthread_mutex_destroy(&cortex->state_mutex);
        pthread_mutex_destroy(&cortex->video_buffer.mutex);
        pthread_mutex_destroy(&cortex->audio_buffer.mutex);
        pthread_mutex_destroy(&cortex->emergency_mutex);
        pthread_mutex_destroy(&cortex->vision_result_mutex);
        pthread_mutex_destroy(&cortex->event_mutex);
        free(cortex);
        return TK_ERROR_SYSTEM_ERROR;
    }
    
    // Initialize event queue
    tk_error_code_t result = event_queue_init(&cortex->event_queue, 128); // 128 event capacity
    if (result != TK_SUCCESS) {
        tk_log_error("Failed to initialize event queue: %d", result);
        pthread_cond_destroy(&cortex->event_cond);
        pthread_mutex_destroy(&cortex->state_mutex);
        pthread_mutex_destroy(&cortex->video_buffer.mutex);
        pthread_mutex_destroy(&cortex->audio_buffer.mutex);
        pthread_mutex_destroy(&cortex->emergency_mutex);
        pthread_mutex_destroy(&cortex->vision_result_mutex);
        pthread_mutex_destroy(&cortex->event_mutex);
        free(cortex);
        return result;
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
    result = cortex_initialize_subsystems(cortex);
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
        // Signal the event condition to wake up the loop
        pthread_cond_signal(&c->event_cond);
        pthread_join(c->main_loop_thread, NULL);
    }
    
    // Cleanup subsystems
    cortex_cleanup_subsystems(c);
    
    // Destroy event queue
    event_queue_destroy(&c->event_queue);
    
    // Free input buffers
    free(c->video_buffer.frames);
    free(c->audio_buffer.samples);
    
    // Destroy condition variables and mutexes
    pthread_cond_destroy(&c->event_cond);
    pthread_mutex_destroy(&c->state_mutex);
    pthread_mutex_destroy(&c->video_buffer.mutex);
    pthread_mutex_destroy(&c->audio_buffer.mutex);
    pthread_mutex_destroy(&c->emergency_mutex);
    pthread_mutex_destroy(&c->vision_result_mutex);
    pthread_mutex_destroy(&c->event_mutex);
    
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
    
    // Enqueue a shutdown event
    tk_cortex_event_t shutdown_event = {
        .type = CORTEX_EVENT_SHUTDOWN,
        .payload = NULL,
        .timestamp_ns = get_current_time_ns()
    };
    event_queue_enqueue(&cortex->event_queue, &shutdown_event);
    
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
    
    // Enqueue a new video frame event
    tk_cortex_event_t video_event = {
        .type = CORTEX_EVENT_NEW_VIDEO_FRAME,
        .payload = NULL, // No payload needed, we'll read from the buffer
        .timestamp_ns = get_current_time_ns()
    };
    event_queue_enqueue(&cortex->event_queue, &video_event);
    
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
    tk_module_register(TK_MODULE_SENSORS, (c_module_executor_fn)c_module_executor);
    
    // Initialize vision pipeline with model paths from config
    tk_vision_pipeline_config_t vision_config = {
        .backend = (cortex->config.gpu_device_id >= 0) ? TK_VISION_BACKEND_CUDA : TK_VISION_BACKEND_CPU,
        .gpu_device_id = cortex->config.gpu_device_id,
        .object_detection_model_path = cortex->config.model_paths.object_detection_model,
        .depth_estimation_model_path = cortex->config.model_paths.depth_estimation_model,
        .tesseract_data_path = cortex->config.model_paths.ocr_data_path,
        .object_confidence_threshold = 0.5f,
        .max_detected_objects = 20
    };
    
    result = tk_vision_pipeline_create(&cortex->vision_pipeline, &vision_config);
    if (result != TK_SUCCESS) {
        tk_log_error("Failed to create vision pipeline: %d", result);
        return result;
    }
    tk_module_register(TK_MODULE_VISION, (c_module_executor_fn)c_module_executor);
    
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
    tk_module_register(TK_MODULE_NAVIGATION, (c_module_executor_fn)c_module_executor);
    
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
    
    // Initialize audio pipeline with model paths from config
    tk_audio_pipeline_config_t audio_config = {
        .input_audio_params = { .sample_rate = 16000, .channels = 1 },
        .asr_model_path = cortex->config.model_paths.asr_model,
        .vad_model_path = cortex->config.model_paths.vad_model,
        .tts_model_dir_path = cortex->config.model_paths.tts_model_dir,
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
    tk_module_register(TK_MODULE_AUDIO, (c_module_executor_fn)c_module_executor);
    
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
    
    // Initialize LLM runner with model path from config
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
    
    tk_module_register(TK_MODULE_CORTEX, (c_module_executor_fn)c_module_executor);

    tk_log_info("All Cortex subsystems initialized successfully");
    return TK_SUCCESS;
}

static void cortex_cleanup_subsystems(tk_cortex_t* cortex) {
    if (!cortex) {
        return;
    }
    
    tk_log_info("Cleaning up Cortex subsystems");
    
    // Clean up cached vision result
    if (cortex->latest_vision_result) {
        tk_vision_result_destroy(&cortex->latest_vision_result);
    }
    
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
    
    while (!cortex->should_stop) {
        // Check for emergency stop
        pthread_mutex_lock(&cortex->emergency_mutex);
        if (cortex->emergency_stop_requested) {
            tk_log_warning("Emergency stop requested - halting main loop");
            pthread_mutex_unlock(&cortex->emergency_mutex);
            break;
        }
        pthread_mutex_unlock(&cortex->emergency_mutex);
        
        // Wait for an event
        tk_cortex_event_t event;
        tk_error_code_t result = event_queue_dequeue(&cortex->event_queue, &event);
        if (result != TK_SUCCESS) {
            tk_log_error("Error dequeuing event: %d", result);
            continue;
        }
        
        // Process the event
        result = cortex_handle_event(cortex, &event);
        if (result != TK_SUCCESS) {
            tk_log_error("Error handling event: %d", result);
            
            // On critical errors, change to error state
            if (result == TK_ERROR_CRITICAL_FAILURE) {
                cortex_change_state(cortex, TK_STATE_FATAL_ERROR);
                break;
            }
        }
        
        // Free event payload using the type-aware free function
        event_payload_free(event.type, event.payload);
    }
    
    cortex_change_state(cortex, TK_STATE_SHUTDOWN);
    tk_log_info("Cortex main processing loop terminated");
    
    return NULL;
}

static tk_error_code_t cortex_handle_event(tk_cortex_t* cortex, const tk_cortex_event_t* event) {
    switch (event->type) {
        case CORTEX_EVENT_NEW_VIDEO_FRAME:
            return cortex_process_vision_input(cortex);
            
        case CORTEX_EVENT_USER_SPEECH_FINAL: {
            // Payload should be a tk_transcription_t*
            tk_transcription_t* transcription = (tk_transcription_t*)event->payload;
            if (transcription) {
                // Add to conversation context
                tk_error_code_t res = tk_contextual_reasoner_add_conversation_turn(
                    cortex->contextual_reasoner,
                    true, // is_user_input
                    transcription->text,
                    transcription->confidence
                );
                
                if (res != TK_SUCCESS) {
                    tk_log_warning("Failed to add conversation turn: %d", res);
                }
                
                // Trigger immediate LLM processing for user input
                return cortex_run_llm_inference(cortex);
            }
            return TK_SUCCESS;
        }
            
        case CORTEX_EVENT_VAD_SPEECH_STARTED:
            cortex_change_state(cortex, TK_STATE_LISTENING);
            return TK_SUCCESS;
            
        case CORTEX_EVENT_SIGNIFICANT_VISION_CHANGE: {
            // Payload should be a tk_vision_result_t*
            tk_vision_result_t* vision_result = (tk_vision_result_t*)event->payload;
            if (vision_result) {
                // Update contextual reasoner with this new visual data
                tk_error_code_t res = tk_contextual_reasoner_update_vision_context(cortex->contextual_reasoner, vision_result);
                if (res != TK_SUCCESS) {
                    tk_log_warning("Failed to update vision context: %d", res);
                }
                
                // Decide if this change is significant enough to trigger LLM inference
                // This is a simplified check - in reality, you'd have more sophisticated logic
                // For example, check if a new object of interest appeared or if an obstacle is close
                bool should_trigger_llm = false;
                // Example logic: if there are detected objects, consider it significant
                if (vision_result->object_detections && vision_result->object_detections_count > 0) {
                    should_trigger_llm = true;
                }
                
                if (should_trigger_llm) {
                    return cortex_run_llm_inference(cortex);
                }
            }
            return TK_SUCCESS;
        }
            
        case CORTEX_EVENT_SYSTEM_TIMER: {
            // This is our main periodic tick.
            // First, process any pending actions from the previous cycle.
            tk_context_summary_t context_summary;
            tk_error_code_t res = tk_contextual_reasoner_get_context_summary(cortex->contextual_reasoner, &context_summary);
            if (res != TK_SUCCESS) {
                tk_log_warning("Failed to get context summary for action processing: %d", res);
                // Continue anyway, some actions might not need context.
            }

            tk_decision_engine_process_actions(
                cortex->decision_engine,
                event->timestamp_ns,
                (res == TK_SUCCESS) ? &context_summary : NULL,
                cortex->audio_pipeline,
                cortex->navigation_engine,
                cortex->contextual_reasoner
            );

            // Then, run LLM inference as a fallback.
            return cortex_run_llm_inference(cortex);
        }
            
        case CORTEX_EVENT_SHUTDOWN:
            cortex->should_stop = true;
            return TK_SUCCESS;
            
        default:
            tk_log_warning("Unknown event type received: %d", event->type);
            return TK_SUCCESS;
    }
}

static tk_error_code_t cortex_process_vision_input(tk_cortex_t* cortex) {
    tk_video_frame_t frame_copy = {0};
    uint8_t* frame_data_copy = NULL;
    bool has_frame_to_process = false;

    pthread_mutex_lock(&cortex->video_buffer.mutex);
    
    if (cortex->video_buffer.has_new_frame && cortex->video_buffer.frame_count > 0) {
        // Get the latest frame from the ring buffer
        size_t read_idx = (cortex->video_buffer.write_index - 1 + cortex->video_buffer.frame_capacity) % cortex->video_buffer.frame_capacity;
        tk_video_frame_t* original_frame = &cortex->video_buffer.frames[read_idx];

        // Perform a deep copy of the frame data while under the lock
        size_t data_size = original_frame->stride * original_frame->height;
        frame_data_copy = malloc(data_size);
        if (frame_data_copy) {
            memcpy(frame_data_copy, original_frame->data, data_size);
            frame_copy = *original_frame;
            frame_copy.data = frame_data_copy;
            has_frame_to_process = true;
        } else {
            tk_log_error("Failed to allocate memory for frame copy in vision processing.");
        }
        
        cortex->video_buffer.has_new_frame = false;
    }
    
    pthread_mutex_unlock(&cortex->video_buffer.mutex);

    if (!has_frame_to_process) {
        return TK_SUCCESS; // Nothing to do
    }
    
    uint64_t start_time = get_current_time_ns();
    
    // Process the frame through vision pipeline using our safe local copy
    cortex_change_state(cortex, TK_STATE_PROCESSING);
    
    tk_vision_result_t* vision_result = NULL;
    tk_vision_analysis_flags_t analysis_flags = TK_VISION_PRESET_ENVIRONMENT_AWARENESS;
    
    tk_error_code_t result = tk_vision_pipeline_process_frame(
        cortex->vision_pipeline,
        &frame_copy,
        analysis_flags,
        start_time,
        &vision_result
    );
    
    if (result != TK_SUCCESS) {
        tk_log_error("Vision pipeline processing failed: %d", result);
        cortex_change_state(cortex, TK_STATE_IDLE);
        return result;
    }
    
    // Update contextual reasoner with vision results
    result = tk_contextual_reasoner_update_vision_context(cortex->contextual_reasoner, vision_result);
    if (result != TK_SUCCESS) {
        tk_log_warning("Failed to update vision context: %d", result);
    }
    
    // Cache the latest vision result for navigation analysis
    pthread_mutex_lock(&cortex->vision_result_mutex);
    if (cortex->latest_vision_result) {
        tk_vision_result_destroy(&cortex->latest_vision_result);
    }
    cortex->latest_vision_result = vision_result;
    pthread_mutex_unlock(&cortex->vision_result_mutex);
    
    cortex->performance_stats.last_vision_process_time_ns = get_current_time_ns() - start_time;
    cortex_change_state(cortex, TK_STATE_IDLE);
    
    // Check if this vision result represents a significant change
    // This is a simplified check - you would implement more sophisticated logic
    bool is_significant_change = false;
    if (vision_result && vision_result->object_detections_count > 0) {
        is_significant_change = true;
    }
    
    // If it's a significant change, enqueue an event to process it
    if (is_significant_change) {
        // Pass the original vision_result pointer. The event queue's deep copy
        // mechanism will handle creating a safe copy for the event.
        tk_cortex_event_t significant_change_event = {
            .type = CORTEX_EVENT_SIGNIFICANT_VISION_CHANGE,
            .payload = vision_result,
            .timestamp_ns = get_current_time_ns()
        };
        event_queue_enqueue(&cortex->event_queue, &significant_change_event);
    }

    // Clean up the local frame copy
    free(frame_data_copy);
    
    return TK_SUCCESS;
}

static tk_error_code_t cortex_process_navigation_analysis(tk_cortex_t* cortex) {
    tk_error_code_t result;
    
    // Get current device orientation from sensor fusion
    tk_quaternion_t orientation;
    result = tk_sensors_fusion_get_orientation(cortex->sensor_fusion, &orientation);
    if (result != TK_SUCCESS) {
        tk_log_warning("Failed to get device orientation: %d", result);
        // Use default orientation
        orientation.w = 1.0f;
        orientation.x = orientation.y = orientation.z = 0.0f;
    }
    
    // Get the latest depth map from cached vision results
    pthread_mutex_lock(&cortex->vision_result_mutex);
    tk_vision_depth_map_t* depth_map = NULL;
    
    if (cortex->latest_vision_result) {
        depth_map = &cortex->latest_vision_result->depth_map;
    }
    pthread_mutex_unlock(&cortex->vision_result_mutex);
    
    if (!depth_map) {
        return TK_SUCCESS; // Skip navigation if no depth data
    }
    
    // Update navigation engine
    result = tk_navigation_engine_update(cortex->navigation_engine, depth_map, &orientation);
    if (result != TK_SUCCESS) {
        tk_log_warning("Navigation engine update failed: %d", result);
        return result;
    }
    
    // Get traversability map
    tk_traversability_map_t traversability_map;
    result = tk_navigation_engine_get_map(cortex->navigation_engine, &traversability_map);
    if (result != TK_SUCCESS) {
        return result;
    }
    
    // Update free space detector
    result = tk_free_space_detector_analyze(cortex->free_space_detector, &traversability_map);
    if (result != TK_SUCCESS) {
        tk_log_warning("Free space analysis failed: %d", result);
    }
    
    // Update obstacle tracker
    float delta_time_s = 1.0f / cortex->config.main_loop_frequency_hz;
    result = tk_obstacle_tracker_update(cortex->obstacle_tracker, &traversability_map, delta_time_s);
    if (result != TK_SUCCESS) {
        tk_log_warning("Obstacle tracking update failed: %d", result);
    }
    
    // Get analysis results
    tk_free_space_analysis_t free_space_analysis;
    const tk_obstacle_t* obstacles = NULL;
    size_t obstacle_count = 0;
    
    tk_free_space_detector_get_analysis(cortex->free_space_detector, &free_space_analysis);
    tk_obstacle_tracker_get_all(cortex->obstacle_tracker, &obstacles, &obstacle_count);
    
    // Update contextual reasoner with navigation data
    result = tk_contextual_reasoner_update_navigation_context(
        cortex->contextual_reasoner,
        &traversability_map,
        &free_space_analysis,
        obstacles,
        obstacle_count
    );
    
    if (result != TK_SUCCESS) {
        tk_log_warning("Failed to update navigation context: %d", result);
    }
    
    return TK_SUCCESS;
}

static tk_error_code_t cortex_run_llm_inference(tk_cortex_t* cortex) {
    uint64_t start_time = get_current_time_ns();
    tk_error_code_t result;
    
    cortex_change_state(cortex, TK_STATE_PROCESSING);
    
    // Generate context string for LLM
    char* context_string = NULL;
    result = tk_contextual_reasoner_generate_context_string(
        cortex->contextual_reasoner,
        &context_string,
        2048 // Token budget
    );
    
    if (result != TK_SUCCESS || !context_string) {
        tk_log_warning("Failed to generate context string: %d", result);
        cortex_change_state(cortex, TK_STATE_IDLE);
        return result;
    }
    
    // Run LLM inference
    char* llm_response = NULL;
    result = tk_model_runner_generate_response(cortex->llm_runner, context_string, &llm_response);
    
    free(context_string);
    
    if (result != TK_SUCCESS) {
        tk_log_error("LLM inference failed: %d", result);
        cortex_change_state(cortex, TK_STATE_IDLE);
        return result;
    }
    
    // Process LLM response through decision engine
    tk_context_summary_t context_summary;
    result = tk_contextual_reasoner_get_context_summary(cortex->contextual_reasoner, &context_summary);
    if (result != TK_SUCCESS) {
        tk_log_warning("Failed to get context summary: %d", result);
    }
    
    tk_llm_response_t* parsed_response = NULL;
    result = tk_decision_engine_process_llm_response(
        cortex->decision_engine,
        llm_response,
        &context_summary,
        &parsed_response
    );
    
    free(llm_response);
    
    if (result == TK_SUCCESS && parsed_response) {
        // Execute the parsed response
        result = tk_decision_engine_execute_response(cortex->decision_engine, parsed_response);
        if (result != TK_SUCCESS) {
            tk_log_warning("Failed to execute LLM response: %d", result);
        }
        
        tk_decision_engine_free_response(&parsed_response);
    }
    
    cortex->performance_stats.last_llm_inference_time_ns = get_current_time_ns() - start_time;
    cortex_change_state(cortex, TK_STATE_IDLE);
    
    return TK_SUCCESS;
}

static void cortex_update_performance_stats(tk_cortex_t* cortex, uint64_t iteration_time_ns) {
    float iteration_time_ms = iteration_time_ns / 1000000.0f;
    
    // Simple moving average
    if (cortex->performance_stats.loop_iteration_count == 0) {
        cortex->performance_stats.average_loop_time_ms = iteration_time_ms;
    } else {
        const float alpha = 0.1f; // Smoothing factor
        cortex->performance_stats.average_loop_time_ms = 
            (1.0f - alpha) * cortex->performance_stats.average_loop_time_ms + 
            alpha * iteration_time_ms;
    }
    
    cortex->performance_stats.last_loop_time_ns = iteration_time_ns;
    
    // Log performance statistics periodically
    if ((cortex->performance_stats.loop_iteration_count % 100) == 0) {
        tk_log_debug("Cortex performance - Avg loop time: %.2fms, Vision: %.2fms, LLM: %.2fms",
            cortex->performance_stats.average_loop_time_ms,
            cortex->performance_stats.last_vision_process_time_ns / 1000000.0f,
            cortex->performance_stats.last_llm_inference_time_ns / 1000000.0f
        );
    }
}

static uint64_t get_current_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void cortex_change_state(tk_cortex_t* cortex, tk_system_state_e new_state) {
    pthread_mutex_lock(&cortex->state_mutex);
    
    if (cortex->current_state != new_state) {
        tk_system_state_e old_state = cortex->current_state;
        cortex->current_state = new_state;
        
        pthread_mutex_unlock(&cortex->state_mutex);
        
        // Notify callback if provided
        if (cortex->callbacks.on_state_change) {
            cortex->callbacks.on_state_change(new_state, cortex->callbacks.user_data);
        }
        
        tk_log_debug("Cortex state changed: %d -> %d", old_state, new_state);
    } else {
        pthread_mutex_unlock(&cortex->state_mutex);
    }
}

//------------------------------------------------------------------------------
// Payload Management Implementation
//------------------------------------------------------------------------------

/**
 * @brief Creates a deep copy of a tk_transcription_t struct.
 */
static tk_error_code_t deep_copy_transcription(tk_transcription_t** dest, const tk_transcription_t* src) {
    if (!dest || !src) return TK_ERROR_INVALID_ARGUMENT;

    *dest = calloc(1, sizeof(tk_transcription_t));
    if (!*dest) return TK_ERROR_OUT_OF_MEMORY;

    (*dest)->is_final = src->is_final;
    (*dest)->confidence = src->confidence;

    if (src->text) {
        (*dest)->text = strdup(src->text);
        if (!(*dest)->text) {
            free(*dest);
            *dest = NULL;
            return TK_ERROR_OUT_OF_MEMORY;
        }
    } else {
        (*dest)->text = NULL;
    }

    return TK_SUCCESS;
}

/**
 * @brief Creates a deep copy of a tk_vision_result_t struct.
 * This is complex due to multiple levels of pointers.
 */
static tk_error_code_t deep_copy_vision_result(tk_vision_result_t** dest, const tk_vision_result_t* src) {
    if (!dest || !src) return TK_ERROR_INVALID_ARGUMENT;

    *dest = calloc(1, sizeof(tk_vision_result_t));
    if (!*dest) return TK_ERROR_OUT_OF_MEMORY;

    // Copy scalar fields
    (*dest)->source_frame_timestamp_ns = src->source_frame_timestamp_ns;
    (*dest)->object_count = 0;
    (*dest)->text_block_count = 0;
    (*dest)->objects = NULL;
    (*dest)->text_blocks = NULL;
    (*dest)->depth_map = NULL;

    // Copy objects
    if (src->object_count > 0 && src->objects) {
        (*dest)->objects = calloc(src->object_count, sizeof(tk_vision_object_t));
        if (!(*dest)->objects) {
            tk_vision_result_destroy(dest); // Safely cleans up partial allocation
            return TK_ERROR_OUT_OF_MEMORY;
        }
        (*dest)->object_count = src->object_count;
        for (size_t i = 0; i < src->object_count; ++i) {
            (*dest)->objects[i] = src->objects[i]; // shallow copy of struct
            if (src->objects[i].label) {
                // The label in tk_vision_object_t is `const char*` and points to a static
                // label map, so we don't need to strdup it.
                (*dest)->objects[i].label = src->objects[i].label;
            }
        }
    }

    // Copy text blocks
    if (src->text_block_count > 0 && src->text_blocks) {
        (*dest)->text_blocks = calloc(src->text_block_count, sizeof(tk_vision_text_block_t));
        if (!(*dest)->text_blocks) {
            tk_vision_result_destroy(dest);
            return TK_ERROR_OUT_OF_MEMORY;
        }
        (*dest)->text_block_count = src->text_block_count;
        for (size_t i = 0; i < src->text_block_count; ++i) {
            (*dest)->text_blocks[i] = src->text_blocks[i]; // shallow copy
            if (src->text_blocks[i].text) {
                (*dest)->text_blocks[i].text = strdup(src->text_blocks[i].text);
                if (!(*dest)->text_blocks[i].text) {
                    tk_vision_result_destroy(dest);
                    return TK_ERROR_OUT_OF_MEMORY;
                }
            }
        }
    }

    // Copy depth map
    if (src->depth_map) {
        (*dest)->depth_map = calloc(1, sizeof(tk_vision_depth_map_t));
        if (!(*dest)->depth_map) {
            tk_vision_result_destroy(dest);
            return TK_ERROR_OUT_OF_MEMORY;
        }
        (*dest)->depth_map->width = src->depth_map->width;
        (*dest)->depth_map->height = src->depth_map->height;
        size_t data_size = src->depth_map->width * src->depth_map->height * sizeof(float);
        if (data_size > 0) {
            (*dest)->depth_map->data = malloc(data_size);
            if (!(*dest)->depth_map->data) {
                tk_vision_result_destroy(dest);
                return TK_ERROR_OUT_OF_MEMORY;
            }
            memcpy((*dest)->depth_map->data, src->depth_map->data, data_size);
        } else {
            (*dest)->depth_map->data = NULL;
        }
    }

    return TK_SUCCESS;
}


/**
 * @brief Copies an event payload based on its type.
 */
static tk_error_code_t event_payload_copy(tk_cortex_event_type_e type, void** dest_payload, const void* src_payload) {
    if (!dest_payload || !src_payload) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    *dest_payload = NULL;
    tk_error_code_t result = TK_SUCCESS;

    switch (type) {
        case CORTEX_EVENT_USER_SPEECH_FINAL:
            result = deep_copy_transcription((tk_transcription_t**)dest_payload, (const tk_transcription_t*)src_payload);
            break;

        case CORTEX_EVENT_SIGNIFICANT_VISION_CHANGE:
            result = deep_copy_vision_result((tk_vision_result_t**)dest_payload, (const tk_vision_result_t*)src_payload);
            break;

        // Events with no payload or simple payloads that don't need deep copy
        case CORTEX_EVENT_NEW_VIDEO_FRAME:
        case CORTEX_EVENT_VAD_SPEECH_STARTED:
        case CORTEX_EVENT_SYSTEM_TIMER:
        case CORTEX_EVENT_SHUTDOWN:
        default:
            // No payload to copy or shallow copy is sufficient (and was already done).
            // This function is for deep copies, so we do nothing.
            break;
    }

    return result;
}

/**
 * @brief Frees an event payload based on its type.
 */
static void event_payload_free(tk_cortex_event_type_e type, void* payload) {
    if (!payload) {
        return;
    }

    switch (type) {
        case CORTEX_EVENT_USER_SPEECH_FINAL: {
            tk_transcription_t* transcription = (tk_transcription_t*)payload;
            // The const_cast is safe here because we are the owner of this memory.
            free((void*)transcription->text);
            free(transcription);
            break;
        }
        case CORTEX_EVENT_SIGNIFICANT_VISION_CHANGE:
            // Use the provided library function to free the complex vision result
            tk_vision_result_destroy((tk_vision_result_t**)&payload);
            break;

        default:
            // For simple payloads that were just malloc'd, a single free is enough.
            // However, our current logic doesn't have any of those.
            // If we had a simple payload, we'd free it here.
            // For now, we only handle the complex types we've deep-copied.
            // free(payload); // This would be for a simple malloc'd payload
            break;
    }
}


//------------------------------------------------------------------------------
// Audio Pipeline Callbacks
//------------------------------------------------------------------------------

static void on_vad_event(tk_vad_event_e event, void* user_data) {
    tk_cortex_t* cortex = (tk_cortex_t*)user_data;
    
    switch (event) {
        case TK_VAD_EVENT_SPEECH_STARTED:
            tk_log_debug("Speech detection started");
            // Enqueue a VAD speech started event
            tk_cortex_event_t vad_event = {
                .type = CORTEX_EVENT_VAD_SPEECH_STARTED,
                .payload = NULL,
                .timestamp_ns = get_current_time_ns()
            };
            event_queue_enqueue(&cortex->event_queue, &vad_event);
            break;
            
        case TK_VAD_EVENT_SPEECH_ENDED:
            tk_log_debug("Speech detection ended");
            cortex_change_state(cortex, TK_STATE_PROCESSING);
            break;
            
        default:
            break;
    }
}

static void on_transcription(const tk_transcription_t* result, void* user_data) {
    tk_cortex_t* cortex = (tk_cortex_t*)user_data;
    
    if (!result || !result->text) {
        return;
    }
    
    tk_log_info("Transcription received: '%s' (confidence: %.2f, final: %s)", 
        result->text, result->confidence, result->is_final ? "yes" : "no");
    
    // Only enqueue an event for final transcriptions
    if (result->is_final) {
        // The original `result` pointer is passed directly. `event_queue_enqueue`
        // is now responsible for creating a deep copy, so we don't need to
        // manually allocate or copy anything here.
        tk_cortex_event_t transcription_event = {
            .type = CORTEX_EVENT_USER_SPEECH_FINAL,
            .payload = (void*)result, // Pass the original pointer
            .timestamp_ns = get_current_time_ns()
        };
        event_queue_enqueue(&cortex->event_queue, &transcription_event);
    }
}

static void on_tts_audio_ready(const int16_t* audio_data, size_t frame_count, uint32_t sample_rate, void* user_data) {
    tk_cortex_t* cortex = (tk_cortex_t*)user_data;
    
    // Forward TTS audio to the host application
    if (cortex->callbacks.on_tts_audio_ready) {
        cortex->callbacks.on_tts_audio_ready(audio_data, frame_count, sample_rate, cortex->callbacks.user_data);
    }
    
    // Change state back to idle after TTS is complete
    cortex_change_state(cortex, TK_STATE_IDLE);
}

//------------------------------------------------------------------------------
// Decision Engine Callbacks
//------------------------------------------------------------------------------

static void on_action_completed(const tk_action_t* action, void* user_data) {
    tk_cortex_t* cortex = (tk_cortex_t*)user_data;
    
    tk_log_debug("Action %lu completed with status %d", action->action_id, action->status);
    
    if (action->status == TK_ACTION_STATUS_FAILED && action->error_message) {
        tk_log_warning("Action %lu failed: %s", action->action_id, action->error_message);
    }
}

static void on_response_ready(const char* response_text, tk_response_priority_e priority, void* user_data) {
    tk_cortex_t* cortex = (tk_cortex_t*)user_data;
    
    if (!response_text) {
        return;
    }
    
    tk_log_info("Response ready (priority %d): '%s'", priority, response_text);
    
    // Add response to conversation context
    tk_error_code_t res = tk_contextual_reasoner_add_conversation_turn(
        cortex->contextual_reasoner,
        false, // is_user_input = false (this is system response)
        response_text,
        1.0f // confidence
    );
    
    if (res != TK_SUCCESS) {
        tk_log_warning("Failed to add response to conversation context: %d", res);
    }
    
    // Send to TTS for speech synthesis
    cortex_change_state(cortex, TK_STATE_RESPONDING);
    
    tk_error_code_t tts_result = tk_audio_pipeline_synthesize_text(cortex->audio_pipeline, response_text);
    if (tts_result != TK_SUCCESS) {
        tk_log_error("Failed to synthesize response: %d", tts_result);
        cortex_change_state(cortex, TK_STATE_IDLE);
    }
}
