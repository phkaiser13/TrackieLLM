/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_cortex_main.h
*
* This header file defines the primary public API for the TrackieLLM Cortex,
* the central reasoning and orchestration engine of the system. The Cortex is
* architected as a stateful, lifecycle-managed object (`tk_cortex_t`) that
* integrates all sensory inputs, manages the AI models, and produces actionable
* outputs.
*
* The design philosophy is centered around a real-time, event-driven main loop
* that operates asynchronously from the data providers (camera, microphone).
* The host application is responsible for feeding sensory data into the Cortex
* via injection functions and receiving outputs (like synthesized speech) via
* a structured callback mechanism.
*
* This API abstracts the immense complexity of:
*   1. Multi-threaded data ingestion and synchronization.
*   2. The main perception-reasoning-action cycle.
*   3. Interaction with the local Large Language Model (LLM), including the
*      critical task of parsing the LLM's output to trigger local "tool use"
*      (emulating function calling).
*   4. Management of the system's operational state (e.g., idle, listening,
*      processing, responding).
*
* SPDX-License-Identifier:
*/

#ifndef TRACKIELLM_CORTEX_TK_CORTEX_MAIN_H
#define TRACKIELLM_CORTEX_TK_CORTEX_MAIN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "internal_tools/tk_file_manager.h" // For tk_path_t

// Forward-declare the main Cortex object as an opaque type to enforce encapsulation.
typedef struct tk_cortex_s tk_cortex_t;

/**
 * @enum tk_system_state_e
 * @brief Defines the major operational states of the Cortex engine.
 *
 * This allows the host application to be aware of the system's activity,
 * for example, to update a UI or control an indicator LED.
 */
typedef enum {
    TK_STATE_UNINITIALIZED, /**< The Cortex object has not been created or has been destroyed. */
    TK_STATE_INITIALIZING,  /**< The Cortex is being created and is loading models. This can be a long process. */
    TK_STATE_IDLE,          /**< The system is running and waiting for a wake word or user input. */
    TK_STATE_LISTENING,     /**< The system has detected a wake word and is actively recording audio for a command. */
    TK_STATE_PROCESSING,    /**< The system is processing sensory input and running inference with the LLM. */
    TK_STATE_RESPONDING,    /**< The system is generating a response, typically via Text-to-Speech. */
    TK_STATE_SHUTDOWN,      /**< The system is in the process of shutting down and releasing resources. */
    TK_STATE_FATAL_ERROR    /**< An unrecoverable error has occurred. The system is in a non-operational state. */
} tk_system_state_e;

/**
 * @struct tk_model_paths_t
 * @brief A structure to hold the filesystem paths for all required AI models.
 *
 * This structure is a member of the main configuration and centralizes all
 * model resource locations.
 */
typedef struct {
    const char* llm_model;          // Path to the Mistral-7B GGUF model file.
    const char* object_detection_model; // Path to the YOLOv5nu ONNX model file.
    const char* depth_estimation_model; // Path to the MiDaS (DPT-SwinV2) ONNX model file.
    const char* asr_model;          // Path to the Whisper.cpp GGML model file.
    const char* tts_model_dir;      // Path to the directory containing Piper TTS model(s).
    const char* vad_model;          // Path to the Silero VAD ONNX model file.
    const char* tesseract_data_dir; // Path to the Tesseract `tessdata` directory.
} tk_model_paths_t;

/**
 * @struct tk_cortex_config_t
 * @brief Comprehensive configuration for initializing the Cortex engine.
 *
 * This structure is passed once during creation and defines the entire
 * operational environment of the Cortex.
 */
typedef struct {
    tk_model_paths_t model_paths;   /**< Paths to all necessary AI models. */
    int gpu_device_id;              /**< The ID of the GPU to use for accelerated tasks. -1 for CPU. */
    float main_loop_frequency_hz;   /**< Target frequency for the main processing loop (e.g., 10.0f for 10Hz). */
    const char* user_language;      /**< User's primary language (e.g., "pt-BR") for TTS selection. */
    void* user_data;                /**< An opaque pointer for the user to associate custom data with callbacks. */
} tk_cortex_config_t;

/**
 * @struct tk_video_frame_t
 * @brief Represents a single frame of video data to be injected into the Cortex.
 */
typedef struct {
    uint32_t width;                 /**< Frame width in pixels. */
    uint32_t height;                /**< Frame height in pixels. */
    uint32_t stride;                /**< Number of bytes per row of pixels. */
    enum { TK_PIXEL_FORMAT_RGB8, TK_PIXEL_FORMAT_RGBA8 } format; /**< Pixel format of the data. */
    const uint8_t* data;            /**< Pointer to the raw pixel data. */
} tk_video_frame_t;

/**
 * @brief Callback function pointer for state change notifications.
 * @param new_state The new state the system has entered.
 * @param user_data The opaque user data pointer from tk_cortex_config_t.
 */
typedef void (*tk_on_state_change_cb)(tk_system_state_e new_state, void* user_data);

/**
 * @brief Callback function pointer for delivering synthesized speech audio.
 * @param audio_data Pointer to the raw PCM audio data (e.g., 16-bit signed mono).
 * @param frame_count The number of audio frames in the buffer.
 * @param sample_rate The sample rate of the audio (e.g., 22050).
 * @param user_data The opaque user data pointer from tk_cortex_config_t.
 */
typedef void (*tk_on_tts_audio_ready_cb)(const int16_t* audio_data, size_t frame_count, uint32_t sample_rate, void* user_data);

/**
 * @struct tk_cortex_callbacks_t
 * @brief A structure to hold all callback functions for the host application.
 */
typedef struct {
    tk_on_state_change_cb on_state_change;
    tk_on_tts_audio_ready_cb on_tts_audio_ready;
} tk_cortex_callbacks_t;


#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Cortex Lifecycle Management
//------------------------------------------------------------------------------

/**
 * @brief Creates and initializes a new Cortex instance.
 *
 * This is the main entry point. It allocates resources, loads all AI models
 * specified in the configuration, and prepares the system for operation. This
 * function can take a significant amount of time to complete.
 *
 * @param[out] out_cortex A pointer to a tk_cortex_t* that will receive the
 *                        address of the newly created Cortex instance.
 * @param[in] config A pointer to the configuration structure. The contents are
 *                   copied, so the caller can free the structure after this call.
 * @param[in] callbacks A structure containing the callback function pointers.
 *
 * @return TK_SUCCESS on successful creation and initialization.
 * @return TK_ERROR_INVALID_ARGUMENT if out_cortex, config, or callbacks are NULL.
 * @return TK_ERROR_OUT_OF_MEMORY if memory allocation fails.
 * @return TK_ERROR_MODEL_LOAD_FAILED if any of the specified AI models cannot be loaded.
 */
TK_NODISCARD tk_error_code_t tk_cortex_create(tk_cortex_t** out_cortex, const tk_cortex_config_t* config, tk_cortex_callbacks_t callbacks);

/**
 * @brief Destroys a Cortex instance and frees all associated resources.
 *
 * This function will gracefully shut down the main loop if it is running before
 * releasing all memory, models, and hardware handles.
 *
 * @param[in,out] cortex A pointer to the tk_cortex_t* object to be destroyed.
 *                       The pointer is set to NULL after destruction.
 */
void tk_cortex_destroy(tk_cortex_t** cortex);

//------------------------------------------------------------------------------
// Cortex Main Loop Control
//------------------------------------------------------------------------------

/**
 * @brief Starts the main processing loop of the Cortex.
 *
 * This function is typically blocking and will run until `tk_cortex_stop` is
 * called from another thread. It should be called from a dedicated thread.
 *
 * @param[in] cortex The Cortex instance to run.
 *
 * @return TK_SUCCESS if the loop completed gracefully.
 * @return TK_ERROR_INVALID_STATE if the Cortex is not in an initialized state.
 * @return An error code corresponding to any fatal error encountered during the loop.
 */
TK_NODISCARD tk_error_code_t tk_cortex_run(tk_cortex_t* cortex);

/**
 * @brief Signals the main processing loop to stop gracefully.
 *
 * This function is thread-safe and can be called from any thread to request
 * the shutdown of the main loop started by `tk_cortex_run`.
 *
 * @param[in] cortex The Cortex instance to stop.
 *
 * @return TK_SUCCESS on success.
 */
TK_NODISCARD tk_error_code_t tk_cortex_stop(tk_cortex_t* cortex);

//------------------------------------------------------------------------------
// Asynchronous Data Injection
//------------------------------------------------------------------------------

/**
 * @brief Injects a buffer of raw audio data into the Cortex.
 *
 * This function is thread-safe and is the primary way for the host application
 * to provide microphone data to the system. The data is copied into an internal
 * ring buffer for processing by the main loop.
 *
 * @param[in] cortex The Cortex instance.
 * @param[in] audio_data Pointer to the raw audio data (16-bit signed mono PCM).
 * @param[in] frame_count The number of audio frames in the buffer.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_INVALID_ARGUMENT if data is NULL or frame_count is zero.
 */
TK_NODISCARD tk_error_code_t tk_cortex_inject_audio_frame(tk_cortex_t* cortex, const int16_t* audio_data, size_t frame_count);

/**
 * @brief Injects a single video frame into the Cortex.
 *
 * This function is thread-safe. It provides the latest visual information to the
 * system. The Cortex will internally manage this frame, potentially making a
 * copy for processing, so the caller can reuse the buffer immediately.
 *
 * @param[in] cortex The Cortex instance.
 * @param[in] frame A pointer to the video frame structure.
 *
 * @return TK_SUCCESS on success.
 */
TK_NODISCARD tk_error_code_t tk_cortex_inject_video_frame(tk_cortex_t* cortex, const tk_video_frame_t* frame);

//------------------------------------------------------------------------------
// State Query
//------------------------------------------------------------------------------

/**
 * @brief Retrieves the current operational state of the Cortex.
 *
 * This function is thread-safe.
 *
 * @param[in] cortex The Cortex instance.
 * @param[out] out_state Pointer to a variable to receive the current state.
 *
 * @return TK_SUCCESS on success.
 */
TK_NODISCARD tk_error_code_t tk_cortex_get_state(const tk_cortex_t* cortex, tk_system_state_e* out_state);


#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_CORTEX_TK_CORTEX_MAIN_H