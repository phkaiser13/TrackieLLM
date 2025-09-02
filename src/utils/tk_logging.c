/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_logging.c
*
* This source file provides the implementation for the TrackieLLM logging
* subsystem. It handles initialization, configuration, thread-safe message
* processing, and shutdown. The implementation uses a single static struct to
* manage state, a mutex for synchronization, and standard C library functions
* for I/O and time formatting.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#include "tk_logging.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Use POSIX threads for mutex. For Windows, this would be replaced with
// #include <windows.h> and CRITICAL_SECTION.
#include <pthread.h>

//------------------------------------------------------------------------------
// Module-level static state
//------------------------------------------------------------------------------

// A static struct to hold the logger's entire state. This is cleaner than
// multiple global static variables.
static struct {
    tk_log_config_t config;         // A copy of the user-provided configuration.
    FILE*           log_file;       // File pointer for the log file, if any.
    pthread_mutex_t mutex;          // Mutex to ensure thread-safe logging.
    bool            initialized;    // Flag to prevent re-initialization.
} g_logger_state;

// Definition of the global log level variable declared as 'extern' in the header.
// This is used by the macros for a fast check before calling the log function.
tk_log_level_t g_tk_log_level = TK_LOG_LEVEL_INFO; // Default level.

//------------------------------------------------------------------------------
// Internal helper functions
//------------------------------------------------------------------------------

/**
 * @brief Converts a log level enum to its string representation.
 * @param level The log level.
 * @return A constant string for the level (e.g., "INFO", "ERROR").
 */
static const char* level_to_string(tk_log_level_t level) {
    switch (level) {
        case TK_LOG_LEVEL_TRACE: return "TRACE";
        case TK_LOG_LEVEL_DEBUG: return "DEBUG";
        case TK_LOG_LEVEL_INFO:  return "INFO";
        case TK_LOG_LEVEL_WARN:  return "WARN";
        case TK_LOG_LEVEL_ERROR: return "ERROR";
        case TK_LOG_LEVEL_FATAL: return "FATAL";
    }
    return "UNKNOWN";
}

/**
 * @brief Shortens a full file path to just the filename.
 * @param path The full path string.
 * @return A pointer to the beginning of the filename within the path string.
 */
static const char* shorten_path(const char* path) {
    if (!path) {
        return "";
    }
    const char* last_slash = strrchr(path, '/');
    const char* last_bslash = strrchr(path, '\\');
    const char* last_separator = (last_slash > last_bslash) ? last_slash : last_bslash;
    return last_separator ? last_separator + 1 : path;
}

//------------------------------------------------------------------------------
// Public API Implementation
//------------------------------------------------------------------------------

TK_NODISCARD tk_error_code_t tk_log_init(const tk_log_config_t* config) {
    if (g_logger_state.initialized) {
        return TK_ERROR_INVALID_STATE;
    }

    // Initialize the mutex.
    if (pthread_mutex_init(&g_logger_state.mutex, NULL) != 0) {
        // This is a critical failure, we can't even log it.
        return TK_ERROR_UNKNOWN;
    }

    // Use default configuration if none is provided.
    if (config) {
        g_logger_state.config = *config;
    } else {
        g_logger_state.config.level = TK_LOG_LEVEL_INFO;
        g_logger_state.config.filename = NULL;
        g_logger_state.config.log_to_console = true;
        g_logger_state.config.use_utc_time = false;
        g_logger_state.config.quiet_mode = false;
    }

    // Set the global log level for macro checks.
    g_tk_log_level = g_logger_state.config.level;

    // Open the log file if a path is provided.
    if (g_logger_state.config.filename) {
        g_logger_state.log_file = fopen(g_logger_state.config.filename, "a");
        if (!g_logger_state.log_file) {
            pthread_mutex_destroy(&g_logger_state.mutex);
            return TK_ERROR_FILE_WRITE;
        }
    }

    g_logger_state.initialized = true;

    TK_LOG_INFO("Logging subsystem initialized.");
    return TK_SUCCESS;
}

void tk_log_shutdown(void) {
    if (!g_logger_state.initialized) {
        return;
    }

    TK_LOG_INFO("Logging subsystem shutting down.");

    pthread_mutex_lock(&g_logger_state.mutex);

    if (g_logger_state.log_file) {
        fflush(g_logger_state.log_file);
        fclose(g_logger_state.log_file);
        g_logger_state.log_file = NULL;
    }

    g_logger_state.initialized = false;

    pthread_mutex_unlock(&g_logger_state.mutex);
    pthread_mutex_destroy(&g_logger_state.mutex);
}

void tk_log_set_level(tk_log_level_t level) {
    if (!g_logger_state.initialized) {
        return;
    }
    pthread_mutex_lock(&g_logger_state.mutex);
    g_logger_state.config.level = level;
    g_tk_log_level = level;
    pthread_mutex_unlock(&g_logger_state.mutex);
}

void tk_log_message(tk_log_level_t level, const char* file, int line, const char* func, const char* fmt, ...) {
    if (!g_logger_state.initialized || level < g_logger_state.config.level) {
        return;
    }

    // Lock to ensure the entire log write is atomic.
    pthread_mutex_lock(&g_logger_state.mutex);

    // 1. Get current time and format it.
    char time_buf[20]; // "YYYY-MM-DD HH:MM:SS" is 19 chars + null.
    time_t raw_time = time(NULL);
    struct tm time_info;
    if (g_logger_state.config.use_utc_time) {
        gmtime_r(&raw_time, &time_info);
    } else {
        localtime_r(&raw_time, &time_info);
    }
    strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", &time_info);

    // 2. Determine output stream and log to console if enabled.
    if (g_logger_state.config.log_to_console && !g_logger_state.config.quiet_mode) {
        FILE* stream = (level >= TK_LOG_LEVEL_ERROR) ? stderr : stdout;
        fprintf(stream, "[%s] [%-5s] [%s:%d (%s)] - ",
                time_buf,
                level_to_string(level),
                shorten_path(file),
                line,
                func);
        va_list args;
        va_start(args, fmt);
        vfprintf(stream, fmt, args);
        va_end(args);
        fprintf(stream, "\n");
        fflush(stream);
    }

    // 3. Log to file if enabled.
    if (g_logger_state.log_file) {
        fprintf(g_logger_state.log_file, "[%s] [%-5s] [%s:%d (%s)] - ",
                time_buf,
                level_to_string(level),
                shorten_path(file),
                line,
                func);
        va_list args;
        va_start(args, fmt);
        vfprintf(g_logger_state.log_file, fmt, args);
        va_end(args);
        fprintf(g_logger_state.log_file, "\n");
        fflush(g_logger_state.log_file);
    }

    // Unlock before potential exit.
    pthread_mutex_unlock(&g_logger_state.mutex);

    // 4. Handle fatal errors.
    if (level == TK_LOG_LEVEL_FATAL) {
        // Shutdown is not called here as the program is terminating immediately.
        // The OS will clean up file handles.
        exit(EXIT_FAILURE);
    }
}