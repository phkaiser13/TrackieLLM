/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: tk_logging.h
*
* This header file defines the public API for the TrackieLLM logging subsystem.
* It provides a flexible, high-performance, and thread-safe logging facility.
* Key features include multiple log levels, configurable output (console/file),
* and performance-conscious macros that avoid evaluating log arguments when the
* corresponding log level is disabled.
*
* SPDX-License-Identifier: AGPL-3.0 license
*/

#ifndef TRACKIELLM_UTILS_TK_LOGGING_H
#define TRACKIELLM_UTILS_TK_LOGGING_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h> // For FILE*
#include <time.h>

#include "tk_error_handling.h"

/**
 * @enum tk_log_level_t
 * @brief Defines the severity levels for log messages.
 */
typedef enum {
    TK_LOG_LEVEL_TRACE = 0, /**< Fine-grained diagnostic information, typically for deep debugging. */
    TK_LOG_LEVEL_DEBUG,     /**< Detailed information useful for debugging. */
    TK_LOG_LEVEL_INFO,      /**< Informational messages about system progress and state. */
    TK_LOG_LEVEL_WARN,      /**< Indicates a potential issue or an unexpected event. */
    TK_LOG_LEVEL_ERROR,     /**< An error that prevents a specific operation from completing. */
    TK_LOG_LEVEL_FATAL      /**< A critical error that will likely lead to application termination. */
} tk_log_level_t;

/**
 * @struct tk_log_config_t
 * @brief Configuration structure for the logging subsystem.
 */
typedef struct {
    tk_log_level_t level;       /**< The minimum log level to be processed. */
    const char*    filename;    /**< Optional: Path to a log file. If NULL, logs are not written to a file. */
    bool           log_to_console;/**< If true, logs will be written to stdout/stderr. */
    bool           use_utc_time;  /**< If true, timestamps will be in UTC; otherwise, local time. */
    bool           quiet_mode;    /**< If true, suppresses all console output regardless of other settings. */
} tk_log_config_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initializes the logging subsystem.
 *
 * This function sets up the logger based on the provided configuration. It must
 * be called once at the beginning of the application's lifecycle before any
 * logging macros are used.
 *
 * @param[in] config A pointer to a tk_log_config_t structure. If NULL, default
 *                   settings (INFO level, console only, local time) will be used.
 *
 * @return TK_SUCCESS on successful initialization.
 * @return TK_ERROR_INVALID_ARGUMENT if the config is valid but contains invalid data.
 * @return TK_ERROR_FILE_WRITE if the specified log file cannot be opened for writing.
 * @return TK_ERROR_INVALID_STATE if the logger is already initialized.
 *
 * @par Thread-Safety
 * This function is not thread-safe and should be called from a single thread
 * during application startup.
 */
TK_NODISCARD tk_error_code_t tk_log_init(const tk_log_config_t* config);

/**
 * @brief Shuts down the logging subsystem.
 *
 * Flushes any buffered log messages and closes the log file if one was opened.
 * This should be called once at the end of the application's lifecycle.
 *
 * @par Thread-Safety
 * This function is not thread-safe and should be called from a single thread
 * during application shutdown.
 */
void tk_log_shutdown(void);

/**
 * @brief Sets the global logging level at runtime.
 *
 * Allows for dynamically changing the verbosity of the logs without restarting
 * the application.
 *
 * @param[in] level The new minimum log level to process.
 *
 * @par Thread-Safety
 * This function is thread-safe.
 */
void tk_log_set_level(tk_log_level_t level);

/**
 * @brief The core logging function (intended for internal use by macros).
 *
 * This function formats and writes a log message if the given level is at or
 * above the currently configured global log level. It should not be called
 * directly; use the TK_LOG_* macros instead.
 *
 * @param[in] level The severity level of the message.
 * @param[in] file The source file where the log was generated (__FILE__).
 * @param[in] line The line number in the source file (__LINE__).
 * @param[in] func The function name where the log was generated (__func__).
 * @param[in] fmt The printf-style format string.
 * @param[in] ... Variadic arguments for the format string.
 *
 * @par Thread-Safety
 * This function is thread-safe.
 */
void tk_log_message(tk_log_level_t level, const char* file, int line, const char* func, const char* fmt, ...);

// External declaration of the global log level for macro optimization.
extern tk_log_level_t g_tk_log_level;

/**
 * @brief Logging macros for different severity levels.
 *
 * These macros are the primary interface for logging. They automatically
 * capture file, line, and function context. Crucially, they check the current
 * log level before evaluating their arguments, which prevents performance
 * penalties from constructing log messages that would be discarded anyway.
 */
#define TK_LOG_TRACE(fmt, ...) \
    do { if (g_tk_log_level <= TK_LOG_LEVEL_TRACE) { tk_log_message(TK_LOG_LEVEL_TRACE, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__); } } while (0)

#define TK_LOG_DEBUG(fmt, ...) \
    do { if (g_tk_log_level <= TK_LOG_LEVEL_DEBUG) { tk_log_message(TK_LOG_LEVEL_DEBUG, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__); } } while (0)

#define TK_LOG_INFO(fmt, ...) \
    do { if (g_tk_log_level <= TK_LOG_LEVEL_INFO) { tk_log_message(TK_LOG_LEVEL_INFO, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__); } } while (0)

#define TK_LOG_WARN(fmt, ...) \
    do { if (g_tk_log_level <= TK_LOG_LEVEL_WARN) { tk_log_message(TK_LOG_LEVEL_WARN, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__); } } while (0)

#define TK_LOG_ERROR(fmt, ...) \
    do { if (g_tk_log_level <= TK_LOG_LEVEL_ERROR) { tk_log_message(TK_LOG_LEVEL_ERROR, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__); } } while (0)

#define TK_LOG_FATAL(fmt, ...) \
    do { if (g_tk_log_level <= TK_LOG_LEVEL_FATAL) { tk_log_message(TK_LOG_LEVEL_FATAL, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__); } } while (0)


#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_UTILS_TK_LOGGING_H