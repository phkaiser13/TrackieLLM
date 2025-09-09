/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/event_bus/event_bus_ffi.h
 *
 * This header file defines the C-facing Foreign Function Interface (FFI) for
 * interacting with the Rust-based central Event Bus. It provides C code
 * with the ability to create, destroy, and publish events to the bus.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#ifndef TRACKIELLM_EVENT_BUS_FFI_H
#define TRACKIELLM_EVENT_BUS_FFI_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Opaque handle to the Rust EventBus.
typedef struct EventBus tk_event_bus_t;

// FFI-safe representation of a vision object to be published.
// This must be kept in sync with the `FfiVisionObject` struct in Rust.
typedef struct {
    const char* label;
    float confidence;
    float distance_meters;
} FfiVisionObject;

// FFI-safe representation of a full vision result to be published.
// This must be kept in sync with the `FfiVisionResult` struct in Rust.
typedef struct {
    const FfiVisionObject* objects;
    size_t object_count;
    uint64_t timestamp_ns;
} FfiVisionResult;


#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a new EventBus instance.
 * @return A pointer to the created event bus. Must be freed with event_bus_destroy.
 */
tk_event_bus_t* event_bus_create();

/**
 * @brief Destroys an EventBus instance.
 * @param bus A pointer to the event bus instance to destroy.
 */
void event_bus_destroy(tk_event_bus_t* bus);

/**
 * @brief Publishes a vision result to the event bus.
 * @param bus A pointer to the event bus instance.
 * @param result A pointer to the FFI-safe vision result to publish.
 */
void vision_publish_result(const tk_event_bus_t* bus, const FfiVisionResult* result);


#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_EVENT_BUS_FFI_H
