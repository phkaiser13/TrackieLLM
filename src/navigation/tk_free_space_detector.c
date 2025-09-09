/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_free_space_detector.c
 * 
 * This file implements the C-side wrapper for the Free Space Detector.
 * It acts as a thin Foreign Function Interface (FFI) layer, delegating all
 * analysis logic and state management to the Rust implementation.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#include "navigation/tk_free_space_detector.h"
#include <stdlib.h>

// Opaque handle to the Rust-managed detector object.
typedef void* FreeSpaceDetectorHandle;

// FFI declarations for the functions implemented in Rust.
extern FreeSpaceDetectorHandle rust_free_space_detector_create(const tk_free_space_config_t* config);
extern void rust_free_space_detector_destroy(FreeSpaceDetectorHandle handle);
extern void rust_free_space_detector_analyze(FreeSpaceDetectorHandle handle, const tk_traversability_map_t* map);
extern void rust_free_space_detector_get_analysis(FreeSpaceDetectorHandle handle, tk_free_space_analysis_t* out_analysis);


// The C-side struct just holds the handle to the Rust object.
struct tk_free_space_detector_s {
    FreeSpaceDetectorHandle rust_handle;
};


//------------------------------------------------------------------------------
// Engine Lifecycle Management
//------------------------------------------------------------------------------

tk_error_code_t tk_free_space_detector_create(
    tk_free_space_detector_t** out_detector,
    const tk_free_space_config_t* config
) {
    if (!out_detector || !config) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    FreeSpaceDetectorHandle handle = rust_free_space_detector_create(config);
    if (!handle) {
        return TK_ERROR_OUT_OF_MEMORY;
    }

    tk_free_space_detector_t* detector = (tk_free_space_detector_t*)calloc(1, sizeof(tk_free_space_detector_t));
    if (!detector) {
        rust_free_space_detector_destroy(handle);
        return TK_ERROR_OUT_OF_MEMORY;
    }

    detector->rust_handle = handle;
    *out_detector = detector;
    return TK_SUCCESS;
}

void tk_free_space_detector_destroy(tk_free_space_detector_t** detector) {
    if (detector && *detector) {
        rust_free_space_detector_destroy((*detector)->rust_handle);
        free(*detector);
        *detector = NULL;
    }
}

//------------------------------------------------------------------------------
// Core Processing and Analysis
//------------------------------------------------------------------------------

tk_error_code_t tk_free_space_detector_analyze(
    tk_free_space_detector_t* detector,
    const tk_traversability_map_t* map
) {
    if (!detector || !map) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    rust_free_space_detector_analyze(detector->rust_handle, map);
    return TK_SUCCESS;
}

//------------------------------------------------------------------------------
// High-Level Queries
//------------------------------------------------------------------------------

tk_error_code_t tk_free_space_detector_get_analysis(
    tk_free_space_detector_t* detector,
    tk_free_space_analysis_t* out_analysis
) {
    if (!detector || !out_analysis) {
        return TK_ERROR_INVALID_ARGUMENT;
    }
    
    // The rust function will fill the analysis struct.
    // Note: The `sectors` pointer inside out_analysis will point to memory
    // managed by the Rust object. It is only valid until the next call to
    // `analyze` or `destroy`. This is documented in the header.
    rust_free_space_detector_get_analysis(detector->rust_handle, out_analysis);
    return TK_SUCCESS;
}
