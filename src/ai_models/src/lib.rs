/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: lib.rs
 *
 * This file is the main library entry point and FFI boundary for the `ai_models`
 * crate. It provides a stateful manager for loading, running, and managing multiple
 * AI model sessions (both ONNX and GGUF). The FFI uses string-based session IDs
 * for referencing models and JSON for data exchange, offering a flexible interface
 * to the C/C++ core.
 *
 * Dependencies:
 *  - `lazy_static`: For the global, thread-safe manager instance.
 *  - `serde_json`: For FFI data exchange.
 *  - `ndarray`: For creating tensors from the FFI input.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Declarations and Imports ---
pub mod gguf_runner;
pub mod onnx_runner;

use gguf_runner::{GgufModel, GgufRunner};
use lazy_static::lazy_static;
use onnx_runner::{OnnxRunner, OnnxSession};
use serde::Serialize;
use std::collections::HashMap;
use std::ffi::{c_char, CStr, CString};
use std::path::Path;
use std::sync::Mutex;
use ndarray::{ArrayD, IxDyn};

// --- Global State Management ---

/// An enum to hold different types of model sessions.
enum ModelSession<'a> {
    Onnx(OnnxSession),
    Gguf(GgufModel<'a>),
}

/// The central manager for all AI model operations.
struct AiModelsManager<'a> {
    onnx_runner: OnnxRunner,
    gguf_runner: GgufRunner,
    sessions: HashMap<String, ModelSession<'a>>,
}

// Note: The lifetime 'a on AiModelsManager and ModelSession is tricky for a
// static variable. For a real-world scenario, you might use `Box::leak` or an
// `unsafe` static mutable variable. For this implementation, we'll simplify
// and assume the manager lives for the duration of the program, but acknowledge
// this is a complex area of FFI design. For now, we'll avoid storing GgufModel
// directly in the static manager to bypass lifetime issues, and instead load it on demand.
// A more robust solution might involve a handle-based system where the C side owns the memory.
struct AiModelsManagerSimple {
    onnx_runner: OnnxRunner,
    gguf_runner: GgufRunner,
    onnx_sessions: HashMap<String, OnnxSession>,
}


lazy_static! {
    static ref MANAGER: Mutex<AiModelsManagerSimple> = Mutex::new(
        AiModelsManagerSimple {
            onnx_runner: OnnxRunner::new().expect("Failed to initialize ONNX Runner"),
            gguf_runner: GgufRunner::new(),
            onnx_sessions: HashMap::new(),
        }
    );
}

// --- FFI Helper Functions ---
fn catch_panic<F, R>(f: F) -> R where F: FnOnce() -> R, R: Default {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|_| R::default())
}

fn serialize_to_c_string<T: Serialize>(data: &T) -> *mut c_char {
    match serde_json::to_string(data) {
        Ok(json_str) => CString::new(json_str).unwrap().into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

// --- FFI Public Interface ---

/// Loads an ONNX model from a file and prepares a session for inference.
///
/// # Arguments
/// - `model_path_c`: C-string with the path to the `.onnx` file.
/// - `use_gpu`: Boolean flag to request GPU execution.
/// - `session_id_c`: C-string to use as a unique identifier for this session.
///
/// # Returns
/// `0` on success, `-1` on failure.
#[no_mangle]
pub extern "C" fn ai_models_load_onnx_session(
    model_path_c: *const c_char,
    use_gpu: bool,
    session_id_c: *const c_char,
) -> i32 {
    catch_panic(|| {
        let model_path = Path::new(unsafe { CStr::from_ptr(model_path_c).to_str().unwrap() });
        let session_id = unsafe { CStr::from_ptr(session_id_c).to_str().unwrap() }.to_string();

        let mut manager = MANAGER.lock().unwrap();
        match manager.onnx_runner.load_session(model_path, use_gpu) {
            Ok(session) => {
                manager.onnx_sessions.insert(session_id, session);
                0
            }
            Err(e) => {
                eprintln!("Failed to load ONNX session: {}", e);
                -1
            }
        }
    })
}

/// Runs inference on a previously loaded ONNX model.
///
/// # Arguments
/// - `session_id_c`: The ID of the session to use.
/// - `inputs_json_c`: A JSON string representing an array of input tensors.
///   Example: `[ { "shape": [1, 3, 224, 224], "data": [0.1, 0.2, ...] } ]`
///
/// # Returns
/// A C-string with the JSON representation of the output tensors. Must be freed.
/// Returns null on error.
#[no_mangle]
pub extern "C" fn ai_models_run_onnx_inference(
    session_id_c: *const c_char,
    inputs_json_c: *const c_char,
) -> *mut c_char {
    catch_panic(|| {
        let session_id = unsafe { CStr::from_ptr(session_id_c).to_str().unwrap() };
        let inputs_json = unsafe { CStr::from_ptr(inputs_json_c).to_str().unwrap() };

        #[derive(serde::Deserialize)]
        struct JsonTensor { shape: Vec<usize>, data: Vec<f32> }

        let json_inputs: Vec<JsonTensor> = match serde_json::from_str(inputs_json) {
            Ok(i) => i,
            Err(e) => { eprintln!("Failed to parse input JSON: {}", e); return std::ptr::null_mut(); }
        };

        let manager = MANAGER.lock().unwrap();
        let session = match manager.onnx_sessions.get(session_id) {
            Some(s) => s,
            None => { eprintln!("ONNX session not found: {}", session_id); return std::ptr::null_mut(); }
        };

        let inputs: Vec<_> = json_inputs.into_iter().map(|jt| {
            let array = ArrayD::from_shape_vec(IxDyn(&jt.shape), jt.data).unwrap();
            ort::Value::from_array(array).unwrap()
        }).collect();

        match session.run(inputs) {
            Ok(outputs) => {
                // For simplicity, we assume outputs can be converted to f32 arrays and serialize them.
                // A full implementation would handle different data types.
                let results: Vec<_> = outputs.iter().map(|v| {
                    let tensor = v.try_extract::<f32>().unwrap();
                    let shape = tensor.view().shape().to_vec();
                    let data = tensor.view().iter().cloned().collect::<Vec<_>>();
                    serde_json::json!({ "shape": shape, "data": data })
                }).collect();
                serialize_to_c_string(&results)
            }
            Err(e) => {
                eprintln!("ONNX inference failed: {}", e);
                std::ptr::null_mut()
            }
        }
    })
}

/// Retrieves the metadata for a GGUF model as a JSON string.
///
/// # Arguments
/// - `model_path_c`: The path to the `.gguf` file.
///
/// # Returns
/// A C-string with the JSON metadata. Must be freed. Returns null on error.
#[no_mangle]
pub extern "C" fn ai_models_get_gguf_metadata(model_path_c: *const c_char) -> *mut c_char {
    catch_panic(|| {
        let model_path = Path::new(unsafe { CStr::from_ptr(model_path_c).to_str().unwrap() });
        let manager = MANAGER.lock().unwrap();

        match manager.gguf_runner.load(model_path) {
            Ok(model) => {
                // Convert GGUF metadata to a serializable HashMap
                let metadata: HashMap<_, _> = model.info().metadata.iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect();
                serialize_to_c_string(&metadata)
            }
            Err(e) => {
                eprintln!("Failed to load GGUF model: {}", e);
                std::ptr::null_mut()
            }
        }
    })
}

/// Frees the resources associated with a loaded ONNX session.
#[no_mangle]
pub extern "C" fn ai_models_close_onnx_session(session_id_c: *const c_char) {
    catch_panic(|| {
        let session_id = unsafe { CStr::from_ptr(session_id_c).to_str().unwrap() };
        let mut manager = MANAGER.lock().unwrap();
        manager.onnx_sessions.remove(session_id);
        std::ptr::null_mut() // Return a value that can be handled by catch_panic
    });
}

/// Frees a C-string that was allocated by this Rust library.
#[no_mangle]
pub extern "C" fn ai_models_free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    unsafe {
        drop(CString::from_raw(s));
    }
}
