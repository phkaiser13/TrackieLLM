/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: lib.rs
 *
 * This file is the main library entry point and FFI boundary for the `networking`
 * crate. It exposes a synchronous C-compatible API that internally drives a
 * high-performance, asynchronous networking stack built with Tokio. The FFI manages
 * client instances, connection pools, and handles the serialization/deserialization
 * of protocol messages.
 *
 * Dependencies:
 *  - `tokio`: For the async runtime.
 *  - `lazy_static`: For the global, thread-safe client manager.
 *  - `serde_json`: For FFI data exchange.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Declarations and Imports ---
pub mod connection_pool;
pub mod protocol;

use connection_pool::{ConnectionPool, PoolConfig};
use lazy_static::lazy_static;
use protocol::{Message, MessageCodec, RequestPayload, ResponsePayload};
use serde::Serialize;
use std::collections::HashMap;
use std::ffi::{c_char, CStr, CString};
use std::net::SocketAddr;
use std::sync::Mutex;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::runtime::{Builder, Runtime};
use bytes::BytesMut;

// --- Global State Management ---

/// A client struct that encapsulates its own Tokio runtime and connection pool.
/// This allows for managing connections to different servers independently.
struct NetworkingClient {
    runtime: Runtime,
    pool: ConnectionPool,
}

lazy_static! {
    static ref CLIENT_MANAGER: Mutex<HashMap<String, Arc<NetworkingClient>>> = Mutex::new(HashMap::new());
}

// --- FFI Helper Functions ---
fn catch_panic<F, R>(f: F) -> R where F: FnOnce() -> R, R: Default {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|_| R::default())
}

// --- FFI Public Interface ---

/// Creates and initializes a new networking client.
///
/// # Arguments
/// - `server_addr_c`: C-string with the server address (e.g., "127.0.0.1:8080").
/// - `client_id_c`: C-string to use as a unique identifier for this client.
///
/// # Returns
/// `0` on success, `-1` on failure.
#[no_mangle]
pub extern "C" fn networking_client_create(
    server_addr_c: *const c_char,
    client_id_c: *const c_char,
) -> i32 {
    catch_panic(|| {
        let client_id = unsafe { CStr::from_ptr(client_id_c).to_str().unwrap() }.to_string();
        let server_addr: SocketAddr = unsafe { CStr::from_ptr(server_addr_c).to_str().unwrap() }.parse().expect("Invalid server address");

        let mut manager = CLIENT_MANAGER.lock().unwrap();
        if manager.contains_key(&client_id) {
            eprintln!("Client with ID '{}' already exists.", client_id);
            return -1;
        }

        let runtime = Builder::new_multi_thread().enable_all().build().unwrap();
        let pool_config = PoolConfig {
            max_size: 10,
            connect_timeout: Duration::from_secs(5),
            get_timeout: Duration::from_secs(5),
        };
        let pool = ConnectionPool::new(server_addr, pool_config);

        let client = Arc::new(NetworkingClient { runtime, pool });
        manager.insert(client_id, client);
        0
    })
}

/// Destroys a networking client and frees its resources.
#[no_mangle]
pub extern "C" fn networking_client_destroy(client_id_c: *const c_char) {
    catch_panic(|| {
        let client_id = unsafe { CStr::from_ptr(client_id_c).to_str().unwrap() };
        let mut manager = CLIENT_MANAGER.lock().unwrap();
        if let Some(client) = manager.remove(client_id) {
            // Dropping the client will shut down its runtime and connection pool.
            drop(client);
        }
        0
    });
}

/// Sends a request to a server using the specified client and waits for a response.
///
/// # Arguments
/// - `client_id_c`: The ID of the client to use.
/// - `request_json_c`: A JSON string representing a `RequestPayload`.
///
/// # Returns
/// A C-string with the JSON representation of the `ResponsePayload`. Must be freed.
/// Returns null on error or timeout.
#[no_mangle]
pub extern "C" fn networking_client_send_request(
    client_id_c: *const c_char,
    request_json_c: *const c_char,
) -> *mut c_char {
    catch_panic(|| {
        let client_id = unsafe { CStr::from_ptr(client_id_c).to_str().unwrap() };
        let request_json = unsafe { CStr::from_ptr(request_json_c).to_str().unwrap() };

        let client = {
            let manager = CLIENT_MANAGER.lock().unwrap();
            manager.get(client_id).cloned().expect("Client not found")
        };

        // This is the bridge from sync (FFI) to async (Tokio).
        // `block_on` will run the async code and wait for it to complete.
        client.runtime.block_on(async {
            let request_payload: RequestPayload = serde_json::from_str(request_json).unwrap();
            let message = Message::Request(request_payload);

            let codec = MessageCodec::new();

            // Get a connection from the pool
            let mut conn = match client.pool.get().await {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Failed to get connection from pool: {}", e);
                    return std::ptr::null_mut();
                }
            };

            // Encode and send the request
            let mut write_buf = BytesMut::new();
            codec.encode(&message, &mut write_buf).unwrap();
            conn.write_all(&write_buf).await.unwrap();

            // Read the response
            let mut read_buf = BytesMut::with_capacity(1024);
            loop {
                if let Ok(Some(response_msg)) = codec.decode(&mut read_buf) {
                    if let Message::Response(payload) = response_msg {
                        return match serde_json::to_string(&payload) {
                            Ok(s) => CString::new(s).unwrap().into_raw(),
                            Err(_) => std::ptr::null_mut(),
                        };
                    }
                }

                if conn.read_buf(&mut read_buf).await.unwrap() == 0 {
                    // Connection closed
                    break;
                }
            }

            std::ptr::null_mut()
        })
    })
}

/// Frees a C-string that was allocated by this Rust library.
#[no_mangle]
pub extern "C" fn networking_free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    unsafe {
        drop(CString::from_raw(s));
    }
}

// --- Unit Tests ---
// Note: FFI tests are complex. These are basic sanity checks.
#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use tokio::net::TcpListener;

    async fn run_test_server() -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let codec = MessageCodec::new();
            let mut buf = BytesMut::new();
            stream.read_buf(&mut buf).await.unwrap();
            if let Ok(Some(Message::Request(RequestPayload::Echo(s)))) = codec.decode(&mut buf) {
                let response = Message::Response(ResponsePayload::Echo(s));
                let mut write_buf = BytesMut::new();
                codec.encode(&response, &mut write_buf).unwrap();
                stream.write_all(&write_buf).await.unwrap();
            }
        });
        addr
    }

    #[test]
    fn test_ffi_client_roundtrip() {
        let rt = Runtime::new().unwrap();
        let server_addr = rt.block_on(run_test_server());

        let client_id = CString::new("test_client").unwrap();
        let server_addr_c = CString::new(server_addr.to_string()).unwrap();

        let res = networking_client_create(server_addr_c.as_ptr(), client_id.as_ptr());
        assert_eq!(res, 0);

        let request_json = CString::new(r#"{"Echo":"hello"}"#).unwrap();

        // Give the server a moment to start
        thread::sleep(Duration::from_millis(100));

        let response_ptr = networking_client_send_request(client_id.as_ptr(), request_json.as_ptr());
        assert!(!response_ptr.is_null());

        let response_str = unsafe { CStr::from_ptr(response_ptr).to_str().unwrap() };
        assert_eq!(response_str, r#"{"Echo":"hello"}"#);

        networking_free_string(response_ptr);
        networking_client_destroy(client_id.as_ptr());
    }
}
