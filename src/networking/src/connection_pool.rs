/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/networking/connection_pool.rs
 *
 * This file implements a connection pool for managing reusable network
 * connections. Establishing a new TCP connection is a relatively expensive
 * operation, involving a three-way handshake and system resource allocation.
 * A connection pool mitigates this cost by maintaining a cache of active,
 * idle connections that can be reused for subsequent requests to the same host.
 *
 * The `ConnectionPool` is designed for an asynchronous environment (like `tokio`)
 * and uses mutexes to ensure thread-safe access to its internal state. It
 * manages connections on a per-host basis, allowing it to handle communication
 * with multiple different servers simultaneously.
 *
 * Key features:
 * - Manages a pool of connections for each remote address.
 * - Limits the maximum number of connections per host to prevent resource exhaustion.
 * - Asynchronously handles connection acquisition and release.
 *
 * Dependencies:
 *   - tokio: For `net::TcpStream` and `sync::Mutex`.
 *   - log: For structured logging.
 *   - thiserror: For ergonomic error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use thiserror::Error;
use tokio::net::TcpStream;
use tokio::sync::Mutex;

// A type alias for clarity. In a real implementation, this might be a
// more complex struct that includes read/write buffers.
type Connection = TcpStream;

/// Represents errors that can occur within the connection pool.
#[derive(Debug, Error)]
pub enum ConnectionPoolError {
    /// The specified host address could not be resolved or connected to.
    #[error("Failed to connect to host '{host}': {source}")]
    ConnectionFailed {
        host: String,
        #[source]
        source: std::io::Error,
    },
    /// The pool has reached its maximum capacity and no connections are available.
    #[error("Connection pool timed out waiting for an available connection to '{host}'.")]
    PoolTimedOut {
        host: String,
    },
}

/// A thread-safe, asynchronous connection pool.
///
/// This struct manages a collection of TCP connections, keyed by their
/// remote address (e.g., "hostname:port").
#[derive(Clone)]
pub struct ConnectionPool {
    /// The internal state of the pool, protected by a Mutex for thread safety.
    state: Arc<Mutex<PoolState>>,
    /// The maximum number of connections allowed per host.
    max_conns_per_host: usize,
}

/// The internal, mutable state of the connection pool.
struct PoolState {
    /// A map where keys are host addresses and values are queues of idle connections.
    connections: HashMap<String, VecDeque<Connection>>,
}

impl ConnectionPool {
    /// Creates a new `ConnectionPool`.
    ///
    /// # Arguments
    ///
    /// * `max_conns_per_host` - The maximum number of idle connections to keep
    ///   for each unique host address.
    pub fn new(max_conns_per_host: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(PoolState {
                connections: HashMap::new(),
            })),
            max_conns_per_host,
        }
    }

    /// Acquires a connection from the pool for the specified address.
    ///
    /// This method first attempts to reuse an existing idle connection. If none
    /// are available, it will create a new one, provided the pool is not at
    /// its maximum capacity for that host.
    ///
    /// # Arguments
    ///
    /// * `addr` - The network address (e.g., "127.0.0.1:8080") to connect to.
    #[allow(dead_code)] // This is a mock, so the function is not used yet.
    pub async fn get(&self, addr: &str) -> Result<Connection, ConnectionPoolError> {
        let mut state = self.state.lock().await;

        let queue = state.connections.entry(addr.to_string()).or_default();

        // If there's an idle connection, reuse it.
        if let Some(conn) = queue.pop_front() {
            log::debug!("Reusing existing connection to {}", addr);
            return Ok(conn);
        }

        // If no idle connection is available, create a new one if we are
        // not at the limit.
        // A real implementation would also track the number of *active* connections.
        // This mock simplifies by just checking the idle queue size.
        if queue.len() < self.max_conns_per_host {
            log::info!("No idle connections to {}. Creating a new one.", addr);
            let conn = TcpStream::connect(addr).await.map_err(|e| {
                ConnectionPoolError::ConnectionFailed {
                    host: addr.to_string(),
                    source: e,
                }
            })?;
            return Ok(conn);
        }

        // In a real implementation, if the pool is full, we would wait for a
        // connection to be released, possibly with a timeout.
        // For this mock, we'll just return an error.
        Err(ConnectionPoolError::PoolTimedOut {
            host: addr.to_string(),
        })
    }

    /// Releases a connection back into the pool.
    ///
    /// The connection is added to the idle queue for its address, making it
    /// available for future requests.
    ///
    /// # Arguments
    ///
    /// * `addr` - The address associated with the connection.
    /// * `conn` - The `Connection` object to be returned to the pool.
    #[allow(dead_code)] // This is a mock, so the function is not used yet.
    pub async fn release(&self, addr: &str, conn: Connection) {
        let mut state = self.state.lock().await;

        let queue = state.connections.entry(addr.to_string()).or_default();

        if queue.len() < self.max_conns_per_host {
            log::debug!("Releasing connection to {} back to the pool.", addr);
            queue.push_back(conn);
        } else {
            log::debug!(
                "Connection pool for {} is full. Dropping released connection.",
                addr
            );
            // The connection is simply dropped here, closing it.
        }
    }
}
