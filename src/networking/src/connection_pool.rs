/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: connection_pool.rs
 *
 * This file implements an asynchronous, high-performance TCP connection pool.
 * Connection pooling is a critical pattern for network clients, as it avoids the
 * high latency and resource cost of establishing a new TCP connection for every
 * request. This implementation uses `tokio` for async I/O and features a
 * smart-pointer-based RAII pattern for safe, ergonomic connection handling.
 *
 * Dependencies:
 *  - `tokio`: For async TCP streams and synchronization primitives.
 *  - `async-trait`: For using async functions in traits.
 *  - `thiserror`: For structured error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Imports ---
use std::net::SocketAddr;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::net::TcpStream;
use tokio::sync::{mpsc, Mutex, Semaphore};
use tokio::time::timeout;

// --- Custom Error and Result Types ---

#[derive(Error, Debug)]
pub enum PoolError {
    #[error("Failed to establish a new TCP connection: {0}")]
    ConnectionError(#[from] std::io::Error),
    #[error("Timed out waiting for an available connection from the pool.")]
    Timeout,
    #[error("The connection pool has been closed and cannot provide new connections.")]
    PoolClosed,
}

pub type PoolResult<T> = Result<T, PoolError>;

// --- Pool Configuration and Structures ---

/// Configuration for the connection pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub max_size: usize,
    pub connect_timeout: Duration,
    pub get_timeout: Duration,
}

/// A wrapper around a `TcpStream` that includes pool-related metadata.
struct ManagedConnection {
    stream: TcpStream,
    created_at: Instant,
}

/// The core state of the connection pool, shared across all users of the pool.
struct PoolInner {
    addr: SocketAddr,
    config: PoolConfig,
    conns: mpsc::Sender<ManagedConnection>,
    // A semaphore is used to limit the total number of connections (both in the
    // pool and currently checked out). This is a more robust way to control
    // concurrency than a simple atomic counter.
    semaphore: Arc<Semaphore>,
}

/// The public-facing connection pool handle.
/// It is a lightweight wrapper that can be cloned to share access to the pool.
#[derive(Clone)]
pub struct ConnectionPool {
    inner: Arc<PoolInner>,
}

impl ConnectionPool {
    /// Creates a new `ConnectionPool`.
    pub fn new(addr: SocketAddr, config: PoolConfig) -> Self {
        let (tx, mut rx) = mpsc::channel(config.max_size);

        let pool = ConnectionPool {
            inner: Arc::new(PoolInner {
                addr,
                config,
                conns: tx,
                semaphore: Arc::new(Semaphore::new(config.max_size)),
            }),
        };

        // This is where you might spawn a background task to maintain a minimum
        // number of connections or to clean up idle ones. For simplicity in this
        // example, we'll focus on the core get/put logic.

        pool
    }

    /// Acquires a connection from the pool.
    ///
    /// This method will first try to get an idle connection from the pool. If none
    /// are available, it will attempt to create a new one, provided the maximum
    /// pool size has not been reached. If the pool is at maximum capacity, it will
    /// wait for a connection to be returned.
    pub async fn get(&self) -> PoolResult<PoolableConnection> {
        let permit = timeout(self.inner.config.get_timeout, self.inner.semaphore.clone().acquire_owned())
            .await
            .map_err(|_| PoolError::Timeout)?
            .expect("Semaphore should not be closed");

        // We have a permit, so we are allowed to have one connection.
        // First, try to get one from the idle queue.
        if let Ok(conn) = self.inner.conns.send_timeout(ManagedConnection{stream: TcpStream::connect(self.inner.addr).await?, created_at: Instant::now()}, self.inner.config.connect_timeout).await {
            return Ok(PoolableConnection {
                conn: Some(conn),
                pool: self.clone(),
                permit: Some(permit),
            });
        }

        // If the queue was empty, create a new connection.
        match timeout(self.inner.config.connect_timeout, TcpStream::connect(self.inner.addr)).await {
            Ok(Ok(stream)) => {
                let conn = ManagedConnection {
                    stream,
                    created_at: Instant::now(),
                };
                Ok(PoolableConnection {
                    conn: Some(conn),
                    pool: self.clone(),
                    permit: Some(permit),
                })
            }
            Ok(Err(e)) => Err(PoolError::ConnectionError(e)),
            Err(_) => Err(PoolError::Timeout),
        }
    }

    /// Returns a connection to the pool. This is typically called by `PoolableConnection`'s `Drop` impl.
    async fn put(&self, conn: ManagedConnection) {
        if let Err(_) = self.inner.conns.send(conn).await {
            // This happens if the receiver side of the channel is dropped,
            // which means the pool is being shut down.
            println!("Could not return connection to closed pool.");
        }
    }
}

/// A smart-pointer-like struct that holds a connection from the pool.
/// When this struct is dropped, the connection is automatically returned to the pool.
/// This RAII pattern makes the pool much safer and easier to use.
pub struct PoolableConnection {
    conn: Option<ManagedConnection>,
    pool: ConnectionPool,
    // The permit is held for the lifetime of the connection. When dropped,
    // it releases its spot in the semaphore.
    permit: Option<tokio::sync::OwnedSemaphorePermit>,
}

impl std::ops::Deref for PoolableConnection {
    type Target = TcpStream;
    fn deref(&self) -> &Self::Target {
        &self.conn.as_ref().unwrap().stream
    }
}

impl std::ops::DerefMut for PoolableConnection {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.conn.as_mut().unwrap().stream
    }
}

impl Drop for PoolableConnection {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            let pool = self.pool.clone();
            // We need to run the `put` async function, but `drop` is synchronous.
            // Spawning a task is the standard way to handle this.
            tokio::spawn(async move {
                pool.put(conn).await;
            });
        }
        // The permit is dropped here automatically, releasing the semaphore.
    }
}


// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    async fn setup_test_server() -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                if let Ok((mut socket, _)) = listener.accept().await {
                    tokio::spawn(async move {
                        let mut buf = [0; 4];
                        if let Ok(_) = socket.read_exact(&mut buf).await {
                            // Echo back the received bytes.
                            socket.write_all(&buf).await.unwrap();
                        }
                    });
                }
            }
        });
        addr
    }

    #[tokio::test]
    async fn test_pool_get_and_put() {
        let server_addr = setup_test_server().await;
        let config = PoolConfig {
            max_size: 2,
            connect_timeout: Duration::from_secs(1),
            get_timeout: Duration::from_secs(1),
        };
        let pool = ConnectionPool::new(server_addr, config);

        {
            let mut conn = pool.get().await.unwrap();
            // Test the connection
            conn.write_all(b"test").await.unwrap();
            let mut buf = [0; 4];
            conn.read_exact(&mut buf).await.unwrap();
            assert_eq!(&buf, b"test");
        } // conn is dropped here and returned to the pool.

        // Get another connection, which should be the one we just returned.
        let _conn2 = pool.get().await.unwrap();
    }

    #[tokio::test]
    async fn test_pool_limit() {
        let server_addr = setup_test_server().await;
        let config = PoolConfig {
            max_size: 1, // Only one connection allowed
            connect_timeout: Duration::from_secs(1),
            get_timeout: Duration::from_millis(100),
        };
        let pool = ConnectionPool::new(server_addr, config);

        // Get the only available connection
        let _conn1 = pool.get().await.unwrap();

        // Try to get another one, this should time out.
        let result = pool.get().await;
        assert!(matches!(result, Err(PoolError::Timeout)));
    }
}
