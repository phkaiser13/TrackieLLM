/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: protocol.rs
 *
 * This file defines the custom network protocol used for communication. It includes
 * the definition of message types and a codec for handling message framing. The
 * protocol uses a simple length-prefix framing mechanism over a binary payload
 * serialized with `bincode` for efficiency.
 *
 * Dependencies:
 *  - `serde`: For serializing and deserializing message structs.
 *  - `bincode`: For efficient binary serialization.
 *  - `bytes`: For high-performance buffer management.
 *  - `thiserror`: For structured error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Imports ---
use bytes::{Buf, BufMut, BytesMut};
use serde::{Deserialize, Serialize};

// --- Custom Error and Result Types ---

/// Represents errors that can occur during protocol encoding or decoding.
#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Bincode serialization/deserialization error: {0}")]
    Bincode(#[from] Box<bincode::ErrorKind>),
    #[error("Message exceeds maximum allowed size of {max_size} bytes. Message size: {actual_size}")]
    MessageTooLarge {
        max_size: usize,
        actual_size: usize,
    },
}

pub type ProtocolResult<T> = Result<T, ProtocolError>;

// --- Protocol Message Definitions ---

/// A top-level enum representing any message that can be sent over the network.
/// Using an enum provides a single, clear definition of the protocol's capabilities.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Message {
    Request(RequestPayload),
    Response(ResponsePayload),
    Heartbeat,
}

/// Represents the payload of a request message.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RequestPayload {
    Echo(String),
    GetStatus,
    // Add other request types here
}

/// Represents the payload of a response message.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ResponsePayload {
    Echo(String),
    Status(StatusInfo),
    Error(String),
}

/// A struct containing status information.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct StatusInfo {
    pub version: String,
    pub uptime_seconds: u64,
    pub active_connections: u32,
}


// --- Protocol Codec ---

const MAX_FRAME_SIZE: usize = 8 * 1024 * 1024; // 8 MB

/// A codec for handling the length-prefixed message framing.
/// This struct encapsulates the logic for converting between `Message` objects
/// and raw byte streams.
#[derive(Default)]
pub struct MessageCodec;

impl MessageCodec {
    pub fn new() -> Self {
        Self::default()
    }

    /// Encodes a `Message` into a `BytesMut` buffer.
    ///
    /// The frame format is: `[4-byte length header (u32, big-endian)] [bincode-serialized payload]`
    ///
    /// # Arguments
    /// * `msg` - The `Message` to encode.
    /// * `dst` - The destination buffer to write the framed message into.
    pub fn encode(&self, msg: &Message, dst: &mut BytesMut) -> ProtocolResult<()> {
        let encoded_payload = bincode::serialize(msg)?;
        let payload_len = encoded_payload.len();

        if payload_len > MAX_FRAME_SIZE {
            return Err(ProtocolError::MessageTooLarge {
                max_size: MAX_FRAME_SIZE,
                actual_size: payload_len,
            });
        }

        // Reserve space in the buffer and write the length header and payload.
        dst.reserve(4 + payload_len);
        dst.put_u32(payload_len as u32);
        dst.put_slice(&encoded_payload);

        Ok(())
    }

    /// Decodes a `Message` from a `BytesMut` buffer.
    ///
    /// This function checks if a complete frame is available in the buffer.
    /// If so, it consumes the frame, deserializes it, and returns the `Message`.
    /// If the frame is incomplete, it returns `Ok(None)`.
    ///
    /// # Arguments
    /// * `src` - The source buffer containing raw byte data from the network.
    ///
    /// # Returns
    /// - `Ok(Some(Message))` if a full message was decoded.
    /// - `Ok(None)` if the buffer does not yet contain a full message.
    /// - `Err(ProtocolError)` if a decoding error occurs.
    pub fn decode(&self, src: &mut BytesMut) -> ProtocolResult<Option<Message>> {
        // 1. Check if we have enough bytes to read the length header.
        if src.len() < 4 {
            return Ok(None);
        }

        // 2. Read the length header without advancing the buffer.
        let mut length_bytes = [0u8; 4];
        length_bytes.copy_from_slice(&src[..4]);
        let payload_len = u32::from_be_bytes(length_bytes) as usize;

        if payload_len > MAX_FRAME_SIZE {
            return Err(ProtocolError::MessageTooLarge {
                max_size: MAX_FRAME_SIZE,
                actual_size: payload_len,
            });
        }

        // 3. Check if the full frame (header + payload) has been received.
        if src.len() < 4 + payload_len {
            // Not enough data yet.
            return Ok(None);
        }

        // 4. We have a full frame, so consume the header and decode the payload.
        src.advance(4); // Consume the length header.
        let payload = src.split_to(payload_len); // Consume the payload.

        let msg = bincode::deserialize(&payload)?;

        Ok(Some(msg))
    }
}


// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let codec = MessageCodec::new();
        let message = Message::Request(RequestPayload::Echo("hello world".to_string()));

        let mut buffer = BytesMut::new();
        codec.encode(&message, &mut buffer).unwrap();

        assert!(buffer.len() > 4); // Should have header + payload

        let decoded_message = codec.decode(&mut buffer).unwrap();

        assert!(decoded_message.is_some());
        assert_eq!(decoded_message.unwrap(), message);
        assert!(buffer.is_empty()); // Buffer should be fully consumed.
    }

    #[test]
    fn test_decode_incomplete_frame() {
        let codec = MessageCodec::new();
        let message = Message::Heartbeat;

        let mut buffer = BytesMut::new();
        codec.encode(&message, &mut buffer).unwrap();

        // Create a partial buffer (missing the last byte)
        let mut partial_buffer = buffer.clone();
        partial_buffer.truncate(buffer.len() - 1);

        let result = codec.decode(&mut partial_buffer).unwrap();
        assert!(result.is_none()); // Should return None as it needs more data
    }

    #[test]
    fn test_decode_multiple_messages() {
        let codec = MessageCodec::new();
        let msg1 = Message::Request(RequestPayload::GetStatus);
        let msg2 = Message::Response(ResponsePayload::Error("test error".to_string()));

        let mut buffer = BytesMut::new();
        codec.encode(&msg1, &mut buffer).unwrap();
        codec.encode(&msg2, &mut buffer).unwrap();

        let decoded1 = codec.decode(&mut buffer).unwrap().unwrap();
        assert_eq!(decoded1, msg1);

        let decoded2 = codec.decode(&mut buffer).unwrap().unwrap();
        assert_eq!(decoded2, msg2);

        assert!(buffer.is_empty());
    }

    #[test]
    fn test_message_too_large_error() {
        // This is a bit tricky to test directly without creating a huge message.
        // We can simulate the error condition in the codec.
        let codec = MessageCodec::new();
        let mut buffer = BytesMut::new();

        // Simulate a huge length header
        let huge_len = MAX_FRAME_SIZE + 1;
        buffer.put_u32(huge_len as u32);

        let result = codec.decode(&mut buffer);
        assert!(matches!(result, Err(ProtocolError::MessageTooLarge { .. })));
    }
}
