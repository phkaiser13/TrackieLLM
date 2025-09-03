/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/networking/protocol.rs
 *
 * This file defines the application-level network protocol used for all
 * client-server communication. A well-defined protocol is essential for
 * reliable and efficient data exchange.
 *
 * The protocol is message-based and follows a simple Request-Response pattern.
 * Each message is framed to allow receivers to easily identify message
 * boundaries in a continuous TCP stream. The framing format is:
 *
 * [4-byte magic number][4-byte payload length][N-byte payload]
 *
 * - Magic Number: A constant value (0x54524B49, "TRKI") to quickly identify
 *   the start of a valid frame and discard corrupted data.
 * - Payload Length: A 32-bit unsigned integer in big-endian format, specifying
 *   the length of the payload that follows.
 * - Payload: The actual message data, serialized using a format like JSON
 *   or a more efficient binary format. This mock uses JSON for readability.
 *
 * Dependencies:
 *   - serde, serde_json: For serializing and deserializing the payload.
 *   - thiserror: For ergonomic error handling.
 *   - bytes: For efficient buffer handling during framing.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A magic number used to identify the start of a valid message frame.
/// "TRKI" in ASCII.
const FRAME_MAGIC_NUMBER: u32 = 0x54524B49;

/// Represents all possible errors that can occur during protocol handling.
#[derive(Debug, Error)]
pub enum ProtocolError {
    /// Failed to serialize a message into its byte representation.
    #[error("Serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),

    /// The received data could not be parsed into a valid message.
    #[error("Deserialization failed: a malformed message was received.")]
    Deserialization,

    /// The received frame has an invalid magic number.
    #[error("Invalid magic number in frame. Expected {expected:X}, got {got:X}")]
    InvalidMagicNumber {
        /// The expected magic number.
        expected: u32,
        /// The magic number that was received.
        got: u32,
    },

    /// The connection was closed while waiting for a complete frame.
    #[error("Connection closed prematurely while reading a frame.")]
    ConnectionClosed,
}

/// Represents a request sent from a client to a server.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Request {
    /// A simple health check to see if the server is responsive.
    Ping,
    /// A request to get the current status of a service.
    GetStatus,
    /// A request to execute a specific command with arguments.
    ExecuteCommand {
        /// The name of the command to execute.
        command: String,
        /// A list of arguments for the command.
        args: Vec<String>,
    },
}

/// Represents a response sent from a server back to a client.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Response {
    /// A positive acknowledgment, often in response to a `Ping`.
    Ack,
    /// A negative acknowledgment, indicating a request could not be processed.
    Nack {
        /// The reason for the failure.
        reason: String,
    },
    /// The current status of the service.
    Status {
        /// A string describing the service's status.
        status_message: String,
    },
    /// The result of a successfully executed command.
    CommandResult {
        /// The standard output from the command.
        stdout: String,
        /// The standard error output from the command.
        stderr: String,
        /// The exit code of the command.
        exit_code: i32,
    },
}

/// Serializes a `Request` and wraps it in a message frame.
///
/// # Arguments
/// * `request` - The request to be serialized.
///
/// # Returns
/// A `Vec<u8>` containing the complete, framed message ready for sending.
pub fn serialize_request(request: &Request) -> Result<Vec<u8>, ProtocolError> {
    // 1. Serialize the payload (the request object) into JSON.
    let payload = serde_json::to_vec(request)?;

    // 2. Construct the frame.
    let payload_len = payload.len() as u32;
    let mut frame = Vec::with_capacity(8 + payload.len());

    // Write the magic number (big-endian).
    frame.extend_from_slice(&FRAME_MAGIC_NUMBER.to_be_bytes());
    // Write the payload length (big-endian).
    frame.extend_from_slice(&payload_len.to_be_bytes());
    // Append the payload.
    frame.extend_from_slice(&payload);

    Ok(frame)
}

/// Parses a byte slice to extract the first complete message frame.
///
/// This function is designed to work with streaming data. It will return
/// information about how many bytes were consumed, allowing the caller to
/// handle buffers that may contain partial frames or multiple frames.
///
/// # Arguments
/// * `buffer` - A byte slice containing the raw data from the network.
///
/// # Returns
/// * `Ok(Some((Response, usize)))` - If a full frame was successfully parsed.
///   The tuple contains the parsed `Response` and the total number of bytes
///   consumed from the buffer for this frame.
/// * `Ok(None)` - If the buffer does not contain enough data for a full frame.
/// * `Err(ProtocolError)` - If the data is malformed (e.g., bad magic number).
pub fn parse_response(buffer: &[u8]) -> Result<Option<(Response, usize)>, ProtocolError> {
    // A frame needs at least 8 bytes for the header (magic + length).
    if buffer.len() < 8 {
        return Ok(None);
    }

    // 1. Read and verify the magic number.
    let mut magic_bytes = [0u8; 4];
    magic_bytes.copy_from_slice(&buffer[0..4]);
    let magic = u32::from_be_bytes(magic_bytes);

    if magic != FRAME_MAGIC_NUMBER {
        return Err(ProtocolError::InvalidMagicNumber {
            expected: FRAME_MAGIC_NUMBER,
            got: magic,
        });
    }

    // 2. Read the payload length.
    let mut len_bytes = [0u8; 4];
    len_bytes.copy_from_slice(&buffer[4..8]);
    let payload_len = u32::from_be_bytes(len_bytes) as usize;

    // 3. Check if the full frame is present in the buffer.
    let frame_len = 8 + payload_len;
    if buffer.len() < frame_len {
        return Ok(None); // Not enough data yet.
    }

    // 4. Deserialize the payload into a Response object.
    let payload = &buffer[8..frame_len];
    let response: Response =
        serde_json::from_slice(payload).map_err(|_| ProtocolError::Deserialization)?;

    Ok(Some((response, frame_len)))
}
