/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: secure_channels.rs
 *
 * This module implements secure communication channels for the TrackieLLM project.
 * It provides encrypted communication channels between different components of the
 * system and handles secure network communication when needed.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::key_management::{EncryptionContext, KeyManagementError};
use crate::ffi::TkErrorCode;

/// Secure channel identifier
pub type ChannelId = u32;

/// Secure channel error types
#[derive(Debug, Clone, PartialEq)]
pub enum ChannelError {
    InvalidChannel,
    EncryptionFailed,
    DecryptionFailed,
    KeyExchangeFailed,
    NetworkError,
    InvalidMessage,
}

impl From<KeyManagementError> for ChannelError {
    fn from(error: KeyManagementError) -> Self {
        match error {
            KeyManagementError::CryptoError => ChannelError::EncryptionFailed,
            _ => ChannelError::KeyExchangeFailed,
        }
    }
}

impl From<ChannelError> for TkErrorCode {
    fn from(error: ChannelError) -> Self {
        match error {
            ChannelError::InvalidChannel => TkErrorCode::InvalidArgument,
            ChannelError::EncryptionFailed => TkErrorCode::Internal,
            ChannelError::DecryptionFailed => TkErrorCode::DecryptionFailed,
            ChannelError::KeyExchangeFailed => TkErrorCode::Internal,
            ChannelError::NetworkError => TkErrorCode::NetworkError,
            ChannelError::InvalidMessage => TkErrorCode::InvalidArgument,
        }
    }
}

/// Message types for secure channels
#[derive(Debug, Clone)]
pub enum MessageType {
    Data,
    KeyExchange,
    Heartbeat,
    Control,
}

/// Secure message structure
#[derive(Debug, Clone)]
pub struct SecureMessage {
    pub msg_type: MessageType,
    pub channel_id: ChannelId,
    pub payload: Vec<u8>,
    pub timestamp: u64,
}

/// Secure channel configuration
#[derive(Debug, Clone)]
pub struct ChannelConfig {
    pub channel_id: ChannelId,
    pub encryption_enabled: bool,
    pub compression_enabled: bool,
    pub max_message_size: usize,
    pub timeout_seconds: u32,
}

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            channel_id: 0,
            encryption_enabled: true,
            compression_enabled: false,
            max_message_size: 1024 * 1024, // 1MB
            timeout_seconds: 30,
        }
    }
}

/// Secure communication channel
pub struct SecureChannel {
    config: ChannelConfig,
    encryption_ctx: Option<EncryptionContext>,
    is_active: bool,
    message_counter: u64,
}

impl SecureChannel {
    /// Creates a new secure channel
    pub fn new(config: ChannelConfig) -> Result<Self, ChannelError> {
        let encryption_ctx = if config.encryption_enabled {
            Some(EncryptionContext::new()?)
        } else {
            None
        };

        Ok(Self {
            config,
            encryption_ctx,
            is_active: false,
            message_counter: 0,
        })
    }

    /// Activates the channel with a master key
    pub fn activate(&mut self) -> Result<(), ChannelError> {
        if let Some(ref mut ctx) = self.encryption_ctx {
            ctx.generate_key()?;
        }
        
        self.is_active = true;
        Ok(())
    }

    /// Deactivates the channel
    pub fn deactivate(&mut self) {
        self.is_active = false;
    }

    /// Sends a message through the channel
    pub fn send_message(&mut self, msg_type: MessageType, payload: &[u8]) -> Result<Vec<u8>, ChannelError> {
        if !self.is_active {
            return Err(ChannelError::InvalidChannel);
        }

        if payload.len() > self.config.max_message_size {
            return Err(ChannelError::InvalidMessage);
        }

        let message = SecureMessage {
            msg_type,
            channel_id: self.config.channel_id,
            payload: payload.to_vec(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        self.message_counter += 1;
        
        // Serialize and optionally encrypt the message
        self.serialize_and_encrypt_message(&message)
    }

    /// Receives and processes a message
    pub fn receive_message(&self, encrypted_data: &[u8]) -> Result<SecureMessage, ChannelError> {
        if !self.is_active {
            return Err(ChannelError::InvalidChannel);
        }

        self.decrypt_and_deserialize_message(encrypted_data)
    }

    /// Serializes and encrypts a message
    fn serialize_and_encrypt_message(&self, message: &SecureMessage) -> Result<Vec<u8>, ChannelError> {
        // Simple serialization (in a real implementation, you'd use proper serialization)
        let serialized = self.serialize_message(message);

        if let Some(ref ctx) = self.encryption_ctx {
            ctx.encrypt(&serialized).map_err(ChannelError::from)
        } else {
            Ok(serialized)
        }
    }

    /// Decrypts and deserializes a message
    fn decrypt_and_deserialize_message(&self, data: &[u8]) -> Result<SecureMessage, ChannelError> {
        let decrypted = if let Some(ref ctx) = self.encryption_ctx {
            ctx.decrypt(data).map_err(ChannelError::from)?
        } else {
            data.to_vec()
        };

        self.deserialize_message(&decrypted)
    }

    /// Simple message serialization (placeholder implementation)
    fn serialize_message(&self, message: &SecureMessage) -> Vec<u8> {
        // This is a simplified implementation
        // In production, you'd use proper serialization like bincode, protobuf, etc.
        let mut serialized = Vec::new();
        
        // Message type (1 byte)
        serialized.push(match message.msg_type {
            MessageType::Data => 0,
            MessageType::KeyExchange => 1,
            MessageType::Heartbeat => 2,
            MessageType::Control => 3,
        });

        // Channel ID (4 bytes)
        serialized.extend_from_slice(&message.channel_id.to_le_bytes());
        
        // Timestamp (8 bytes)
        serialized.extend_from_slice(&message.timestamp.to_le_bytes());
        
        // Payload length (4 bytes)
        serialized.extend_from_slice(&(message.payload.len() as u32).to_le_bytes());
        
        // Payload
        serialized.extend_from_slice(&message.payload);

        serialized
    }

    /// Simple message deserialization (placeholder implementation)
    fn deserialize_message(&self, data: &[u8]) -> Result<SecureMessage, ChannelError> {
        if data.len() < 17 { // Minimum size: 1 + 4 + 8 + 4
            return Err(ChannelError::InvalidMessage);
        }

        let mut offset = 0;

        // Message type
        let msg_type = match data[offset] {
            0 => MessageType::Data,
            1 => MessageType::KeyExchange,
            2 => MessageType::Heartbeat,
            3 => MessageType::Control,
            _ => return Err(ChannelError::InvalidMessage),
        };
        offset += 1;

        // Channel ID
        let channel_id = u32::from_le_bytes([
            data[offset], data[offset + 1], data[offset + 2], data[offset + 3]
        ]);
        offset += 4;

        // Timestamp
        let timestamp = u64::from_le_bytes([
            data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
            data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7]
        ]);
        offset += 8;

        // Payload length
        let payload_len = u32::from_le_bytes([
            data[offset], data[offset + 1], data[offset + 2], data[offset + 3]
        ]) as usize;
        offset += 4;

        // Validate payload length
        if offset + payload_len != data.len() {
            return Err(ChannelError::InvalidMessage);
        }

        // Payload
        let payload = data[offset..offset + payload_len].to_vec();

        Ok(SecureMessage {
            msg_type,
            channel_id,
            payload,
            timestamp,
        })
    }
}

/// Secure channel manager
pub struct SecureChannelManager {
    channels: Arc<Mutex<HashMap<ChannelId, SecureChannel>>>,
    next_channel_id: Arc<Mutex<ChannelId>>,
}

impl SecureChannelManager {
    /// Creates a new channel manager
    pub fn new() -> Self {
        Self {
            channels: Arc::new(Mutex::new(HashMap::new())),
            next_channel_id: Arc::new(Mutex::new(1)),
        }
    }

    /// Creates a new secure channel
    pub fn create_channel(&self, mut config: ChannelConfig) -> Result<ChannelId, ChannelError> {
        let channel_id = {
            let mut next_id = self.next_channel_id.lock().unwrap();
            let id = *next_id;
            *next_id += 1;
            id
        };

        config.channel_id = channel_id;
        let mut channel = SecureChannel::new(config)?;
        channel.activate()?;

        {
            let mut channels = self.channels.lock().unwrap();
            channels.insert(channel_id, channel);
        }

        Ok(channel_id)
    }

    /// Removes a secure channel
    pub fn remove_channel(&self, channel_id: ChannelId) -> Result<(), ChannelError> {
        let mut channels = self.channels.lock().unwrap();
        if let Some(mut channel) = channels.remove(&channel_id) {
            channel.deactivate();
            Ok(())
        } else {
            Err(ChannelError::InvalidChannel)
        }
    }

    /// Sends a message through a specific channel
    pub fn send_message(&self, channel_id: ChannelId, msg_type: MessageType, payload: &[u8]) -> Result<Vec<u8>, ChannelError> {
        let mut channels = self.channels.lock().unwrap();
        if let Some(channel) = channels.get_mut(&channel_id) {
            channel.send_message(msg_type, payload)
        } else {
            Err(ChannelError::InvalidChannel)
        }
    }

    /// Receives a message from a specific channel
    pub fn receive_message(&self, channel_id: ChannelId, encrypted_data: &[u8]) -> Result<SecureMessage, ChannelError> {
        let channels = self.channels.lock().unwrap();
        if let Some(channel) = channels.get(&channel_id) {
            channel.receive_message(encrypted_data)
        } else {
            Err(ChannelError::InvalidChannel)
        }
    }

    /// Gets the number of active channels
    pub fn get_active_channel_count(&self) -> usize {
        let channels = self.channels.lock().unwrap();
        channels.values().filter(|c| c.is_active).count()
    }

    /// Lists all active channel IDs
    pub fn get_active_channels(&self) -> Vec<ChannelId> {
        let channels = self.channels.lock().unwrap();
        channels
            .iter()
            .filter(|(_, channel)| channel.is_active)
            .map(|(id, _)| *id)
            .collect()
    }
}

impl Default for SecureChannelManager {
    fn default() -> Self {
        Self::new()
    }
}

// Global channel manager instance
static mut GLOBAL_CHANNEL_MANAGER: Option<SecureChannelManager> = None;
static INIT_MANAGER: std::sync::Once = std::sync::Once::new();

/// Gets the global channel manager instance
pub fn get_global_channel_manager() -> &'static SecureChannelManager {
    INIT_MANAGER.call_once(|| {
        unsafe {
            GLOBAL_CHANNEL_MANAGER = Some(SecureChannelManager::new());
        }
    });

    unsafe { GLOBAL_CHANNEL_MANAGER.as_ref().unwrap() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_creation() {
        let config = ChannelConfig::default();
        let channel = SecureChannel::new(config);
        assert!(channel.is_ok());
    }

    #[test]
    fn test_channel_activation() {
        let config = ChannelConfig::default();
        let mut channel = SecureChannel::new(config).unwrap();
        let result = channel.activate();
        assert!(result.is_ok());
        assert!(channel.is_active);
    }

    #[test]
    fn test_message_serialization() {
        let config = ChannelConfig {
            encryption_enabled: false,
            ..Default::default()
        };
        
        let channel = SecureChannel::new(config).unwrap();
        
        let message = SecureMessage {
            msg_type: MessageType::Data,
            channel_id: 1,
            payload: b"test message".to_vec(),
            timestamp: 1234567890,
        };

        let serialized = channel.serialize_message(&message);
        let deserialized = channel.deserialize_message(&serialized).unwrap();

        assert_eq!(deserialized.channel_id, message.channel_id);
        assert_eq!(deserialized.payload, message.payload);
        assert_eq!(deserialized.timestamp, message.timestamp);
    }

    #[test]
    fn test_channel_manager() {
        let manager = SecureChannelManager::new();
        let config = ChannelConfig::default();
        
        let channel_id = manager.create_channel(config).unwrap();
        assert!(channel_id > 0);
        
        let count = manager.get_active_channel_count();
        assert_eq!(count, 1);
        
        let result = manager.remove_channel(channel_id);
        assert!(result.is_ok());
        
        let count = manager.get_active_channel_count();
        assert_eq!(count, 0);
    }
}