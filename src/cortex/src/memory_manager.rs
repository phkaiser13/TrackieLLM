/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/cortex/memory_manager.rs
 *
 * This file implements the `MemoryManager`, a high-level, Rust-native component
 * responsible for the AI's long-term memory. While the `ContextualReasoner`
 * handles short-term, situational awareness, the `MemoryManager` is designed
 * to store, retrieve, and synthesize information over extended periods.
 *
 * This component is crucial for enabling the AI to learn, remember user
 * preferences, and maintain a consistent personality. It acts as an abstraction
 * layer over a persistent storage backend (which is mocked in this
 * implementation).
 *
 * Key responsibilities:
 * - **Archiving**: Deciding which pieces of short-term context are important
 *   enough to be committed to long-term memory.
 * - **Summarization**: Condensing conversational history and events into
 *   succinct summaries or "key facts".
 * - **Retrieval**: Searching long-term memory for information relevant to the
 *   current context or a specific query.
 *
 * This implementation simulates a simple in-memory key-value store for memories,
 * but could be backed by a database or a vector store in a real application.
 *
 * Dependencies:
 *   - log: For structured logging.
 *   - thiserror: For ergonomic error handling.
 *   - chrono: For timestamping memories.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use chrono::{DateTime, Utc};
use std::collections::HashMap;
use thiserror::Error;

/// Represents the different types of long-term memories that can be stored.
#[derive(Debug, Clone)]
pub enum MemoryType {
    /// A key fact about the user, the environment, or the AI itself.
    Fact,
    /// A summary of a past conversation or event.
    Episodic,
    /// A user preference.
    Preference,
}

/// A single, retrievable piece of information in long-term memory.
#[derive(Debug, Clone)]
pub struct MemoryFragment {
    /// A unique identifier for this memory.
    pub id: u64,
    /// The type of memory.
    pub memory_type: MemoryType,
    /// The timestamp when the memory was created or last updated.
    pub timestamp: DateTime<Utc>,
    /// The content of the memory.
    pub content: String,
    /// A list of keywords or tags for efficient retrieval.
    pub keywords: Vec<String>,
    /// A score indicating the importance or relevance of the memory.
    pub importance: f32,
}

// --- Persistence Layer Design ---

/// User-specific preferences that can be saved and loaded.
#[derive(Debug, Clone)]
pub struct UserPreferences {
    /// The user's preferred language for TTS.
    pub language: String,
    /// The default level of detail for descriptions (0.0 to 1.0).
    pub detail_level: f32,
    // Add other preferences here, e.g., voice selection.
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            language: "en-US".to_string(),
            detail_level: 0.5,
        }
    }
}

/// Represents a location known to the system, with associated memories.
#[derive(Debug, Clone)]
pub struct KnownPlace {
    pub name: String,
    /// A list of memory fragment IDs associated with this place.
    pub memory_ids: Vec<u64>,
}

/// Container for all data that is persisted to disk.
#[derive(Debug, Clone, Default)]
pub struct LongTermMemory {
    pub preferences: UserPreferences,
    pub known_places: Vec<KnownPlace>,
}

/// Represents errors that can occur within the Memory Manager.
#[derive(Debug, Error)]
pub enum MemoryError {
    /// The requested memory could not be found.
    #[error("Memory fragment with ID {0} not found.")]
    NotFound(u64),
    /// An error occurred during the storage or retrieval process.
    #[error("Storage backend failed: {0}")]
    StorageFailed(String),
}

/// Manages very recent events to avoid repetition (e.g., repeated alerts).
#[derive(Debug, Clone, Default)]
pub struct ShortTermMemory {
    /// Maps an alert key to the last time it was triggered.
    recent_alerts: HashMap<String, DateTime<Utc>>,
}

impl ShortTermMemory {
    /// Records that an alert has just been triggered.
    pub fn record_alert(&mut self, alert_key: String) {
        self.recent_alerts.insert(alert_key, Utc::now());
    }

    /// Checks if an alert with the given key has been triggered within the
    /// specified duration.
    pub fn has_been_alerted_recently(&self, alert_key: &str, duration_secs: i64) -> bool {
        if let Some(last_alert_time) = self.recent_alerts.get(alert_key) {
            let now = Utc::now();
            let duration = chrono::Duration::seconds(duration_secs);
            return now.signed_duration_since(*last_alert_time) < duration;
        }
        false
    }

    /// Records that an object has been mentioned to the user.
    pub fn record_object_mentioned(&mut self, object_id: u64) {
        let key = format!("object_mentioned_{}", object_id);
        self.recent_alerts.insert(key, Utc::now());
    }

    /// Checks if an object has been mentioned within the specified duration.
    pub fn was_object_mentioned_recently(&self, object_id: u64, duration_secs: i64) -> bool {
        let key = format!("object_mentioned_{}", object_id);
        self.has_been_alerted_recently(&key, duration_secs)
    }
}

/// The main service for managing the AI's long-term memory.
pub struct MemoryManager {
    /// The in-memory storage for memory fragments.
    /// The key is the memory ID. In a real implementation, this would be
    /// an interface to a persistent database.
    storage: HashMap<u64, MemoryFragment>,
    /// The short-term memory component for handling recent events.
    pub short_term_memory: ShortTermMemory,
    /// The container for all data that gets saved to/loaded from disk.
    long_term_memory: LongTermMemory,
    /// A simple counter to generate unique memory IDs.
    next_id: u64,
}

impl MemoryManager {
    /// Creates a new, empty `MemoryManager`.
    pub fn new() -> Self {
        log::info!("Initializing new Memory Manager.");
        Self {
            storage: HashMap::new(),
            short_term_memory: ShortTermMemory::default(),
            long_term_memory: LongTermMemory::default(),
            next_id: 1,
        }
    }

    /// (Design) Saves the long-term memory to a file.
    ///
    /// In a real implementation, this would serialize the `LongTermMemory`
    /// struct (likely as JSON or another format) and write it to disk.
    pub fn save_to_disk(&self, _path: &std::path::Path) -> Result<(), MemoryError> {
        log::info!("Simulating save of long-term memory to disk.");
        // TODO: Implement serialization (e.g., with serde_json) and file I/O.
        Ok(())
    }

    /// (Design) Loads the long-term memory from a file.
    ///
    /// In a real implementation, this would read from a file, deserialize
    /// the data into the `LongTermMemory` struct, and populate the manager's state.
    pub fn load_from_disk(&mut self, _path: &std::path::Path) -> Result<(), MemoryError> {
        log::info!("Simulating load of long-term memory from disk.");
        // TODO: Implement file reading and deserialization.
        Ok(())
    }

    /// Archives a new piece of information into long-term memory.
    ///
    /// # Arguments
    /// * `memory_type` - The category of the new memory.
    /// * `content` - The textual content of the memory.
    /// * `keywords` - A list of keywords for future retrieval.
    /// * `importance` - A score from 0.0 to 1.0 indicating the memory's importance.
    ///
    /// # Returns
    /// The unique ID of the newly created memory fragment.
    pub fn archive_memory(
        &mut self,
        memory_type: MemoryType,
        content: String,
        keywords: Vec<String>,
        importance: f32,
    ) -> Result<u64, MemoryError> {
        let id = self.next_id;
        let memory = MemoryFragment {
            id,
            memory_type,
            timestamp: Utc::now(),
            content,
            keywords,
            importance,
        };

        log::debug!("Archiving new memory (ID: {}) with importance {:.2}", id, importance);
        
        // Mock storage operation.
        self.storage.insert(id, memory);
        self.next_id += 1;

        Ok(id)
    }

    /// Retrieves a memory fragment by its unique ID.
    pub fn retrieve_by_id(&self, id: u64) -> Result<&MemoryFragment, MemoryError> {
        self.storage
            .get(&id)
            .ok_or(MemoryError::NotFound(id))
    }

    /// Searches long-term memory for fragments matching a set of keywords.
    ///
    /// This is a simplified mock of a memory retrieval system. A real implementation
    /// would use more sophisticated techniques like vector similarity search.
    ///
    /// # Arguments
    /// * `query_keywords` - A slice of keywords to search for.
    ///
    /// # Returns
    /// A `Vec` of matching `MemoryFragment`s, sorted by importance.
    pub fn retrieve_relevant_memories(&self, query_keywords: &[&str]) -> Vec<&MemoryFragment> {
        log::debug!("Retrieving memories relevant to keywords: {:?}", query_keywords);
        
        let mut relevant_memories: Vec<_> = self
            .storage
            .values()
            .filter(|fragment| {
                query_keywords
                    .iter()
                    .any(|query_kw| fragment.keywords.iter().any(|mem_kw| mem_kw == *query_kw))
            })
            .collect();

        // Sort the results by importance, descending.
        relevant_memories.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());

        relevant_memories
    }

    /// Forgets a memory fragment, permanently removing it from storage.
    pub fn forget_memory(&mut self, id: u64) -> Result<(), MemoryError> {
        if self.storage.remove(&id).is_some() {
            log::info!("Forgot memory fragment with ID: {}", id);
            Ok(())
        } else {
            Err(MemoryError::NotFound(id))
        }
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}
