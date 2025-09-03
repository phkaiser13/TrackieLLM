/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: memory_manager.rs
 *
 * This file implements the memory management system for the AI's "cortex". It
 * provides a simplified vector store for storing and retrieving "memory fragments"
 * based on semantic similarity. This is a foundational component for Retrieval-
 * Augmented Generation (RAG) systems, allowing the AI to recall relevant
 * information when reasoning.
 *
 * Dependencies:
 *  - `ndarray`: For efficient vector (embedding) operations and similarity calculations.
 *  - `thiserror`: For structured error handling.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Imports ---
use ndarray::{Array1, Axis, linalg::Dot};
use std::collections::HashMap;
use uuid::Uuid;

// --- Custom Error and Result Types ---

#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("A memory fragment with ID '{0}' already exists.")]
    DuplicateMemoryId(Uuid),
    #[error("Embedding vector has a dimension of zero, which is invalid.")]
    ZeroDimensionEmbedding,
}

pub type MemoryResult<T> = Result<T, MemoryError>;

// --- Core Data Structures ---

/// Represents a single, addressable piece of information in the AI's memory.
#[derive(Debug, Clone)]
pub struct MemoryFragment {
    pub id: Uuid,
    pub text_content: String,
    pub embedding: Array1<f32>,
    pub metadata: HashMap<String, String>,
}

/// A search result, containing a reference to a memory fragment and its similarity score.
#[derive(Debug, Clone)]
pub struct MemorySearchResult<'a> {
    pub fragment: &'a MemoryFragment,
    pub similarity_score: f32,
}

// --- Memory Management Service ---

/// Manages the storage and retrieval of memory fragments.
/// This acts as a simple in-memory vector database.
#[derive(Default)]
pub struct MemoryManager {
    // A simple Vec-based store. For larger applications, this could be replaced
    // with a more sophisticated data structure like an HNSW index for faster lookups.
    fragments: Vec<MemoryFragment>,
    // A HashMap for quick lookups by ID.
    id_map: HashMap<Uuid, usize>, // Maps ID to index in the `fragments` Vec
}

impl MemoryManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a new memory fragment to the store.
    ///
    /// # Arguments
    /// * `text_content` - The raw text of the memory.
    /// * `embedding` - The vector embedding representing the text's semantics.
    ///
    /// # Returns
    /// The unique ID of the newly created memory fragment.
    pub fn add_memory(&mut self, text_content: String, embedding: Vec<f32>) -> MemoryResult<Uuid> {
        let id = Uuid::new_v4();
        let embedding_array = Array1::from(embedding);

        let fragment = MemoryFragment {
            id,
            text_content,
            embedding: embedding_array,
            metadata: HashMap::new(),
        };

        if self.id_map.contains_key(&id) {
            return Err(MemoryError::DuplicateMemoryId(id));
        }

        self.fragments.push(fragment);
        self.id_map.insert(id, self.fragments.len() - 1);

        Ok(id)
    }

    /// Finds the `top_k` most similar memory fragments to a given query embedding.
    ///
    /// # Arguments
    /// * `query_embedding` - The vector embedding of the search query.
    /// * `top_k` - The maximum number of similar fragments to return.
    ///
    /// # Returns
    /// A `Vec` of `MemorySearchResult`s, sorted from most to least similar.
    pub fn find_similar<'a>(
        &'a self,
        query_embedding: &Array1<f32>,
        top_k: usize,
    ) -> MemoryResult<Vec<MemorySearchResult<'a>>> {
        if query_embedding.len() == 0 {
            return Err(MemoryError::ZeroDimensionEmbedding);
        }

        let mut results: Vec<_> = self.fragments
            .iter()
            .filter_map(|fragment| {
                cosine_similarity(query_embedding, &fragment.embedding)
                    .map(|score| MemorySearchResult { fragment, similarity_score: score })
            })
            .collect();

        // Sort by similarity score in descending order.
        results.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap_or(std::cmp::Ordering::Equal));

        // Truncate to `top_k` results.
        results.truncate(top_k);

        Ok(results)
    }
}

/// Calculates the cosine similarity between two 1-D vectors.
/// Returns `None` if the vectors have zero magnitude.
pub fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> Option<f32> {
    let dot_product = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        None
    } else {
        Some(dot_product / (norm_a * norm_b))
    }
}


// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_add_and_find_memory() {
        let mut manager = MemoryManager::new();
        let id1 = manager.add_memory("cat".to_string(), vec![1.0, 0.0, 0.0]).unwrap();
        let id2 = manager.add_memory("dog".to_string(), vec![0.0, 1.0, 0.0]).unwrap();
        let id3 = manager.add_memory("kitty".to_string(), vec![0.9, 0.1, 0.0]).unwrap();

        let query = Array1::from(vec![0.95, 0.05, 0.0]); // A query very similar to "cat" / "kitty"

        let results = manager.find_similar(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        // The most similar should be "kitty"
        assert_eq!(results[0].fragment.id, id3);
        // The second most similar should be "cat"
        assert_eq!(results[1].fragment.id, id1);
    }

    #[test]
    fn test_cosine_similarity_logic() {
        let vec_a = Array1::from(vec![1.0, 2.0, 3.0]);
        let vec_b = Array1::from(vec![1.0, 2.0, 3.0]); // Identical
        let vec_c = Array1::from(vec![-1.0, -2.0, -3.0]); // Opposite
        let vec_d = Array1::from(vec![3.0, -2.0, 1.0]); // Orthogonal

        assert_relative_eq!(cosine_similarity(&vec_a, &vec_b).unwrap(), 1.0);
        assert_relative_eq!(cosine_similarity(&vec_a, &vec_c).unwrap(), -1.0);
        // Dot product is 3 - 4 + 3 = 2. It's not perfectly orthogonal.
        // Let's use a better orthogonal vector.
        let vec_e = Array1::from(vec![-2.0, 1.0, 0.0]); // Orthogonal to a
        assert_relative_eq!(cosine_similarity(&vec_a, &vec_e).unwrap(), 0.0);
    }

    #[test]
    fn test_find_empty_returns_empty() {
        let manager = MemoryManager::new();
        let query = Array1::from(vec![1.0, 2.0]);
        let results = manager.find_similar(&query, 5).unwrap();
        assert!(results.is_empty());
    }
}
