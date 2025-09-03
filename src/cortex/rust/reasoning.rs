/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: reasoning.rs
 *
 * This file implements the core reasoning engine of the cortex. It orchestrates
 * the process of Retrieval-Augmented Generation (RAG). The engine takes a query,
 * retrieves relevant context from the `MemoryManager`, constructs a rich prompt,
 * and then (conceptually) uses a language model to generate a final, context-
 * aware response. The design uses traits to abstract the embedding and model
 * running components, making it modular and testable.
 *
 * Dependencies:
 *  - `memory_manager`: For retrieving relevant memory fragments.
 *  - `ndarray`: For handling embedding vectors.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

// --- Module Imports ---
use crate::memory_manager::{MemoryManager, MemoryResult};
use ndarray::Array1;
use thiserror::Error;

// --- Custom Error and Result Types ---

#[derive(Error, Debug)]
pub enum ReasoningError {
    #[error("Failed to generate embedding for query: {0}")]
    EmbeddingError(String),
    #[error("Failed to retrieve memories: {0}")]
    MemoryRetrievalError(#[from] crate::memory_manager::MemoryError),
    #[error("Model runner failed to generate a response: {0}")]
    ModelError(String),
}

pub type ReasoningResult<T> = Result<T, ReasoningError>;

// --- Abstractions for External Components ---

/// A trait for a service that can generate vector embeddings from text.
pub trait EmbeddingGenerator: Send + Sync {
    fn generate_embedding(&self, text: &str) -> Result<Array1<f32>, ReasoningError>;
}

/// A trait for a service that can run a language model to generate a response.
pub trait ModelRunner: Send + Sync {
    fn run_inference(&self, prompt: &str) -> Result<String, ReasoningError>;
}

// --- Reasoning Engine Service ---

/// The core reasoning engine.
pub struct ReasoningEngine {
    memory_manager: MemoryManager,
    embedding_generator: Box<dyn EmbeddingGenerator>,
    model_runner: Box<dyn ModelRunner>,
}

impl ReasoningEngine {
    /// Creates a new `ReasoningEngine` with its required components.
    /// This uses dependency injection for modularity and testability.
    pub fn new(
        memory_manager: MemoryManager,
        embedding_generator: Box<dyn EmbeddingGenerator>,
        model_runner: Box<dyn ModelRunner>,
    ) -> Self {
        Self {
            memory_manager,
            embedding_generator,
            model_runner,
        }
    }

    /// Processes a query using the full Retrieval-Augmented Generation (RAG) pipeline.
    ///
    /// # Arguments
    /// * `query_text` - The input query from the user.
    ///
    /// # Returns
    /// A `String` containing the final, context-aware response from the language model.
    pub fn query(&self, query_text: &str) -> ReasoningResult<String> {
        // 1. Generate an embedding for the input query.
        let query_embedding = self.embedding_generator.generate_embedding(query_text)?;

        // 2. Retrieve relevant memory fragments based on the query embedding.
        let relevant_memories = self.memory_manager.find_similar(&query_embedding, 3)?; // Get top 3

        // 3. Construct a rich prompt for the language model.
        let prompt = self.construct_prompt(query_text, &relevant_memories);
        println!("--- Constructed Prompt ---\n{}\n--------------------------", prompt);

        // 4. Run the language model with the enriched prompt.
        let response = self.model_runner.run_inference(&prompt)?;

        // 5. (Optional) Here, you could add logic to store the interaction
        //    (query + response) as a new memory in the `MemoryManager`.

        Ok(response)
    }

    /// Constructs a prompt by combining the original query with retrieved context.
    fn construct_prompt(
        &self,
        query: &str,
        context: &[crate::memory_manager::MemorySearchResult],
    ) -> String {
        let mut prompt = "You are a helpful AI assistant. Answer the user's question based on the following context. If the context is not relevant, use your own knowledge.\n\n".to_string();

        if !context.is_empty() {
            prompt.push_str("--- CONTEXT ---\n");
            for (i, item) in context.iter().enumerate() {
                prompt.push_str(&format!("{}. {}\n", i + 1, item.fragment.text_content));
            }
            prompt.push_str("--- END CONTEXT ---\n\n");
        }

        prompt.push_str(&format!("User Question: {}\n\nAI Answer:", query));

        prompt
    }
}


// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use uuid::Uuid;

    // --- Mock Implementations for Testing ---

    struct MockEmbeddingGenerator;
    impl EmbeddingGenerator for MockEmbeddingGenerator {
        fn generate_embedding(&self, text: &str) -> Result<Array1<f32>, ReasoningError> {
            // Simple mock: return a vector based on the text's content
            match text {
                "What is the capital of France?" => Ok(Array1::from(vec![1.0, 0.0])),
                _ => Ok(Array1::from(vec![0.0, 0.0])),
            }
        }
    }

    struct MockModelRunner;
    impl ModelRunner for MockModelRunner {
        fn run_inference(&self, prompt: &str) -> Result<String, ReasoningError> {
            // Simple mock: check if the prompt contains expected context
            if prompt.contains("Paris is the capital of France.") {
                Ok("Based on the context, the capital of France is Paris.".to_string())
            } else {
                Ok("I'm sorry, I don't have that information.".to_string())
            }
        }
    }

    #[test]
    fn test_reasoning_engine_query_pipeline() {
        // 1. Setup: Create a memory manager and populate it.
        let mut memory_manager = MemoryManager::new();
        memory_manager.add_memory(
            "Paris is the capital of France.".to_string(),
            vec![0.98, 0.01] // Similar to the query embedding
        ).unwrap();
        memory_manager.add_memory(
            "The sky is blue.".to_string(),
            vec![0.0, 1.0] // Dissimilar
        ).unwrap();

        // 2. Create the engine with mock components.
        let engine = ReasoningEngine::new(
            memory_manager,
            Box::new(MockEmbeddingGenerator),
            Box::new(MockModelRunner),
        );

        // 3. Run a query that should trigger the RAG pipeline.
        let query = "What is the capital of France?";
        let response = engine.query(query).unwrap();

        // 4. Assert: The response should be the one generated from the context.
        assert_eq!(response, "Based on the context, the capital of France is Paris.");
    }

    #[test]
    fn test_prompt_construction() {
        let engine = ReasoningEngine::new(
            MemoryManager::new(),
            Box::new(MockEmbeddingGenerator),
            Box::new(MockModelRunner),
        );

        let fragment = crate::memory_manager::MemoryFragment {
            id: Uuid::new_v4(),
            text_content: "Test context.".to_string(),
            embedding: Array1::from(vec![]),
            metadata: HashMap::new(),
        };
        let context = vec![crate::memory_manager::MemorySearchResult {
            fragment: &fragment,
            similarity_score: 0.9,
        }];

        let prompt = engine.construct_prompt("Test question?", &context);

        assert!(prompt.contains("--- CONTEXT ---"));
        assert!(prompt.contains("1. Test context."));
        assert!(prompt.contains("User Question: Test question?"));
    }
}
