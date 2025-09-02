/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_model_loader.h
 *
 * This header file defines the public API for the Model Loader module.
 * This component is responsible for loading, validating, and preparing
 * various AI models (LLM, vision, audio) for use in the TrackieLLM system.
 *
 * The loader supports multiple model formats (GGUF, ONNX, etc.) and provides
 * a unified interface for model initialization across different frameworks.
 *
 * Key architectural features:
 *   - Opaque handle for managing model contexts
 *   - Format-agnostic loading interface
 *   - Model validation and metadata extraction
 *   - Memory mapping and optimization settings
 *   - Multi-framework support (llama.cpp, ONNX Runtime, etc.)
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

#ifndef TRACKIELLM_AI_MODELS_TK_MODEL_LOADER_H
#define TRACKIELLM_AI_MODELS_TK_MODEL_LOADER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "utils/tk_error_handling.h"
#include "internal_tools/tk_file_manager.h"

// Forward-declare the primary loader object as an opaque type.
typedef struct tk_model_loader_s tk_model_loader_t;

/**
 * @enum tk_model_format_e
 * @brief Supported model formats
 */
typedef enum {
    TK_MODEL_FORMAT_UNKNOWN = 0,
    TK_MODEL_FORMAT_GGUF,      /**< GGUF format (llama.cpp) */
    TK_MODEL_FORMAT_ONNX,      /**< ONNX format */
    TK_MODEL_FORMAT_TFLITE,    /**< TensorFlow Lite format */
    TK_MODEL_FORMAT_TENSORRT,  /**< TensorRT format */
    TK_MODEL_FORMAT_COREML,    /**< CoreML format (Apple) */
    TK_MODEL_FORMAT_OPENVINO,  /**< OpenVINO format (Intel) */
    TK_MODEL_FORMAT_TORCH,     /**< PyTorch format */
    TK_MODEL_FORMAT_SAFETENSORS /**< Safetensors format */
} tk_model_format_e;

/**
 * @enum tk_model_type_e
 * @brief Types of models supported
 */
typedef enum {
    TK_MODEL_TYPE_UNKNOWN = 0,
    TK_MODEL_TYPE_LLM,         /**< Large Language Model */
    TK_MODEL_TYPE_VISION,      /**< Computer Vision model */
    TK_MODEL_TYPE_AUDIO,       /**< Audio processing model */
    TK_MODEL_TYPE_MULTIMODAL,  /**< Multimodal model */
    TK_MODEL_TYPE_EMBEDDING,   /**< Embedding model */
    TK_MODEL_TYPE_CLASSIFIER,  /**< Classification model */
    TK_MODEL_TYPE_DETECTOR,    /**< Object detection model */
    TK_MODEL_TYPE_SEGMENTER,   /**< Image segmentation model */
    TK_MODEL_TYPE_GENERATOR,   /**< Content generation model */
    TK_MODEL_TYPE_RETRIEVER,   /**< Information retrieval model */
    TK_MODEL_TYPE_RANKER,      /**< Ranking model */
    TK_MODEL_TYPE_TRANSLATOR,  /**< Translation model */
    TK_MODEL_TYPE_SUMMARIZER,  /**< Text summarization model */
    TK_MODEL_TYPE_SENTIMENT,   /**< Sentiment analysis model */
    TK_MODEL_TYPE_NER,         /**< Named Entity Recognition model */
    TK_MODEL_TYPE_SPEECH_RECOGNITION, /**< Speech recognition model */
    TK_MODEL_TYPE_TEXT_TO_SPEECH,     /**< Text-to-speech model */
    TK_MODEL_TYPE_VOICE_CLONING,      /**< Voice cloning model */
    TK_MODEL_TYPE_MUSIC_GENERATION    /**< Music generation model */
} tk_model_type_e;

/**
 * @struct tk_model_metadata_t
 * @brief Metadata information about a loaded model
 */
typedef struct {
    char* name;                /**< Model name */
    char* version;             /**< Model version */
    char* author;              /**< Model author */
    char* description;         /**< Model description */
    char* license;             /**< Model license */
    char* architecture;        /**< Model architecture (e.g., "LLaMA", "YOLOv5") */
    tk_model_type_e type;      /**< Model type */
    tk_model_format_e format;  /**< Model format */
    uint64_t size_bytes;       /**< Model file size in bytes */
    uint32_t input_dim_count;  /**< Number of input dimensions */
    uint32_t* input_dims;      /**< Input dimensions array */
    uint32_t output_dim_count; /**< Number of output dimensions */
    uint32_t* output_dims;     /**< Output dimensions array */
    uint32_t parameter_count;  /**< Number of parameters (in millions) */
    uint32_t context_length;   /**< Context length for LLMs */
    uint32_t embedding_dim;    /**< Embedding dimension */
    uint32_t vocab_size;       /**< Vocabulary size for LLMs */
    uint32_t num_layers;       /**< Number of layers */
    uint32_t hidden_size;      /**< Hidden layer size */
    uint32_t num_heads;        /**< Number of attention heads */
    float quantization_level;  /**< Quantization level (0.0 = none, 1.0 = maximum) */
    bool is_quantized;         /**< Whether the model is quantized */
    bool is_multilingual;      /**< Whether the model supports multiple languages */
    char** supported_languages;/**< Array of supported languages */
    uint32_t language_count;   /**< Number of supported languages */
    char* creation_date;       /**< Model creation date */
    char* last_modified;       /**< Model last modified date */
    char* framework;           /**< Framework used to create the model */
    char* framework_version;   /**< Framework version */
    char* hardware_target;     /**< Target hardware (CPU, GPU, etc.) */
    bool supports_gpu;         /**< Whether the model supports GPU acceleration */
    uint32_t min_memory_mb;    /**< Minimum memory required (MB) */
    uint32_t recommended_memory_mb; /**< Recommended memory (MB) */
    char* dependencies;        /**< Model dependencies */
    char* checksum;            /**< Model file checksum */
    bool is_valid;             /**< Whether the model passed validation */
    char* validation_message;  /**< Validation result message */
} tk_model_metadata_t;

/**
 * @struct tk_model_loader_config_t
 * @brief Configuration for initializing the Model Loader
 */
typedef struct {
    uint32_t max_models;       /**< Maximum number of models that can be loaded simultaneously */
    uint32_t cache_size_mb;    /**< Size of model cache in MB */
    bool enable_memory_mapping;/**< Enable memory mapping for large models */
    bool enable_model_caching; /**< Enable model caching for faster loading */
    uint32_t num_threads;      /**< Number of threads for model loading */
    uint32_t gpu_device_id;    /**< GPU device ID for GPU-accelerated loading */
    bool enable_gpu_loading;   /**< Enable GPU-accelerated model loading */
    uint32_t timeout_seconds;  /**< Timeout for model loading operations */
    bool enable_validation;    /**< Enable model validation after loading */
    bool enable_preprocessing; /**< Enable automatic model preprocessing */
    char* temp_dir;            /**< Temporary directory for model extraction */
    char* cache_dir;           /**< Cache directory for loaded models */
    uint32_t max_retries;      /**< Maximum number of retry attempts for failed loads */
    bool enable_compression;   /**< Enable model compression during loading */
    float compression_ratio;   /**< Target compression ratio (0.0-1.0) */
    bool enable_integrity_check; /**< Enable model file integrity checking */
    char* ssl_cert_file;       /**< SSL certificate file for secure downloads */
    bool enable_progress_reporting; /**< Enable progress reporting during loading */
    uint32_t buffer_size_kb;   /**< Buffer size for file operations (KB) */
    bool enable_parallel_loading; /**< Enable parallel loading of model components */
    uint32_t parallel_threads; /**< Number of threads for parallel loading */
    bool enable_model_optimization; /**< Enable automatic model optimization */
    char* optimization_profile; /**< Optimization profile name */
    bool enable_model_conversion; /**< Enable automatic model format conversion */
    char* target_format;       /**< Target format for conversion */
    bool enable_model_fusion;  /**< Enable model fusion optimization */
    bool enable_model_pruning; /**< Enable model pruning for size reduction */
    float pruning_ratio;       /**< Target pruning ratio (0.0-1.0) */
    bool enable_model_quantization; /**< Enable model quantization */
    uint32_t quantization_bits; /**< Target quantization bits (8, 4, etc.) */
    bool enable_model_compilation; /**< Enable model compilation for target hardware */
    char* target_hardware;     /**< Target hardware for compilation */
    bool enable_model_signing; /**< Enable model signing for security */
    char* signing_key_file;    /**< Signing key file path */
    bool enable_model_encryption; /**< Enable model encryption */
    char* encryption_key_file; /**< Encryption key file path */
    bool enable_model_verification; /**< Enable model verification */
    char* verification_cert_file; /**< Verification certificate file path */
    bool enable_model_profiling; /**< Enable model profiling during loading */
    uint32_t profiling_interval_ms; /**< Profiling interval in milliseconds */
    bool enable_model_warmup;  /**< Enable model warmup after loading */
    uint32_t warmup_iterations; /**< Number of warmup iterations */
    bool enable_model_precompilation; /**< Enable model precompilation */
    char* precompilation_cache_dir; /**< Precompilation cache directory */
    bool enable_model_streaming; /**< Enable model streaming for large models */
    uint32_t streaming_chunk_size_kb; /**< Streaming chunk size in KB */
    bool enable_model_checkpointing; /**< Enable model checkpointing */
    uint32_t checkpoint_interval_seconds; /**< Checkpoint interval in seconds */
    bool enable_model_resilience; /**< Enable model resilience features */
    uint32_t resilience_retry_count; /**< Resilience retry count */
    uint32_t resilience_timeout_ms; /**< Resilience timeout in milliseconds */
    bool enable_model_monitoring; /**< Enable model monitoring */
    uint32_t monitoring_interval_ms; /**< Monitoring interval in milliseconds */
    bool enable_model_telemetry; /**< Enable model telemetry collection */
    char* telemetry_endpoint;  /**< Telemetry collection endpoint */
    bool enable_model_debugging; /**< Enable model debugging features */
    uint32_t debug_verbosity;  /**< Debug verbosity level */
    char* log_file;            /**< Log file path */
    bool enable_model_tracing; /**< Enable model execution tracing */
    char* trace_file;          /**< Trace output file */
    bool enable_model_benchmarking; /**< Enable model benchmarking */
    uint32_t benchmark_iterations; /**< Number of benchmark iterations */
    bool enable_model_analytics; /**< Enable model analytics collection */
    char* analytics_endpoint;  /**< Analytics collection endpoint */
    bool enable_model_security; /**< Enable model security features */
    char* security_policy;     /**< Security policy file */
    bool enable_model_isolation; /**< Enable model process isolation */
    char* isolation_method;    /**< Isolation method */
    bool enable_model_sandboxing; /**< Enable model sandboxing */
    char* sandbox_config;      /**< Sandboxing configuration */
    bool enable_model_containerization; /**< Enable model containerization */
    char* container_runtime;   /**< Container runtime */
    bool enable_model_virtualization; /**< Enable model virtualization */
    char* virtualization_platform; /**< Virtualization platform */
    bool enable_model_compatibility_check; /**< Enable compatibility checking */
    char* compatibility_profile; /**< Compatibility profile */
    bool enable_model_dependency_resolution; /**< Enable dependency resolution */
    char* dependency_resolver; /**< Dependency resolver */
    bool enable_model_update_check; /**< Enable automatic update checking */
    char* update_server;       /**< Update server URL */
    bool enable_model_rollback; /**< Enable automatic rollback on failure */
    char* rollback_policy;     /**< Rollback policy */
    bool enable_model_backup;  /**< Enable automatic backup */
    char* backup_location;     /**< Backup location */
    bool enable_model_restore; /**< Enable automatic restore */
    char* restore_policy;      /**< Restore policy */
    bool enable_model_migration; /**< Enable model migration */
    char* migration_target;    /**< Migration target */
    bool enable_model_replication; /**< Enable model replication */
    uint32_t replication_factor; /**< Replication factor */
    bool enable_model_sharding; /**< Enable model sharding */
    uint32_t shard_count;      /**< Number of shards */
    bool enable_model_federation; /**< Enable model federation */
    char* federation_config;   /**< Federation configuration */
    bool enable_model_synchronization; /**< Enable model synchronization */
    uint32_t sync_interval_seconds; /**< Synchronization interval in seconds */
    bool enable_model_consistency; /**< Enable model consistency checking */
    char* consistency_protocol; /**< Consistency protocol */
    bool enable_model_transaction; /**< Enable model transaction support */
    char* transaction_manager; /**< Transaction manager */
    bool enable_model_caching_strategy; /**< Enable advanced caching strategy */
    char* caching_strategy;    /**< Caching strategy */
    bool enable_model_compression_strategy; /**< Enable advanced compression strategy */
    char* compression_strategy; /**< Compression strategy */
    bool enable_model_optimization_strategy; /**< Enable advanced optimization strategy */
    char* optimization_strategy; /**< Optimization strategy */
    bool enable_model_security_strategy; /**< Enable advanced security strategy */
    char* security_strategy;   /**< Security strategy */
    bool enable_model_resilience_strategy; /**< Enable advanced resilience strategy */
    char* resilience_strategy; /**< Resilience strategy */
    bool enable_model_monitoring_strategy; /**< Enable advanced monitoring strategy */
    char* monitoring_strategy; /**< Monitoring strategy */
    bool enable_model_analytics_strategy; /**< Enable advanced analytics strategy */
    char* analytics_strategy;  /**< Analytics strategy */
    bool enable_model_debugging_strategy; /**< Enable advanced debugging strategy */
    char* debugging_strategy;  /**< Debugging strategy */
    bool enable_model_tracing_strategy; /**< Enable advanced tracing strategy */
    char* tracing_strategy;    /**< Tracing strategy */
    bool enable_model_benchmarking_strategy; /**< Enable advanced benchmarking strategy */
    char* benchmarking_strategy; /**< Benchmarking strategy */
    bool enable_model_custom_strategy; /**< Enable custom strategy */
    char* custom_strategy;     /**< Custom strategy */
} tk_model_loader_config_t;

/**
 * @struct tk_model_load_params_t
 * @brief Parameters for loading a specific model
 */
typedef struct {
    tk_path_t* model_path;     /**< Path to the model file or directory */
    tk_model_type_e model_type; /**< Type of model to load */
    bool force_reload;         /**< Force reload even if already loaded */
    uint32_t gpu_layers;       /**< Number of layers to offload to GPU */
    uint32_t cpu_threads;      /**< Number of CPU threads to use */
    uint32_t context_length;   /**< Context length for LLMs */
    uint32_t batch_size;       /**< Batch size for inference */
    bool use_mmap;             /**< Use memory mapping */
    bool use_mlock;            /**< Lock model in memory */
    uint32_t memory_limit_mb;  /**< Memory limit in MB */
    float rope_freq_base;      /**< RoPE frequency base */
    float rope_freq_scale;     /**< RoPE frequency scale */
    bool enable_flash_attn;    /**< Enable flash attention */
    uint32_t n_gpu_layers;     /**< Number of GPU layers */
    bool low_vram;             /**< Optimize for low VRAM */
    bool numa;                 /**< Enable NUMA support */
    uint32_t seed;             /**< Random seed */
    bool verbose;              /**< Enable verbose logging */
    char* lora_adapter;        /**< LoRA adapter path */
    char* lora_base;           /**< LoRA base model path */
    float tensor_split[128];   /**< Tensor split configuration */
    char* main_gpu;            /**< Main GPU to use */
    bool no_kv_offload;        /**< Disable KV offloading */
    char* cache_type_k;        /**< Cache type for K */
    char* cache_type_v;        /**< Cache type for V */
    bool dry_run;              /**< Perform dry run without actual loading */
    char* export_dir;          /**< Export directory for converted models */
    bool prefer_cpu;           /**< Prefer CPU over GPU */
    bool prefer_gpu;           /**< Prefer GPU over CPU */
    char* device_placement;    /**< Device placement strategy */
    bool enable_model_fusion;  /**< Enable model fusion */
    bool enable_model_pruning; /**< Enable model pruning */
    float pruning_ratio;       /**< Pruning ratio */
    bool enable_model_quantization; /**< Enable model quantization */
    uint32_t quantization_bits; /**< Quantization bits */
    bool enable_model_compilation; /**< Enable model compilation */
    char* target_hardware;     /**< Target hardware for compilation */
    bool enable_model_signing; /**< Enable model signing */
    char* signing_key_file;    /**< Signing key file */
    bool enable_model_encryption; /**< Enable model encryption */
    char* encryption_key_file; /**< Encryption key file */
    bool enable_model_verification; /**< Enable model verification */
    char* verification_cert_file; /**< Verification certificate file */
    bool enable_model_profiling; /**< Enable model profiling */
    uint32_t profiling_interval_ms; /**< Profiling interval */
    bool enable_model_warmup;  /**< Enable model warmup */
    uint32_t warmup_iterations; /**< Warmup iterations */
    bool enable_model_precompilation; /**< Enable precompilation */
    char* precompilation_cache_dir; /**< Precompilation cache directory */
    bool enable_model_streaming; /**< Enable model streaming */
    uint32_t streaming_chunk_size_kb; /**< Streaming chunk size */
    bool enable_model_checkpointing; /**< Enable checkpointing */
    uint32_t checkpoint_interval_seconds; /**< Checkpoint interval */
    bool enable_model_resilience; /**< Enable resilience */
    uint32_t resilience_retry_count; /**< Resilience retry count */
    uint32_t resilience_timeout_ms; /**< Resilience timeout */
    bool enable_model_monitoring; /**< Enable monitoring */
    uint32_t monitoring_interval_ms; /**< Monitoring interval */
    bool enable_model_telemetry; /**< Enable telemetry */
    char* telemetry_endpoint;  /**< Telemetry endpoint */
    bool enable_model_debugging; /**< Enable debugging */
    uint32_t debug_verbosity;  /**< Debug verbosity */
    char* log_file;            /**< Log file */
    bool enable_model_tracing; /**< Enable tracing */
    char* trace_file;          /**< Trace file */
    bool enable_model_benchmarking; /**< Enable benchmarking */
    uint32_t benchmark_iterations; /**< Benchmark iterations */
    bool enable_model_analytics; /**< Enable analytics */
    char* analytics_endpoint;  /**< Analytics endpoint */
    bool enable_model_security; /**< Enable security */
    char* security_policy;     /**< Security policy */
    bool enable_model_isolation; /**< Enable isolation */
    char* isolation_method;    /**< Isolation method */
    bool enable_model_sandboxing; /**< Enable sandboxing */
    char* sandbox_config;      /**< Sandbox configuration */
    bool enable_model_containerization; /**< Enable containerization */
    char* container_runtime;   /**< Container runtime */
    bool enable_model_virtualization; /**< Enable virtualization */
    char* virtualization_platform; /**< Virtualization platform */
    bool enable_model_compatibility_check; /**< Enable compatibility check */
    char* compatibility_profile; /**< Compatibility profile */
    bool enable_model_dependency_resolution; /**< Enable dependency resolution */
    char* dependency_resolver; /**< Dependency resolver */
    bool enable_model_update_check; /**< Enable update check */
    char* update_server;       /**< Update server */
    bool enable_model_rollback; /**< Enable rollback */
    char* rollback_policy;     /**< Rollback policy */
    bool enable_model_backup;  /**< Enable backup */
    char* backup_location;     /**< Backup location */
    bool enable_model_restore; /**< Enable restore */
    char* restore_policy;      /**< Restore policy */
    bool enable_model_migration; /**< Enable migration */
    char* migration_target;    /**< Migration target */
    bool enable_model_replication; /**< Enable replication */
    uint32_t replication_factor; /**< Replication factor */
    bool enable_model_sharding; /**< Enable sharding */
    uint32_t shard_count;      /**< Shard count */
    bool enable_model_federation; /**< Enable federation */
    char* federation_config;   /**< Federation configuration */
    bool enable_model_synchronization; /**< Enable synchronization */
    uint32_t sync_interval_seconds; /**< Sync interval */
    bool enable_model_consistency; /**< Enable consistency */
    char* consistency_protocol; /**< Consistency protocol */
    bool enable_model_transaction; /**< Enable transaction */
    char* transaction_manager; /**< Transaction manager */
} tk_model_load_params_t;

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Loader Lifecycle Management
//------------------------------------------------------------------------------

/**
 * @brief Creates and initializes a new Model Loader instance.
 *
 * @param[out] out_loader Pointer to receive the address of the new loader instance.
 * @param[in] config The configuration for the loader.
 *
 * @return TK_SUCCESS on success.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL or config is invalid.
 * @return TK_ERROR_OUT_OF_MEMORY on memory allocation failure.
 */
TK_NODISCARD tk_error_code_t tk_model_loader_create(tk_model_loader_t** out_loader, const tk_model_loader_config_t* config);

/**
 * @brief Destroys a Model Loader instance and frees all associated resources.
 *
 * @param[in,out] loader Pointer to the loader instance to be destroyed.
 */
void tk_model_loader_destroy(tk_model_loader_t** loader);

//------------------------------------------------------------------------------
// Model Loading and Management
//------------------------------------------------------------------------------

/**
 * @brief Loads a model from the specified path with given parameters.
 *
 * @param[in] loader The model loader instance.
 * @param[in] params The parameters for loading the model.
 * @param[out] out_model_handle Pointer to receive the model handle.
 *
 * @return TK_SUCCESS on successful loading.
 * @return TK_ERROR_INVALID_ARGUMENT if required pointers are NULL.
 * @return TK_ERROR_MODEL_LOAD_FAILED if the model cannot be loaded.
 * @return TK_ERROR_OUT_OF_MEMORY if memory allocation fails.
 */
TK_NODISCARD tk_error_code_t tk_model_loader_load_model(
    tk_model_loader_t* loader,
    const tk_model_load_params_t* params,
    void** out_model_handle
);

/**
 * @brief Unloads a previously loaded model.
 *
 * @param[in] loader The model loader instance.
 * @param[in,out] model_handle Pointer to the model handle to unload.
 *
 * @return TK_SUCCESS on successful unloading.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL.
 */
TK_NODISCARD tk_error_code_t tk_model_loader_unload_model(tk_model_loader_t* loader, void** model_handle);

/**
 * @brief Gets metadata information about a loaded model.
 *
 * @param[in] loader The model loader instance.
 * @param[in] model_handle The model handle.
 * @param[out] out_metadata Pointer to receive the model metadata.
 *
 * @return TK_SUCCESS on successful retrieval.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL.
 * @return TK_ERROR_NOT_FOUND if the model handle is invalid.
 */
TK_NODISCARD tk_error_code_t tk_model_loader_get_model_metadata(
    tk_model_loader_t* loader,
    void* model_handle,
    tk_model_metadata_t** out_metadata
);

/**
 * @brief Frees metadata information.
 *
 * @param[in,out] metadata Pointer to the metadata to free.
 */
void tk_model_loader_free_metadata(tk_model_metadata_t** metadata);

/**
 * @brief Validates a model file without loading it.
 *
 * @param[in] loader The model loader instance.
 * @param[in] model_path Path to the model file.
 * @param[out] out_is_valid Pointer to receive validation result.
 * @param[out] out_format Pointer to receive detected model format.
 *
 * @return TK_SUCCESS on successful validation.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL.
 * @return TK_ERROR_IO_ERROR if file cannot be accessed.
 */
TK_NODISCARD tk_error_code_t tk_model_loader_validate_model(
    tk_model_loader_t* loader,
    const tk_path_t* model_path,
    bool* out_is_valid,
    tk_model_format_e* out_format
);

/**
 * @brief Converts a model from one format to another.
 *
 * @param[in] loader The model loader instance.
 * @param[in] source_path Path to the source model.
 * @param[in] target_path Path for the converted model.
 * @param[in] target_format Target format for conversion.
 *
 * @return TK_SUCCESS on successful conversion.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL.
 * @return TK_ERROR_IO_ERROR if file operations fail.
 * @return TK_ERROR_UNSUPPORTED_OPERATION if conversion is not supported.
 */
TK_NODISCARD tk_error_code_t tk_model_loader_convert_model(
    tk_model_loader_t* loader,
    const tk_path_t* source_path,
    const tk_path_t* target_path,
    tk_model_format_e target_format
);

/**
 * @brief Optimizes a loaded model for better performance.
 *
 * @param[in] loader The model loader instance.
 * @param[in,out] model_handle The model handle.
 * @param[in] optimization_level Level of optimization (0-3).
 *
 * @return TK_SUCCESS on successful optimization.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL.
 * @return TK_ERROR_NOT_FOUND if the model handle is invalid.
 */
TK_NODISCARD tk_error_code_t tk_model_loader_optimize_model(
    tk_model_loader_t* loader,
    void* model_handle,
    uint32_t optimization_level
);

/**
 * @brief Quantizes a loaded model to reduce size and improve performance.
 *
 * @param[in] loader The model loader instance.
 * @param[in,out] model_handle The model handle.
 * @param[in] bits Number of bits for quantization (4, 8, etc.).
 *
 * @return TK_SUCCESS on successful quantization.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL.
 * @return TK_ERROR_NOT_FOUND if the model handle is invalid.
 * @return TK_ERROR_UNSUPPORTED_OPERATION if quantization is not supported.
 */
TK_NODISCARD tk_error_code_t tk_model_loader_quantize_model(
    tk_model_loader_t* loader,
    void* model_handle,
    uint32_t bits
);

//------------------------------------------------------------------------------
// Model Caching and Management
//------------------------------------------------------------------------------

/**
 * @brief Clears the model cache.
 *
 * @param[in] loader The model loader instance.
 *
 * @return TK_SUCCESS on successful cache clearing.
 * @return TK_ERROR_INVALID_ARGUMENT if loader is NULL.
 */
TK_NODISCARD tk_error_code_t tk_model_loader_clear_cache(tk_model_loader_t* loader);

/**
 * @brief Gets cache statistics.
 *
 * @param[in] loader The model loader instance.
 * @param[out] out_cache_hit_rate Pointer to receive cache hit rate (0.0-1.0).
 * @param[out] out_cache_size_mb Pointer to receive current cache size in MB.
 * @param[out] out_cache_entries Pointer to receive number of cached models.
 *
 * @return TK_SUCCESS on successful retrieval.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL.
 */
TK_NODISCARD tk_error_code_t tk_model_loader_get_cache_stats(
    tk_model_loader_t* loader,
    float* out_cache_hit_rate,
    uint32_t* out_cache_size_mb,
    uint32_t* out_cache_entries
);

/**
 * @brief Preloads a model into cache.
 *
 * @param[in] loader The model loader instance.
 * @param[in] model_path Path to the model file.
 *
 * @return TK_SUCCESS on successful preloading.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL.
 * @return TK_ERROR_IO_ERROR if file cannot be accessed.
 */
TK_NODISCARD tk_error_code_t tk_model_loader_preload_model(
    tk_model_loader_t* loader,
    const tk_path_t* model_path
);

//------------------------------------------------------------------------------
// Utility Functions
//------------------------------------------------------------------------------

/**
 * @brief Detects the format of a model file.
 *
 * @param[in] loader The model loader instance.
 * @param[in] model_path Path to the model file.
 * @param[out] out_format Pointer to receive detected format.
 *
 * @return TK_SUCCESS on successful detection.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL.
 * @return TK_ERROR_IO_ERROR if file cannot be accessed.
 */
TK_NODISCARD tk_error_code_t tk_model_loader_detect_format(
    tk_model_loader_t* loader,
    const tk_path_t* model_path,
    tk_model_format_e* out_format
);

/**
 * @brief Gets a list of supported model formats.
 *
 * @param[out] out_formats Pointer to receive array of supported formats.
 * @param[out] out_count Pointer to receive number of supported formats.
 *
 * @return TK_SUCCESS on successful retrieval.
 * @return TK_ERROR_INVALID_ARGUMENT if pointers are NULL.
 * @return TK_ERROR_OUT_OF_MEMORY if memory allocation fails.
 */
TK_NODISCARD tk_error_code_t tk_model_loader_get_supported_formats(
    tk_model_format_e** out_formats,
    size_t* out_count
);

/**
 * @brief Frees the array of supported formats.
 *
 * @param[in,out] formats Pointer to the formats array to free.
 */
void tk_model_loader_free_formats(tk_model_format_e** formats);

#ifdef __cplusplus
}
#endif

#endif // TRACKIELLM_AI_MODELS_TK_MODEL_LOADER_H
