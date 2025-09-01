/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: trackiellm.h
*
* TrackieLLM Public API - Stable C ABI Interface
*
* This header defines the complete public interface for the TrackieLLM system.
* It provides a stable C ABI for cross-language interoperability with real-time
* computer vision processing, LLM integration, and assistive technology features.
*
* Key Design Principles:
* - Pure C ABI with extern "C" linkage for universal FFI compatibility
* - Opaque handle pattern for complete implementation encapsulation
* - Rich error handling with enumerated result codes
* - Asynchronous callback-based architecture for real-time performance
* - Thread-safe operations with explicit synchronization guarantees
* - Memory management with clear ownership semantics
*
* Target Platforms: Linux ARM64/x86_64, macOS (Metal), Windows (optional)
* GPU Acceleration: CUDA (primary), Metal (Apple), ROCm (AMD)
*
* SPDX-License-Identifier: 
*/

#ifndef TRACKIELLM_H
#define TRACKIELLM_H

#ifdef __cplusplus
extern "C" {
#endif

/* Standard library dependencies for stable ABI */
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* Version information - defined in trackie_version.h */
#include "trackie_version.h"

/*
 * =============================================================================
 * FORWARD DECLARATIONS AND OPAQUE TYPES
 * =============================================================================
 */

/**
 * TkContext - Main system context handle
 *
 * Opaque handle representing the complete TrackieLLM system state.
 * Contains all subsystems: vision processing, audio analysis, LLM interface,
 * GPU acceleration contexts, and thread pools.
 */
typedef struct TkContext TkContext;

/**
 * TkConfig - System configuration structure
 *
 * Contains initialization parameters for the TrackieLLM system.
 * Always initialize with tk_config_init_default() before use.
 */
typedef struct TkConfig TkConfig;

/**
 * TkStream - Media stream handle
 *
 * Represents an active media stream (video/audio) being processed.
 * Supports multiple concurrent streams with independent processing pipelines.
 */
typedef struct TkStream TkStream;

/*
 * =============================================================================
 * RESULT CODES AND ERROR HANDLING
 * =============================================================================
 */

/**
 * TkResult - Comprehensive result code enumeration
 *
 * Covers all possible success and failure states in the system.
 * Use tk_result_to_string() for human-readable error messages.
 */
typedef enum TkResult {
    /* Success codes */
    TK_SUCCESS = 0,                    /* Operation completed successfully */
    TK_SUCCESS_PARTIAL = 1,            /* Partial success (some data processed) */
    TK_SUCCESS_CACHED = 2,             /* Result retrieved from cache */

    /* General errors (100-199) */
    TK_ERROR_GENERIC = 100,            /* Unspecified error */
    TK_ERROR_INVALID_ARGUMENT = 101,   /* Invalid function argument */
    TK_ERROR_NULL_POINTER = 102,       /* Unexpected null pointer */
    TK_ERROR_INVALID_HANDLE = 103,     /* Invalid or corrupted handle */
    TK_ERROR_INVALID_STATE = 104,      /* Operation invalid in current state */
    TK_ERROR_TIMEOUT = 105,            /* Operation timed out */
    TK_ERROR_INTERRUPTED = 106,        /* Operation was interrupted */

    /* Memory management errors (200-299) */
    TK_ERROR_OUT_OF_MEMORY = 200,      /* Memory allocation failed */
    TK_ERROR_BUFFER_TOO_SMALL = 201,   /* Provided buffer insufficient */
    TK_ERROR_MEMORY_CORRUPTION = 202,  /* Memory corruption detected */

    /* System/platform errors (300-399) */
    TK_ERROR_SYSTEM_INIT_FAILED = 300, /* System initialization failed */
    TK_ERROR_THREAD_CREATE_FAILED = 301, /* Thread creation failed */
    TK_ERROR_MUTEX_LOCK_FAILED = 302,  /* Mutex lock operation failed */
    TK_ERROR_SIGNAL_FAILED = 303,      /* Signal handling setup failed */

    /* GPU/acceleration errors (400-499) */
    TK_ERROR_GPU_NOT_AVAILABLE = 400,  /* No compatible GPU found */
    TK_ERROR_GPU_INIT_FAILED = 401,    /* GPU initialization failed */
    TK_ERROR_GPU_OUT_OF_MEMORY = 402,  /* GPU memory exhausted */
    TK_ERROR_GPU_KERNEL_FAILED = 403,  /* GPU kernel execution failed */
    TK_ERROR_GPU_DRIVER_ERROR = 404,   /* GPU driver error */
    TK_ERROR_CUDA_ERROR = 410,         /* CUDA-specific error */
    TK_ERROR_METAL_ERROR = 420,        /* Metal-specific error */
    TK_ERROR_ROCM_ERROR = 430,         /* ROCm-specific error */

    /* Model/LLM errors (500-599) */
    TK_ERROR_MODEL_NOT_FOUND = 500,    /* Model file not found */
    TK_ERROR_MODEL_LOAD_FAILED = 501,  /* Model loading failed */
    TK_ERROR_MODEL_INVALID_FORMAT = 502, /* Invalid model file format */
    TK_ERROR_MODEL_VERSION_MISMATCH = 503, /* Model version incompatible */
    TK_ERROR_LLM_REQUEST_FAILED = 510, /* LLM inference request failed */
    TK_ERROR_LLM_CONTEXT_OVERFLOW = 511, /* LLM context length exceeded */
    TK_ERROR_LLM_RATE_LIMITED = 512,   /* LLM service rate limited */

    /* Vision/audio processing errors (600-699) */
    TK_ERROR_VISION_INIT_FAILED = 600, /* Vision subsystem init failed */
    TK_ERROR_INVALID_VIDEO_FORMAT = 601, /* Unsupported video format */
    TK_ERROR_AUDIO_INIT_FAILED = 610,  /* Audio subsystem init failed */
    TK_ERROR_INVALID_AUDIO_FORMAT = 611, /* Unsupported audio format */

    /* Network/communication errors (700-799) */
    TK_ERROR_NETWORK_UNAVAILABLE = 700, /* Network not available */
    TK_ERROR_CONNECTION_FAILED = 701,   /* Connection establishment failed */
    TK_ERROR_REQUEST_TIMEOUT = 702,     /* Network request timed out */
    TK_ERROR_AUTH_FAILED = 703,         /* Authentication failed */

    /* Configuration errors (800-899) */
    TK_ERROR_CONFIG_INVALID = 800,      /* Invalid configuration */
    TK_ERROR_CONFIG_FILE_NOT_FOUND = 801, /* Config file not found */
    TK_ERROR_CONFIG_PARSE_ERROR = 802,  /* Config parsing error */

    /* Internal errors (900-999) */
    TK_ERROR_INTERNAL_ERROR = 900,      /* Internal system error */
    TK_ERROR_NOT_IMPLEMENTED = 901,     /* Feature not implemented */
    TK_ERROR_DEPRECATED = 902,          /* Deprecated function called */

    /* Sentinel value for range checking */
    TK_RESULT_MAX = 999
} TkResult;

/*
 * =============================================================================
 * LOGGING AND DIAGNOSTICS
 * =============================================================================
 */

/**
 * TkLogLevel - Logging severity levels
 */
typedef enum TkLogLevel {
    TK_LOG_TRACE = 0,    /* Verbose debug information */
    TK_LOG_DEBUG = 1,    /* Debug information */
    TK_LOG_INFO = 2,     /* General information */
    TK_LOG_WARN = 3,     /* Warning conditions */
    TK_LOG_ERROR = 4,    /* Error conditions */
    TK_LOG_FATAL = 5,    /* Fatal errors */
    TK_LOG_OFF = 6       /* Logging disabled */
} TkLogLevel;

/**
 * TkLogCallback - Log message callback function
 *
 * @param level: Severity level of the log message
 * @param timestamp_us: Microsecond timestamp since system start
 * @param thread_id: ID of the thread generating the message
 * @param module: Source module name (null-terminated string)
 * @param message: Log message content (null-terminated string)
 * @param user_data: User-provided context data
 *
 * Thread Safety: This callback may be invoked from multiple threads
 * concurrently. Implementations must be thread-safe.
 */
typedef void (*TkLogCallback)(
    TkLogLevel level,
    uint64_t timestamp_us,
    uint32_t thread_id,
    const char* module,
    const char* message,
    void* user_data
);

/*
 * =============================================================================
 * GEOMETRIC AND SPATIAL DATA TYPES
 * =============================================================================
 */

/**
 * TkPoint2D - 2D point with floating-point coordinates
 */
typedef struct TkPoint2D {
    float x;  /* X coordinate */
    float y;  /* Y coordinate */
} TkPoint2D;

/**
 * TkPoint3D - 3D point with floating-point coordinates
 */
typedef struct TkPoint3D {
    float x;  /* X coordinate */
    float y;  /* Y coordinate */
    float z;  /* Z coordinate */
} TkPoint3D;

/**
 * TkBoundingBox - Axis-aligned bounding rectangle
 */
typedef struct TkBoundingBox {
    float x;       /* Left edge X coordinate */
    float y;       /* Top edge Y coordinate */
    float width;   /* Width of bounding box */
    float height;  /* Height of bounding box */
} TkBoundingBox;

/**
 * TkBoundingBox3D - 3D axis-aligned bounding box
 */
typedef struct TkBoundingBox3D {
    float x, y, z;           /* Minimum corner coordinates */
    float width, height, depth; /* Dimensions */
} TkBoundingBox3D;

/*
 * =============================================================================
 * VISION AND DETECTION DATA TYPES
 * =============================================================================
 */

/**
 * TkObjectClass - Predefined object classification types
 */
typedef enum TkObjectClass {
    TK_OBJECT_UNKNOWN = 0,
    
    /* People and body parts */
    TK_OBJECT_PERSON = 1,
    TK_OBJECT_FACE = 2,
    TK_OBJECT_HAND = 3,
    TK_OBJECT_GESTURE = 4,
    
    /* Vehicles */
    TK_OBJECT_CAR = 10,
    TK_OBJECT_TRUCK = 11,
    TK_OBJECT_BUS = 12,
    TK_OBJECT_MOTORCYCLE = 13,
    TK_OBJECT_BICYCLE = 14,
    
    /* Navigation obstacles */
    TK_OBJECT_OBSTACLE = 20,
    TK_OBJECT_POLE = 21,
    TK_OBJECT_BARRIER = 22,
    TK_OBJECT_STEP = 23,
    TK_OBJECT_CURB = 24,
    
    /* Assistive landmarks */
    TK_OBJECT_DOOR = 30,
    TK_OBJECT_WINDOW = 31,
    TK_OBJECT_STAIRS = 32,
    TK_OBJECT_ELEVATOR = 33,
    TK_OBJECT_SIGN = 34,
    TK_OBJECT_CROSSWALK = 35,
    
    /* Indoor navigation */
    TK_OBJECT_CHAIR = 40,
    TK_OBJECT_TABLE = 41,
    TK_OBJECT_COUNTER = 42,
    
    /* Custom/extended classes start at 1000 */
    TK_OBJECT_CUSTOM_BASE = 1000
} TkObjectClass;

/**
 * TkDetection - Single object detection result
 */
typedef struct TkDetection {
    uint32_t detection_id;     /* Unique detection identifier */
    TkObjectClass class_id;    /* Detected object class */
    float confidence_score;    /* Detection confidence [0.0, 1.0] */
    TkBoundingBox bbox_2d;     /* 2D bounding box in image space */
    TkBoundingBox3D bbox_3d;   /* 3D bounding box in world space */
    TkPoint3D center_point;    /* 3D center point in world coordinates */
    float distance_meters;     /* Estimated distance from camera */
    float velocity_mps;        /* Estimated velocity (meters per second) */
    uint64_t timestamp_us;     /* Detection timestamp (microseconds) */
    uint32_t tracking_id;      /* Multi-frame tracking ID (0 if not tracked) */
    
    /* Extended attributes */
    float orientation_degrees; /* Object orientation (-180 to 180 degrees) */
    bool is_moving;           /* Whether object is in motion */
    bool is_occluded;         /* Whether object is partially occluded */
} TkDetection;

/**
 * TkSceneAnalysis - High-level scene understanding results
 */
typedef struct TkSceneAnalysis {
    uint64_t timestamp_us;        /* Analysis timestamp */
    uint32_t total_objects;       /* Total number of detected objects */
    uint32_t people_count;        /* Number of people detected */
    uint32_t vehicle_count;       /* Number of vehicles detected */
    uint32_t obstacle_count;      /* Number of obstacles detected */
    
    /* Navigation assistance */
    bool path_clear_ahead;        /* Whether forward path is clear */
    float nearest_obstacle_distance; /* Distance to nearest obstacle (meters) */
    float recommended_heading;    /* Suggested heading angle (degrees) */
    
    /* Environmental factors */
    float lighting_level;         /* Ambient lighting estimate [0.0, 1.0] */
    bool low_visibility;          /* Poor visibility conditions detected */
    float crowd_density;          /* Estimated crowd density [0.0, 1.0] */
    
    /* Scene description for LLM */
    char scene_description[512];  /* Natural language scene description */
} TkSceneAnalysis;

/*
 * =============================================================================
 * AUDIO PROCESSING DATA TYPES
 * =============================================================================
 */

/**
 * TkAudioFormat - Audio format specification
 */
typedef enum TkAudioFormat {
    TK_AUDIO_FORMAT_UNKNOWN = 0,
    TK_AUDIO_FORMAT_PCM_S16LE = 1,    /* 16-bit signed little-endian PCM */
    TK_AUDIO_FORMAT_PCM_S32LE = 2,    /* 32-bit signed little-endian PCM */
    TK_AUDIO_FORMAT_PCM_F32LE = 3,    /* 32-bit float little-endian PCM */
    TK_AUDIO_FORMAT_PCM_F64LE = 4     /* 64-bit float little-endian PCM */
} TkAudioFormat;

/**
 * TkAudioChunk - Audio data chunk for processing
 */
typedef struct TkAudioChunk {
    const void* data;            /* Pointer to audio sample data */
    size_t data_size;            /* Size of audio data in bytes */
    TkAudioFormat format;        /* Audio format specification */
    uint32_t sample_rate;        /* Sample rate in Hz */
    uint32_t channels;           /* Number of audio channels */
    uint32_t samples_per_channel; /* Number of samples per channel */
    uint64_t timestamp_us;       /* Audio chunk timestamp */
} TkAudioChunk;

/**
 * TkAudioAnalysis - Audio scene analysis results
 */
typedef struct TkAudioAnalysis {
    uint64_t timestamp_us;       /* Analysis timestamp */
    
    /* Sound classification */
    bool speech_detected;        /* Human speech detected */
    bool music_detected;         /* Music or structured audio */
    bool noise_detected;         /* Unstructured noise */
    bool vehicle_sounds;         /* Vehicle engine/horn sounds */
    bool warning_sounds;         /* Sirens, alarms, etc. */
    
    /* Spatial audio information */
    float dominant_direction;    /* Direction of strongest audio source */
    float sound_level_db;        /* Sound pressure level in dB */
    uint32_t sound_source_count; /* Number of distinct sound sources */
    
    /* Voice activity detection */
    bool voice_activity;         /* Voice activity detected */
    float speech_clarity;        /* Speech clarity score [0.0, 1.0] */
    char transcribed_text[1024]; /* Transcribed speech (if available) */
} TkAudioAnalysis;

/*
 * =============================================================================
 * VIDEO PROCESSING DATA TYPES  
 * =============================================================================
 */

/**
 * TkPixelFormat - Supported pixel formats
 */
typedef enum TkPixelFormat {
    TK_PIXEL_FORMAT_UNKNOWN = 0,
    TK_PIXEL_FORMAT_RGB24 = 1,       /* 24-bit RGB */
    TK_PIXEL_FORMAT_RGBA32 = 2,      /* 32-bit RGBA */
    TK_PIXEL_FORMAT_BGR24 = 3,       /* 24-bit BGR */
    TK_PIXEL_FORMAT_BGRA32 = 4,      /* 32-bit BGRA */
    TK_PIXEL_FORMAT_YUV420P = 5,     /* Planar YUV 4:2:0 */
    TK_PIXEL_FORMAT_NV12 = 6,        /* Semi-planar YUV 4:2:0 */
    TK_PIXEL_FORMAT_GRAY8 = 7        /* 8-bit grayscale */
} TkPixelFormat;

/**
 * TkVideoFrame - Video frame data structure
 */
typedef struct TkVideoFrame {
    const uint8_t* data[4];      /* Plane data pointers (up to 4 planes) */
    uint32_t linesize[4];        /* Bytes per line for each plane */
    uint32_t width;              /* Frame width in pixels */
    uint32_t height;             /* Frame height in pixels */
    TkPixelFormat format;        /* Pixel format */
    uint64_t timestamp_us;       /* Frame timestamp in microseconds */
    uint64_t frame_number;       /* Sequential frame number */
    
    /* Camera intrinsics (for 3D reconstruction) */
    float focal_length_x;        /* Focal length X (pixels) */
    float focal_length_y;        /* Focal length Y (pixels) */
    float principal_point_x;     /* Principal point X (pixels) */
    float principal_point_y;     /* Principal point Y (pixels) */
    
    /* Depth information (if available) */
    const uint16_t* depth_data;  /* Depth map data (millimeters) */
    uint32_t depth_width;        /* Depth map width */
    uint32_t depth_height;       /* Depth map height */
} TkVideoFrame;

/*
 * =============================================================================
 * CALLBACK FUNCTION TYPES
 * =============================================================================
 */

/**
 * TkVisionResultCallback - Vision processing result callback
 *
 * @param context: System context that generated the results
 * @param detections: Array of detection results
 * @param detection_count: Number of detections in array
 * @param scene_analysis: High-level scene analysis results
 * @param user_data: User-provided context data
 *
 * Thread Safety: May be called from worker threads. Implementation must be
 * thread-safe and should not block for extended periods.
 */
typedef void (*TkVisionResultCallback)(
    TkContext* context,
    const TkDetection* detections,
    size_t detection_count,
    const TkSceneAnalysis* scene_analysis,
    void* user_data
);

/**
 * TkAudioResultCallback - Audio processing result callback
 *
 * @param context: System context that generated the results
 * @param audio_analysis: Audio analysis results
 * @param user_data: User-provided context data
 */
typedef void (*TkAudioResultCallback)(
    TkContext* context,
    const TkAudioAnalysis* audio_analysis,
    void* user_data
);

/**
 * TkLLMResponseCallback - LLM response callback
 *
 * @param context: System context that generated the response
 * @param request_id: Unique identifier for the original request
 * @param response_text: LLM response text (null-terminated)
 * @param response_length: Length of response text in bytes
 * @param is_final: Whether this is the final response chunk
 * @param user_data: User-provided context data
 */
typedef void (*TkLLMResponseCallback)(
    TkContext* context,
    uint64_t request_id,
    const char* response_text,
    size_t response_length,
    bool is_final,
    void* user_data
);

/**
 * TkErrorCallback - System error callback
 *
 * @param context: System context (may be NULL if error occurred during init)
 * @param error_code: TkResult error code
 * @param error_message: Human-readable error description
 * @param user_data: User-provided context data
 */
typedef void (*TkErrorCallback)(
    TkContext* context,
    TkResult error_code,
    const char* error_message,
    void* user_data
);

/*
 * =============================================================================
 * CONFIGURATION STRUCTURES
 * =============================================================================
 */

/**
 * TkGpuAcceleration - GPU acceleration backend selection
 */
typedef enum TkGpuAcceleration {
    TK_GPU_NONE = 0,        /* CPU-only processing */
    TK_GPU_AUTO = 1,        /* Automatic backend selection */
    TK_GPU_CUDA = 2,        /* NVIDIA CUDA */
    TK_GPU_METAL = 3,       /* Apple Metal */
    TK_GPU_ROCM = 4,        /* AMD ROCm */
    TK_GPU_OPENCL = 5       /* OpenCL (fallback) */
} TkGpuAcceleration;

/**
 * TkThreadingConfig - Threading configuration
 */
typedef struct TkThreadingConfig {
    uint32_t worker_thread_count;    /* Number of worker threads (0 = auto) */
    uint32_t gpu_stream_count;       /* Number of GPU streams for parallelism */
    bool enable_thread_affinity;     /* Enable CPU thread affinity */
    uint32_t priority_boost;         /* Thread priority boost (0-99) */
} TkThreadingConfig;

/**
 * TkVisionConfig - Vision processing configuration
 */
typedef struct TkVisionConfig {
    bool enable_detection;           /* Enable object detection */
    bool enable_tracking;            /* Enable multi-frame object tracking */
    bool enable_depth_estimation;    /* Enable depth estimation */
    bool enable_scene_analysis;      /* Enable high-level scene analysis */
    
    float detection_threshold;       /* Detection confidence threshold */
    float tracking_threshold;        /* Tracking confidence threshold */
    uint32_t max_detections_per_frame; /* Maximum detections to process */
    uint32_t tracking_history_frames;  /* Frames to keep for tracking */
    
    /* Model paths */
    const char* detection_model_path;   /* Path to detection model */
    const char* depth_model_path;       /* Path to depth estimation model */
    const char* tracking_model_path;    /* Path to tracking model */
} TkVisionConfig;

/**
 * TkAudioConfig - Audio processing configuration
 */
typedef struct TkAudioConfig {
    bool enable_processing;          /* Enable audio processing */
    bool enable_transcription;       /* Enable speech transcription */
    bool enable_spatial_audio;       /* Enable spatial audio analysis */
    
    TkAudioFormat preferred_format;  /* Preferred audio input format */
    uint32_t sample_rate;           /* Target sample rate */
    uint32_t buffer_size_ms;        /* Audio buffer size in milliseconds */
    
    float voice_activity_threshold; /* Voice activity detection threshold */
    float noise_suppression_level;  /* Noise suppression strength */
    
    /* Model paths */
    const char* transcription_model_path; /* Path to speech recognition model */
    const char* audio_classification_model_path; /* Path to audio classifier */
} TkAudioConfig;

/**
 * TkLLMConfig - Large Language Model configuration
 */
typedef struct TkLLMConfig {
    bool enable_llm;                 /* Enable LLM integration */
    
    /* Model configuration */
    const char* model_path;          /* Path to LLM model */
    const char* model_type;          /* Model type (e.g., "mistral", "llama") */
    uint32_t context_length;         /* Maximum context length */
    uint32_t max_response_tokens;    /* Maximum response length */
    
    /* Inference parameters */
    float temperature;               /* Sampling temperature */
    float top_p;                    /* Nucleus sampling parameter */
    uint32_t top_k;                 /* Top-k sampling parameter */
    
    /* Performance settings */
    uint32_t batch_size;            /* Inference batch size */
    bool use_gpu_inference;         /* Use GPU for LLM inference */
    uint32_t gpu_layers;            /* Number of layers to run on GPU */
} TkLLMConfig;

/*
 * =============================================================================
 * MAIN CONFIGURATION STRUCTURE
 * =============================================================================
 */

/**
 * TkConfig - Complete system configuration
 *
 * This structure must be initialized with tk_config_init_default() before use.
 * All string pointers must remain valid for the lifetime of the TkContext.
 */
struct TkConfig {
    /* System identification */
    uint32_t config_version;         /* Config structure version */
    const char* application_name;    /* Application name for logging */
    
    /* Resource management */
    size_t max_memory_usage_mb;      /* Maximum memory usage in MB */
    const char* cache_directory;     /* Directory for model caching */
    const char* temp_directory;      /* Temporary file directory */
    
    /* GPU acceleration */
    TkGpuAcceleration gpu_backend;   /* GPU acceleration backend */
    uint32_t gpu_device_id;         /* GPU device ID to use */
    
    /* Threading configuration */
    TkThreadingConfig threading;     /* Threading parameters */
    
    /* Subsystem configurations */
    TkVisionConfig vision;           /* Vision processing config */
    TkAudioConfig audio;             /* Audio processing config */
    TkLLMConfig llm;                /* LLM integration config */
    
    /* Logging and diagnostics */
    TkLogLevel log_level;           /* Minimum log level to output */
    TkLogCallback log_callback;     /* Log message callback */
    void* log_user_data;            /* User data for log callback */
    
    /* Event callbacks */
    TkVisionResultCallback vision_callback;     /* Vision results callback */
    void* vision_user_data;                     /* Vision callback user data */
    
    TkAudioResultCallback audio_callback;       /* Audio results callback */
    void* audio_user_data;                      /* Audio callback user data */
    
    TkLLMResponseCallback llm_callback;         /* LLM response callback */
    void* llm_user_data;                        /* LLM callback user data */
    
    TkErrorCallback error_callback;             /* Error callback */
    void* error_user_data;                      /* Error callback user data */
    
    /* Advanced options */
    bool enable_metrics;            /* Enable performance metrics collection */
    bool enable_debug_output;       /* Enable debug file output */
    uint32_t processing_timeout_ms; /* Processing timeout in milliseconds */
};

/*
 * =============================================================================
 * API FUNCTIONS - SYSTEM LIFECYCLE
 * =============================================================================
 */

/**
 * tk_config_create - Create a new configuration structure
 *
 * Allocates and initializes a new TkConfig structure with default values.
 * The returned configuration must be freed with tk_config_destroy().
 *
 * @param config: Pointer to receive the allocated configuration
 *
 * Returns: TK_SUCCESS on success, error code on failure
 *
 * Thread Safety: Thread-safe
 * Memory: Caller must call tk_config_destroy() to free memory
 */
TkResult tk_config_create(TkConfig** config)
    __attribute__((warn_unused_result)) __attribute__((nonnull(1)));

/**
 * tk_config_init_default - Initialize configuration with default values
 *
 * Fills the provided configuration structure with sensible default values.
 * This function should always be called before modifying config parameters.
 *
 * @param config: Configuration structure to initialize
 *
 * Returns: TK_SUCCESS on success, error code on failure
 *
 * Thread Safety: Not thread-safe (config structure access)
 */
TkResult tk_config_init_default(TkConfig* config)
    __attribute__((warn_unused_result)) __attribute__((nonnull(1)));

/**
 * tk_config_destroy - Free configuration structure
 *
 * Releases memory allocated for a configuration structure.
 * The pointer will be set to NULL after destruction.
 *
 * @param config: Pointer to configuration structure to destroy
 *
 * Thread Safety: Not thread-safe (config structure access)
 */
void tk_config_destroy(TkConfig** config) __attribute__((nonnull(1)));

/**
 * tk_context_create - Create and initialize TrackieLLM system context
 *
 * Initializes the complete TrackieLLM system with the provided configuration.
 * This includes GPU initialization, model loading, thread pool creation,
 * and subsystem startup. The operation may take several seconds.
 *
 * @param config: System configuration (must not be NULL)
 * @param context: Pointer to receive the created context
 *
 * Returns: TK_SUCCESS on success, specific error code on failure
 *
 * Error Conditions:
 * - TK_ERROR_NULL_POINTER: config or context is NULL
 * - TK_ERROR_CONFIG_INVALID: Invalid configuration parameters
 * - TK_ERROR_GPU_INIT_FAILED: GPU initialization failed
 * - TK_ERROR_MODEL_LOAD_FAILED: Model loading failed
 * - TK_ERROR_SYSTEM_INIT_FAILED: System initialization failed
 * - TK_ERROR_OUT_OF_MEMORY: Insufficient memory for initialization
 *
 * Thread Safety: Thread-safe
 * Memory: Caller must call tk_context_destroy() to free resources
 * Performance: May block for several seconds during initialization
 */
TkResult tk_context_create(const TkConfig* config, TkContext** context)
    __attribute__((warn_unused_result)) __attribute__((nonnull(1, 2)));

/**
 * tk_context_destroy - Destroy TrackieLLM system context
 *
 * Cleanly shuts down the TrackieLLM system, stopping all processing threads,
 * releasing GPU resources, unloading models, and freeing all associated memory.
 * Any pending operations will be cancelled. The pointer will be set to NULL.
 *
 * @param context: Pointer to context to destroy
 *
 * Thread Safety: Thread-safe, but blocks until all operations complete
 * Performance: May block for up to several seconds during cleanup
 */
void tk_context_destroy(TkContext** context) __attribute__((nonnull(1)));

/**
 * tk_context_start - Start system processing
 *
 * Activates all configured subsystems and begins processing incoming data.
 * Must be called after tk_context_create() and before processing functions.
 *
 * @param context: System context
 *
 * Returns: TK_SUCCESS on success, error code on failure
 *
 * Thread Safety: Thread-safe
 */
TkResult tk_context_start(TkContext* context)
    __attribute__((warn_unused_result)) __attribute__((nonnull(1)));

/**
 * tk_context_stop - Stop system processing
 *
 * Gracefully stops all processing threads and subsystems. Pending operations
 * will be completed before stopping. Can be restarted with tk_context_start().
 *
 * @param context: System context
 *
 * Returns: TK_SUCCESS on success, error code on failure
 *
 * Thread Safety: Thread-safe
 * Performance: May block until current operations complete
 */
TkResult tk_context_stop(TkContext* context)
    __attribute__((warn_unused_result)) __attribute__((nonnull(1)));

/**
 * tk_context_is_running - Check if system is actively processing
 *
 * @param context: System context
 *
 * Returns: true if system is running, false otherwise
 *
 * Thread Safety: Thread-safe
 */
bool tk_context_is_running(const TkContext* context) __attribute__((nonnull(1)));

/*
 * =============================================================================
 * API FUNCTIONS - STREAM MANAGEMENT
 * =============================================================================
 */

/**
 * TkStreamType - Type of media stream
 */
typedef enum TkStreamType {
    TK_STREAM_VIDEO = 1,      /* Video stream */
    TK_STREAM_AUDIO = 2,      /* Audio stream */
    TK_STREAM_MULTIMODAL = 3  /* Combined video and audio */
} TkStreamType;

/**
 * TkStreamConfig - Stream configuration
 */
typedef struct TkStreamConfig {
    TkStreamType stream_type;     /* Type of stream */
    const char* stream_name;      /* Human-readable stream name */
    uint32_t buffer_size_frames;  /* Internal buffer size in frames */
    bool enable_realtime_mode;    /* Enable real-time processing priority */
    float processing_fps_limit;   /* Maximum processing FPS (0 = unlimited) */
    void* stream_user_data;       /* User data passed to callbacks */
} TkStreamConfig;

/**
 * tk_stream_create - Create a new media stream
 *
 * Creates a new processing stream with the specified configuration.
 * Multiple streams can be processed concurrently by the same context.
 *
 * @param context: System context
 * @param config: Stream configuration
 * @param stream: Pointer to receive created stream handle
 *
 * Returns: TK_SUCCESS on success, error code on failure
 *
 * Thread Safety: Thread-safe
 * Memory: Caller must call tk_stream_destroy() to free resources
 */
TkResult tk_stream_create(TkContext* context, const TkStreamConfig* config, TkStream** stream)
    __attribute__((warn_unused_result)) __attribute__((nonnull(1, 2, 3)));

/**
 * tk_stream_destroy - Destroy a media stream
 *
 * Stops stream processing and releases all associated resources.
 * The pointer will be set to NULL after destruction.
 *
 * @param stream: Pointer to stream to destroy
 *
 * Thread Safety: Thread-safe
 */
void tk_stream_destroy(TkStream** stream) __attribute__((nonnull(1)));

/**
 * tk_stream_start - Start stream processing
 *
 * Begins processing data submitted to this stream. Callbacks will begin
 * firing as results become available.
 *
 * @param stream: Stream to start
 *
 * Returns: TK_SUCCESS on success, error code on failure
 *
 * Thread Safety: Thread-safe
 */
TkResult tk_stream_start(TkStream* stream)
    __attribute__((warn_unused_result)) __attribute__((nonnull(1)));

/**
 * tk_stream_stop - Stop stream processing
 *
 * Stops processing for this stream. Pending operations will complete.
 *
 * @param stream: Stream to stop
 *
 * Returns: TK_SUCCESS on success, error code on failure
 *
 * Thread Safety: Thread-safe
 */
TkResult tk_stream_stop(TkStream* stream)
    __attribute__((warn_unused_result)) __attribute__((nonnull(1)));

/*
 * =============================================================================
 * API FUNCTIONS - DATA PROCESSING
 * =============================================================================
 */

/**
 * tk_process_video_frame_async - Process video frame asynchronously
 *
 * Submits a video frame for asynchronous processing. Results will be delivered
 * via the registered TkVisionResultCallback. The function returns immediately
 * without blocking.
 *
 * @param stream: Target stream for processing
 * @param frame: Video frame data (must remain valid until processing completes)
 *
 * Returns: TK_SUCCESS if frame was queued, error code on failure
 *
 * Error Conditions:
 * - TK_ERROR_NULL_POINTER: stream or frame is NULL
 * - TK_ERROR_INVALID_HANDLE: stream is invalid or destroyed
 * - TK_ERROR_INVALID_STATE: stream is not started
 * - TK_ERROR_BUFFER_FULL: Processing queue is full
 * - TK_ERROR_INVALID_VIDEO_FORMAT: Unsupported video format
 *
 * Thread Safety: Thread-safe
 * Performance: Non-blocking, returns immediately
 * Memory: Frame data must remain valid until callback is invoked
 */
TkResult tk_process_video_frame_async(TkStream* stream, const TkVideoFrame* frame)
    __attribute__((warn_unused_result)) __attribute__((nonnull(1, 2)));

/**
 * tk_process_audio_chunk_async - Process audio chunk asynchronously
 *
 * Submits an audio chunk for asynchronous processing. Results will be delivered
 * via the registered TkAudioResultCallback.
 *
 * @param stream: Target stream for processing
 * @param chunk: Audio chunk data (must remain valid until processing completes)
 *
 * Returns: TK_SUCCESS if chunk was queued, error code on failure
 *
 * Thread Safety: Thread-safe
 * Performance: Non-blocking, returns immediately
 * Memory: Chunk data must remain valid until callback is invoked
 */
TkResult tk_process_audio_chunk_async(TkStream* stream, const TkAudioChunk* chunk)
    __attribute__((warn_unused_result)) __attribute__((nonnull(1, 2)));

/**
 * tk_process_multimodal_async - Process synchronized audio/video data
 *
 * Submits synchronized audio and video data for joint processing.
 * Enables advanced multimodal analysis and scene understanding.
 *
 * @param stream: Target stream for processing
 * @param frame: Video frame data (can be NULL for audio-only processing)
 * @param chunk: Audio chunk data (can be NULL for video-only processing)
 *
 * Returns: TK_SUCCESS if data was queued, error code on failure
 *
 * Thread Safety: Thread-safe
 * Performance: Non-blocking, returns immediately
 */
TkResult tk_process_multimodal_async(TkStream* stream, const TkVideoFrame* frame, const TkAudioChunk* chunk)
    __attribute__((warn_unused_result)) __attribute__((nonnull(1)));

/*
 * =============================================================================
 * API FUNCTIONS - LLM INTEGRATION
 * =============================================================================
 */

/**
 * TkLLMRequest - LLM inference request
 */
typedef struct TkLLMRequest {
    uint64_t request_id;         /* Unique request identifier */
    const char* prompt;          /* Input prompt text */
    size_t prompt_length;        /* Length of prompt in bytes */
    const char* system_prompt;   /* Optional system prompt */
    float temperature;           /* Sampling temperature (0.0 = deterministic) */
    uint32_t max_tokens;        /* Maximum response tokens */
    bool stream_response;        /* Enable streaming response */
    void* user_data;            /* User data for callback */
} TkLLMRequest;

/**
 * tk_llm_inference_async - Submit LLM inference request
 *
 * Submits a text prompt to the configured LLM for inference. Results will be
 * delivered via the registered TkLLMResponseCallback. Supports streaming
 * responses for real-time interaction.
 *
 * @param context: System context
 * @param request: LLM inference request
 *
 * Returns: TK_SUCCESS if request was queued, error code on failure
 *
 * Thread Safety: Thread-safe
 * Performance: Non-blocking, returns immediately
 */
TkResult tk_llm_inference_async(TkContext* context, const TkLLMRequest* request)
    __attribute__((warn_unused_result)) __attribute__((nonnull(1, 2)));

/**
 * tk_llm_cancel_request - Cancel pending LLM request
 *
 * Attempts to cancel a pending LLM inference request. If the request is
 * already being processed, cancellation may not be immediate.
 *
 * @param context: System context
 * @param request_id: ID of request to cancel
 *
 * Returns: TK_SUCCESS if cancellation was successful
 *
 * Thread Safety: Thread-safe
 */
TkResult tk_llm_cancel_request(TkContext* context, uint64_t request_id)
    __attribute__((warn_unused_result)) __attribute__((nonnull(1)));

/*
 * =============================================================================
 * API FUNCTIONS - SYSTEM QUERY AND CONTROL
 * =============================================================================
 */

/**
 * TkSystemStats - System performance statistics
 */
typedef struct TkSystemStats {
    /* Processing statistics */
    uint64_t frames_processed;       /* Total video frames processed */
    uint64_t audio_chunks_processed; /* Total audio chunks processed */
    uint64_t llm_requests_processed; /* Total LLM requests processed */
    
    /* Performance metrics */
    float average_frame_time_ms;     /* Average frame processing time */
    float average_fps;               /* Average processing FPS */
    uint32_t dropped_frames;         /* Number of dropped frames */
    uint32_t queue_depth;            /* Current processing queue depth */
    
    /* Resource utilization */
    float cpu_usage_percent;         /* CPU usage percentage */
    float gpu_usage_percent;         /* GPU usage percentage */
    size_t memory_usage_mb;          /* Memory usage in MB */
    size_t gpu_memory_usage_mb;      /* GPU memory usage in MB */
    
    /* Error statistics */
    uint32_t total_errors;           /* Total number of errors */
    uint32_t recent_errors;          /* Errors in last minute */
} TkSystemStats;

/**
 * tk_get_system_stats - Get system performance statistics
 *
 * Retrieves current system performance and resource utilization statistics.
 *
 * @param context: System context
 * @param stats: Pointer to receive statistics
 *
 * Returns: TK_SUCCESS on success, error code on failure
 *
 * Thread Safety: Thread-safe
 */
TkResult tk_get_system_stats(TkContext* context, TkSystemStats* stats)
    __attribute__((warn_unused_result)) __attribute__((nonnull(1, 2)));

/**
 * tk_reset_system_stats - Reset performance statistics
 *
 * Resets all performance counters and statistics to zero.
 *
 * @param context: System context
 *
 * Returns: TK_SUCCESS on success, error code on failure
 *
 * Thread Safety: Thread-safe
 */
TkResult tk_reset_system_stats(TkContext* context)
    __attribute__((warn_unused_result)) __attribute__((nonnull(1)));

/**
 * TkSystemInfo - System capability information
 */
typedef struct TkSystemInfo {
    /* Version information */
    uint32_t version_major;          /* Major version number */
    uint32_t version_minor;          /* Minor version number */
    uint32_t version_patch;          /* Patch version number */
    const char* version_string;      /* Full version string */
    const char* build_timestamp;     /* Build timestamp */
    const char* git_commit;          /* Git commit hash */
    
    /* GPU capabilities */
    TkGpuAcceleration available_gpu_backends; /* Available GPU backends */
    TkGpuAcceleration active_gpu_backend;     /* Currently active backend */
    const char* gpu_device_name;     /* GPU device name */
    size_t gpu_memory_total_mb;      /* Total GPU memory in MB */
    
    /* System capabilities */
    uint32_t cpu_core_count;         /* Number of CPU cores */
    size_t system_memory_mb;         /* Total system memory in MB */
    bool supports_depth_estimation;  /* Depth estimation support */
    bool supports_audio_transcription; /* Audio transcription support */
    bool supports_llm_inference;     /* LLM inference support */
} TkSystemInfo;

/**
 * tk_get_system_info - Get system capability information
 *
 * Retrieves information about system capabilities, hardware, and build version.
 *
 * @param info: Pointer to receive system information
 *
 * Returns: TK_SUCCESS on success, error code on failure
 *
 * Thread Safety: Thread-safe
 * Note: This function can be called without a context
 */
TkResult tk_get_system_info(TkSystemInfo* info)
    __attribute__((warn_unused_result)) __attribute__((nonnull(1)));

/*
 * =============================================================================
 * API FUNCTIONS - UTILITY AND HELPER FUNCTIONS
 * =============================================================================
 */

/**
 * tk_result_to_string - Convert result code to human-readable string
 *
 * Converts a TkResult error code to a descriptive string for logging
 * and error reporting.
 *
 * @param result: Result code to convert
 *
 * Returns: Null-terminated string describing the result code
 *
 * Thread Safety: Thread-safe
 * Memory: Returned string is statically allocated and does not need to be freed
 */
const char* tk_result_to_string(TkResult result) __attribute__((const));

/**
 * tk_log_level_to_string - Convert log level to string
 *
 * @param level: Log level to convert
 *
 * Returns: String representation of log level
 *
 * Thread Safety: Thread-safe
 */
const char* tk_log_level_to_string(TkLogLevel level) __attribute__((const));

/**
 * tk_object_class_to_string - Convert object class to string
 *
 * @param class_id: Object class to convert
 *
 * Returns: String representation of object class
 *
 * Thread Safety: Thread-safe
 */
const char* tk_object_class_to_string(TkObjectClass class_id) __attribute__((const));

/**
 * tk_pixel_format_to_string - Convert pixel format to string
 *
 * @param format: Pixel format to convert
 *
 * Returns: String representation of pixel format
 *
 * Thread Safety: Thread-safe
 */
const char* tk_pixel_format_to_string(TkPixelFormat format) __attribute__((const));

/**
 * tk_calculate_frame_size - Calculate frame data size
 *
 * Calculates the total size in bytes required for a video frame with
 * the specified dimensions and pixel format.
 *
 * @param width: Frame width in pixels
 * @param height: Frame height in pixels
 * @param format: Pixel format
 *
 * Returns: Required size in bytes, or 0 if format is invalid
 *
 * Thread Safety: Thread-safe
 */
size_t tk_calculate_frame_size(uint32_t width, uint32_t height, TkPixelFormat format)
    __attribute__((const));

/**
 * tk_validate_video_frame - Validate video frame structure
 *
 * Validates that a video frame structure contains consistent and valid data.
 * Useful for debugging and ensuring data integrity.
 *
 * @param frame: Video frame to validate
 *
 * Returns: TK_SUCCESS if frame is valid, error code describing problem
 *
 * Thread Safety: Thread-safe
 */
TkResult tk_validate_video_frame(const TkVideoFrame* frame)
    __attribute__((warn_unused_result)) __attribute__((nonnull(1)));

/**
 * tk_validate_audio_chunk - Validate audio chunk structure
 *
 * Validates that an audio chunk structure contains consistent and valid data.
 *
 * @param chunk: Audio chunk to validate
 *
 * Returns: TK_SUCCESS if chunk is valid, error code describing problem
 *
 * Thread Safety: Thread-safe
 */
TkResult tk_validate_audio_chunk(const TkAudioChunk* chunk)
    __attribute__((warn_unused_result)) __attribute__((nonnull(1)));

/**
 * tk_get_timestamp_us - Get current timestamp in microseconds
 *
 * Returns a high-resolution timestamp suitable for frame timing and
 * performance measurement.
 *
 * Returns: Current timestamp in microseconds since system start
 *
 * Thread Safety: Thread-safe
 */
uint64_t tk_get_timestamp_us(void) __attribute__((const));

/**
 * tk_sleep_ms - Sleep for specified milliseconds
 *
 * High-resolution sleep function that works across platforms.
 *
 * @param milliseconds: Number of milliseconds to sleep
 *
 * Thread Safety: Thread-safe
 */
void tk_sleep_ms(uint32_t milliseconds);

/*
 * =============================================================================
 * API FUNCTIONS - ADVANCED FEATURES
 * =============================================================================
 */

/**
 * tk_set_gpu_memory_limit - Set GPU memory usage limit
 *
 * Dynamically adjusts the maximum GPU memory that TrackieLLM will use.
 * Useful for systems with shared GPU memory or multiple applications.
 *
 * @param context: System context
 * @param limit_mb: Memory limit in megabytes (0 = no limit)
 *
 * Returns: TK_SUCCESS on success, error code on failure
 *
 * Thread Safety: Thread-safe
 */
TkResult tk_set_gpu_memory_limit(TkContext* context, size_t limit_mb)
    __attribute__((warn_unused_result)) __attribute__((nonnull(1)));

/**
 * tk_force_gpu_synchronize - Force GPU synchronization
 *
 * Forces synchronization with the GPU, ensuring all pending operations
 * are completed. Useful for timing measurements and debugging.
 *
 * @param context: System context
 *
 * Returns: TK_SUCCESS on success, error code on failure
 *
 * Thread Safety: Thread-safe
 * Performance: Blocking operation, may take several milliseconds
 */
TkResult tk_force_gpu_synchronize(TkContext* context)
    __attribute__((warn_unused_result)) __attribute__((nonnull(1)));

/**
 * tk_enable_debug_mode - Enable/disable debug mode
 *
 * Enables or disables debug mode, which provides additional logging,
 * validation, and diagnostic information at the cost of performance.
 *
 * @param context: System context
 * @param enable: true to enable debug mode, false to disable
 *
 * Returns: TK_SUCCESS on success, error code on failure
 *
 * Thread Safety: Thread-safe
 */
TkResult tk_enable_debug_mode(TkContext* context, bool enable)
    __attribute__((warn_unused_result)) __attribute__((nonnull(1)));

/*
 * =============================================================================
 * VERSION INFORMATION FUNCTIONS
 * =============================================================================
 */

/**
 * trackiellm_version_major - Get major version number
 *
 * Returns: Major version number
 *
 * Thread Safety: Thread-safe
 */
uint32_t trackiellm_version_major(void) __attribute__((const));

/**
 * trackiellm_version_minor - Get minor version number
 *
 * Returns: Minor version number
 *
 * Thread Safety: Thread-safe
 */
uint32_t trackiellm_version_minor(void) __attribute__((const));

/**
 * trackiellm_version_patch - Get patch version number
 *
 * Returns: Patch version number
 *
 * Thread Safety: Thread-safe
 */
uint32_t trackiellm_version_patch(void) __attribute__((const));

/**
 * trackiellm_version_string - Get full version string
 *
 * Returns: Full version string (e.g., "1.0.0-beta.1")
 *
 * Thread Safety: Thread-safe
 * Memory: Returned string is statically allocated
 */
const char* trackiellm_version_string(void) __attribute__((const));

/**
 * trackiellm_build_timestamp - Get build timestamp
 *
 * Returns: ISO 8601 timestamp of when the library was built
 *
 * Thread Safety: Thread-safe
 * Memory: Returned string is statically allocated
 */
const char* trackiellm_build_timestamp(void) __attribute__((const));

/**
 * trackiellm_git_commit - Get git commit hash
 *
 * Returns: Git commit hash of the build (short format)
 *
 * Thread Safety: Thread-safe
 * Memory: Returned string is statically allocated
 */
const char* trackiellm_git_commit(void) __attribute__((const));

#ifdef __cplusplus
}
#endif

#endif /* TRACKIELLM_H */