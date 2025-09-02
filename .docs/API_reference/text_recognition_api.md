<!--
This documentation was written by Jules - Google labs bot.
Original code by phkaiser13.
-->

# Text Recognition API Reference

This document provides a detailed reference for the Text Recognition (OCR) module's C API. This API allows you to integrate and control the Tesseract-based OCR functionality within the TrackieLLM platform.

## Developer and Author

*   **Original Code:** phkaiser13
*   **Documentation:** Jules - Google labs bot

---

## Enums

### `tk_ocr_language_e`

Specifies the supported languages for OCR.

| Value                          | Description        |
| ------------------------------ | ------------------ |
| `TK_OCR_LANG_UNKNOWN`          | Unknown language   |
| `TK_OCR_LANG_ENGLISH`          | English            |
| `TK_OCR_LANG_PORTUGUESE`       | Portuguese         |
| `TK_OCR_LANG_SPANISH`          | Spanish            |
| `TK_OCR_LANG_FRENCH`           | French             |
| `TK_OCR_LANG_GERMAN`           | German             |
| ...                            | (and so on)        |
| `TK_OCR_LANG_BRAZILIAN_PORTUGUESE` | Brazilian Portuguese (alias for `TK_OCR_LANG_PORTUGUESE`) |

### `tk_ocr_engine_mode_e`

Defines the OCR engine mode to be used by Tesseract.

| Value                      | Description                             |
| -------------------------- | --------------------------------------- |
| `TK_OCR_ENGINE_DEFAULT`    | Default engine mode                     |
| `TK_OCR_ENGINE_LSTM_ONLY`  | Use only the LSTM neural network engine |
| `TK_OCR_ENGINE_TESSERACT_ONLY` | Use only the legacy Tesseract engine    |
| `TK_OCR_ENGINE_TESSERACT_LSTM` | Use both the legacy and LSTM engines  |

### `tk_ocr_page_seg_mode_e`

Defines the page segmentation mode, which controls how Tesseract processes the image.

| Value                        | Description                                      |
| ---------------------------- | ------------------------------------------------ |
| `TK_OCR_PSM_OSD_ONLY`        | Orientation and script detection only.           |
| `TK_OCR_PSM_AUTO_OSD`        | Automatic page segmentation with OSD.            |
| `TK_OCR_PSM_AUTO`            | Fully automatic page segmentation.               |
| `TK_OCR_PSM_SINGLE_COLUMN`   | Assume a single column of text.                  |
| `TK_OCR_PSM_SINGLE_BLOCK`    | Assume a single uniform block of text.           |
| `TK_OCR_PSM_SINGLE_LINE`     | Treat the image as a single text line.           |
| `TK_OCR_PSM_SINGLE_WORD`     | Treat the image as a single word.                |
| `TK_OCR_PSM_SINGLE_CHAR`     | Treat the image as a single character.           |
| ...                          | (and other modes)                                |

---

## Data Structures

### `tk_text_recognition_context_t`

An opaque handle representing the Text Recognition context. It holds the internal state of the OCR engine, including loaded models and configuration. It is created by `tk_text_recognition_create` and destroyed by `tk_text_recognition_destroy`.

### `tk_ocr_config_t`

A large structure that holds the configuration for the OCR module. This structure is passed to `tk_text_recognition_create` to initialize the context. It contains over 50 fields to fine-tune every aspect of the OCR process, from preprocessing and language selection to result caching and debugging.

**Key fields:**
*   `data_path`: Path to Tesseract's `.traineddata` files.
*   `language`: The primary OCR language.
*   `engine_mode`: The OCR engine mode to use.
*   `psm`: The page segmentation mode.
*   `enable_preprocessing`: A boolean to enable/disable automatic image preprocessing.
*   `enable_gpu_acceleration`: A boolean to enable/disable GPU acceleration.
*   ... and many more.

### `tk_ocr_image_params_t`

Describes the image to be processed.

| Field        | Type           | Description                                  |
| ------------ | -------------- | -------------------------------------------- |
| `image_data` | `const uint8_t*` | Pointer to the raw image data.               |
| `width`      | `uint32_t`     | The width of the image in pixels.            |
| `height`     | `uint32_t`     | The height of the image in pixels.           |
| `channels`   | `uint32_t`     | Number of color channels (1, 3, or 4).       |
| `stride`     | `uint32_t`     | The number of bytes per row of the image.    |
| ...          |                |                                              |

### `tk_ocr_text_region_t`

Represents a single recognized region of text in the image.

| Field        | Type       | Description                                      |
| ------------ | ---------- | ------------------------------------------------ |
| `x`, `y`, `width`, `height` | `uint32_t` | The bounding box of the region.      |
| `confidence` | `float`    | The confidence score of the recognition (0-1).   |
| `text`       | `char*`    | The recognized text string.                      |
| ...          |            |                                                  |

### `tk_ocr_result_t`

Contains the complete result of an OCR operation.

| Field              | Type                      | Description                                        |
| ------------------ | ------------------------- | -------------------------------------------------- |
| `full_text`        | `char*`                   | The full recognized text from the entire image.    |
| `regions`          | `tk_ocr_text_region_t*`   | An array of recognized text regions.               |
| `region_count`     | `size_t`                  | The number of regions in the `regions` array.      |
| `average_confidence` | `float`                 | The average confidence score for all regions.      |
| `processing_time_ms`| `float`                  | The time taken for the OCR process in milliseconds.|
| ...                |                           |                                                    |

---

## Functions

### Context Lifecycle

#### `tk_text_recognition_create()`

Creates and initializes a new Text Recognition context.

```c
TK_NODISCARD tk_error_code_t tk_text_recognition_create(
    tk_text_recognition_context_t** out_context,
    const tk_ocr_config_t* config
);
```

*   **`out_context`**: A pointer to receive the address of the newly created context.
*   **`config`**: The configuration for the OCR module.
*   **Returns**: `TK_SUCCESS` on success, or an error code on failure.

#### `tk_text_recognition_destroy()`

Destroys a Text Recognition context and frees all associated resources.

```c
void tk_text_recognition_destroy(tk_text_recognition_context_t** context);
```

*   **`context`**: A pointer to the context to be destroyed.

### OCR Processing

#### `tk_text_recognition_process_image()`

Recognizes text in an entire image.

```c
TK_NODISCARD tk_error_code_t tk_text_recognition_process_image(
    tk_text_recognition_context_t* context,
    const tk_ocr_image_params_t* image_params,
    tk_ocr_result_t** out_result
);
```

*   **`context`**: The OCR context.
*   **`image_params`**: Parameters describing the image to process.
*   **`out_result`**: A pointer to receive the OCR result. The result must be freed with `tk_text_recognition_free_result`.
*   **Returns**: `TK_SUCCESS` on success, or an error code on failure.

#### `tk_text_recognition_process_region()`

Recognizes text in a specific region of an image.

```c
TK_NODISCARD tk_error_code_t tk_text_recognition_process_region(
    tk_text_recognition_context_t* context,
    const tk_ocr_image_params_t* image_params,
    uint32_t region_x,
    uint32_t region_y,
    uint32_t region_width,
    uint32_t region_height,
    tk_ocr_result_t** out_result
);
```

*   ... (arguments similar to `process_image`, with added region parameters)

#### `tk_text_recognition_free_result()`

Frees the memory allocated for an OCR result.

```c
void tk_text_recognition_free_result(tk_ocr_result_t** result);
```

*   **`result`**: A pointer to the result to be freed.

### Configuration and Management

#### `tk_text_recognition_set_language()`

Changes the language used for OCR at runtime.

```c
TK_NODISCARD tk_error_code_t tk_text_recognition_set_language(
    tk_text_recognition_context_t* context,
    tk_ocr_language_e language
);
```

#### `tk_text_recognition_get_model_info()`

Retrieves information about the currently loaded OCR models.

```c
TK_NODISCARD tk_error_code_t tk_text_recognition_get_model_info(
    tk_text_recognition_context_t* context,
    tk_ocr_language_e* out_language,
    char** out_engine_version,
    char** out_model_path
);
```

#### `tk_text_recognition_update_config()`

Updates the OCR configuration at runtime.

```c
TK_NODISCARD tk_error_code_t tk_text_recognition_update_config(
    tk_text_recognition_context_t* context,
    const tk_ocr_config_t* config
);
```
