divert(-1)
#
# This file contains M4 macros for generating documentation for the
# data structures used in the Text Recognition API.
#
# To generate the markdown, process this file with M4:
#   m4 text_recognition_structures.m4 > text_recognition_structures.md
#
# This documentation was written by Jules - Google labs bot.
# Original code by phkaiser13.
#

# Macro to generate a documentation section for a C struct.
# Arguments:
#   $1: The name of the struct (e.g., tk_ocr_config_t)
#   $2: A short description of the struct.
#   $3: The body of the documentation, typically a markdown table
#       describing the fields of the struct.
define(`GENERATE_STRUCT_DOC',
`### `$1`

`$2`

`$3`
')

divert(0)dnl
# Data Structure Documentation

GENERATE_STRUCT_DOC(`tk_ocr_config_t`,
`Configuration for initializing the OCR module. This structure is passed to tk_text_recognition_create.`,
`| Field                   | Type     | Description                                         |
| ----------------------- | -------- | --------------------------------------------------- |
| data_path               | tk_path_t* | Path to Tesseract data files (.traineddata).        |
| language                | tk_ocr_language_e | Primary language for OCR.                    |
| additional_languages    | char*    | Comma-separated list of additional languages.       |
| engine_mode             | tk_ocr_engine_mode_e | OCR engine mode to use.                  |
| psm                     | tk_ocr_page_seg_mode_e | Page segmentation mode.               |
| enable_preprocessing    | bool     | If true, enables automatic image preprocessing.     |
| ...                     | ...      | (many more configuration options)                   |`
)

GENERATE_STRUCT_DOC(`tk_ocr_image_params_t`,
`Parameters describing the image to be processed by an OCR function.`,
`| Field        | Type           | Description                                  |
| ------------ | -------------- | -------------------------------------------- |
| image_data   | const uint8_t* | Pointer to the raw image data.               |
| width        | uint32_t       | The width of the image in pixels.            |
| height       | uint32_t       | The height of the image in pixels.           |
| channels     | uint32_t       | Number of color channels (1, 3, or 4).       |
| stride       | uint32_t       | The number of bytes per row of the image.    |`
)

GENERATE_STRUCT_DOC(`tk_ocr_text_region_t`,
`Represents a single recognized region of text within the image.`,
`| Field        | Type     | Description                                      |
| ------------ | -------- | ------------------------------------------------ |
| x, y, width, height | uint32_t | The bounding box of the text region.        |
| confidence   | float    | The confidence score of the recognition (0-1).   |
| text         | char*    | The recognized text string for this region.      |
| is_handwritten | bool   | True if the text is likely handwritten.        |`
)

GENERATE_STRUCT_DOC(`tk_ocr_result_t`,
`Contains the complete result of an OCR operation on an image.`,
`| Field              | Type                      | Description                                        |
| ------------------ | ------------------------- | -------------------------------------------------- |
| full_text          | char*                     | The full recognized text from the entire image.    |
| regions            | tk_ocr_text_region_t*     | An array of recognized text regions.               |
| region_count       | size_t                    | The number of regions in the `regions` array.      |
| average_confidence | float                     | The average confidence score for all regions.      |
| processing_time_ms | float                     | The time taken for the OCR process in milliseconds.|`
)
