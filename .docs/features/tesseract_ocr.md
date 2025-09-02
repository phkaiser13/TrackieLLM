<!--
This documentation was written by Jules - Google labs bot.
Original code by phkaiser13.
-->

# Text Recognition (OCR) with Tesseract

## Overview

The Text Recognition module provides Optical Character Recognition (OCR) capabilities to the TrackieLLM platform. It is designed to extract textual information from images captured by the device's camera, enabling a new layer of environmental understanding for the user. This feature is crucial for reading signs, documents, labels, and other text in the real world.

The core of this module is built upon the powerful **Tesseract OCR engine**, a widely respected open-source OCR library. Our implementation is highly optimized for performance on embedded devices, ensuring low-latency processing without sacrificing accuracy.

## Key Features

*   **High-Accuracy Text Extraction:** Leverages Tesseract's advanced LSTM-based OCR engine to accurately recognize text in a variety of fonts, languages, and conditions.
*   **Multi-Language Support:** The module is designed to be multilingual, with initial support for several languages including English, Portuguese, Spanish, and more. The language can be configured at runtime.
*   **Extensive Preprocessing Options:** A rich set of image preprocessing options are available to improve OCR accuracy in challenging conditions. These include:
    *   Automatic image rotation and deskewing.
    *   Contrast enhancement, noise reduction, and sharpening.
    *   Image binarization and inversion (for light text on dark backgrounds).
*   **Detailed Result Analysis:** The OCR results are not just plain text. The API provides detailed information about the recognized text, including:
    *   Bounding boxes for text regions, lines, and words.
    *   Confidence scores for each recognition.
    *   Font attributes and text orientation.
*   **Flexible Configuration:** The module is highly configurable through the `tk_ocr_config_t` structure, allowing developers to fine-tune the OCR process for specific use cases and hardware capabilities.
*   **GPU Acceleration:** Where available, the module can leverage GPU acceleration to further speed up the OCR process.

## Integration with the Vision Pipeline

The Text Recognition module is designed to be a component of the main `vision_pipeline`. It can be invoked on-demand to process frames from the camera feed. The typical workflow is as follows:

1.  An image frame is captured from the camera.
2.  The frame is passed to the `tk_text_recognition_process_image` function.
3.  The module preprocesses the image and runs the Tesseract OCR engine.
4.  The recognized text and associated metadata are returned in a `tk_ocr_result_t` structure.
5.  The `cortex` (reasoning engine) can then use this information to provide assistance to the user, for example, by reading the text aloud.

## Developer and Author

*   **Original Code:** phkaiser13
*   **Documentation:** Jules - Google labs bot
