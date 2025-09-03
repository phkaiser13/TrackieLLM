# Mirror-Trad: A Simple Translation Tool

This tool is a simple Python script that scans a repository, extracts comments and docstrings, and generates translated versions of the files.

## Setup

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The tool is configured using the `config.yaml` file. Here are the available options:

*   `source_dir`: The directory to scan for files to translate.
*   `output_dir`: The directory where the translated files will be saved.
*   `target_language`: The target language for translation (e.g., 'pt' for Portuguese).
*   `translation_api_key`: Your API key for the translation service.
*   `exclude`: A list of files and directories to exclude from translation.
*   `include_extensions`: A list of file extensions to include in the translation.
*   `single_file`: (Optional) If you want to process only a single file, you can specify its path here. This is useful for testing.

## How to Run

To run the translation tool, simply execute the `translate.py` script from the root of the repository:

```bash
python3 .docs/mirror-trad/translate.py
```

## Integrating a Real Translation API

The `translate_text` function in `translate.py` currently uses a simulated translation. To use a real translation service, you will need to modify this function to call your chosen API.

For example, you could use a library like `deepl` or `google-cloud-translate` to make the API calls. You will need to pass your API key from the `config.yaml` file to the translation function.

## Limitations

This tool uses a simple string replacement method to substitute the original comments with their translated versions. This approach has some limitations:

*   If a comment appears multiple times in a file, all occurrences will be replaced with the same translation.
*   If a comment is a substring of another part of the file (e.g., a string literal), it might also be replaced.

For a more robust solution, a more sophisticated parsing and replacement mechanism would be required, such as using an Abstract Syntax Tree (AST) for code files.
