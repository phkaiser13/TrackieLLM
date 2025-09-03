import os
import re
import time
import yaml
import chardet
from markdown_it import MarkdownIt

def load_config():
    """Loads the configuration from config.yaml."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def translate_text(text, target_language, api_key):
    """
    Translates a given text to the target language.
    This is a simulated function that does not actually call the API.
    """
    if not text.strip():
        return text
    # In a real scenario, you would use the api_key to authenticate with a service.
    # For example:
    # import deepl
    # translator = deepl.Translator(api_key)
    # return translator.translate_text(text, target_lang=target_language).text
    return f"[Translated] {text.strip()}"

def render_markdown_tokens(tokens, target_language, api_key):
    """Renders a token stream back to Markdown."""
    result = ""
    for i, token in enumerate(tokens):
        if token.type == 'heading_open':
            result += '#' * int(token.tag[1:]) + ' '
        elif token.type == 'paragraph_open':
            pass
        elif token.type == 'paragraph_close':
            result += '\n\n'
        elif token.type == 'inline':
            for child in token.children:
                if child.type == 'text':
                    result += translate_text(child.content, target_language, api_key)
                else:
                    result += child.content
            if i > 0 and tokens[i-1].type == 'heading_open':
                result += '\n\n'

        elif token.type == 'code_block':
            result += f"```{token.info}\n{token.content}```\n"
        elif token.type == 'fence':
             result += f"```{token.info}\n{token.content}```\n"
        else:
            result += token.content if token.content else ''
    return result


def process_markdown_file(filepath, config, target_language, api_key):
    """Processes a single markdown file."""
    source_dir = config['source_dir']
    output_dir = os.path.join(config['output_dir'], target_language)


    try:
        with open(filepath, 'rb') as f:
            raw_data = f.read()
            encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
        
        original_content = raw_data.decode(encoding, errors='ignore')

        md = MarkdownIt()
        tokens = md.parse(original_content)
        
        new_content = render_markdown_tokens(tokens, target_language, api_key)

        relative_path = os.path.relpath(filepath, source_dir)
        output_filepath = os.path.join(output_dir, relative_path)
        output_dir_for_file = os.path.dirname(output_filepath)
        os.makedirs(output_dir_for_file, exist_ok=True)

        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        # print(f"  -> Saved translated file to: {output_filepath}")

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")

def process_file(filepath, config, target_language, api_key):
    """Processes a single file."""
    _, extension = os.path.splitext(filepath)
    if extension == '.md':
        process_markdown_file(filepath, config, target_language, api_key)
        return

    source_dir = config['source_dir']
    output_dir = os.path.join(config['output_dir'], target_language)


    try:
        with open(filepath, 'rb') as f:
            raw_data = f.read()
            encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
        
        original_content = raw_data.decode(encoding, errors='ignore')
        
        matches = list(re.finditer(r'(?://\s*(.*))|(?:\/\*([\s\S]*?)\*\/)|(?:#\s*(.*))|(?:"""([\s\S]*?)""")|(?:\'\'\'([\s\S]*?)\'\'\')', original_content))
        if not matches:
            new_content = original_content
        else:
            # print(f"Found {len(matches)} comments/text blocks in {filepath}")
            
            new_content = original_content
            for match in reversed(matches):
                start, end = match.span()
                
                comment_text = ""
                for group in match.groups():
                    if group is not None:
                        comment_text = group
                        break
                
                translated_text = translate_text(comment_text, target_language, api_key)

                original_comment = match.group(0)
                if original_comment.startswith('//'):
                    reconstructed_comment = f"// {translated_text}"
                elif original_comment.startswith('/*'):
                    lines = comment_text.strip().split('\n')
                    translated_lines = [f" * {line.strip()}" for line in translated_text.split('\n')]
                    reconstructed_comment = f"/*\n{'\n'.join(translated_lines)}\n */"
                elif original_comment.startswith('#'):
                    reconstructed_comment = f"# {translated_text}"
                elif original_comment.startswith('"""'):
                    reconstructed_comment = f'"""{translated_text}"""'
                elif original_comment.startswith("'''"):
                    reconstructed_comment = f"'''{translated_text}'''"
                else:
                    reconstructed_comment = translated_text

                new_content = new_content[:start] + reconstructed_comment + new_content[end:]

        relative_path = os.path.relpath(filepath, source_dir)
        output_filepath = os.path.join(output_dir, relative_path)
        output_dir_for_file = os.path.dirname(output_filepath)
        os.makedirs(output_dir_for_file, exist_ok=True)

        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        # print(f"  -> Saved translated file to: {output_filepath}")

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")

def scan_files(config, target_language, api_key):
    """Scans for files to translate based on the configuration."""
    source_dir = config['source_dir']
    
    exclude_list = [os.path.normpath(os.path.join(source_dir, path)) for path in config['exclude']]
    include_extensions = tuple(config['include_extensions'])
    
    for root, dirs, files in os.walk(source_dir, topdown=True):
        dirs[:] = [d for d in dirs if os.path.normpath(os.path.join(root, d)) not in exclude_list]
        
        for file in files:
            filepath = os.path.join(root, file)
            if file.endswith(include_extensions) and not any(os.path.normpath(filepath).startswith(ex_path) for ex_path in exclude_list):
                process_file(filepath, config, target_language, api_key)


def main():
    """
    Main function to run the translation process.
    """
    print("Starting translation process...")
    config = load_config()
    
    api_key = os.getenv("TRANSLATION_API_KEY")
    if not api_key:
        print("Error: TRANSLATION_API_KEY environment variable not set.")
        return

    for mirror in config.get('mirrors', []):
        target_language = mirror.get('lang')
        if not target_language:
            continue
        
        print(f"Translating to {target_language}...")
        scan_files(config, target_language, api_key)

    print("File scanning and translation complete.")

if __name__ == "__main__":
    main()
