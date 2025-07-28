import re

def preprocess_lines(lines_data):
    """
    Cleans and preprocesses extracted text lines.

    This function performs the following operations:
    1. Removes extra whitespace from the beginning and end of each line.
    2. Normalizes internal spacing (e.g., multiple spaces to a single space).
    3. Filters out lines that are empty after cleaning.

    Args:
        lines_data (list): A list of dictionaries, where each dictionary represents
                           a line with its 'text' and other features, as extracted
                           by pdf_parser.py.

    Returns:
        list: A new list of dictionaries with cleaned text lines and potentially
              additional metadata if needed in the future.
    """
    processed_lines = []
    for line in lines_data:
        original_text = line.get("text", "")
        # Remove leading/trailing whitespace and normalize internal spaces
        cleaned_text = re.sub(r'\s+', ' ', original_text).strip()

        if cleaned_text:  # Only keep lines that are not empty after cleaning
            # Create a new dictionary to avoid modifying the original 'line' dict directly
            # This is good practice if the original 'lines_data' might be used elsewhere
            processed_line = line.copy()
            processed_line["text"] = cleaned_text
            processed_lines.append(processed_line)
    return processed_lines

if __name__ == "__main__":
    # Simple test case for preprocessing.py
    # This simulates the output from pdf_parser.py

    sample_raw_lines = [
        {"text": "  This is a sample title.  ", "page_number": 0, "font_size": 24, "is_bold": True},
        {"text": "Section   1: Introduction", "page_number": 0, "font_size": 18, "is_bold": True},
        {"text": "  This is some body text.   ", "page_number": 0, "font_size": 12, "is_bold": False},
        {"text": "  ", "page_number": 0, "font_size": 10, "is_bold": False}, # Empty line
        {"text": "Another line. ", "page_number": 1, "font_size": 12, "is_bold": False},
        {"text": "\tFinal  Line\n", "page_number": 1, "font_size": 14, "is_bold": True}
    ]

    print("Original lines:")
    for line in sample_raw_lines:
        print(f"  '{line['text']}'")

    cleaned_lines = preprocess_lines(sample_raw_lines)

    print("\nCleaned lines:")
    if cleaned_lines:
        for line in cleaned_lines:
            print(f"  '{line['text']}' (Page: {line['page_number']}, Size: {line['font_size']})")
    else:
        print("No lines after cleaning.")

    # Test with an empty input list
    print("\nTesting with an empty input list:")
    empty_input = []
    cleaned_empty = preprocess_lines(empty_input)
    print(f"Result for empty input: {cleaned_empty}")
