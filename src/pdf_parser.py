import fitz  # PyMuPDF
import os

def parse_pdf(pdf_path):
    """
    Extracts text and layout information (font, size, position) from a PDF document.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a line
              and contains its text, page number, and layout features.
              Returns an empty list if the PDF cannot be opened or processed.
    """
    lines_data = []
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            # Get text blocks with detailed information
            # 'flags' parameter can be used to control the output, 11 for detailed text
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block["type"] == 0:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            # Extract relevant features for each span (part of a line with consistent formatting)
                            text = span["text"].strip()
                            if text:  # Only process non-empty text
                                lines_data.append({
                                    "text": text,
                                    "page_number": page_num,
                                    "font_size": span["size"],
                                    "font_name": span["font"],
                                    "is_bold": "bold" in span["font"].lower() or span["flags"] & 16, # Flag 16 indicates bold
                                    "is_italic": "italic" in span["font"].lower() or span["flags"] & 32, # Flag 32 indicates italic
                                    "bbox": span["bbox"],  # Bounding box (x0, y0, x1, y1)
                                    "origin": span["origin"], # (x, y) coordinates of the start of the text
                                    "line_height": line["bbox"][3] - line["bbox"][1] # Height of the line bounding box
                                })
        doc.close()
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
    return lines_data

if __name__ == "__main__":
    # This block is for testing the pdf_parser.py module directly.
    # It assumes there's an 'input' directory at the same level as 'src'.
    # For actual hackathon submission, main.py will handle iterating through
    # the /app/input directory.

    input_dir = "input" # Assuming 'input' folder is at the root level of your project
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        print("Please ensure your PDF files are in an 'input' folder relative to this script's execution.")
    else:
        pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
        if not pdf_files:
            print(f"No PDF files found in '{input_dir}'. Please add some PDFs to test.")
        else:
            print(f"Found {len(pdf_files)} PDF(s) in '{input_dir}'. Parsing the first one...")
            first_pdf_path = os.path.join(input_dir, pdf_files[0])
            extracted_lines = parse_pdf(first_pdf_path)

            if extracted_lines:
                print(f"Successfully extracted {len(extracted_lines)} lines from {pdf_files[0]}.")
                print("\nSample of extracted lines (first 5):")
                for i, line in enumerate(extracted_lines[:5]): # Print first 5 lines
                    print(f"  Line {i+1}: Text='{line['text']}', Page={line['page_number']}, Size={line['font_size']:.1f}, Bold={line['is_bold']}")
            else:
                print(f"No lines extracted or an error occurred for {pdf_files[0]}.")

    # You can also test with a non-existent PDF to see error handling
    print("\nTesting with a non-existent PDF:")
    non_existent_pdf = os.path.join(input_dir, "non_existent.pdf")
    extracted_lines_error = parse_pdf(non_existent_pdf)
    if not extracted_lines_error:
        print(f"Correctly handled non-existent PDF: {non_existent_pdf}")
