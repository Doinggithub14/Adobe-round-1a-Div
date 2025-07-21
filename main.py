import fitz  # PyMuPDF
import os
import json

def extract_headings_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    headings = []
    title = ""

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    font_size = 0
                    is_bold = False

                    for span in line["spans"]:
                        line_text += span["text"].strip() + " "
                        font_size = span["size"]
                        if "bold" in span.get("font", "").lower():
                            is_bold = True

                    line_text = line_text.strip()

                    if len(line_text) < 3:
                        continue

                    # Decide heading level
                    level = None
                    if font_size > 15:
                        level = "H1"
                        if page_num == 1 and not title:
                            title = line_text
                    elif 13 <= font_size <= 15:
                        level = "H2"
                    elif 11 <= font_size < 13:
                        level = "H3"

                    if level:
                        headings.append({
                            "level": level,
                            "text": line_text,
                            "page": page_num
                        })

    return {
        "title": title or "Untitled Document",
        "outline": headings
    }

def main():
    input_dir = "input"
    output_dir = "output"

    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            input_path = os.path.join(input_dir, filename)
            result = extract_headings_from_pdf(input_path)

            output_filename = filename.replace(".pdf", ".json")
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()