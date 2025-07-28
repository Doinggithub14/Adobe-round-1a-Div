import os
import json

def validate_synthetic_data():
    pdf_dir = "../input"
    json_dir = "../ground_truth"
    
    # Check file pairs
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith(".pdf"):
            json_file = pdf_file.replace(".pdf", ".json")
            json_path = os.path.join(json_dir, json_file)
            if not os.path.exists(json_path):
                raise ValueError(f"Missing ground truth JSON for {pdf_file}")
    
    print("All synthetic PDFs have valid ground truth JSONs.")

if __name__ == "__main__":
    validate_synthetic_data()