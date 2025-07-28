import json
import os

def write_json_output(output_path, title, outline_data):
        """
        Generates and saves the JSON output for Round 1A.

        Args:
            output_path (str): The full path to the output JSON file (e.g., "output/document.json").
            title (str): The identified document title.
            outline_data (list): A list of dictionaries, each representing a heading:
                                 {"level": "H1", "text": "Section Title", "page": 0}
        """
        output_data = {
            "title": title,
            "outline": outline_data
        }

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"JSON output successfully written to {output_path}")
        except Exception as e:
            print(f"Error writing JSON output to {output_path}: {e}")

if __name__ == "__main__":
        # Simple test case for json_writer.py
        test_output_dir = "output_test"
        os.makedirs(test_output_dir, exist_ok=True)
        test_output_path = os.path.join(test_output_dir, "test_document_outline.json")

        sample_title = "Navigating the Adobe Document Cloud Hackathon"
        sample_outline = [
            {"level": "H1", "text": "Understanding the \"Connecting the Dots\" Vision", "page": 1},
            {"level": "H2", "text": "Purpose and Relevance to Adobe Document Cloud", "page": 1},
            {"level": "H3", "text": "The Overarching Theme: Connecting Disparate Information", "page": 2},
            {"level": "H1", "text": "Deep Dive into Round 1A: Document Section Identification", "page": 2}
        ]

        print(f"Writing sample JSON output to {test_output_path}...")
        write_json_output(test_output_path, sample_title, sample_outline)

        # Verify the file content
        if os.path.exists(test_output_path):
            with open(test_output_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                print("\nContent of generated JSON:")
                print(json.dumps(content, indent=2, ensure_ascii=False))
            os.remove(test_output_path) # Clean up test file
            os.rmdir(test_output_dir) # Clean up test directory
            print(f"\nCleaned up test file and directory: {test_output_path}, {test_output_dir}")
        else:
            print("Failed to create JSON file for verification.")
    