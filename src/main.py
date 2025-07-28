import os
import numpy as np
import joblib
import re

# Import all custom modules
from pdf_parser import parse_pdf
from preprocessing import preprocess_lines
from feature_engineering import engineer_features
from model import HeadingClassifier # Imports our classifier wrapper
from json_writer import write_json_output
from semantic_filter import SemanticFilter # Imports our semantic post-processor


def identify_headings_in_document(pdf_path, model_path):
    """
    Orchestrates the process of identifying hierarchical headings in a single PDF document.
    This function integrates all pipeline stages: parsing, preprocessing, feature engineering,
    binary classification, and dynamic semantic level assignment.

    Args:
        pdf_path (str): The path to the input PDF file.
        model_path (str): The path to the trained binary classification model (Decision Tree).

    Returns:
        tuple: A tuple containing the identified document title (str) and a list of outline
               dictionaries for headings, or (None, []) if an error occurs or no headings found.
    """
    print(f"Starting processing for: {os.path.basename(pdf_path)}")

    # Stage 1: PDF Parsing - Extract raw text and layout information
    print("  1/6: Parsing PDF for raw text and layout...")
    raw_lines = parse_pdf(pdf_path)
    if not raw_lines:
        print(f"  Warning: No lines extracted from {pdf_path}. Skipping document.")
        return None, []

    # Stage 2: Preprocessing - Clean and normalize text lines
    print("  2/6: Preprocessing extracted lines...")
    processed_lines = preprocess_lines(raw_lines)
    if not processed_lines:
        print(f"  Warning: No lines left after preprocessing for {pdf_path}. Skipping document.")
        return None, []

    # Stage 3: Feature Engineering - Transform lines into discriminative features
    print("  3/6: Engineering rich features for classification...")
    engineered_features = engineer_features(processed_lines)
    if not engineered_features:
        print(f"  Warning: No features engineered for {pdf_path}. Skipping document.")
        return None, []

    # Stage 4: Binary Classification - Predict if a line is a heading or not
    print("  4/6: Classifying lines to identify if they are headings (binary prediction)...")
    classifier = HeadingClassifier(model_path=model_path)
    if not classifier.model:
        print("  Error: Binary classification model could not be loaded. Cannot classify headings.")
        return None, []

    # Get binary predictions from the model (0: Not Heading, 1: Is Heading)
    binary_predictions = classifier.predict(engineered_features)

    # Combine features with binary predictions for the semantic filter
    lines_for_filter = []
    for i, features in enumerate(engineered_features):
        line_data = features.copy()
        line_data['predicted_label'] = int(binary_predictions[i]) # Store binary prediction
        lines_for_filter.append(line_data)

    # Stage 5: Semantic Filtering & Dynamic Level Assignment
    print("  5/6: Applying semantic filters and dynamically assigning H-levels...")
    semantic_filter = SemanticFilter()
    # This function now returns the actual H-levels (1, 2, 3, 4, ...) or 0 (None)
    final_h_levels = semantic_filter.apply_filters_and_assign_levels(lines_for_filter)

    # Stage 6: Post-processing and Outline Generation
    print("  6/6: Generating final structured outline...")
    document_title = None
    outline = []
    
    # Title Detection Heuristic:
    # Prioritize lines marked as 'is_title_candidate' from feature engineering.
    title_candidates = [line for line in engineered_features if line.get('is_title_candidate', 0) == 1 and line['page_number'] == 0]
    if title_candidates:
        # Sort by font size (desc) and then y-position (asc) to pick the most prominent top line
        title_candidates.sort(key=lambda x: (x['font_size'], -x['normalized_y_pos']), reverse=True)
        document_title = title_candidates[0]['text']
    elif raw_lines: # Fallback to largest font on page 0 if no explicit title candidate
        potential_titles_raw = sorted(
            [line for line in raw_lines if line['page_number'] == 0],
            key=lambda x: x['font_size'], reverse=True
        )
        if potential_titles_raw:
            document_title = potential_titles_raw[0]['text']

    # Build the outline from dynamically assigned H-levels
    for i, h_level in enumerate(final_h_levels):
        if h_level > 0: # Only include actual headings (H1, H2, H3, H4, ...)
            line_info = engineered_features[i] # Use engineered_features for original text and page_number
            outline.append({
                "level": f"H{h_level}", # Dynamically format the level string
                "text": line_info["text"],
                "page": line_info["page_number"]
            })
    
    # Ensure the outline is correctly sorted by page number and then by vertical position on page.
    # The `engineered_features` list is already ordered by page and then y-position.
    # We'll use the original index to preserve this order.
    outline.sort(key=lambda x: (x['page'], 
                                next(idx for idx, ef in enumerate(engineered_features) 
                                     if ef['text'] == x['text'] and ef['page_number'] == x['page'])))


    print(f"  Identified {len(outline)} final headings.")
    return document_title, outline

def main(input_dir, output_dir, model_path):
    """
    Main function to process all PDFs in the input directory.
    This is the primary entry point for the hackathon submission.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found. Please ensure your PDFs are in this directory.")
        return
    os.makedirs(output_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in '{input_dir}'. Please add some PDFs to process.")
        return

    print(f"\nProcessing {len(pdf_files)} PDF(s) from '{input_dir}'...")

    for pdf_filename in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_filename)
        json_filename = os.path.splitext(pdf_filename)[0] + ".json"
        output_json_path = os.path.join(output_dir, json_filename)

        title, outline = identify_headings_in_document(pdf_path, model_path)

        if title is not None or outline:
            write_json_output(output_json_path, title, outline)
        else:
            print(f"  Skipping JSON output for {pdf_filename} due to empty or erroneous results.")
    print("\nAll PDF documents processed.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    input_directory = os.path.join(project_root, "input")
    output_directory = os.path.join(project_root, "output")
    model_file_path = os.path.join(project_root, "models", "heading_classifier.pkl")

    # --- Dummy Model Creation for Initial Local Testing ---
    # This block ensures that `main.py` can run even if you haven't
    # trained your actual model yet. It creates a very basic Decision Tree
    # that makes binary predictions based on a simple rule.
    # **REMEMBER TO REPLACE THIS WITH YOUR ACTUALLY TRAINED MODEL FOR HACKATHON SUBMISSION!**
    if not os.path.exists(model_file_path):
        print(f"Dummy model not found at {model_file_path}.")
        print("Creating a placeholder dummy Decision Tree model for initial testing. "
              "**REMEMBER TO REPLACE THIS WITH YOUR TRAINED MODEL FOR HACKATHON SUBMISSION!**")
        
        from sklearn.tree import DecisionTreeClassifier
        # Define dummy binary training data for the dummy model
        # Features should align with feature_engineering.py
        # [normalized_font_size_doc_max, is_bold, is_italic, is_uppercase,
        #  starts_with_heading_pattern, word_count, char_count, ends_with_period,
        #  ends_with_colon, font_size_ratio_to_avg, font_size_ratio_to_median,
        #  font_size_ratio_to_most_common, normalized_x_pos, normalized_y_pos,
        #  font_size_diff_prev, vertical_gap_prev, normalized_vertical_gap_prev,
        #  is_title_candidate]
        dummy_X_train = np.array([
            [0.9, 1, 0, 1, 0, 2, 13, 0, 0, 2.0, 2.3, 2.5, 0.08, 0.05, 0.0, 0.0, 0.0, 1], # Is Heading (Title-like)
            [0.7, 1, 0, 0, 1, 2, 15, 0, 0, 1.5, 1.8, 1.7, 0.08, 0.12, -8.0, 30.0, 1.5, 0], # Is Heading (H1-like)
            [0.5, 1, 0, 0, 1, 3, 17, 0, 0, 1.2, 1.4, 1.3, 0.12, 0.18, -4.0, 15.0, 0.9, 0], # Is Heading (H2-like)
            [0.3, 1, 0, 0, 1, 3, 17, 0, 0, 1.1, 1.2, 1.1, 0.15, 0.25, -2.0, 10.0, 0.7, 0], # Is Heading (H3-like)
            [0.2, 0, 0, 0, 0, 5, 20, 1, 0, 0.8, 0.9, 1.0, 0.08, 0.15, -8.0, 10.0, 0.83, 0], # Not Heading (Body-like)
        ])
        dummy_y_train = np.array([1, 1, 1, 1, 0]) # Binary labels for training (1=Is Heading, 0=Not Heading)

        dummy_dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        dummy_dt_model.fit(dummy_X_train, dummy_y_train)

        try:
            os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
            joblib.dump(dummy_dt_model, model_file_path)
            print("Placeholder dummy model created for testing.")
        except Exception as e:
            print(f"Could not create dummy model: {e}")
            model_file_path = None

    if model_file_path and os.path.exists(model_file_path):
        main(input_directory, output_directory, model_file_path)
    else:
        print("Cannot proceed without a model file. Please ensure 'models/heading_classifier.pkl' exists or is created.")
