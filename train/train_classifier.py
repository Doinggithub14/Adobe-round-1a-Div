import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier 
import joblib
import re 
from collections import Counter 
import numpy as np 

# Ensure sys.path is correctly set to import from src
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from pdf_parser import parse_pdf
from preprocessing import preprocess_lines
from feature_engineering import engineer_features # This will use the enhanced features

def load_ground_truth(json_path):
    """Loads and parses a ground truth JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading ground truth from {json_path}: {e}")
        return None

def normalize_text_for_matching(text):
    """Normalizes text by stripping whitespace and converting to lowercase."""
    return re.sub(r'\s+', ' ', text).strip().lower()

def prepare_training_data(input_dir, ground_truth_dir):
    """
    Prepares features and binary labels (Is Heading / Not Heading) for training.

    Iterates through input PDFs, extracts features, and matches them with
    corresponding ground truth labels. Any H1, H2, H3, H4, etc., in ground truth
    will be mapped to '1' (Is Heading). 'None' will be mapped to '0' (Not Heading).
    """
    X = [] # Features
    y = [] # Labels (0: Not Heading, 1: Is Heading)
    
    # Map any 'H' level to 1 (Is Heading), 'None' to 0 (Not Heading)
    heading_to_binary_label = {
        "H1": 1, "H2": 1, "H3": 1, "H4": 1, "H5": 1, "H6": 1, 
        "None": 0 
    }

    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    
    print(f"Preparing training data from {len(pdf_files)} PDFs and ground truths...")

    all_engineered_features = []
    for pdf_filename in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_filename)
        json_filename = os.path.splitext(pdf_filename)[0] + ".json"
        ground_truth_path = os.path.join(ground_truth_dir, json_filename)

        if not os.path.exists(ground_truth_path):
            print(f"  Warning: No ground truth JSON found for {pdf_filename}. Skipping.")
            continue
        
        raw_lines = parse_pdf(pdf_path)
        if not raw_lines:
            print(f"  No raw lines extracted from {pdf_filename}.")
            continue

        processed_lines = preprocess_lines(raw_lines)
        if not processed_lines:
            print(f"  No processed lines after preprocessing for {pdf_filename}.")
            continue

        doc_engineered_features = engineer_features(processed_lines)
        if not doc_engineered_features:
            print(f"  No features engineered for {pdf_filename}.")
            continue
        
        ground_truth = load_ground_truth(ground_truth_path)
        if not ground_truth or "outline" not in ground_truth:
            print(f"  Warning: Ground truth for {pdf_filename} is invalid or missing 'outline'. Skipping.")
            continue

        gt_outline = ground_truth["outline"]
        
        gt_headings = {}
        for item in gt_outline:
            normalized_gt_text = normalize_text_for_matching(item['text'])
            gt_headings[(item['page'], normalized_gt_text)] = item['level']

        for line_features in doc_engineered_features:
            line_text = line_features["text"]
            line_page = line_features["page_number"]
            
            normalized_extracted_text = normalize_text_for_matching(line_text)

            binary_label = heading_to_binary_label["None"] # Default to 0 (Not Heading)
            gt_level = gt_headings.get((line_page, normalized_extracted_text))
            
            if gt_level is not None:
                if gt_level in heading_to_binary_label and heading_to_binary_label[gt_level] == 1:
                    binary_label = 1
                else:
                    if gt_level.startswith('H') and gt_level[1:].isdigit():
                         binary_label = 1
                    else:
                         print(f"  Warning: Unexpected ground truth level '{gt_level}' for '{line_text}'. Treating as Not Heading (0).")
                         binary_label = 0
            
            all_engineered_features.append(line_features)
            y.append(binary_label)
    
    feature_keys = []
    if all_engineered_features:
        non_feature_keys = ['text', 'page_number', 'font_name', 'bbox', 'origin', 'line_height']
        feature_keys = [key for key in all_engineered_features[0].keys() if key not in non_feature_keys]
    
    if feature_keys:
        X = pd.DataFrame([{key: feat_dict.get(key, 0.0) for key in feature_keys} for feat_dict in all_engineered_features], columns=feature_keys)
    else:
        X = pd.DataFrame()

    y = pd.Series(y)
    
    print(f"Finished preparing data. Total samples: {len(X)}")
    return X, y

def train_model(X, y, model_save_path):
    """Trains a RandomForestClassifier for binary (Is Heading / Not Heading) classification."""
    if X.empty or y.empty:
        print("No data to train the model.")
        return

    class_counts = y.value_counts().sort_index()
    print("\nClass distribution in training data (0:Not Heading, 1:Is Heading):")
    print(class_counts)

    can_stratify = True
    for class_label, count in class_counts.items():
        if count < 2:
            can_stratify = False
            break

    if can_stratify:
        print("Performing stratified train-test split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        print("\nWARNING: One or more classes have fewer than 2 samples. Cannot perform stratified split for binary classification.")
        print("Proceeding with non-stratified split. For better model performance, add more training data.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples.")
    
    # Initialize RandomForestClassifier with optimized parameters
    # n_estimators: Increased for more robust ensemble.
    # max_depth: Kept at 10, can be tuned.
    # class_weight: Tuned to a custom value to balance precision and recall.
    #               {0: 1, 1: 15} means that misclassifying a heading is 15 times
    #               more costly than misclassifying a non-heading.
    model = RandomForestClassifier(
        random_state=42,
        n_estimators=300, # Increased from 200 for more robustness
        max_depth=10,     # Keep at 10, or try 12 if recall needs more boost
        class_weight={0: 1, 1: 15} # Tuned from 'balanced' (which was ~39)
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test) 

    print("\nEvaluating model performance on test set:")
    print(classification_report(y_test, y_pred, target_names=["Not Heading", "Is Heading"], zero_division='warn'))

    # Optional: Print Feature Importances (useful for further feature engineering)
    print("\nFeature Importances (Top 10):")
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(feature_importances.head(10))

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")
    return model

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    input_data_dir = os.path.join(project_root, "input")
    ground_truth_data_dir = os.path.join(project_root, "ground_truth")
    model_output_path = os.path.join(project_root, "models", "heading_classifier.pkl")

    features_df, labels_series = prepare_training_data(input_data_dir, ground_truth_data_dir)

    trained_model = train_model(features_df, labels_series, model_output_path)
    
    if trained_model:
        print("\nTraining process completed.")
        print(f"You can now use the model '{os.path.basename(model_output_path)}' in your main.py for Round 1A.")
    else:
        print("\nModel training failed or no data to train.")
