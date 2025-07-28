import joblib
import os
import numpy as np
import pandas as pd # Import pandas

class HeadingClassifier:
    """
    A wrapper class for loading and using the trained RandomForestClassifier.
    Designed for robustness and clear error handling.
    """
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path
        # Store feature names if available from the loaded model (RandomForest stores them)
        self.feature_names_in_ = None 
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        """
        Loads a pre-trained machine learning model (e.g., scikit-learn RandomForestClassifier).
        Handles FileNotFoundError and other loading exceptions gracefully.
        """
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at: {model_path}. "
                  "Ensure the model is trained or the path is correct.")
            self.model = None
            return
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded successfully from {model_path}")
            # Try to get feature names from the loaded model
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names_in_ = self.model.feature_names_in_
            elif hasattr(self.model, 'feature_names'): # Older scikit-learn versions might use .feature_names
                 self.feature_names_in_ = self.model.feature_names
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            self.model = None

    def predict(self, features_df_list): # Renamed argument to clarify it's a list of dicts
        """
        Makes predictions (heading levels) based on engineered features.

        Args:
            features_df_list (list of dicts): List of dictionaries, each containing
                                         engineered features for a line.

        Returns:
            np.array: Predicted heading levels (e.g., 0 for None, 1 for H1, etc.).
                      Returns an array of zeros if the model is not loaded or
                      prediction fails.
        """
        if self.model is None:
            print("Error: Model not loaded. Cannot make predictions.")
            return np.array([0] * len(features_df_list))

        if not features_df_list:
            return np.array([])
        
        # Dynamically determine feature keys from the first dictionary if not already known
        # This ensures the order of features matches what was used during training.
        if self.feature_names_in_ is None:
            # If model didn't provide feature names (e.g., dummy model or older sklearn)
            # Infer from the first data point, excluding non-feature keys.
            non_feature_keys = ['text', 'page_number', 'font_name', 'bbox', 'origin', 'line_height', 'predicted_label']
            self.feature_names_in_ = [key for key in features_df_list[0].keys() if key not in non_feature_keys]
            print("Warning: Feature names not found in loaded model. Inferring from input data. "
                  "Ensure feature order and names are consistent with training data to avoid issues.")

        # Convert list of dictionaries to a Pandas DataFrame, ensuring column order
        # and handling missing features by filling with 0.0
        X_predict = pd.DataFrame(features_df_list)
        # Ensure only the features used for training are present and in the correct order
        # Use .reindex for robustness: it reorders columns and adds missing ones (filled with NaN, then fillna)
        X_predict = X_predict.reindex(columns=self.feature_names_in_).fillna(0.0)

        try:
            predictions = self.model.predict(X_predict)
            return predictions
        except Exception as e:
            print(f"Error during prediction: {e}")
            return np.array([0] * len(features_df_list))

if __name__ == "__main__":
    # Self-contained test for model.py:
    temp_model_dir = "temp_models"
    dummy_model_path = os.path.join(temp_model_dir, "dummy_rf_model.pkl") 
    os.makedirs(temp_model_dir, exist_ok=True)

    from sklearn.ensemble import RandomForestClassifier
    
    # Dummy binary training data - MUST match expected features from feature_engineering.py
    dummy_feature_names = [
        "normalized_font_size_doc_max", "is_bold", "is_uppercase",
        "starts_with_heading_pattern", "word_count", "normalized_x_pos",
        "font_size_ratio_to_avg", "vertical_gap_prev",
        "prev_text_ends_with_colon", "next_line_indented",
        "line_has_common_heading_phrase", "font_consistent_with_dominant_h1",
        "is_italic", "char_count", "ends_with_period", "ends_with_colon",
        "font_size", "font_size_ratio_to_median", "font_size_ratio_to_most_common",
        "normalized_y_pos", "font_size_diff_prev", "normalized_vertical_gap_prev",
        "is_title_candidate"
    ]
    # Create a DataFrame directly with named columns for dummy training
    dummy_X_train = pd.DataFrame([
        [0.9, 1, 1, 0, 2, 0.08, 2.0, 0.0, 0, 0, 0, 0, 0, 13, 0, 0, 28.0, 2.3, 2.5, 0.05, 0.0, 0.0, 1], # Is Heading
        [0.7, 1, 0, 1, 2, 0.08, 1.5, 30.0, 1, 0, 1, 0, 0, 15, 0, 0, 20.0, 1.8, 1.7, 0.12, -8.0, 1.5, 0], # Is Heading
        [0.2, 0, 0, 0, 5, 0.08, 0.8, 10.0, 0, 0, 0, 0, 0, 18, 1, 0, 12.0, 0.9, 1.0, 0.15, -8.0, 0.83, 0], # Not Heading
    ], columns=dummy_feature_names)
    dummy_y_train = np.array([1, 1, 0])

    dummy_rf_model = RandomForestClassifier(max_depth=2, random_state=42)
    dummy_rf_model.fit(dummy_X_train, dummy_y_train)

    try:
        joblib.dump(dummy_rf_model, dummy_model_path)
        print(f"Dummy RandomForest model saved to {dummy_model_path}")

        print("\nTesting HeadingClassifier with dummy RandomForest model:")
        classifier = HeadingClassifier(model_path=dummy_model_path)

        # Sample engineered features - MUST match the full set from feature_engineering.py
        # These are now passed as a list of dicts, and the predict method handles DataFrame conversion.
        sample_features = [
            {"text": "TITLE EXAMPLE", "page_number": 0, "font_size": 28.0, "is_bold": 1, "is_italic": 0, "is_uppercase": 1, "starts_with_heading_pattern": 0, "word_count": 2, "char_count": 13, "ends_with_period": 0, "ends_with_colon": 0, "normalized_font_size_doc_max": 0.9, "font_size_ratio_to_avg": 2.0, "font_size_ratio_to_median": 2.3, "font_size_ratio_to_most_common": 2.5, "normalized_x_pos": 0.08, "normalized_y_pos": 0.05, "font_size_diff_prev": 0.0, "vertical_gap_prev": 0.0, "normalized_vertical_gap_prev": 0.0, "is_title_candidate": 1, "prev_text_ends_with_colon": 0, "next_line_indented": 0, "line_has_common_heading_phrase": 0, "font_consistent_with_dominant_h1": 0},
            {"text": "1. Introduction", "page_number": 0, "font_size": 20.0, "is_bold": 1, "is_italic": 0, "is_uppercase": 0, "starts_with_heading_pattern": 1, "word_count": 2, "char_count": 15, "ends_with_period": 0, "ends_with_colon": 0, "normalized_font_size_doc_max": 0.7, "font_size_ratio_to_avg": 1.5, "font_size_ratio_to_median": 1.8, "font_size_ratio_to_most_common": 1.7, "normalized_x_pos": 0.08, "normalized_y_pos": 0.12, "font_size_diff_prev": -8.0, "vertical_gap_prev": 30.0, "normalized_vertical_gap_prev": 1.5, "is_title_candidate": 0, "prev_text_ends_with_colon": 0, "next_line_indented": 0, "line_has_common_heading_phrase": 0, "font_consistent_with_dominant_h1": 0},
            {"text": "Body text example.", "page_number": 0, "font_size": 12.0, "is_bold": 0, "is_italic": 0, "is_uppercase": 0, "starts_with_heading_pattern": 0, "word_count": 3, "char_count": 18, "ends_with_period": 1, "ends_with_colon": 0, "normalized_font_size_doc_max": 0.2, "font_size_ratio_to_avg": 0.8, "font_size_ratio_to_median": 0.9, "font_size_ratio_to_most_common": 1.0, "normalized_x_pos": 0.08, "normalized_y_pos": 0.15, "font_size_diff_prev": -8.0, "vertical_gap_prev": 10.0, "normalized_vertical_gap_prev": 0.83, "is_title_candidate": 0, "prev_text_ends_with_colon": 0, "next_line_indented": 0, "line_has_common_heading_phrase": 0, "font_consistent_with_dominant_h1": 0},
        ]

        predictions = classifier.predict(sample_features)
        print("Predictions:", predictions)

    except Exception as e:
        print(f"Error during dummy model test: {e}")
    finally:
        if os.path.exists(dummy_model_path):
            os.remove(dummy_model_path)
            print(f"Cleaned up dummy model file: {dummy_model_path}")
        if os.path.exists(temp_model_dir):
            os.rmdir(temp_model_dir)
            print(f"Cleaned up dummy model directory: {temp_model_dir}")

