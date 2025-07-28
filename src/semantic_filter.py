import numpy as np
from collections import Counter

class SemanticFilter:
    """
    Refines heading predictions and dynamically assigns hierarchical levels
    (H1, H2, H3, H4, ...) based on document structure and feature prominence.
    This module is crucial for handling variable heading depths in PDFs.
    It also enforces logical hierarchy rules.
    """
    def __init__(self):
        pass

    def _get_prominent_font_sizes(self, heading_lines):
        """
        Identifies distinct and prominent font sizes among detected heading lines.
        This helps in dynamically assigning H-levels.
        """
        if not heading_lines:
            return []

        heading_font_sizes = sorted(list(set([line['font_size'] for line in heading_lines])), reverse=True)
        
        prominent_sizes = []
        if heading_font_sizes:
            prominent_sizes.append(heading_font_sizes[0]) # Always take the largest
            for size in heading_font_sizes[1:]:
                # Add if significantly different from the last added prominent size
                # The 10% threshold determines how many distinct levels are identified.
                # Tune this value (e.g., 0.08 to 0.15) based on your document's typical
                # font size variations between heading levels.
                if (prominent_sizes[-1] - size) / prominent_sizes[-1] > 0.10: # 10% difference
                    prominent_sizes.append(size)
        
        return sorted(prominent_sizes, reverse=True)

    def apply_filters_and_assign_levels(self, lines_with_features_and_binary_preds):
        """
        Applies post-processing rules and dynamically assigns H-levels
        to lines predicted as headings.

        Args:
            lines_with_features_and_binary_preds (list): A list of dictionaries,
                each containing engineered features and the binary prediction
                ('predicted_label': 0 or 1).

        Returns:
            list: A list of refined predicted labels (numeric: 0 for None,
                  1 for H1, 2 for H2, 3 for H3, 4 for H4, etc.).
        """
        if not lines_with_features_and_binary_preds:
            return []

        # Step 1: Filter out lines not predicted as headings by the model
        potential_headings = [
            item for item in lines_with_features_and_binary_preds
            if item['predicted_label'] == 1
        ]
        
        # Sort potential headings by page and then by vertical position
        # This is critical for correct hierarchical assignment
        potential_headings.sort(key=lambda x: (x['page_number'], x['normalized_y_pos']))

        # Step 2: Dynamically determine heading levels based on prominent font sizes
        prominent_font_sizes = self._get_prominent_font_sizes(potential_headings)
        
        if not prominent_font_sizes:
            # If no prominent sizes (e.g., no headings detected), return all as None
            return [0] * len(lines_with_features_and_binary_preds)

        # Create a mapping from prominent font size to a relative H-level
        # The largest font size gets H1, second largest H2, etc.
        font_size_to_h_level = {
            size: level + 1 for level, size in enumerate(prominent_font_sizes)
        }
        max_h_level_detected = len(prominent_font_sizes)
        
        # Step 3: Assign initial H-levels to potential_headings
        for item in potential_headings:
            assigned_level = 0
            # Find the closest prominent font size that is less than or equal to current line's font size
            for size_idx, p_size in enumerate(prominent_font_sizes):
                if item['font_size'] >= p_size:
                    assigned_level = size_idx + 1
                    break
            
            if assigned_level == 0 and item['font_size'] > 0:
                # If it's a heading but smaller than all prominent sizes, assign it the lowest detected H-level
                assigned_level = max_h_level_detected if max_h_level_detected > 0 else 1
            
            item['assigned_h_level'] = assigned_level

        # Step 4: Apply hierarchical and contextual rules for refinement
        final_predictions_map = {} 
        last_valid_h_level = 0 # Track the last heading level encountered (0 for None)

        for i, item in enumerate(potential_headings):
            current_h_level = item['assigned_h_level']
            
            # Rule A: Demote very long lines that are predicted as headings
            # Tunable threshold: Headings are usually concise.
            if current_h_level > 0 and item['word_count'] > 15: 
                current_h_level = 0 

            # Rule B: Demote if it's all uppercase but very long (often body text in ALL CAPS)
            # Tunable threshold:
            if current_h_level > 0 and item['is_uppercase'] and item['word_count'] > 10:
                current_h_level = 0

            # Rule C: Demote lines ending with a period (strong indicator it's not a heading)
            if current_h_level > 0 and item['ends_with_period'] == 1:
                current_h_level = 0

            # Rule D: Enforce strict hierarchy (main improvement for structural integrity)
            # Prevent jumping down more than one level (e.g., H1 -> H3 directly)
            if current_h_level > 0: # If it's still considered a heading after A, B, C
                if last_valid_h_level == 0: # If the previous line was not a heading
                    # This is the start of a new section. Allow H1/H2, but if it's H3/H4 starting
                    # a new major section, it might be an error or a very deep start.
                    # For now, we'll allow it, but this is a tuning point if you see issues.
                    pass
                else: # Previous line was a valid heading
                    # If jumping down more than one level (e.g., H1 to H3), promote to next logical level
                    if current_h_level - last_valid_h_level > 1:
                        current_h_level = last_valid_h_level + 1
                    
                    # If jumping up more than one level (e.g., H3 to H1) without a new page,
                    # it might be an error. This rule is complex and can be aggressive.
                    # For now, we'll allow it if the font size is very prominent, assuming
                    # it's a valid new major section. If issues, could add:
                    # if current_h_level < last_valid_h_level - 1 and item['page_number'] == potential_headings[i-1]['page_number']:
                    #    # Demote current_h_level if it's an illogical jump UP on the same page
                    #    current_h_level = last_valid_h_level # Or last_valid_h_level - 1
                    pass # Current implementation allows large jumps up, which is often correct for new sections.

            item['final_h_level'] = current_h_level
            
            # Update last_valid_h_level only if the current line is a valid heading
            if current_h_level > 0:
                last_valid_h_level = current_h_level
            else: # If current line was demoted to None
                last_valid_h_level = 0 # Reset if we hit body text

            final_predictions_map[(item['page_number'], item['text'])] = current_h_level

        # Step 5: Construct the final list of predictions matching the original input order
        final_predictions_in_original_order = []
        for original_item in lines_with_features_and_binary_preds:
            # Look up the refined level for this line. If not a potential heading, it's 0.
            level = final_predictions_map.get((original_item['page_number'], original_item['text']), 0)
            final_predictions_in_original_order.append(level)

        return final_predictions_in_original_order

if __name__ == "__main__":
    # Sample data for testing semantic filter with binary predictions
    sample_data = [
        {'text': 'DOCUMENT TITLE', 'page_number': 0, 'font_size': 28.0, 'is_bold': 1, 'predicted_label': 1, 'starts_with_heading_pattern': 0, 'font_size_ratio_to_most_common': 2.0, 'word_count': 2, 'is_uppercase': 1, 'normalized_x_pos': 0.08, 'normalized_y_pos': 0.05, 'ends_with_period': 0},
        {'text': '1. Introduction', 'page_number': 0, 'font_size': 20.0, 'is_bold': 1, 'predicted_label': 1, 'starts_with_heading_pattern': 1, 'font_size_ratio_to_most_common': 1.5, 'word_count': 2, 'is_uppercase': 0, 'normalized_x_pos': 0.08, 'normalized_y_pos': 0.12, 'ends_with_period': 0},
        {'text': 'This is some body text.', 'page_number': 0, 'font_size': 12.0, 'is_bold': 0, 'predicted_label': 0, 'starts_with_heading_pattern': 0, 'font_size_ratio_to_most_common': 1.0, 'word_count': 5, 'is_uppercase': 0, 'normalized_x_pos': 0.08, 'normalized_y_pos': 0.15, 'ends_with_period': 1},
        {'text': '1.1 Sub-section A', 'page_number': 0, 'font_size': 16.0, 'is_bold': 1, 'predicted_label': 1, 'starts_with_heading_pattern': 1, 'font_size_ratio_to_most_common': 1.2, 'word_count': 3, 'is_uppercase': 0, 'normalized_x_pos': 0.12, 'normalized_y_pos': 0.18, 'ends_with_period': 0},
        {'text': 'More body text.', 'page_number': 0, 'font_size': 12.0, 'is_bold': 0, 'predicted_label': 0, 'starts_with_heading_pattern': 0, 'font_size_ratio_to_most_common': 1.0, 'word_count': 3, 'is_uppercase': 0, 'normalized_x_pos': 0.12, 'normalized_y_pos': 0.21, 'ends_with_period': 1},
        {'text': '2. Another Section', 'page_number': 1, 'font_size': 20.0, 'is_bold': 1, 'predicted_label': 1, 'starts_with_heading_pattern': 1, 'font_size_ratio_to_most_common': 1.5, 'word_count': 3, 'is_uppercase': 0, 'normalized_x_pos': 0.08, 'normalized_y_pos': 0.05, 'ends_with_period': 0},
        {'text': '2.1.1 A Deeper Level', 'page_number': 1, 'font_size': 14.0, 'is_bold': 1, 'predicted_label': 1, 'starts_with_heading_pattern': 1, 'font_size_ratio_to_most_common': 1.1, 'word_count': 4, 'is_uppercase': 0, 'normalized_x_pos': 0.15, 'normalized_y_pos': 0.08, 'ends_with_period': 0},
        {'text': 'A very long, all-caps sentence that is actually body text but might be mistaken for a heading by the model.', 'page_number': 1, 'font_size': 14.0, 'is_bold': 0, 'predicted_label': 1, 'starts_with_heading_pattern': 0, 'font_size_ratio_to_most_common': 1.1, 'word_count': 25, 'is_uppercase': 1, 'normalized_x_pos': 0.08, 'normalized_y_pos': 0.12, 'ends_with_period': 1}, # Should be demoted
        {'text': '3. Conclusion', 'page_number': 2, 'font_size': 20.0, 'is_bold': 1, 'predicted_label': 1, 'starts_with_heading_pattern': 1, 'font_size_ratio_to_most_common': 1.5, 'word_count': 2, 'is_uppercase': 0, 'normalized_x_pos': 0.08, 'normalized_y_pos': 0.05, 'ends_with_period': 0},
        {'text': 'Summary of findings.', 'page_number': 2, 'font_size': 12.0, 'is_bold': 0, 'predicted_label': 0, 'starts_with_heading_pattern': 0, 'font_size_ratio_to_most_common': 1.0, 'word_count': 3, 'is_uppercase': 0, 'normalized_x_pos': 0.08, 'normalized_y_pos': 0.08, 'ends_with_period': 1},
    ]

    print("Initial Binary Predictions (simulated):")
    for item in sample_data:
        print(f"  '{item['text']}' -> Binary Pred: {item['predicted_label']}")

    filter = SemanticFilter()
    refined_preds = filter.apply_filters_and_assign_levels(sample_data)

    print("\nRefined H-Level Predictions (0:None, 1:H1, 2:H2, etc.):")
    for i, pred in enumerate(refined_preds):
        print(f"  '{sample_data[i]['text']}' -> Refined H-Level: {pred}")
