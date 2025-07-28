import numpy as np
import re
from collections import Counter

def engineer_features(processed_lines):
    """
    Converts each preprocessed line into a vector of highly discriminative features
    for heading classification, with enhanced cross-page consistency considerations
    and features for visually prominent text in non-standard documents.
    Handles potential NoneType errors by providing safe default values for all accessed attributes.
    """
    if not processed_lines:
        return []

    features_list = []

    # --- Global Document-Level Feature Calculation ---
    # Ensure all_font_sizes only contains valid numbers
    all_font_sizes = [line.get('font_size') for line in processed_lines if line.get('font_size') is not None]
    
    if not all_font_sizes:
        # Fallback if no valid font sizes are found (e.g., empty or malformed PDF)
        doc_max_font_size = 1.0
        doc_min_font_size = 1.0
        doc_avg_font_size = 1.0
        doc_median_font_size = 1.0
        most_common_font_size = 1.0
    else:
        doc_max_font_size = max(all_font_sizes)
        doc_min_font_size = min(all_font_sizes)
        doc_avg_font_size = np.mean(all_font_sizes)
        doc_median_font_size = np.median(all_font_sizes)
        font_size_counts = Counter(all_font_sizes)
        most_common_font_size = font_size_counts.most_common(1)[0][0]

    # Identify dominant styles for potential H1, H2, H3 across the document
    potential_heading_styles = []
    for line in processed_lines:
        # Safely get bbox, font_size, is_bold, and text
        line_bbox = line.get('bbox')
        line_font_size = line.get('font_size')
        line_is_bold = line.get('is_bold', False)
        line_text = line.get('text', '')

        if line_bbox is not None and line_font_size is not None:
            # Heuristic: large font, possibly bold, or starts with pattern
            if (line_font_size > doc_avg_font_size * 1.2 and line_is_bold) or \
               (re.match(r'^((\d+(\.\d+)*)|([A-Za-z]\.?)|(\([ivxlcdm]+\)))\s+', line_text)):
                potential_heading_styles.append({
                    'font_size': line_font_size,
                    'is_bold': line_is_bold,
                    'normalized_x_pos': line_bbox[0] / 612.0, # Using fixed width for consistency
                    'font_name': line.get('font_name', '')
                })
    
    dominant_h_styles = {} 
    if potential_heading_styles: # Only sort if there are elements
        potential_heading_styles.sort(key=lambda x: x['font_size'], reverse=True)

        # Heuristic to pick dominant styles for H1, H2, H3
        h1_candidates = [s for s in potential_heading_styles if s['font_size'] >= doc_max_font_size * 0.8 and s['is_bold'] and s['normalized_x_pos'] < 0.1]
        if h1_candidates: dominant_h_styles['H1'] = h1_candidates[0]
        h2_candidates = [s for s in potential_heading_styles if s['font_size'] >= doc_avg_font_size * 1.4 and s['is_bold'] and s['normalized_x_pos'] > 0.05 and s['normalized_x_pos'] < 0.2]
        if h2_candidates: dominant_h_styles['H2'] = h2_candidates[0]
        h3_candidates = [s for s in potential_heading_styles if s['font_size'] >= doc_avg_font_size * 1.1 and s['is_bold'] and s['normalized_x_pos'] > 0.1 and s['normalized_x_pos'] < 0.3]
        if h3_candidates: dominant_h_styles['H3'] = h3_candidates[0]

    # Calculate page-level text density
    page_text_lengths = Counter()
    for line in processed_lines:
        page_text_lengths[line.get('page_number', 0)] += len(line.get('text', ''))
    
    # Estimate average page height (for density calculation)
    valid_bbox_heights = [line['bbox'][3] - line['bbox'][1] for line in processed_lines if 'bbox' in line and line['bbox'] is not None]
    avg_page_height = np.mean(valid_bbox_heights) if valid_bbox_heights else 792.0


    # --- Feature Engineering for Each Line ---
    for i, line in enumerate(processed_lines):
        # Safely get line properties, defaulting to safe values if missing or None
        text = line.get("text", "")
        font_size = line.get("font_size", 0.0) # Default to 0.0 if font_size is None
        is_bold = line.get("is_bold", False)
        is_italic = line.get("is_italic", False)
        page_num = line.get('page_number', 0)
        
        # Safely get bbox coordinates, default to (0,0,0,0) if bbox is None or missing
        bbox = line.get('bbox')
        x0, y0, x1, y1 = bbox if bbox is not None else (0,0,0,0)

        # Text-based features
        is_uppercase = int(text.isupper() and len(text.split()) > 1 and len(text) > 5)
        starts_with_heading_pattern = int(bool(re.match(r'^((\d+(\.\d+)*)|([A-Za-z]\.?)|(\([ivxlcdm]+\)))\s+', text)))
        word_count = len(text.split())
        char_count = len(text)
        ends_with_period = int(text.endswith('.'))
        ends_with_colon = int(text.endswith(':'))

        # Relative Font Sizes (handle division by zero if doc_min/max_font_size are same)
        normalized_font_size_doc_max = (font_size - doc_min_font_size) / (doc_max_font_size - doc_min_font_size + 1e-6) if (doc_max_font_size - doc_min_font_size) > 0 else 0.0
        font_size_ratio_to_avg = font_size / (doc_avg_font_size + 1e-6)
        font_size_ratio_to_median = font_size / (doc_median_font_size + 1e-6)
        font_size_ratio_to_most_common = font_size / (most_common_font_size + 1e-6)

        # Positional Features
        page_width_estimate = 612.0 # Standard A4/Letter width in points
        normalized_x_pos = x0 / page_width_estimate if page_width_estimate > 0 else 0.0
        page_height_estimate = 792.0 # Standard A4/Letter height in points
        normalized_y_pos = y0 / page_height_estimate if page_height_estimate > 0 else 0.0

        # Contextual Features (Requires looking at previous/next lines)
        prev_line = processed_lines[i-1] if i > 0 else None
        next_line = processed_lines[i+1] if i < len(processed_lines) - 1 else None

        # Safely calculate font_size_diff_prev and vertical_gap_prev
        prev_font_size = prev_line.get('font_size', 0.0) if prev_line else 0.0
        font_size_diff_prev = (font_size - prev_font_size)

        prev_bbox_bottom = prev_line.get('bbox', [0,0,0,0])[3] if prev_line and prev_line.get('bbox') is not None else 0.0
        vertical_gap_prev = (y0 - prev_bbox_bottom)
        normalized_vertical_gap_prev = vertical_gap_prev / (font_size + 1e-6) if font_size > 0 else 0.0 # Avoid division by zero


        is_title_candidate = int(
            page_num == 0 and 
            normalized_y_pos < 0.2 and 
            font_size_ratio_to_avg > 1.8 and 
            word_count < 10 
        )
        prev_text_ends_with_colon = int(prev_line.get('text', '').strip().endswith(':')) if prev_line else 0
        
        next_line_indented = 0
        next_bbox = next_line.get('bbox') if next_line else None
        next_bbox_x0 = next_bbox[0] if next_bbox is not None else 0.0
        if next_line and (next_bbox_x0 - x0) / page_width_estimate > 0.02:
            next_line_indented = 1

        line_has_common_heading_phrase = int(bool(re.search(r"^(Chapter|Section|Appendix|Introduction|Conclusion|Methodology|Results|Discussion|References)\b", text, re.IGNORECASE)))
        
        # CROSS-PAGE CONSISTENCY FEATURES (MORE ROBUST):
        # Use .get() and check if the style dict exists AND its values are not None
        h1_style = dominant_h_styles.get('H1')
        is_h1_style_consistent = int(
            h1_style is not None and # Check if H1 style was identified
            h1_style.get('font_size') is not None and h1_style.get('is_bold') is not None and h1_style.get('normalized_x_pos') is not None and # Check individual values
            abs(font_size - h1_style.get('font_size', 0.0)) < 2 and # Within 2 points
            is_bold == h1_style.get('is_bold', False) and
            abs(normalized_x_pos - h1_style.get('normalized_x_pos', 0.0)) < 0.01 # Close indentation
        )
        h2_style = dominant_h_styles.get('H2')
        is_h2_style_consistent = int(
            h2_style is not None and
            h2_style.get('font_size') is not None and h2_style.get('is_bold') is not None and h2_style.get('normalized_x_pos') is not None and
            abs(font_size - h2_style.get('font_size', 0.0)) < 2 and
            is_bold == h2_style.get('is_bold', False) and
            abs(normalized_x_pos - h2_style.get('normalized_x_pos', 0.0)) < 0.01
        )
        h3_style = dominant_h_styles.get('H3')
        is_h3_style_consistent = int(
            h3_style is not None and
            h3_style.get('font_size') is not None and h3_style.get('is_bold') is not None and h3_style.get('normalized_x_pos') is not None and
            abs(font_size - h3_style.get('font_size', 0.0)) < 2 and
            is_bold == h3_style.get('is_bold', False) and
            abs(normalized_x_pos - h3_style.get('normalized_x_pos', 0.0)) < 0.01
        )
        
        # NEW FEATURES for visually-driven documents:
        is_centered = int(abs((x0 + x1) / 2 - page_width_estimate / 2) < 20)
        
        # Safely calculate vertical_gap_next
        next_bbox = next_line.get('bbox') if next_line else None
        next_bbox_y1 = next_bbox[1] if next_bbox is not None else 0.0
        vertical_gap_next = (next_bbox_y1 - y1)
        # Ensure font_size is not zero before division
        is_isolated = int(normalized_vertical_gap_prev > 0.5 and (vertical_gap_next / (font_size + 1e-6)) > 0.5) if font_size > 0 else 0

        absolute_font_size = font_size
        
        page_total_text_length = page_text_lengths.get(page_num, 0)
        normalized_page_text_density = page_total_text_length / (avg_page_height * page_width_estimate + 1e-6) if (avg_page_height * page_width_estimate + 1e-6) > 0 else 0.0
        
        line_width_ratio = (x1 - x0) / (page_width_estimate + 1e-6) if page_width_estimate > 0 else 0.0


        features_list.append({
            "text": text,
            "page_number": page_num,
            "font_size": font_size,
            "is_bold": is_bold,
            "is_italic": is_italic,
            "is_uppercase": is_uppercase,
            "starts_with_heading_pattern": starts_with_heading_pattern,
            "word_count": word_count,
            "char_count": char_count,
            "ends_with_period": ends_with_period,
            "ends_with_colon": ends_with_colon,
            "normalized_font_size_doc_max": normalized_font_size_doc_max,
            "font_size_ratio_to_avg": font_size_ratio_to_avg,
            "font_size_ratio_to_median": font_size_ratio_to_median,
            "font_size_ratio_to_most_common": font_size_ratio_to_most_common,
            "normalized_x_pos": normalized_x_pos,
            "normalized_y_pos": normalized_y_pos,
            "font_size_diff_prev": font_size_diff_prev,
            "vertical_gap_prev": vertical_gap_prev,
            "normalized_vertical_gap_prev": normalized_vertical_gap_prev,
            "is_title_candidate": is_title_candidate,
            "prev_text_ends_with_colon": prev_text_ends_with_colon,
            "next_line_indented": next_line_indented,
            "line_has_common_heading_phrase": line_has_common_heading_phrase,
            "is_h1_style_consistent": is_h1_style_consistent,
            "is_h2_style_consistent": is_h2_style_consistent,
            "is_h3_style_consistent": is_h3_style_consistent,
            "is_centered": is_centered,
            "is_isolated": is_isolated,
            "absolute_font_size": absolute_font_size,
            "normalized_page_text_density": normalized_page_text_density,
            "line_width_ratio": line_width_ratio,
        })
    return features_list

if __name__ == "__main__":
    # Sample processed lines for testing feature engineering
    sample_processed_lines = [
        {"text": "THE DOCUMENT TITLE", "page_number": 0, "font_size": 28.0, "font_name": "Arial-Bold", "is_bold": True, "origin": [50.0, 50.0], "bbox": [50.0, 30.0, 200.0, 60.0]},
        {"text": "A SUBTITLE:", "page_number": 0, "font_size": 14.0, "font_name": "Arial", "is_bold": False, "origin": [50.0, 70.0], "bbox": [50.0, 60.0, 150.0, 75.0]},
        {"text": "1. Introduction to AI", "page_number": 0, "font_size": 20.0, "font_name": "Arial-Bold", "is_bold": True, "origin": [50.0, 100.0], "bbox": [50.0, 80.0, 250.0, 110.0]},
        {"text": "  Artificial intelligence (AI) is a rapidly expanding field.", "page_number": 0, "font_size": 12.0, "font_name": "Arial", "is_bold": False, "origin": [70.0, 130.0], "bbox": [70.0, 115.0, 400.0, 135.0]},
        {"text": "1.1 History of AI", "page_number": 0, "font_size": 16.0, "font_name": "Arial-Bold", "is_bold": True, "origin": [50.0, 160.0], "bbox": [50.0, 145.0, 200.0, 165.0]},
        {"text": "Early AI research began in the 1950s. This is a longer body text line.", "page_number": 0, "font_size": 12.0, "font_name": "Arial", "is_bold": False, "origin": [70.0, 180.0], "bbox": [70.0, 165.0, 350.0, 185.0]},
        {"text": "Section 2: Modern Approaches", "page_number": 1, "font_size": 20.0, "font_name": "Arial-Bold", "is_bold": True, "origin": [50.0, 50.0], "bbox": [50.0, 30.0, 280.0, 60.0]},
        {"text": "2.1 Machine Learning", "page_number": 1, "font_size": 16.0, "font_name": "Arial-Bold", "is_bold": True, "origin": [70.0, 80.0], "bbox": [70.0, 65.0, 250.0, 85.0]},
        {"text": "This is a very long line of text that is all in uppercase, but it should not be a heading because it is too long and likely just emphasized body content.", "page_number": 1, "font_size": 14.0, "is_bold": False, "origin": [50.0, 100.0], "bbox": [50.0, 85.0, 450.0, 105.0]},
        {"text": "Conclusion.", "page_number": 1, "font_size": 18.0, "font_name": "Arial-Bold", "is_bold": True, "origin": [50.0, 150.0], "bbox": [50.0, 130.0, 200.0, 160.0]},
    ]

    print("Original processed lines (sample):")
    for line in sample_processed_lines:
        print(f"  Text: '{line['text']}', Size: {line['font_size']}, Bold: {line['is_bold']}, Origin: {line['origin']}")

    engineered_data = engineer_features(sample_processed_lines)

    print("\nEngineered features (sample of first 3 lines, showing new features):")
    if engineered_data:
        for i, feat in enumerate(engineered_data[:3]):
            print(f"  Line {i+1}:")
            print(f"    Text: '{feat['text']}'")
            print(f"    font_size: {feat['font_size']:.1f}, is_bold: {feat['is_bold']}")
            print(f"    is_centered: {feat['is_centered']}")
            print(f"    is_isolated: {feat['is_isolated']}")
            print(f"    absolute_font_size: {feat['absolute_font_size']:.1f}")
            print(f"    normalized_page_text_density: {feat['normalized_page_text_density']:.4f}")
            print(f"    line_width_ratio: {feat['line_width_ratio']:.2f}")
            print("-" * 20)
    else:
        print("No features engineered.")
