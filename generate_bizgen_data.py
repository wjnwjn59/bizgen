#!/usr/bin/env python3
"""
BizGen Data Generator for Colab
Generates test data in BizGen format from narrator-generated infographic data.
This script is designed to be run in Google Colab environment.
"""

import json
import os
import random
import re
import argparse
from typing import List, Dict, Any, Tuple


def load_json(filepath: str) -> Any:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, filepath: str):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def calculate_bbox_area(bbox: Dict) -> int:
    """Calculate area of a bounding box."""
    top_left = bbox['top_left']
    bottom_right = bbox['bottom_right']
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]
    
    if width <= 0 or height <= 0:
        return 0
        
    return width * height


def bboxes_overlap(bbox1: Dict, bbox2: Dict, threshold: float = 0.3) -> bool:
    """Check if two bounding boxes overlap significantly."""
    x1_min, y1_min = bbox1['top_left']
    x1_max, y1_max = bbox1['bottom_right']
    x2_min, y2_min = bbox2['top_left']
    x2_max, y2_max = bbox2['bottom_right']
    
    # Calculate intersection
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection = x_overlap * y_overlap
    
    # Calculate areas
    area1 = calculate_bbox_area(bbox1)
    area2 = calculate_bbox_area(bbox2)
    
    # Check if intersection is significant relative to smaller bbox
    min_area = min(area1, area2)
    return (intersection / min_area) > threshold if min_area > 0 else False


def select_largest_non_overlapping_bboxes(bboxes: List[Dict], category: str, count: int) -> List[Dict]:
    """Select the largest non-overlapping bounding boxes of a specific category."""
    # Filter by category
    filtered = [b for b in bboxes if b.get('category') == category]
    
    # Sort by area (largest first)
    filtered.sort(key=calculate_bbox_area, reverse=True)
    
    selected = []
    for bbox in filtered:
        if len(selected) >= count:
            break
        
        # Check if this bbox overlaps with any already selected bbox
        overlaps = any(bboxes_overlap(bbox, sel_bbox) for sel_bbox in selected)
        if not overlaps:
            selected.append(bbox)
    
    return selected


def get_random_colors(color_idx: Dict, num_colors: int) -> List[str]:
    """Get random colors from color index."""
    colors = list(color_idx.keys())
    # Exclude white for better visibility
    colors = [c for c in colors if c not in ['white']]
    return random.sample(colors, min(num_colors, len(colors)))


def get_random_font(font_idx: Dict) -> str:
    """Get a random English font from font index."""
    # Always use English fonts only
    fonts = [k for k in font_idx.keys() if k.startswith('en-')]
    
    if not fonts:
        print("Warning: No English fonts found in font index")
        return "en-YACgEQNAr7w,1"
    
    return random.choice(fonts)


def extract_text_elements(full_caption: str) -> List[str]:
    """Extract text content from caption (quoted text)."""
    text_elements = []
    
    # Pattern to match quoted text
    pattern = r'"([^"]+)"'
    matches = re.findall(pattern, full_caption)
    
    for text_content in matches:
        text_elements.append(text_content.strip())
    
    return text_elements


def extract_images_from_caption(full_caption: str) -> List[str]:
    """Extract image descriptions from caption using Figure: format."""
    image_elements = []
    
    # Pattern to match "Figure: " followed by description up to a period
    pattern = r'Figure:\s+([^.]+\.)'
    matches = re.findall(pattern, full_caption, re.IGNORECASE)
    
    for description in matches:
        clean_desc = description.strip()
        if clean_desc:
            image_elements.append(clean_desc)
    
    return image_elements


def extract_images_from_figures(figures: List[Dict]) -> List[Dict]:
    """Extract image descriptions from figures data by randomly selecting from ideas."""
    image_elements = []
    
    for figure in figures:
        if 'ideas' in figure and figure['ideas']:
            # Randomly select one idea from the figure
            selected_idea = random.choice(figure['ideas'])
            image_elements.append({
                'caption': selected_idea,
                'type': 'figure'
            })
    
    return image_elements


def sort_by_reading_order(bboxes: List[Dict]) -> List[Dict]:
    """Sort bboxes by reading order (left to right, top to bottom)."""
    def reading_order_key(bbox):
        top_left = bbox['top_left']
        return (top_left[1] // 100, top_left[0])  # Group by y-coordinate, then sort by x
    
    return sorted(bboxes, key=reading_order_key)


def create_bizgen_format_data(
    infographic_data: List[Dict],
    extracted_bboxes: List[Dict],
    color_idx: Dict,
    font_idx: Dict,
    max_samples: int = 10
) -> List[Dict]:
    """
    Create data in BizGen format from narrator-generated infographic data.
    
    Args:
        infographic_data: List of infographic data with captions and figures
        extracted_bboxes: List of bbox data
        color_idx: Color index mapping
        font_idx: Font index mapping
        max_samples: Maximum number of samples to generate
    
    Returns:
        List of data in BizGen format
    """
    result = []
    
    # Create a mapping of indices from extracted_bboxes
    bbox_by_index = {item['index']: item for item in extracted_bboxes}
    available_indices = list(bbox_by_index.keys())
    
    for idx, infographic in enumerate(infographic_data[:max_samples]):
        print(f"Processing infographic {idx + 1}/{min(len(infographic_data), max_samples)}")
        
        # Get basic info
        full_caption = infographic.get('full_image_caption', '')
        background_caption = infographic.get('background_caption', '')
        figures = infographic.get('figures', [])
        
        # Extract text and image elements
        text_elements = extract_text_elements(full_caption)
        image_elements_from_caption = extract_images_from_caption(full_caption)
        image_elements_from_figures = extract_images_from_figures(figures)
        
        # Combine image elements
        all_image_elements = image_elements_from_caption + [img['caption'] for img in image_elements_from_figures]
        
        if not text_elements and not all_image_elements:
            print(f"  Skipping infographic {idx + 1}: No text or image elements found")
            continue
        
        # Randomly select a bbox set
        if not available_indices:
            print("  Warning: No more bbox indices available")
            break
        
        selected_idx = random.choice(available_indices)
        bbox_data = bbox_by_index[selected_idx]
        
        # Get bboxes for layout
        layout_bboxes = bbox_data.get('bboxes', [])
        if not layout_bboxes:
            print(f"  Skipping infographic {idx + 1}: No bboxes found for index {selected_idx}")
            continue
        
        # Filter and select bboxes
        text_bboxes = [b for b in layout_bboxes if b.get('category') == 'text']
        element_bboxes = [b for b in layout_bboxes if b.get('category') == 'element']
        
        # Select non-overlapping bboxes
        max_text_count = min(len(text_elements), len(text_bboxes), 5)  # Limit to 5 text elements
        max_element_count = min(len(all_image_elements), len(element_bboxes), 3)  # Limit to 3 image elements
        
        selected_text_bboxes = select_largest_non_overlapping_bboxes(text_bboxes, 'text', max_text_count)
        selected_element_bboxes = select_largest_non_overlapping_bboxes(element_bboxes, 'element', max_element_count)
        
        if not selected_text_bboxes and not selected_element_bboxes:
            print(f"  Skipping infographic {idx + 1}: No suitable bboxes selected")
            continue
        
        # Sort bboxes by reading order
        selected_text_bboxes = sort_by_reading_order(selected_text_bboxes)
        selected_element_bboxes = sort_by_reading_order(selected_element_bboxes)
        
        # Create layers
        layers = []
        
        # Add background layer (full canvas)
        layers.append({
            "category": "background",
            "top_left": [0, 0],
            "bottom_right": [896, 2240],
            "color": random.choice(list(color_idx.keys())),
            "caption": background_caption or "Background"
        })
        
        # Add text layers
        font_token = get_random_font(font_idx)
        text_colors = get_random_colors(color_idx, len(selected_text_bboxes))
        
        for i, (text_bbox, text_content) in enumerate(zip(selected_text_bboxes, text_elements[:len(selected_text_bboxes)])):
            layers.append({
                "category": "text",
                "top_left": text_bbox["top_left"],
                "bottom_right": text_bbox["bottom_right"],
                "font": font_token,
                "color": text_colors[i % len(text_colors)],
                "text": text_content,
                "caption": f'Text "{text_content}"'
            })
        
        # Add image/element layers
        element_colors = get_random_colors(color_idx, len(selected_element_bboxes))
        
        for i, (elem_bbox, image_desc) in enumerate(zip(selected_element_bboxes, all_image_elements[:len(selected_element_bboxes)])):
            layers.append({
                "category": "element",
                "top_left": elem_bbox["top_left"],
                "bottom_right": elem_bbox["bottom_right"],
                "color": element_colors[i % len(element_colors)],
                "caption": f"Figure: {image_desc}"
            })
        
        # Create the final data entry
        infographic_id = f"demo_{idx + 1:06d}"
        
        bizgen_entry = {
            "infographic_id": infographic_id,
            "layers": layers,
            "full_image_caption": full_caption,
            "background_caption": background_caption,
            "text_count": len([l for l in layers if l["category"] == "text"]),
            "element_count": len([l for l in layers if l["category"] == "element"]),
        }
        
        result.append(bizgen_entry)
        print(f"  Generated infographic {infographic_id} with {len(layers)} layers")
    
    return result


def validate_input_format(data: List[Dict]) -> Tuple[bool, str]:
    """
    Validate the input data format.
    
    Args:
        data: List of infographic data entries
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(data, list):
        return False, "Input data should be a list"
    
    if not data:
        return False, "Input data is empty"
    
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            return False, f"Entry {i} is not a dictionary"
        
        required_fields = ['full_image_caption', 'background_caption', 'figures']
        for field in required_fields:
            if field not in entry:
                return False, f"Entry {i} missing required field: {field}"
        
        # Check if full_image_caption has some content
        if not entry['full_image_caption'].strip():
            return False, f"Entry {i} has empty full_image_caption"
        
        # Check figures format
        if not isinstance(entry['figures'], list):
            return False, f"Entry {i} figures field should be a list"
    
    return True, "Format is valid"


def generate_bizgen_data_from_string(
    input_json_string: str,
    max_samples: int = 10,
    output_path: str = "meta/test_infographics.json"
) -> List[Dict]:
    """
    Generate BizGen format data from JSON string input.
    
    Args:
        input_json_string: JSON string containing infographic data
        max_samples: Maximum number of samples to generate
        output_path: Path to save the output file
    
    Returns:
        List of BizGen format data
    """
    print("=== BizGen Data Generator ===")
    print("Generating test data for BizGen inference from JSON string...")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # File paths (assuming we're in the bizgen directory)
    extracted_bboxes_path = "glyph/extracted_bboxes.json"
    color_idx_path = "glyph/color_idx.json"
    font_idx_path = "glyph/font_uni_10-lang_idx.json"
    
    # Check if required files exist
    required_files = [extracted_bboxes_path, color_idx_path, font_idx_path]
    for filepath in required_files:
        if not os.path.exists(filepath):
            print(f"Error: Required file not found: {filepath}")
            print("Please ensure you have cloned BizGen and are in the correct directory.")
            return []
    
    # Load required data
    print("Loading required data files...")
    try:
        extracted_bboxes = load_json(extracted_bboxes_path)
        color_idx = load_json(color_idx_path)
        font_idx = load_json(font_idx_path)
        
        # Parse input JSON string
        input_data = json.loads(input_json_string)
        
        # Validate input format
        is_valid, error_msg = validate_input_format(input_data)
        if not is_valid:
            print(f"Error: Invalid input format - {error_msg}")
            return []
        
        print(f"  - Loaded {len(extracted_bboxes)} bbox sets")
        print(f"  - Loaded {len(color_idx)} colors")
        print(f"  - Loaded {len(font_idx)} fonts")
        print(f"  - Parsed {len(input_data)} infographic entries from JSON string")
        print(f"  - Input format validation: {error_msg}")
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON string: {e}")
        return []
    except Exception as e:
        print(f"Error loading data files: {e}")
        return []
    
    # Generate BizGen format data
    print("\nGenerating BizGen format data...")
    try:
        bizgen_data = create_bizgen_format_data(
            infographic_data=input_data,
            extracted_bboxes=extracted_bboxes,
            color_idx=color_idx,
            font_idx=font_idx,
            max_samples=max_samples
        )
        
        if not bizgen_data:
            print("No data generated. Please check your input data format.")
            return []
        
        # Save results
        save_json(bizgen_data, output_path)
        print(f"\nSuccess! Generated {len(bizgen_data)} test samples")
        print(f"Output saved to: {output_path}")
        print("\nYou can now run BizGen inference with:")
        print(f"!python inference.py --ckpt_dir checkpoints/lora/infographic --output_dir infographic --sample_list {output_path}")
        
        return bizgen_data
        
    except Exception as e:
        print(f"Error generating data: {e}")
        return []


def main():
    """Main function for Colab environment (legacy support for file input)."""
    print("=== BizGen Data Generator ===")
    print("Generating test data for BizGen inference...")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # File paths (assuming we're in the bizgen directory)
    extracted_bboxes_path = "glyph/extracted_bboxes.json"
    color_idx_path = "glyph/color_idx.json"
    font_idx_path = "glyph/font_uni_10-lang_idx.json"
    
    # Input data path (will be provided by user)
    input_data_path = "input_infographic_data.json"  # This should be downloaded via gdown
    
    # Output path
    output_path = "meta/test_infographics.json"
    
    # Check if required files exist
    required_files = [extracted_bboxes_path, color_idx_path, font_idx_path]
    for filepath in required_files:
        if not os.path.exists(filepath):
            print(f"Error: Required file not found: {filepath}")
            print("Please ensure you have cloned BizGen and are in the correct directory.")
            return
    
    # Check if input data exists
    if not os.path.exists(input_data_path):
        print(f"Error: Input data file not found: {input_data_path}")
        print("Please download your input data file using gdown and name it 'input_infographic_data.json'")
        return
    
    # Load required data
    print("Loading required data files...")
    try:
        extracted_bboxes = load_json(extracted_bboxes_path)
        color_idx = load_json(color_idx_path)
        font_idx = load_json(font_idx_path)
        input_data = load_json(input_data_path)
        
        print(f"  - Loaded {len(extracted_bboxes)} bbox sets")
        print(f"  - Loaded {len(color_idx)} colors")
        print(f"  - Loaded {len(font_idx)} fonts")
        print(f"  - Loaded {len(input_data)} infographic entries")
        
    except Exception as e:
        print(f"Error loading data files: {e}")
        return
    
    # Generate BizGen format data
    print("\nGenerating BizGen format data...")
    try:
        bizgen_data = create_bizgen_format_data(
            infographic_data=input_data,
            extracted_bboxes=extracted_bboxes,
            color_idx=color_idx,
            font_idx=font_idx,
            max_samples=10  # Generate 10 test samples
        )
        
        if not bizgen_data:
            print("No data generated. Please check your input data format.")
            return
        
        # Save results
        save_json(bizgen_data, output_path)
        print(f"\nSuccess! Generated {len(bizgen_data)} test samples")
        print(f"Output saved to: {output_path}")
        print("\nYou can now run BizGen inference with:")
        print(f"!python inference.py --ckpt_dir checkpoints/lora/infographic --output_dir infographic --sample_list {output_path}")
        
    except Exception as e:
        print(f"Error generating data: {e}")
        return


if __name__ == '__main__':
    main()