#!/usr/bin/env python3
"""
Script to sort paragraph IDs by numeric suffix and save to sorted folder
"""

import json
import re
from pathlib import Path

def extract_numeric_suffix(key: str) -> int:
    """Extract numeric suffix from key like 'taichinhnganhang_14' -> 14"""
    match = re.search(r'_(\d+)$', key)
    return int(match.group(1)) if match else -1

def sort_paragraphs_by_id(data: dict) -> dict:
    """Sort dictionary by paragraph ID numeric suffix"""
    sorted_keys = sorted(data.keys(), key=extract_numeric_suffix)
    sorted_data = {k: data[k] for k in sorted_keys}
    return sorted_data

def sort_and_fix_file(input_file: str):
    """Sort and fix the specific file"""
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return
    
    # Create sorted directory
    sorted_dir = input_path.parent / "sorted"
    sorted_dir.mkdir(exist_ok=True)
    
    try:
        # Load the file
        print(f"Loading file: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Found {len(data)} paragraphs")
        print(f"Current order (first 5): {list(data.keys())[:5]}...")
        
        # Sort by paragraph ID
        sorted_data = sort_paragraphs_by_id(data)
        
        print(f"Sorted order (first 5): {list(sorted_data.keys())[:5]}...")
        
        # Save sorted file to new location
        output_file = sorted_dir / input_path.name
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Successfully sorted and saved to: {output_file}")
        
        # Show the complete order
        print(f"\nComplete sorted order:")
        for i, key in enumerate(list(sorted_data.keys())[:10]):  # Show first 10
            print(f"  {i+1}. {key}")
        if len(sorted_data) > 10:
            print(f"  ... and {len(sorted_data) - 10} more")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return False

def main():
    """Main function"""
    # file_path = "/home/hien2706/school/nam3_hk2/ds200/labeling-dataset/data/processed/agentA/tokenized_data_1000.json_agentA.json"
    file_path = "/home/hien2706/school/nam3_hk2/ds200/labeling-dataset/data/processed/agentB/tokenized_data_1000.json_final.json"
    
    print(f"Sorting and fixing {file_path}")
    print("=" * 60)
    
    success = sort_and_fix_file(file_path)
    
    if success:
        print("\n✅ File successfully sorted!")
        print(f"📁 Original file preserved at: {file_path}")
        print(f"📁 Sorted file saved to: {Path(file_path).parent}/sorted/{Path(file_path).name}")
    else:
        print("\n❌ Failed to sort file")

if __name__ == "__main__":
    main()
