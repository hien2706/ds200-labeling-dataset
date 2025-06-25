import os
import json
import shutil
from collections import defaultdict

def scale_down_dataset_ordered(source_dir='data/raw', target_dir='data/raw_new', keep_files=None, taichinhnganhang_existing=1000):
    """
    Scale down dataset to have exactly 2200 samples per category.
    Save in specified category order with ascending IDs within each category.
    Files can contain mixed categories with exactly 500 entries each.
    Keep specified files unchanged.
    
    Args:
        source_dir (str): Source directory containing JSON files
        target_dir (str): Target directory for scaled dataset
        keep_files (list): Files to keep unchanged
        taichinhnganhang_existing (int): Existing samples count for 'taichinhnganhang'
    """
    if keep_files is None:
        keep_files = ['tokenized_data_500.json', 'tokenized_data_1000.json']

    # Define the desired category order
    category_order = ['taichinhnganhang', 'taichinhquocte', 'doanhnghiep', 'vimo', 'thitruongchungkhoan']

    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist!")
        return

    # Create target directory if not exists
    os.makedirs(target_dir, exist_ok=True)
    print(f"Created target directory: {target_dir}")

    # Copy keep_files as is
    for keep_file in keep_files:
        src_path = os.path.join(source_dir, keep_file)
        dst_path = os.path.join(target_dir, keep_file)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copied {keep_file} unchanged")

    # Load all data except keep_files
    all_data = defaultdict(list)  # category -> list of (original_key, data)

    # Read all other files
    files = [f for f in os.listdir(source_dir) if f.endswith('.json') and f not in keep_files]
    print(f"Processing {len(files)} files...")

    for file in files:
        filepath = os.path.join(source_dir, file)
        print(f"Reading {file}...")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for key, value in data.items():
                    # Skip empty para_doc
                    if 'para_doc' in value and len(value['para_doc']) == 0:
                        continue
                    
                    # Extract category
                    if '_' in key:
                        category = key.rsplit('_', 1)[0]
                        all_data[category].append((key, value))
        except Exception as e:
            print(f"Error reading {file}: {e}")

    # For taichinhnganhang, only add 1200 more samples (to reach 2200 total)
    taichinhnganhang_needed = 2200 - taichinhnganhang_existing  # 1200
    # For other categories, need 2200 samples each
    samples_per_other_category = 2200

    print(f"\nFound categories: {list(all_data.keys())}")
    print(f"Processing in order: {category_order}")
    print(f"Existing taichinhnganhang samples in keep_files: {taichinhnganhang_existing}")
    print(f"Additional taichinhnganhang samples needed: {taichinhnganhang_needed}")
    print(f"Samples per other category: {samples_per_other_category}")

    # Process categories in specified order and collect entries
    ordered_entries = []  # List of (new_key, value) tuples in desired order

    for category in category_order:
        if category not in all_data:
            print(f"Warning: Category '{category}' not found in data")
            continue
            
        available_samples = len(all_data[category])
        if category == 'taichinhnganhang':
            samples_to_take = min(taichinhnganhang_needed, available_samples)
            start_id = taichinhnganhang_existing  # Start from 1000
        else:
            samples_to_take = min(samples_per_other_category, available_samples)
            start_id = 0  # Start from 0 for other categories
        
        selected_entries = all_data[category][:samples_to_take]
        
        # Create ordered entries for this category
        category_entries = []
        for i, (old_key, value) in enumerate(selected_entries):
            new_id = i + start_id
            new_key = f"{category}_{new_id}"
            category_entries.append((new_key, value))
        
        # Sort by ID within category (ascending order)
        category_entries.sort(key=lambda x: int(x[0].rsplit('_', 1)[1]))
        ordered_entries.extend(category_entries)
        
        print(f"{category}: selected {samples_to_take} out of {available_samples} samples")

    # Distribute entries across files with exactly 500 entries each
    # Maintain the order: categories in specified order, IDs ascending within each category
    entries_per_file = 500
    file_counter = 1500  # Start from 1500 to avoid conflict with keep_files

    for i in range(0, len(ordered_entries), entries_per_file):
        batch = ordered_entries[i:i + entries_per_file]
        filename = f"tokenized_data_{file_counter}.json"
        filepath = os.path.join(target_dir, filename)
        
        # Create ordered dictionary to maintain order in JSON file
        file_data = {}
        for new_key, value in batch:
            file_data[new_key] = value

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(file_data, f, ensure_ascii=False, indent=2)

        # Count categories in this file
        category_count = defaultdict(int)
        for new_key in file_data.keys():
            cat = new_key.rsplit('_', 1)[0]
            category_count[cat] += 1

        # Display category summary in the specified order
        category_summary = ", ".join([f"{cat}: {category_count[cat]}" for cat in category_order if cat in category_count])
        print(f"Saved {filename} with {len(file_data)} entries ({category_summary})")

        file_counter += 500

    print(f"\nScaling complete!")
    print(f"Total entries saved: {len(ordered_entries)}")
    print(f"Scaled down dataset saved to {target_dir}")

    # Print final summary by category in specified order
    print("\nFinal summary by category:")
    category_counts = defaultdict(int)
    for new_key, _ in ordered_entries:
        category = new_key.rsplit('_', 1)[0]
        category_counts[category] += 1

    # Add existing samples from keep_files for taichinhnganhang
    category_counts['taichinhnganhang'] += taichinhnganhang_existing

    for category in category_order:
        if category in category_counts:
            print(f"  {category}: {category_counts[category]} entries")

if __name__ == '__main__':
    scale_down_dataset_ordered()
