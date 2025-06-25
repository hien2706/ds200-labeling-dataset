import os
import json

def analyze_dataset(directory='data/raw'):
    """
    Analyze tokenized dataset to count categories, their numbers, and empty para_doc entries.
    
    Args:
        directory (str): Path to the directory containing JSON files
    
    Returns:
        tuple: (category_counts, empty_paradoc_count, total_entries)
    """
    category_counts = {}
    empty_paradoc_count = 0
    total_entries = 0
    
    # Get all JSON files in the directory
    try:
        files = [f for f in os.listdir(directory) if f.endswith('.json')]
        print(f"Found {len(files)} JSON files to process...")
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found!")
        return {}, 0, 0
    
    # Process each file
    for file in files:
        filepath = os.path.join(directory, file)
        print(f"Processing {file}...")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Process each entry in the file
                for key, value in data.items():
                    total_entries += 1
                    
                    # Extract category and number from key (e.g., "taichinhnganhang_3")
                    if '_' in key:
                        category, number = key.rsplit('_', 1)
                        
                        # Initialize category set if not exists
                        if category not in category_counts:
                            category_counts[category] = set()
                        
                        # Add the number to the category set
                        category_counts[category].add(number)
                    
                    # Check if para_doc is empty
                    if 'para_doc' in value and len(value['para_doc']) == 0:
                        empty_paradoc_count += 1
                        
        except json.JSONDecodeError:
            print(f"Error: Could not parse JSON in file {file}")
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
    
    # Convert sets to counts for final output
    category_final_counts = {cat: len(nums) for cat, nums in category_counts.items()}
    
    return category_final_counts, empty_paradoc_count, total_entries

def print_analysis_results(category_counts, empty_paradoc_count, total_entries):
    """Print formatted analysis results."""
    print("\n" + "="*60)
    print("DATASET ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\nTotal entries processed: {total_entries}")
    print(f"Entries with empty para_doc: {empty_paradoc_count}")
    print(f"Percentage of empty entries: {(empty_paradoc_count/total_entries)*100:.2f}%" if total_entries > 0 else "N/A")
    
    print(f"\nCategories found: {len(category_counts)}")
    print("-" * 40)
    
    for category, count in sorted(category_counts.items()):
        print(f"{category}: {count} entries")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # Run the analysis
    category_counts, empty_count, total = analyze_dataset(directory='data/raw_new')
    
    # Print results
    print_analysis_results(category_counts, empty_count, total)
    
    # Optional: Save results to a summary file
    summary = {
        "total_entries": total,
        "empty_paradoc_count": empty_count,
        "category_counts": category_counts,
        "empty_percentage": (empty_count/total)*100 if total > 0 else 0
    }
    
    # with open('dataset_analysis_summary.json', 'w', encoding='utf-8') as f:
    #     json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # print("Analysis summary saved to 'dataset_analysis_summary.json'")
