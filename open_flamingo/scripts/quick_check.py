#!/usr/bin/env python3
"""
Quick script to check dataset structure and find filename columns
"""

from datasets import load_from_disk
import json

def quick_check():
    local_path = "/mnt/data/maund/open_med_flamingo/open_flamingo/data/llavamed_dataset"
    
    print("Loading dataset...")
    ds = load_from_disk(local_path)
    
    print(f"Dataset splits: {list(ds.keys())}")
    
    for split_name in ds.keys():
        split_data = ds[split_name]
        print(f"\n=== {split_name.upper()} ===")
        print(f"Size: {len(split_data)}")
        print(f"Columns: {split_data.column_names}")
        
        if len(split_data) > 0:
            first_example = split_data[0]
            print("\nFirst example preview:")
            for key, value in first_example.items():
                print(f"  {key}: {type(value)} - {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")

if __name__ == "__main__":
    quick_check()
