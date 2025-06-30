#!/usr/bin/env python3
"""
Script to examine the downloaded dataset structure and extract images
"""

from datasets import load_from_disk
import os
from PIL import Image
import json

def examine_dataset():
    # Load the dataset from local path
    local_path = "/mnt/data/maund/open_med_flamingo/open_flamingo/data/llavamed_dataset"
    
    print("Loading dataset from:", local_path)
    ds = load_from_disk(local_path)
    
    print("\n=== Dataset Info ===")
    print(f"Dataset: {ds}")
    print(f"Available splits: {list(ds.keys())}")
    
    # Examine each split
    for split_name in ds.keys():
        split_data = ds[split_name]
        print(f"\n=== {split_name.upper()} Split ===")
        print(f"Number of examples: {len(split_data)}")
        print(f"Features: {split_data.features}")
        print(f"Column names: {split_data.column_names}")
        
        # Look at first example
        if len(split_data) > 0:
            first_example = split_data[0]
            print(f"\nFirst example keys: {first_example.keys()}")
            
            # Print sample data for each field
            for key, value in first_example.items():
                print(f"\n{key}:")
                if hasattr(value, '__len__') and len(str(value)) > 200:
                    print(f"  Type: {type(value)}")
                    print(f"  Length/Size: {len(value) if hasattr(value, '__len__') else 'N/A'}")
                    if isinstance(value, str):
                        print(f"  Preview: {str(value)[:200]}...")
                else:
                    print(f"  Value: {value}")
                    print(f"  Type: {type(value)}")

def extract_images(output_dir="extracted_images", max_images=10):
    """Extract images from the dataset and save them locally with original filenames"""
    
    local_path = "/mnt/data/maund/open_med_flamingo/open_flamingo/data/llavamed_dataset"
    ds = load_from_disk(local_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name in ds.keys():
        split_data = ds[split_name]
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        print(f"\n=== Extracting images from {split_name} split ===")
        print(f"Available columns: {split_data.column_names}")
        
        # Look for filename/path columns
        filename_cols = []
        for col in split_data.column_names:
            if any(keyword in col.lower() for keyword in ['filename', 'file_name', 'path', 'name', 'id']):
                filename_cols.append(col)
        
        print(f"Potential filename columns: {filename_cols}")
        
        # Look for image columns
        image_cols = []
        for col in split_data.column_names:
            if 'image' in col.lower() or 'img' in col.lower():
                image_cols.append(col)
        
        print(f"Image columns: {image_cols}")
        
        # Extract images
        for i, example in enumerate(split_data):
            if i >= max_images:
                break
            
            print(f"\n--- Processing example {i} ---")
            
            # Try to find the original filename
            original_filename = None
            for fname_col in filename_cols:
                if fname_col in example:
                    potential_name = example[fname_col]
                    if isinstance(potential_name, str) and potential_name:
                        original_filename = potential_name
                        print(f"  Found filename in '{fname_col}': {original_filename}")
                        break
            
            # If no filename found, create one
            if not original_filename:
                original_filename = f"image_{i}"
                print(f"  No filename found, using: {original_filename}")
            
            # Clean filename (remove invalid characters)
            import re
            clean_filename = re.sub(r'[^\w\-_\.]', '_', original_filename)
            if not clean_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                clean_filename += '.png'
            
            # Extract image from each image column
            for img_col in image_cols:
                if img_col in example:
                    image_data = example[img_col]
                    
                    # Create filename with column prefix if multiple image columns
                    if len(image_cols) > 1:
                        final_filename = f"{img_col}_{clean_filename}"
                    else:
                        final_filename = clean_filename
                    
                    image_path = os.path.join(split_dir, final_filename)
                    
                    try:
                        # Handle different image formats
                        if hasattr(image_data, 'save'):  # PIL Image
                            image_data.save(image_path)
                            print(f"  ✓ Saved PIL image: {image_path}")
                            
                        elif isinstance(image_data, dict):
                            if 'bytes' in image_data:
                                # Image stored as bytes
                                from io import BytesIO
                                image = Image.open(BytesIO(image_data['bytes']))
                                image.save(image_path)
                                print(f"  ✓ Saved bytes image: {image_path}")
                            elif 'path' in image_data:
                                print(f"  Found image path: {image_data['path']}")
                            else:
                                print(f"  Image dict keys: {list(image_data.keys())}")
                                
                        elif isinstance(image_data, str):
                            # Could be base64 or file path
                            if image_data.startswith('data:image') or len(image_data) > 100:
                                print(f"  Possible base64 image: {image_data[:50]}...")
                                # Try to decode base64
                                try:
                                    import base64
                                    from io import BytesIO
                                    if 'base64,' in image_data:
                                        base64_data = image_data.split('base64,')[1]
                                    else:
                                        base64_data = image_data
                                    
                                    img_bytes = base64.b64decode(base64_data)
                                    image = Image.open(BytesIO(img_bytes))
                                    image.save(image_path)
                                    print(f"  ✓ Saved base64 image: {image_path}")
                                except Exception as e:
                                    print(f"  ✗ Failed to decode base64: {e}")
                            else:
                                print(f"  Image path/filename: {image_data}")
                                
                        else:
                            print(f"  Unknown image format: {type(image_data)}")
                            print(f"  Value preview: {str(image_data)[:100]}")
                            
                    except Exception as e:
                        print(f"  ✗ Error processing {img_col}: {e}")

def extract_all_images(output_dir="all_images"):
    """Extract ALL images from the dataset with original filenames"""
    
    local_path = "/mnt/data/maund/open_med_flamingo/open_flamingo/data/llavamed_dataset"
    ds = load_from_disk(local_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name in ds.keys():
        split_data = ds[split_name]
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        print(f"\n=== Extracting ALL images from {split_name} split ===")
        print(f"Total examples: {len(split_data)}")
        
        # Extract all images
        extract_images(split_dir, max_images=len(split_data))

def save_metadata():
    """Save dataset metadata and sample data as JSON"""
    
    local_path = "/mnt/data/maund/open_med_flamingo/open_flamingo/data/llavamed_dataset"
    ds = load_from_disk(local_path)
    
    metadata = {}
    
    for split_name in ds.keys():
        split_data = ds[split_name]
        
        # Get sample without images (to avoid huge JSON)
        sample_data = []
        for i, example in enumerate(split_data):
            if i >= 3:  # Only first 3 examples
                break
                
            sample_example = {}
            for key, value in example.items():
                if 'image' not in key.lower():
                    sample_example[key] = value
                else:
                    sample_example[key] = f"<IMAGE_DATA: {type(value)}>"
            
            sample_data.append(sample_example)
        
        metadata[split_name] = {
            "num_examples": len(split_data),
            "features": str(split_data.features),
            "column_names": split_data.column_names,
            "sample_data": sample_data
        }
    
    # Save metadata
    with open("dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print("Metadata saved to dataset_metadata.json")

if __name__ == "__main__":
    print("=== Examining Dataset ===")
    examine_dataset()
    
    print("\n=== Extracting Sample Images ===")
    extract_images(max_images=5)
    
    print("\n=== Saving Metadata ===")
    save_metadata()
    
    print("\nDone! Check the 'extracted_images' folder and 'dataset_metadata.json' file.")
    print("\nTo extract ALL images, run:")
    print("extract_all_images()")
