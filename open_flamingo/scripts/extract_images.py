#!/usr/bin/env python3
"""
Extract images from the dataset using original filenames
"""

from datasets import load_from_disk
import os
from PIL import Image

def extract_images_with_original_names(output_dir="extracted_images", max_images=None):
    """Extract images from the dataset and save them with original filenames"""
    
    local_path = "/mnt/data/maund/open_med_flamingo/open_flamingo/data/llavamed_dataset"
    ds = load_from_disk(local_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name in ds.keys():
        split_data = ds[split_name]
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        print(f"\n=== Extracting images from {split_name} split ===")
        print(f"Total examples: {len(split_data)}")
        
        # Extract images
        extracted_count = 0
        for i, example in enumerate(split_data):
            if max_images and i >= max_images:
                break
            
            # Get original filename
            original_filename = example.get('Deliverable', f'image_{i}.jpg')
            
            # Clean filename (remove any invalid characters)
            import re
            clean_filename = re.sub(r'[^\w\-_\.]', '_', original_filename)
            
            # Extract regular image
            if 'image' in example:
                image_data = example['image']
                if hasattr(image_data, 'save'):  # PIL Image
                    image_path = os.path.join(split_dir, clean_filename)
                    image_data.save(image_path)
                    extracted_count += 1
                    if i % 100 == 0:  # Progress update every 100 images
                        print(f"  Extracted {extracted_count} images...")
            
            # Extract image with bboxes
            if 'image_with_bboxes' in example:
                image_data = example['image_with_bboxes']
                if hasattr(image_data, 'save'):  # PIL Image
                    # Add suffix for bbox version
                    name_parts = clean_filename.rsplit('.', 1)
                    if len(name_parts) == 2:
                        bbox_filename = f"{name_parts[0]}_with_bboxes.{name_parts[1]}"
                    else:
                        bbox_filename = f"{clean_filename}_with_bboxes.png"
                    
                    bbox_image_path = os.path.join(split_dir, bbox_filename)
                    image_data.save(bbox_image_path)
        
        print(f"  ✓ Extracted {extracted_count} images from {split_name} split")
        print(f"  Saved to: {split_dir}")

def extract_sample_images(num_samples=5):
    """Extract just a few sample images to check"""
    print("Extracting sample images...")
    extract_images_with_original_names("sample_images", max_images=num_samples)

def extract_all_images():
    """Extract ALL images from the dataset"""
    print("Extracting ALL images...")
    extract_images_with_original_names("all_extracted_images")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        extract_all_images()
    else:
        extract_sample_images(10)
    
    print("\nDone!")
    print("To extract all images, run: python extract_images.py all")
