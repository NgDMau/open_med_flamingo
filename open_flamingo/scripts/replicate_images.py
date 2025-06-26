#!/usr/bin/env python3
"""
Script to replicate a sample image with sequential IDs
"""

import os
import shutil
from pathlib import Path

def replicate_image():
    NUM_IMAGES = 20000  # Total images to create (1.jpg to 20000.jpg)
    
    # Paths
    source_image = "/mnt/data/maund/open_med_flamingo/open_flamingo/data/sample_image.jpg"
    target_dir = "/mnt/data/maund/open_med_flamingo/open_flamingo/data/images"
    
    # Check if source image exists
    if not os.path.exists(source_image):
        print(f"❌ Source image not found: {source_image}")
        return False
    
    print(f"📁 Source: {source_image}")
    print(f"📁 Target directory: {target_dir}")
    print(f"🔄 Creating {NUM_IMAGES} copies with IDs 1-{NUM_IMAGES}...")
    
    # Create copies
    success_count = 0
    error_count = 0
    
    for i in range(1, 20001):  # 1 to 10000
        target_path = os.path.join(target_dir, f"{i}.jpg")
        
        try:
            shutil.copy2(source_image, target_path)
            success_count += 1
            
            # Progress update every 1000 images
            if i % 1000 == 0:
                print(f"  ✅ Copied {i} images...")
                
        except Exception as e:
            print(f"  ❌ Error copying {i}.jpg: {e}")
            error_count += 1
    
    print(f"\n🎉 Replication complete!")
    print(f"  ✅ Successfully created: {success_count} images")
    print(f"  ❌ Errors: {error_count}")
    print(f"  📊 Total images in folder: {success_count + 1} (including original)")
    
    # Verify some files exist
    print(f"\n🔍 Verification:")
    test_files = [1, 100, 1000, 5000, 10000]
    for test_id in test_files:
        test_path = os.path.join(target_dir, f"{test_id}.jpg")
        exists = "✅" if os.path.exists(test_path) else "❌"
        print(f"  {exists} {test_id}.jpg")
    
    return success_count == 10000

def check_disk_space():
    """Check if there's enough disk space"""
    source_image = "/mnt/data/maund/open_med_flamingo/open_flamingo/data/sample_image.jpg"
    target_dir = "/mnt/data/maund/open_med_flamingo/open_flamingo/data/images"
    
    if not os.path.exists(source_image):
        print(f"❌ Source image not found: {source_image}")
        return False
    
    # Get file size
    file_size = os.path.getsize(source_image)
    total_size_needed = file_size * 10000  # 10,000 copies
    
    # Get available disk space
    statvfs = os.statvfs(target_dir)
    available_space = statvfs.f_frsize * statvfs.f_bavail
    
    print(f"📊 Disk Space Analysis:")
    print(f"  Source image size: {file_size / (1024*1024):.2f} MB")
    print(f"  Total space needed: {total_size_needed / (1024*1024):.2f} MB")
    print(f"  Available space: {available_space / (1024*1024):.2f} MB")
    
    if total_size_needed > available_space:
        print(f"❌ Not enough disk space!")
        return False
    else:
        print(f"✅ Sufficient disk space available")
        return True

def clean_generated_images():
    """Remove all generated images (1.jpg to 10000.jpg) but keep sample_image.jpg"""
    target_dir = "/mnt/data/maund/open_med_flamingo/open_flamingo/data/images"
    
    print(f"🧹 Cleaning generated images from {target_dir}...")
    
    removed_count = 0
    for i in range(1, 10001):
        target_path = os.path.join(target_dir, f"{i}.jpg")
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
                removed_count += 1
                if i % 1000 == 0:
                    print(f"  🗑️ Removed {removed_count} images...")
            except Exception as e:
                print(f"  ❌ Error removing {i}.jpg: {e}")
    
    print(f"✅ Cleanup complete! Removed {removed_count} images")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "clean":
            clean_generated_images()
            exit(0)
        elif sys.argv[1] == "check":
            check_disk_space()
            exit(0)
    
    print("=== Image Replication Script ===")
    print("This will create 10,000 copies of sample_image.jpg")
    print("Named as: 1.jpg, 2.jpg, 3.jpg, ..., 10000.jpg")
    print()
    
    # Check disk space first
    if not check_disk_space():
        print("❌ Aborting due to insufficient disk space")
        exit(1)
    
    # Ask for confirmation
    response = input("\n🤔 Do you want to proceed? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Operation cancelled")
        exit(0)
    
    # Replicate images
    success = replicate_image()
    
    if success:
        print("\n🎉 All done! You now have 10,001 images in the folder.")
        print("💡 To clean up generated images, run: python replicate_images.py clean")
    else:
        print("\n❌ Replication failed or incomplete")
        print("💡 To clean up any partial files, run: python replicate_images.py clean")
