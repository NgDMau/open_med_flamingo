import os
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
DATASET_ID = "tungvu3196/vlm-project-with-images-with-bbox-images-v4"
OUTPUT_DIR = "images"

# --- Main Script ---
print(f"Loading dataset '{DATASET_ID}'...")
# This downloads the dataset to a local cache if not already present
dataset = load_dataset(DATASET_ID)
print("Dataset loaded successfully.")

def save_images_from_split(dataset_split, split_name, base_output_dir):
    """
    Saves images from a dataset split to a specified directory.
    The filename for each image is derived from the 'No.' field.
    """
    # Define the specific output directory for this split (e.g., 'images/train')
    split_dir = os.path.join(base_output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    print(f"\nProcessing '{split_name}' split. Saving images to '{split_dir}'...")

    # Use tqdm for a progress bar
    for example in tqdm(dataset_split, desc=f"Saving {split_name} images"):
        try:
            # Get the image object (it's a PIL.Image object)
            image = example['image']
            
            # Get the number for the filename and ensure it's an integer
            file_number = int(example['No.'])
            
            # Construct the full output path
            output_path = os.path.join(split_dir, f"{file_number}.jpg")
            
            # Convert to RGB mode before saving as JPEG to prevent potential errors
            # This handles images with alpha channels (RGBA) or other modes
            image.convert("RGB").save(output_path, "JPEG")
            
        except (KeyError, TypeError) as e:
            # Handle cases where 'image' or 'No.' might be missing or invalid
            print(f"Skipping an item due to an error: {e}")
            continue

    print(f"Finished processing '{split_name}' split.")


# Process both the 'train' and 'test' splits
save_images_from_split(dataset['train'], 'train', OUTPUT_DIR)
save_images_from_split(dataset['test'], 'test', OUTPUT_DIR)

print(f"\nExtraction complete. All images are saved in the '{OUTPUT_DIR}' directory.")