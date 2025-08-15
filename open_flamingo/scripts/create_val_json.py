import json
import random
import os

def split_data_for_validation(input_file, output_folder, validation_ratio=0.01):
    """
    Split JSON data into training and validation sets.
    
    Args:
        input_file (str): Path to the input JSON file
        output_folder (str): Folder where val.json will be saved
        validation_ratio (float): Ratio of data to use for validation (default: 0.1 for 10%)
    """
    
    # Read the original JSON file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{input_file}'.")
        return
    
    # Ensure data is a list
    if not isinstance(data, list):
        print("Error: JSON file should contain a list of objects.")
        return
    
    # Calculate validation set size
    total_items = len(data)
    val_size = max(1, int(total_items * validation_ratio))  # At least 1 item
    
    print(f"Total items: {total_items}")
    print(f"Validation items: {val_size}")
    print(f"Training items: {total_items - val_size}")
    
    # Randomly select validation data
    random.seed(42)  # For reproducible results
    val_data = random.sample(data, val_size)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save validation data
    val_file_path = os.path.join(output_folder, 'llava_med_mri_bbox_val_CoT_new.json')
    try:
        with open(val_file_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        print(f"Validation data saved to: {val_file_path}")
    except Exception as e:
        print(f"Error saving validation data: {e}")
        return
    
    # Optionally, save the remaining training data
    train_data = [item for item in data if item not in val_data]
    train_file_path = os.path.join(output_folder, 'train.json')
    try:
        with open(train_file_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        print(f"Training data saved to: {train_file_path}")
    except Exception as e:
        print(f"Error saving training data: {e}")

# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual file paths
    input_json_file = "/home/mau_nguyen_dinh_caddi_jp/projects/dataset/vlm-project-with-images-with-bbox-images-v4/llava_med_mri_bbox_train_CoT_new.json"  # Path to your input JSON file
    output_directory = "/home/mau_nguyen_dinh_caddi_jp/projects/dataset/vlm-project-with-images-with-bbox-images-v4"  # Folder where val.json will be saved
    
    split_data_for_validation(input_json_file, output_directory)