import json

def count_json_entries(file_path):
    """
    Opens a JSON file containing a list of objects and counts how many objects are in the list.

    Args:
        file_path (str): The full path to the .json file.

    Returns:
        int: The number of entries in the JSON list, or -1 if an error occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Load the entire JSON structure from the file
            data = json.load(f)
            
            # Check if the loaded data is a list
            if isinstance(data, list):
                # Use the len() function to get the number of items
                num_entries = len(data)
                print(f"✅ Successfully loaded the file.")
                print(f"   The list contains {num_entries} JSON objects.")
                return num_entries
            else:
                print("❌ Error: The JSON file does not contain a list at the top level.")
                return -1

    except FileNotFoundError:
        print(f"❌ Error: The file at '{file_path}' was not found.")
        return -1
    except json.JSONDecodeError:
        print(f"❌ Error: The file at '{file_path}' is not a valid JSON file.")
        return -1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return -1

# --- Example Usage ---
# To use this script:
# 1. Save it as a Python file (e.g., `count_json.py`).
# 2. Make sure you have a .json file in the same directory.
#    For example, a file named `my_data.json` could look like this:
#    [
#      {"id": "a1", "value": 100},
#      {"id": "b2", "value": 200},
#      {"id": "c3", "value": 300}
#    ]
# 3. Change the `file_to_check` variable below to the name of your file.

if __name__ == "__main__":
    file_to_check = "./llava_med_mri_bbox_test_CoT_new.json"  # <--- CHANGE THIS TO YOUR FILENAME
    count_json_entries(file_to_check)
    file_to_check = "./llava_med_mri_bbox_train_CoT_new.json"  # <--- CHANGE THIS TO YOUR FILENAME
    count_json_entries(file_to_check)