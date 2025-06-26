import pandas as pd
import numpy as np
import os
import os.path as osp
import json
import cv2
from tqdm import tqdm
from typing import Optional


def load_img(str_bytes: str):
    """
    Load image from bytes string.
    From review_mri.py
    """
    img_bytes = bytes(str_bytes['bytes'])
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)  # BGR
    return img

from datasets import load_dataset
def load_data_hgf():
    dataset = load_dataset("tungvu3196/vlm-project-with-images-with-bbox-images-v4")
    df_train = dataset["train"].to_pandas()
    df_test = dataset["test"].to_pandas()
    
    return df_train, df_test


def save_image(image, output_path):
    """
    Save an image to the specified path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    return output_path

def convert_dataset_to_sharegpt_format(
    df: pd.DataFrame,
    output_path: str,
    format_type: str = "sharegpt",
    image_dir: str = 'data/images',
    max_samples: Optional[int] = None,
    multi_turn: bool = True,
    cot_format: bool = False  # New parameter for CoT format
) -> None:
    """
    Convert a dataframe to the LLaMA-Factory format with both CoT conversation
    and bounding box annotations.
    
    Args:
        df: Pandas DataFrame with the dataset
        output_path: Where to save the converted dataset
        format_type: Either "alpaca" or "sharegpt"
        image_dir: Directory where images are stored
        max_samples: Maximum number of samples to convert (None for all)
        multi_turn: Whether to include multi-turn conversation (ignored if cot_format=True)
        cot_format: Whether to use Chain of Thought format
    """
    if format_type not in ["alpaca", "sharegpt"]:
        raise ValueError("format_type must be either 'alpaca' or 'sharegpt'")
    
    if max_samples:
        df = df.head(max_samples)
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    samples = []
    
    # Process each row in the dataframe
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting dataset"):
        try:
            img = load_img(row['image'])
            
            # Generate a unique filename using the No. field
            filename = f"{row['No.']}.jpg"
            image_path = os.path.join(image_dir, filename)
            save_image(img, image_path)
            
            # Process bounding boxes from A1
            boxes = eval(row['A1'])  # Convert string representation of list to actual list
            if len(boxes) > 0 and isinstance(boxes[0], list) and len(boxes[0]) == 1:
                boxes = boxes[0]

            bbox_annotations = []
            
            # Convert normalized coordinates to absolute coordinates
            H, W = img.shape[:2]  # Get image dimensions
            for box in boxes:
                x1, y1, x2, y2 = box
                # Convert normalized coordinates to absolute coordinates
                x1 = int(x1 * W)
                y1 = int(y1 * H)
                x2 = int(x2 * W)
                y2 = int(y2 * H)
                
                # Create bbox annotation in the required format
                bbox_annotation = {
                    "bbox_2d": [x1, y1, x2, y2],
                    "label": "disease area",  # Using Q1 as the label for the bounding box
                }
                bbox_annotations.append(bbox_annotation)
            
            # Format the bbox annotations as JSON string
            bbox_json = json.dumps(bbox_annotations)
            bbox_response = f"```json\n{bbox_json}\n```"
            
            if cot_format:
                # Create Chain of Thought format
                reasoning_parts = []
                
                # Step 1: Q1-A1 (bounding box detection)
                reasoning_parts.append(f"- Step 1: {row['Q1']}")
                reasoning_parts.append(f"- Answer: {bbox_response}")
                
                # Step 2: Q2-A2 (if available)
                if row['A2'] and str(row['A2']).strip():
                    reasoning_parts.append(f"- Step 2: {row['Q2']}")
                    reasoning_parts.append(f"- Answer: {row['A2']}")
                
                # Step 3: Q3-A3 (if available)
                if row['A3'] and str(row['A3']).strip():
                    reasoning_parts.append(f"- Step 3: {row['Q3']}")
                    reasoning_parts.append(f"- Answer: {row['A3']}")
                
                # Combine reasoning and final answer
                reasoning_text = "\n".join(reasoning_parts)
                cot_response = f"\n\n###Reasoning:\n{reasoning_text}\n\n###Final Answer: {row['Status']}"
                
                # Create the sample with Q4 as main question
                sample = {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{row['Q4']}\n<image>"
                        },
                        {
                            "role": "assistant",
                            "content": cot_response
                        }
                    ],
                    "images": [image_path]
                }
                
            elif multi_turn:
                # Original multi-turn format
                messages = [
                    {
                        "role": "user",
                        "content": f"<image>{row['Q1']}"
                    },
                    {
                        "role": "assistant",
                        "content": f"{bbox_response}"
                    }
                ]
                
                # Check if A2 is not empty before adding Q2/A2
                if row['A2'] and str(row['A2']).strip():
                    messages.extend([
                        {
                            "role": "user",
                            "content": f"{row['Q2']}"
                        },
                        {
                            "role": "assistant",
                            "content": f"{row['A2']}"
                        }
                    ])
                
                # Check if A3 is not empty before adding Q3/A3
                if row['A3'] and str(row['A3']).strip():
                    messages.extend([
                        {
                            "role": "user",
                            "content": f"{row['Q3']}"
                        },
                        {
                            "role": "assistant",
                            "content": f"{row['A3']}"
                        }
                    ])
                
                # Check if A4 is not empty before adding Q4/A4
                if row['A4'] and str(row['A4']).strip():
                    messages.extend([
                        {
                            "role": "user",
                            "content": f"{row['Q4']}"
                        },
                        {
                            "role": "assistant",
                            "content": f"{row['Status']}"
                        }
                    ])
                
                # Create the sample
                sample = {
                    "messages": messages,
                    "images": [image_path]
                }
            else:
                # Just create the classification sample
                sample = {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"<image>{row['Q4']}"
                        },
                        {
                            "role": "assistant",
                            "content": f"{row['Status']}"#classification response
                        }
                    ],
                    "images": [image_path]
                }
            
            samples.append(sample)
            
        except (SyntaxError, ValueError, KeyError) as e:
            print(f"Error processing row {idx} (No. {row.get('No.', 'unknown')}), value: {row['A1']}: {e}")
            continue
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    # Create dataset_info.json entry
    dataset_name = os.path.splitext(os.path.basename(output_path))[0]
    
    dataset_info = {
        dataset_name: {
            "file_name": os.path.basename(output_path),
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "images"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
            }
        },
    }
    
    print(f"Converted {len(samples)} samples to {output_path}")
    print("\nAdd this to your dataset_info.json:")
    print(json.dumps(dataset_info, indent=2))
    
    # Also save the dataset_info as a separate file for reference
    info_path = os.path.join(os.path.dirname(output_path), f"{dataset_name}_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset info saved to {info_path}")

def main():
    # Configuration
    output_dir = "LLaMA-Factory/data"
    data_dir = "/netscratch/duynguyen/Research/phatnt/VLM-R1/LLaMA-Factory/data"
    image_dir = osp.join(data_dir, "data/images")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    # Load data using the function from review_mri.py
    # df_train, df_test = load_data()
    # print(f"Loaded {len(df_train)} training samples and {len(df_test)} test samples")
    
    df_train, df_test = load_data_hgf()
    print(f"Loaded {len(df_train)} training samples and {len(df_test)} test samples from HGF")

    # Convert training data with CoT format
    print("\nConverting training data with CoT format...")
    convert_dataset_to_sharegpt_format(
        df=df_train,
        output_path=os.path.join(output_dir, "mri_bbox_train_cot.json"),
        format_type="sharegpt",
        image_dir=image_dir,
        max_samples=None,  # Set to an integer if you want to limit the number of samples
        multi_turn=False,  # This will be ignored when cot_format=True
        cot_format=True  # Enable CoT format
    )
    
    # Convert test data with CoT format
    print("\nConverting test data with CoT format...")
    convert_dataset_to_sharegpt_format(
        df=df_test,
        output_path=os.path.join(output_dir, "mri_bbox_test_cot.json"),
        format_type="sharegpt",
        image_dir=image_dir,
        max_samples=None,  # Set to an integer if you want to limit the number of samples
        multi_turn=False,  # This will be ignored when cot_format=True
        cot_format=True  # Enable CoT format
    )
    
    print("Conversion complete!")

if __name__ == "__main__":
    main()