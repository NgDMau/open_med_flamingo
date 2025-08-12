#!/usr/bin/env python3
"""
Download and load OpenFlamingo pretrained models from Hugging Face Hub
"""

import os
import torch
from huggingface_hub import hf_hub_download
import argparse

def download_openflamingo_checkpoint(model_name="openflamingo/OpenFlamingo-3B-vitl-mpt1b", cache_dir=None):
    """
    Download OpenFlamingo checkpoint from Hugging Face Hub
    
    Args:
        model_name (str): The model name on Hugging Face Hub
        cache_dir (str): Local cache directory (optional)
    
    Returns:
        str: Path to the downloaded checkpoint
    """
    print(f"Downloading checkpoint from {model_name}...")
    
    try:
        checkpoint_path = hf_hub_download(
            repo_id=model_name,
            filename="checkpoint.pt",
            cache_dir=cache_dir
        )
        print(f"Checkpoint downloaded successfully to: {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        print(f"Error downloading checkpoint: {e}")
        return None

def load_checkpoint(checkpoint_path, device="cpu"):
    """
    Load checkpoint from file
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        device (str): Device to load checkpoint on
    
    Returns:
        dict: Loaded checkpoint state dict
    """
    try:
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("Checkpoint loaded successfully!")
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download OpenFlamingo pretrained models")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="openflamingo/OpenFlamingo-3B-vitl-mpt1b",
        help="Model name on Hugging Face Hub"
    )
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default=None,
        help="Local cache directory"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu",
        help="Device to load checkpoint on (cpu/cuda)"
    )
    parser.add_argument(
        "--download_only", 
        action="store_true",
        help="Only download, don't load the checkpoint"
    )
    
    args = parser.parse_args()
    
    # Download checkpoint
    checkpoint_path = download_openflamingo_checkpoint(args.model_name, args.cache_dir)
    
    if checkpoint_path is None:
        print("Failed to download checkpoint")
        return 1
    
    # Load checkpoint if requested
    if not args.download_only:
        checkpoint = load_checkpoint(checkpoint_path, args.device)
        if checkpoint is not None:
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            if 'model_state_dict' in checkpoint:
                print("Found model_state_dict in checkpoint")
            elif 'state_dict' in checkpoint:
                print("Found state_dict in checkpoint")
            else:
                print("Checkpoint structure:", type(checkpoint))
    
    print(f"Checkpoint path: {checkpoint_path}")
    return 0

if __name__ == "__main__":
    exit(main())