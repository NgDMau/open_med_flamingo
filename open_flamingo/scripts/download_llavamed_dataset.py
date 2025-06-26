from datasets import load_dataset

# Load the dataset
ds = load_dataset("tungvu3196/vlm-project-with-images-with-bbox-images-v4")

# Save to local path
local_path = "/mnt/data/maund/open_med_flamingo/open_flamingo/data/llavamed_dataset"
ds.save_to_disk(local_path)

# Later, you can load it from local path
# ds = load_from_disk(local_path)