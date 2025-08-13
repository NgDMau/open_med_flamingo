from PIL import Image
import requests
import torch
import os
from datetime import datetime
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1
)

checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)

# Move model to device
model = model.to(device)
model.eval()  # Set to evaluation mode

"""
Step 1: Load images
"""
demo_image_one = Image.open(
    requests.get(
        "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
    ).raw
)

demo_image_two = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
        stream=True
    ).raw
)

query_image = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028352.jpg", 
        stream=True
    ).raw
)

query_image = Image.open("/home/mau_nguyen_dinh_caddi_jp/projects/dataset/vlm-project-with-images-with-bbox-images-v4/images/test/238.jpg")  # Local image for testing


"""
Step 2: Preprocessing images
Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
 batch_size x num_media x num_frames x channels x height x width. 
 In this case batch_size = 1, num_media = 3, num_frames = 1,
 channels = 3, height = 224, width = 224.
"""
vision_x = [
    # image_processor(demo_image_one).unsqueeze(0), 
    # image_processor(demo_image_two).unsqueeze(0), 
    image_processor(query_image).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim=0)
vision_x = vision_x.unsqueeze(1).unsqueeze(0).to(device)

"""
Step 3: Preprocessing text
Details: In the text we expect an <image> special token to indicate where an image is.
 We also expect an <|endofchunk|> special token to indicate the end of the text 
 portion associated with an image.
"""
tokenizer.padding_side = "left" # For generation padding tokens should be on the left
lang_x = tokenizer(
    # ["<image>What is it? An image of two cats.<|endofchunk|><image>What is it? An image of a bathroom sink.<|endofchunk|><image>What is it?"],
    ["<image>What is it?"],    
    return_tensors="pt",
)

# Move text tensors to device
lang_x = {k: v.to(device) for k, v in lang_x.items()}


"""
Step 4: Generate text
"""
# Generation parameters
generation_params = {
    "max_new_tokens": 128,           # Default: None
    "num_beams": 3,                 # Default: 1
    "no_repeat_ngram_size": 3,      # Default: 0
    "temperature": 1.0,             # Default: 1.0
    "do_sample": False,             # Default: False
    "top_k": 50,                    # Default: 50
    "top_p": 1.0,                   # Default: 1.0
}

generated_text = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    **generation_params
)

decoded_text = tokenizer.decode(generated_text[0])
print("Generated text: ", decoded_text)

# Save results to file
output_folder = "../../inference_results"  # Change this to your desired folder path
os.makedirs(output_folder, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"generation_result_{timestamp}.txt"
filepath = os.path.join(output_folder, filename)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write("=== OpenFlamingo Generation Results ===\n")
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: OpenFlamingo-3B-vitl-mpt1b\n")
    f.write(f"Device: {device}\n\n")
    
    f.write("=== Input Prompt ===\n")
    f.write(f"{tokenizer.decode(lang_x['input_ids'][0])}\n\n")
    
    f.write("=== Generation Parameters ===\n")
    for param, value in generation_params.items():
        f.write(f"{param}: {value}\n")
    f.write("\n")
    
    f.write("=== Generated Text ===\n")
    f.write(f"{decoded_text}\n\n")
    
    f.write("=== Full Output (with special tokens) ===\n")
    f.write(f"{tokenizer.decode(generated_text[0], skip_special_tokens=False)}\n")

print(f"Results saved to: {filepath}")