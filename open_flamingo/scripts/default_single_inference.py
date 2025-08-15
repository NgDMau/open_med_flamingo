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

# checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
checkpoint_path = "/home/mau_nguyen_dinh_caddi_jp/projects/open_med_flamingo/open_flamingo/MedFlamingo-MRI-CoT-Finetune/checkpoint_1.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)

# Move model to device
model = model.to(device)
model.eval()  # Set to evaluation mode


query_image = Image.open("/home/mau_nguyen_dinh_caddi_jp/projects/dataset/vlm-project-with-images-with-bbox-images-v4/images/test/238.jpg")  # Local image for testing
query_prompt = """<image>What's the potency of this disease?

+ Reasoning:
- Step 1: identify region of disease
- Answer:```json
[{"bbox_2d": [139, 214, 188, 251], "label": "disease area"}]
```
- Step 2: What are the visual signatures of this lesion?
- Answer: No atrophy.
- Step 3: What grade designation would you give to this lesion?
- Answer: MTA = 0

+ Final Answer:"""

# query_prompt = """<image>What's the potency of this disease?

# Analysis:
# - Step 1: Disease regions identified in temporal areas
# - Step 2: No cortical atrophy visible
# - Step 3: Severity grade MTA = 0

# Final diagnosis:"""

vision_x = [
    image_processor(query_image).unsqueeze(0)
]
vision_x = torch.cat(vision_x, dim=0)
vision_x = vision_x.unsqueeze(1).unsqueeze(0).to(device)


tokenizer.padding_side = "left" # For generation padding tokens should be on the left
tokenizer.add_eos_token = False

lang_x = tokenizer(
    [query_prompt],    
    return_tensors="pt",
    padding=True
)

# Move text tensors to device
lang_x = {k: v.to(device) for k, v in lang_x.items()}

# Generation parameters
generation_params = {
    "max_new_tokens": 8,           # Increased from 64
    "num_beams": 3,                  
    "no_repeat_ngram_size": 2,       # KEY: Prevent repetition
    "temperature": 0.8,              # Add randomness
    "do_sample": True,               # Enable sampling  
    "top_k": 50,
    "top_p": 0.9,                    # Nucleus sampling
    # "eos_token_id": 50277,
    # "pad_token_id": tokenizer.pad_token_id,
}


generated_text = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    **generation_params
)

mask_token = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens([1]))
predicted_tokens = tokenizer.convert_ids_to_tokens(generated_text[0].squeeze().tolist())
predicted_text = tokenizer.convert_tokens_to_string(predicted_tokens).replace(mask_token, ' ')
            
decoded_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)



print("Generated text: ", decoded_text)
print("predicted_text: ", predicted_text)

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

# Add this debug code to your inference script:
print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
print(f"End of chunk token ID: {tokenizer.encode('<|endofchunk|>', add_special_tokens=False)}")
print(f"End of text token ID: {tokenizer.encode('<|endoftext|>', add_special_tokens=False)}")
print(f"Results saved to: {filepath}")