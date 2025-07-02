from PIL import Image
import requests
import torch
import numpy as np

import os

from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from open_flamingo import create_model_and_transforms

from huggingface_hub import hf_hub_download
import torch

    # --vision_encoder_path ViT-L-14 \
    # --vision_encoder_pretrained openai\
    # --lm_path anas-awadalla/mpt-1b-redpajama-200b \
    # --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b \
    # --cross_attn_every_n_layers 1 \
    # --checkpoint_path "/mnt/data/maund/open_med_flamingo/open_flamingo/MedFlamingo-MRI-CoT/checkpoint_9.pt" \
    # --results_file "results.json" \
    # --precision amp_bf16 \
    # --batch_size 8 \
    # --shots 0 \
    # --eval_llavamed \
    # --llavamed_image_dir_path "/mnt/data/maund/open_med_flamingo/open_flamingo/scripts/all_extracted_images/test" \
    # --llavamed_test_json_path "/mnt/data/maund/open_med_flamingo/open_flamingo/data/CoT/llava_med_mri_bbox_test_CoT_new.json" \

# model, image_processor, tokenizer = create_model_and_transforms(
#         "ViT-L-14", # args.vision_encoder_path,
#         "openai", # args.vision_encoder_pretrained,
#         "anas-awadalla/mpt-1b-redpajama-200b", # args.lm_path,
#         "anas-awadalla/mpt-1b-redpajama-200b", # args.tokenizer_path if args.tokenizer_path else args.lm_path,
#         cross_attn_every_n_layers=1 ,#args.cross_attn_every_n_layers,
#         use_local_files= True,#args.offline,
#         gradient_checkpointing=False,#args.gradient_checkpointing,
#         freeze_lm_embeddings=True,#args.freeze_lm_embeddings,
#     )

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1
)

checkpoint_path = "/mnt/data/maund/open_med_flamingo/open_flamingo/MedFlamingo-MRI-CoT-10-epochs/checkpoint_9.pt"

checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    print("Checkpoint loaded successfully")
else:
    print(f"Warning: Checkpoint not found at {checkpoint_path}")

# Set device and move model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Model loaded on {device}")

"""
Step 1: Load images
"""

sample_data_entry = {
        "image": "844.jpg",
        "id": "844",
        "conversations": [
            {
                "from": "human",
                "value": "What's the potency of this disease?\n<image>"
            },
            {
                "from": "gpt",
                "value": "\n\n+ Reasoning:\n- Step 1: Determine the affected region\n- Answer: ```json\n[{\"bbox_2d\": [134, 10, 245, 66], \"label\": \"disease area\"}, {\"bbox_2d\": [91, 132, 200, 180], \"label\": \"disease area\"}, {\"bbox_2d\": [157, 170, 264, 217], \"label\": \"disease area\"}]\n```\n- Step 2: How would you identify the visual hallmarks of this lesion?\n- Answer: Moderate atrophy volume loss of gyri. Substantial widening of parietal sulci. Central atrophy, enlarged lateral ventricular body width.\n- Step 3: How would you characterize the grade of this lesion?\n- Answer: GCA = 2\n\n+ Final Answer: Moderate-Dementia"
            }
        ],
        "conversations_long": [
            {
                "from": "human",
                "value": "What's the potency of this disease?\n<image>"
            },
            {
                "from": "gpt",
                "value": "\n\n+ Reasoning:\n- Step 1: Determine the affected region\n- Answer: ```json\n[{\"bbox_2d\": [134, 10, 245, 66], \"label\": \"disease area\"}, {\"bbox_2d\": [91, 132, 200, 180], \"label\": \"disease area\"}, {\"bbox_2d\": [157, 170, 264, 217], \"label\": \"disease area\"}]\n```\n- Step 2: How would you identify the visual hallmarks of this lesion?\n- Answer: Moderate atrophy volume loss of gyri. Substantial widening of parietal sulci. Central atrophy, enlarged lateral ventricular body width.\n- Step 3: How would you characterize the grade of this lesion?\n- Answer: GCA = 2\n\n+ Final Answer: Moderate-Dementia"
            }
        ]
    }

image_test_dir = "/mnt/data/maund/open_med_flamingo/open_flamingo/scripts/all_extracted_images/test"

demo_image_fn = sample_data_entry["image"]
demo_image_path = f"{image_test_dir}/{demo_image_fn}" 
demo_image_one = Image.open(demo_image_path)

# demo_image_two = Image.open(
#     requests.get(
#         "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
#         stream=True
#     ).raw
# )

# query_image = Image.open(
#     requests.get(
#         "http://images.cocodataset.org/test-stuff2017/000000028352.jpg", 
#         stream=True
#     ).raw
# )


"""
Step 2: Preprocessing images
Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
 batch_size x num_media x num_frames x channels x height x width. 
 In this case batch_size = 1, num_media = 3, num_frames = 1,
 channels = 3, height = 224, width = 224.
"""
# vision_x = [image_processor(demo_image_one).unsqueeze(0), image_processor(demo_image_two).unsqueeze(0), image_processor(query_image).unsqueeze(0)]
vision_x = [image_processor(demo_image_one).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim=0)
vision_x = vision_x.unsqueeze(1).unsqueeze(0)
vision_x = vision_x.to(device)
print(f"Vision tensor shape: {vision_x.shape}")

"""
Step 3: Preprocessing text
Details: In the text we expect an <image> special token to indicate where an image is.
 We also expect an <|endofchunk|> special token to indicate the end of the text 
 portion associated with an image.
"""
tokenizer.padding_side = "left" # For generation padding tokens should be on the left
lang_x = tokenizer(
    ["What's the potency of this disease?\n<image>\n\n+ Reasoning:\n- Step 1: Determine the affected region\n- Answer: ```json\n[{\"bbox_2d\": [134, 10, 245, 66], \"label\": \"disease area\"}, {\"bbox_2d\": [91, 132, 200, 180], \"label\": \"disease area\"}, {\"bbox_2d\": [157, 170, 264, 217], \"label\": \"disease area\"}]\n```\n- Step 2: How would you identify the visual hallmarks of this lesion?\n- Answer:"],
    return_tensors="pt",
)


# Move language tensors to device
lang_x = {k: v.to(device) for k, v in lang_x.items()}
print(f"Language input shape: {lang_x['input_ids'].shape}")

"""
Step 4: Generate text
"""
generated_text = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=1024,
    num_beams=3,
)

print("Generated text: ", tokenizer.decode(generated_text[0]))