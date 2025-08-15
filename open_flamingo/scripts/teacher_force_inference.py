from PIL import Image
import torch
import torch.nn.functional as F
import os
from datetime import datetime
from open_flamingo import create_model_and_transforms

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model_with_checkpoint(checkpoint_path, model_name=""):
    """Load model with specified checkpoint"""
    print(f"Loading {model_name} model...")
    
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
        tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
        cross_attn_every_n_layers=1
    )
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"{model_name} model loaded successfully!")
    return model, image_processor, tokenizer

# Load both models
finetuned_checkpoint = "/home/mau_nguyen_dinh_caddi_jp/projects/open_med_flamingo/open_flamingo/MedFlamingo-MRI-CoT-Finetune/checkpoint_1.pt"
pretrained_checkpoint = "/home/mau_nguyen_dinh_caddi_jp/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-3B-vitl-mpt1b/snapshots/ed3a0c3190b2fc2d1c39630738896d4e73ce1bbc/checkpoint.pt"

# Load finetuned model
finetuned_model, image_processor, tokenizer = load_model_with_checkpoint(
    finetuned_checkpoint, "Finetuned"
)

# Load pretrained model  
pretrained_model, _, _ = load_model_with_checkpoint(
    pretrained_checkpoint, "Pretrained"
)

query_image = Image.open("/home/mau_nguyen_dinh_caddi_jp/projects/dataset/vlm-project-with-images-with-bbox-images-v4/images/test/238.jpg")

# Define both prompt and expected complete response for teacher forcing
query_prompt = """<image>What's the potency of this disease?

+ Reasoning:
- Step 1: identify region of disease
- Answer:```json
[{\"bbox_2d\": [139, 214, 188, 251], \"label\": \"disease area\"}, {\"bbox_2d\": [64, 218, 107, 255], \"label\": \"disease area\"}]
```
- Step 2: What are the visual signatures of this lesion?
- Answer: No atrophy.
- Step 3: What grade designation would you give to this lesion?
- Answer: MTA = 0

+ Final Answer:"""

# Ground truth completion for teacher forcing
ground_truth_completion = " Non-Dementia<|endofchunk|>"

# Full sequence for teacher forcing (prompt + ground truth)
full_sequence = query_prompt + ground_truth_completion

def teacher_force_inference(model, vision_x, full_text, tokenizer, device):
    """
    Perform teacher forcing inference - use ground truth tokens as input
    and see what the model predicts at each position
    """
    
    # Tokenize the full sequence
    tokenizer.padding_side = "left"
    tokenizer.add_eos_token = False
    
    full_tokens = tokenizer(
        [full_text],
        return_tensors="pt",
        padding=True
    )
    
    # Move to device
    input_ids = full_tokens["input_ids"].to(device)
    attention_mask = full_tokens["attention_mask"].to(device)
    
    with torch.no_grad():
        # Forward pass - this is like training but without computing loss
        outputs = model(
            vision_x=vision_x,
            lang_x=input_ids,
            attention_mask=attention_mask,
        )
        
        # Get logits (predictions at each position)
        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
        
        # Get probabilities and predictions
        probs = F.softmax(logits, dim=-1)
        predicted_token_ids = torch.argmax(logits, dim=-1)
        
        return {
            'input_ids': input_ids,
            'logits': logits,
            'probs': probs,
            'predicted_token_ids': predicted_token_ids,
            'attention_mask': attention_mask
        }

def analyze_teacher_forcing_results(results, tokenizer, prompt_text, ground_truth_completion, model_name=""):
    """
    Analyze the teacher forcing results and compare predictions vs ground truth
    """
    input_ids = results['input_ids'][0]  # Remove batch dimension
    predicted_ids = results['predicted_token_ids'][0]
    attention_mask = results['attention_mask'][0]
    
    # Convert to tokens for analysis
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids.tolist())
    
    # Find where the prompt ends and completion begins
    prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)['input_ids']
    prompt_length = len(prompt_tokens)
    
    print(f"==================== {model_name} Teacher Forcing Analysis ====================")
    print(f"Prompt length: {prompt_length} tokens")
    print(f"Total sequence length: {len(input_tokens)} tokens")
    
    print(f"\n==================== {model_name} Ground Truth vs Predictions (Completion Part Only) ====================")
    
    # Analyze only the completion part
    completion_start = prompt_length
    
    correct_predictions = 0
    total_completion_tokens = 0
    
    for i in range(completion_start, len(input_tokens)):
        if attention_mask[i] == 1:  # Only analyze non-padded tokens
            gt_token = input_tokens[i]
            pred_token = predicted_tokens[i-1] if i > 0 else "<START>"  # Predictions are shifted by 1
            
            is_correct = (i > 0 and input_ids[i] == predicted_ids[i-1])
            
            status = "✓" if is_correct else "✗"
            
            print(f"Position {i}: GT='{gt_token}' | Pred='{pred_token}' | {status}")
            
            if i > 0:  # Skip first position (no prediction for first token)
                if is_correct:
                    correct_predictions += 1
                total_completion_tokens += 1
    
    if total_completion_tokens > 0:
        accuracy = correct_predictions / total_completion_tokens
        print(f"\n==================== {model_name} Summary ====================")
        print(f"Completion accuracy: {correct_predictions}/{total_completion_tokens} = {accuracy:.2%}")
    
    # Generate the predicted completion text
    predicted_completion_text = ""
    if total_completion_tokens > 0:
        predicted_completion_ids = predicted_ids[completion_start-1:completion_start-1+total_completion_tokens]
        predicted_completion_text = tokenizer.decode(predicted_completion_ids, skip_special_tokens=True)
        print(f"Ground truth completion: '{ground_truth_completion.strip()}'")
        print(f"Predicted completion: '{predicted_completion_text}'")
    
    return {
        'accuracy': accuracy if total_completion_tokens > 0 else 0,
        'correct_predictions': correct_predictions,
        'total_tokens': total_completion_tokens,
        'predicted_completion': predicted_completion_text,
        'model_name': model_name
    }

def autoregressive_inference(model, vision_x, prompt_text, tokenizer, device, generation_params=None, model_name=""):
    """
    Perform autoregressive inference with the given model and parameters
    """
    
    if generation_params is None:
        generation_params = {
            "max_new_tokens": 16,
            "num_beams": 3,
            "no_repeat_ngram_size": 2,
            "temperature": 0.8,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.9,
        }
    
    print(f"Running {model_name} autoregressive inference...")
    
    # Setup tokenizer for generation
    tokenizer.padding_side = "left"
    tokenizer.add_eos_token = False
    
    # Tokenize input prompt
    lang_x = tokenizer([prompt_text], return_tensors="pt", padding=True)
    lang_x = {k: v.to(device) for k, v in lang_x.items()}
    
    # Generate text
    with torch.no_grad():
        generated_text = model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"],
            **generation_params
        )
    
    # Decode results
    full_output = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    full_output_with_tokens = tokenizer.decode(generated_text[0], skip_special_tokens=False)
    
    # Extract only the completion (remove original prompt)
    prompt_text_no_media = prompt_text.replace("<image>", "").strip()
    completion_only = full_output.replace(prompt_text_no_media, "").strip()

    return {
        'full_output': full_output,
        'full_output_with_tokens': full_output_with_tokens,
        'completion_only': completion_only,
        'generated_token_ids': generated_text[0],
        'input_prompt': prompt_text,
        'generation_params': generation_params,
        'model_name': model_name
    }

# Prepare vision input
vision_x = [image_processor(query_image).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim=0)
vision_x = vision_x.unsqueeze(1).unsqueeze(0).to(device)

# Define generation parameters
generation_params = {
    "max_new_tokens": 16,
    "num_beams": 3,
    "no_repeat_ngram_size": 2,
    "temperature": 0.8,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.9,
}

print("="*100)
print("STARTING COMPREHENSIVE MODEL COMPARISON")
print("="*100)

# 1. Finetuned model - Teacher forcing
print("\n" + "="*80)
print("1. FINETUNED MODEL - TEACHER FORCING")
print("="*80)
finetuned_tf_results = teacher_force_inference(finetuned_model, vision_x, full_sequence, tokenizer, device)
finetuned_tf_analysis = analyze_teacher_forcing_results(
    finetuned_tf_results, tokenizer, query_prompt, ground_truth_completion, "Finetuned"
)

# 2. Finetuned model - Autoregressive
print("\n" + "="*80)
print("2. FINETUNED MODEL - AUTOREGRESSIVE")
print("="*80)
finetuned_ar_results = autoregressive_inference(
    finetuned_model, vision_x, query_prompt, tokenizer, device, generation_params, "Finetuned"
)

# 3. Pretrained model - Teacher forcing  
print("\n" + "="*80)
print("3. PRETRAINED MODEL - TEACHER FORCING")
print("="*80)
pretrained_tf_results = teacher_force_inference(pretrained_model, vision_x, full_sequence, tokenizer, device)
pretrained_tf_analysis = analyze_teacher_forcing_results(
    pretrained_tf_results, tokenizer, query_prompt, ground_truth_completion, "Pretrained"
)

# 4. Pretrained model - Autoregressive
print("\n" + "="*80)
print("4. PRETRAINED MODEL - AUTOREGRESSIVE")
print("="*80)
pretrained_ar_results = autoregressive_inference(
    pretrained_model, vision_x, query_prompt, tokenizer, device, generation_params, "Pretrained"
)

# Comprehensive comparison
print("\n" + "="*100)
print("COMPREHENSIVE COMPARISON SUMMARY")
print("="*100)

print(f"Ground Truth: '{ground_truth_completion.strip()}'")
print("\n--- Teacher Forcing Results ---")
print(f"Finetuned TF:  '{finetuned_tf_analysis['predicted_completion']}' (Accuracy: {finetuned_tf_analysis['accuracy']:.2%})")
print(f"Pretrained TF: '{pretrained_tf_analysis['predicted_completion']}' (Accuracy: {pretrained_tf_analysis['accuracy']:.2%})")

print("\n--- Autoregressive Results ---")
print(f"Finetuned AR:  '{finetuned_ar_results['completion_only']}'")
print(f"Pretrained AR: '{pretrained_ar_results['completion_only']}'")

print("\n--- Teacher Forcing Accuracy Comparison ---")
print(f"Finetuned:  {finetuned_tf_analysis['accuracy']:.2%} ({finetuned_tf_analysis['correct_predictions']}/{finetuned_tf_analysis['total_tokens']})")
print(f"Pretrained: {pretrained_tf_analysis['accuracy']:.2%} ({pretrained_tf_analysis['correct_predictions']}/{pretrained_tf_analysis['total_tokens']})")

improvement = finetuned_tf_analysis['accuracy'] - pretrained_tf_analysis['accuracy']
print(f"Finetuning improvement: {improvement:.2%}")

# Save comprehensive results
output_folder = "../../inference_results"
os.makedirs(output_folder, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"comprehensive_comparison_{timestamp}.txt"
filepath = os.path.join(output_folder, filename)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write("="*100 + "\n")
    f.write("COMPREHENSIVE MODEL COMPARISON: PRETRAINED vs FINETUNED × TEACHER FORCING vs AUTOREGRESSIVE\n")
    f.write("="*100 + "\n")
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("="*50 + " INPUT PROMPT " + "="*50 + "\n")
    f.write(f"{query_prompt}\n\n")
    
    f.write("="*50 + " GROUND TRUTH " + "="*50 + "\n")
    f.write(f"{ground_truth_completion}\n\n")
    
    f.write("="*50 + " TEACHER FORCING RESULTS " + "="*50 + "\n")
    f.write(f"$$$ Finetuned Model:\n")
    f.write(f"======>Predicted completion:\n'{finetuned_tf_analysis['predicted_completion']}'\n")
    f.write(f"  Accuracy: {finetuned_tf_analysis['accuracy']:.2%} ({finetuned_tf_analysis['correct_predictions']}/{finetuned_tf_analysis['total_tokens']})\n\n")
    f.write("="*50 + "="*50 + "\n")
    f.write(f"$$$ Pretrained Model:\n")
    f.write(f"======> Predicted completion:\n '{pretrained_tf_analysis['predicted_completion']}'\n")
    f.write(f"  Accuracy: {pretrained_tf_analysis['accuracy']:.2%} ({pretrained_tf_analysis['correct_predictions']}/{pretrained_tf_analysis['total_tokens']})\n\n")
    
    f.write(f"Teacher Forcing Improvement: {improvement:.2%}\n\n")
    
    f.write("="*50 + " AUTOREGRESSIVE RESULTS " + "="*50 + "\n")
    f.write(f"$$$ Finetuned Model:\n")
    f.write(f"======>  Generated completion:\n '{finetuned_ar_results['completion_only']}'\n")
    f.write("="*20+ "\n")
    f.write(f"======>  Full output:\n {finetuned_ar_results['full_output']}\n\n")
    f.write("="*50 + "="*50 + "\n\n")
    f.write(f"$$$ Pretrained Model:\n")
    f.write(f"======> Generated completion:\n '{pretrained_ar_results['completion_only']}'\n")
    f.write("="*20+ "\n")
    f.write(f"======> Full output:\n {pretrained_ar_results['full_output']}\n\n")
    
    f.write("="*50 + " GENERATION PARAMETERS " + "="*50 + "\n")
    for param, value in generation_params.items():
        f.write(f"{param}: {value}\n")
    f.write("\n")
    
    f.write("="*50 + " DETAILED OUTPUTS " + "="*50 + "\n")
    f.write("Finetuned Autoregressive (with tokens):\n")
    f.write(f"{finetuned_ar_results['full_output_with_tokens']}\n\n")
    
    f.write("Pretrained Autoregressive (with tokens):\n")
    f.write(f"{pretrained_ar_results['full_output_with_tokens']}\n\n")

print(f"\nComprehensive results saved to: {filepath}")

# Debug token information
print(f"\nEOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
print(f"End of chunk token ID: {tokenizer.encode('<|endofchunk|>', add_special_tokens=False)}")
print(f"End of text token ID: {tokenizer.encode('<|endoftext|>', add_special_tokens=False)}")