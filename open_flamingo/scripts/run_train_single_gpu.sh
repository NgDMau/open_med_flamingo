#!/bin/bash

# Single GPU training script - fallback option if distributed training fails
# This avoids NCCL entirely by using only one GPU

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "=== Single GPU Training ==="
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Run without torchrun for single GPU
python train/train.py \
    --lm_path anas-awadalla/mpt-1b-redpajama-200b \
    --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b \
    --cross_attn_every_n_layers 1 \
    --dataset_type llavamed \
    --batch_size 4 \
    --max_tokens 256 \
    --workers 2 \
    --run_name MedFlamingo-MRI-CoT-SingleGPU \
    --train_json_path "/mnt/data/maund/open_med_flamingo/open_flamingo/data/CoT/llava_med_mri_bbox_train_CoT_new.json" \
    --image_dir "/mnt/data/maund/open_med_flamingo/open_flamingo/data/images" \
    --num_epochs 10 \
    --gradient_checkpointing \
    --learning_rate 1e-5 \
    --warmup_steps 50 \
    --weight_decay 0.05 \
    --gradient_accumulation_steps 8 \
    --precision fp16 \
    --logging_steps 10
