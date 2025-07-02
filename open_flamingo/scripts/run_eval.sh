#!/bin/bash
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(shuf -i 0-65535 -n 1)
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

export PYTHONPATH="$PYTHONPATH:open_flamingo"

echo go $COUNT_NODE
echo $HOSTNAMES

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi

torchrun --nnodes=1 --nproc_per_node=2 open_flamingo/eval/evaluate.py \
    --vision_encoder_path ViT-L-14 \
    --vision_encoder_pretrained openai\
    --lm_path anas-awadalla/mpt-1b-redpajama-200b \
    --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b \
    --cross_attn_every_n_layers 1 \
    --checkpoint_path "/mnt/data/maund/open_med_flamingo/open_flamingo/MedFlamingo-MRI-CoT/checkpoint_9.pt" \
    --results_file "results.json" \
    --precision amp_bf16 \
    --batch_size 8 \
    --shots 0 \
    --eval_llavamed \
    --llavamed_image_dir_path "/mnt/data/maund/open_med_flamingo/open_flamingo/scripts/all_extracted_images/test" \
    --llavamed_test_json_path "/mnt/data/maund/open_med_flamingo/open_flamingo/data/CoT/llava_med_mri_bbox_test_CoT_new.json" \