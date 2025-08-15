#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1

# Enable CUDA error debugging
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export PYTHONFAULTHANDLER=1

# Set NCCL timeout and retry settings
export NCCL_TIMEOUT=600
export NCCL_RETRY_COUNT=3

# Optimize NCCL performance
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

# Memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Set SLURM environment variables if using SLURM
# export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_PORT=15000
# export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

# export PYTHONPATH="$PYTHONPATH:open_flamingo"

# Check GPU availability before starting
echo "Checking GPU availability..."
# nvidia-smi

# train_num_samples is at max 10783

# Reduce number of processes initially to test
torchrun --nnodes=1 --nproc_per_node=1 train/train.py \
    --lm_path anas-awadalla/mpt-1b-redpajama-200b \
    --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b \
    --cross_attn_every_n_layers 1 \
    --dataset_name llavamed \
    --batch_size 2 \
    --val_batch_size 8 \
    --max_tokens 256 \
    --workers 1 \
    --run_name MedFlamingo-MRI-CoT-Finetune \
    --resume_from_checkpoint "/home/mau_nguyen_dinh_caddi_jp/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-3B-vitl-mpt1b/snapshots/ed3a0c3190b2fc2d1c39630738896d4e73ce1bbc/checkpoint.pt" \
    --train_num_samples 2 \
    --train_json_path "/home/mau_nguyen_dinh_caddi_jp/projects/dataset/vlm-project-with-images-with-bbox-images-v4/llava_med_mri_bbox_train_CoT_new.json" \
    --val_json_path "/home/mau_nguyen_dinh_caddi_jp/projects/dataset/vlm-project-with-images-with-bbox-images-v4/llava_med_mri_bbox_val_CoT_new.json" \
    --image_dir "/home/mau_nguyen_dinh_caddi_jp/projects/dataset/vlm-project-with-images-with-bbox-images-v4/images/train" \
    --num_epochs 3 \
    --gradient_checkpointing \
    --learning_rate 5e-6 \
    --warmup_steps 100 \
    --weight_decay 0.05 \
    --gradient_accumulation_steps 1
