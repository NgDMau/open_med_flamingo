#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1

# Conservative training script with better error handling
# Use this if the main script fails with NCCL errors

# Enable debugging and conservative settings
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export NCCL_RETRY_COUNT=5

# Disable potentially problematic NCCL features
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4

# Memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

# Check system before starting
echo "=== System Check ==="
nvidia-smi
echo "Available GPUs: $(nvidia-smi --list-gpus | wc -l)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run diagnostic script first
echo "=== Running diagnostics ==="
python debug_cuda.py

echo "=== Starting training ==="

# Use fewer GPUs and smaller batch size for stability
torchrun --nnodes=1 --nproc_per_node=2 train/train.py \
    --lm_path anas-awadalla/mpt-1b-redpajama-200b \
    --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b \
    --cross_attn_every_n_layers 1 \
    --dataset_type llavamed \
    --batch_size 8 \
    --max_tokens 256 \
    --workers 2 \
    --run_name MedFlamingo-MRI-CoT-Conservative \
    --train_json_path "/mnt/data/maund/open_med_flamingo/open_flamingo/data/CoT/llava_med_mri_bbox_train_CoT_new.json" \
    --image_dir "/mnt/data/maund/open_med_flamingo/open_flamingo/data/images" \
    --num_epochs 10 \
    --gradient_checkpointing \
    --learning_rate 1e-5 \
    --warmup_steps 50 \
    --weight_decay 0.05 \
    --gradient_accumulation_steps 2 \
    --precision fp16 \
    --logging_steps 10
