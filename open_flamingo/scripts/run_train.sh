#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1

# Enable CUDA error debugging
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export PYTHONFAULTHANDLER=1

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
nvidia-smi

# Reduce number of processes initially to test
torchrun --nnodes=1 --nproc_per_node=4 train/train.py \
    --lm_path anas-awadalla/mpt-1b-redpajama-200b \
    --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b \
    --cross_attn_every_n_layers 1 \
    --dataset_type llavamed \
    --batch_size 32 \
    --max_tokens 256 \
    --workers 4 \
    --run_name MedFlamingo-MRI-CoT \
    --train_num_samples 100 \
    --train_json_path "/mnt/data/maund/open_med_flamingo/open_flamingo/data/CoT/llava_med_mri_bbox_train_CoT_new.json" \
    --image_dir "/mnt/data/maund/open_med_flamingo/open_flamingo/data/images" \
    --num_epochs 2 \
    --gradient_checkpointing \
    --learning_rate 1e-5 \
    --warmup_steps 100 \
    --weight_decay 0.05 \
    --gradient_accumulation_steps 1 


# torchrun --nnodes=1 --nproc_per_node=8 open_flamingo/open_flamingo/train/train.py \
#     --lm_path anas-awadalla/mpt-1b-redpajama-200b \
#     --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b \
#     --cross_attn_every_n_layers 1 \
#     --dataset_resampled \
#     --batch_size_mmc4 32 \
#     --batch_size_laion 64 \
#     --train_num_samples_mmc4 125000\
#     --train_num_samples_laion 250000 \
#     --loss_multiplier_laion 0.2 \
#     --workers=4 \
#     --run_name OpenFlamingo-3B-vitl-mpt1b \
#     --num_epochs 480 \
#     --warmup_steps  1875 \
#     --mmc4_textsim_threshold 0.24 \
#     --laion_shards "/path/to/shards/shard-{0000..0999}.tar" \
#     --mmc4_shards "/path/to/shards/shard-{0000..0999}.tar" \
#     --gradient_checkpointing \
# --report_to_wandb \
