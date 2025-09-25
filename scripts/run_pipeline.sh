#!/bin/bash

# ==============================================================================
# Pipeline Script for Training, Inference, and Evaluation
# ==============================================================================
# This script orchestrates the training, inference, and evaluation steps for a
# machine learning model.
#
# Usage:
#   ./run_pipeline.sh [options]
#
# Options:
#   -t, --train     Run the training step.
#   -i, --inference Run the inference step.
#   -e, --evaluate  Run the evaluation step.
#   -h, --help      Display this help message and exit.
#
# Example:
#   ./run_pipeline.sh -t -i -e  # Runs all three steps
#   ./run_pipeline.sh -i -e      # Runs only inference and evaluation
#
# ==============================================================================

# ==============================================================================
# --- Configuration Variables (Modify these as needed) ---
# ==============================================================================
# Training Parameters
ENV_NAME="instruct_flamingo" # Name of the conda environment
MIX_PRECISION="bf16" #{no,fp16,bf16,fp8} (str)
TRAIN_BATCH_SIZE=1
TEST_BATCH_SIZE=128
LR=1e-4
EPOCHS=35
NUM_PROCESS=5
LR_SCHEDULER="consine"
CHECKPOINT_EPOCH=14 # The epoch number to use for inference and evaluation
WARMUP_STEPS=10
# RUN_CODE=09223_medrag_cot_500tok for medrag cot 500tok v1
RUN_CODE=0924_medragv2_no_cot_resize_bbox_img336
MODEL_SIZE="3b" # Options: "3b", "7b"
MAX_LENGTH=1024
SEED=42
CUDA_VISIBLE_DEVICES='1,2,3,4,5'
# ------------------------------------------------------------------------------

# Path Configuration
INSTRUCT_FLAMINGO_ROOT="/app/baseline_models/instruct_flamingo"
DATA_PATH="/app/baseline_models/sample_data/llama_mri_cot/instruct_flamingo"
# MODEL_PATH="/app/baseline_models/models/med-flamingo/model.pt" # For 3b model A.K.A --resume_from_checkpoint
MODEL_PATH="/app/baseline_models/models/OpenFlamingo-9B-vitl-mpt7b/checkpoint.pt" # For 9b model A.K.A --resume_from_checkpoint
LM_PATH="anas-awadalla/mpt-7b"
# LM_PATH="/mnt/data/nict/maund/baseline_models/models/Meta-Llama-3-8B-Instruct"
# LM_PATH="mosaicml/mpt-7b-8k"
VISION_ENCODER="ViT-L-14-336"
TUNING_MODE="sft" # Options: "sft", "perceiver", "lora[lm+xqttn]+perceiver.json"
TUNING_CONFIG="${INSTRUCT_FLAMINGO_ROOT}/open_flamingo/instruction_tuning/tuning_config/${TUNING_MODE}.json"
RESULTS_DIR="predictions_validation"

# Derived Variables (Do not edit)
RUN_DIR="${INSTRUCT_FLAMINGO_ROOT}/runs"
RUN_NAME="${RUN_CODE}-clever_flamingo_v2_${MODEL_SIZE}-${TUNING_MODE}-maxlength${MAX_LENGTH}-batch${TRAIN_BATCH_SIZE}-lr${LR}-epochs${EPOCHS}-lrsched${LR_SCHEDULER}-resume-from-mpt7b"
CHECKPOINT_PATH="${RUN_DIR}/${RUN_NAME}/checkpoint_${CHECKPOINT_EPOCH}.pt"
INFERENCE_RESULT_FILE="${INSTRUCT_FLAMINGO_ROOT}/${RESULTS_DIR}/${RUN_NAME}-checkpoint_${CHECKPOINT_EPOCH}/eval_dataset_config_-1/llava-mri-cot-1k-test_all.json"
# ------------------------------------------------------------------------------

# Functions
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -t, --train     Run the training step."
    echo "  -i, --inference Run the inference step."
    echo "  -e, --evaluate  Run the evaluation step."
    echo "  -h, --help      Display this help message and exit."
    exit 1
}

# Parse command-line arguments
train_flag=false
inference_flag=false
evaluation_flag=false

while getopts "tieh" opt; do
    case ${opt} in
        t ) train_flag=true ;;
        i ) inference_flag=true ;;
        e ) evaluation_flag=true ;;
        h ) usage ;;
        * ) usage ;;
    esac
done

# If no options are provided, show usage
if [ "$#" -eq 0 ]; then
    usage
fi

# ==============================================================================
# --- Step 1: Training ---
# ==============================================================================
if [ "$train_flag" = true ]; then
    echo "Starting Training..."
    cd "${INSTRUCT_FLAMINGO_ROOT}" || { echo "Failed to change directory to ${INSTRUCT_FLAMINGO_ROOT}"; exit 1; }
    export HF_DATASETS_CACHE="/app/.cache/"
    export HF_HOME="/app/.cache/"
    echo "activating virtual environment"
    source ~/.bashrc
    eval "$(conda shell.bash hook)"
    conda activate ${ENV_NAME}
    which python

    export PYTHONPATH="$PYTHONPATH:open_flamingo"
    export NCCL_NVLS_ENABLE=0
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    # CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' torchrun --nnodes=1 --nproc_per_node=8 --master_port=29502 open_flamingo/instruction_tuning/train.py \
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --num_processes ${NUM_PROCESS} --mixed-precision ${MIX_PRECISION} open_flamingo/instruction_tuning/train.py \
        --instruction_data "${DATA_PATH}/dataset_config.json" \
        --instruction_prompt_templete 'guanaco-no-prompt' \
        --run_name "${RUN_DIR}/${RUN_NAME}" \
        --seed "${SEED}" \
        --vision_encoder_path "${VISION_ENCODER}" \
        --lm_path "${LM_PATH}" \
        --tokenizer_path "${LM_PATH}" \
        --cross_attn_every_n_layers 1 \
        --freeze_lm_embeddings \
        --tuning_config "${TUNING_CONFIG}" \
        --resume_from_checkpoint "${MODEL_PATH}" \
        --continue_training \
        --max_length "${MAX_LENGTH}" \
        --multiturn_augmentation 0 \
        --max_img 16 \
        --skip_check_overlength \
        --train_num_samples 10000 \
        --epoch_num_samples 50 \
        --batch_size "${TRAIN_BATCH_SIZE}" \
        --learning_rate "${LR}" \
        --gradient_accumulation_steps 1 \
        --precision 'bf16' \
        --workers 8 \
        --num_epochs "${EPOCHS}" \
        --lr_scheduler "${LR_SCHEDULER}" \
        --warmup_steps "${WARMUP_STEPS}" \
        --logging_steps 1
    
    echo "Training step completed."
    echo ""
fi

# ==============================================================================
# --- Step 2: Inference ---
# ==============================================================================
if [ "$inference_flag" = true ]; then
    echo "Starting Inference..."
    cd "${INSTRUCT_FLAMINGO_ROOT}" || { echo "Failed to change directory to ${INSTRUCT_FLAMINGO_ROOT}"; exit 1; }
    export HF_HOME="/app/.cache/"
    export NCCL_NVLS_ENABLE=0
    echo "activating virtual environment"
    source ~/.bashrc
    eval "$(conda shell.bash hook)"
    conda activate ${ENV_NAME}
    which python

    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} torchrun --nnodes=1 --nproc_per_node=${NUM_PROCESS} --master_port=29502 open_flamingo/instruction_tuning/instruction_dataset_inference_llava_cot_distributed.py \
        --lm_path "${LM_PATH}" \
        --vision_encoder_path "${VISION_ENCODER}" \
        --vision_encoder_pretrained "openai" \
        --tuning_config "${TUNING_CONFIG}" \
        --checkpoint_paths "${CHECKPOINT_PATH}" \
        --cross_attn_every_n_layers 4 \
        --instruction_path "${DATA_PATH}/eval_dataset_config.json" \
        --instruction_prompt_templete 'guanaco-no-prompt' \
        --num_samples -1 \
        --max_new_token 16 \
        --no_repeat_ngram_size 3 \
        --num_beams 1 \
        --do_sample False \
        --seed 42 \
        --results_dir "${RESULTS_DIR}"
        
    echo "Inference step completed."
    echo ""
fi


# ==============================================================================
# --- Step 3: Evaluation ---
# ==============================================================================
if [ "$evaluation_flag" = true ]; then
    echo "Starting Evaluation..."
    cd "${INSTRUCT_FLAMINGO_ROOT}" || { echo "Failed to change directory to ${INSTRUCT_FLAMINGO_ROOT}"; exit 1; }
    export HF_HOME="/app/.cache/"
    export NCCL_NVLS_ENABLE=0
    echo "activating virtual environment"
    source ~/.bashrc
    eval "$(conda shell.bash hook)"
    conda activate ${ENV_NAME}
    which python
    
    python open_flamingo/instruction_tuning/evaluation.py \
        --inference_result "${INFERENCE_RESULT_FILE}"
        
    echo "Evaluation step completed."
    echo ""
fi

if ! "$train_flag" && ! "$inference_flag" && ! "$evaluation_flag"; then
    echo "No options selected. Please specify at least one of -t, -i, or -e."
    usage
fi
