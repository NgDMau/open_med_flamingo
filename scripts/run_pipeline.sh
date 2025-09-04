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
BATCH_SIZE=256
LR=1e-4
EPOCHS=20
LR_SCHEDULER="consine"
CHECKPOINT_EPOCH=19 # The epoch number to use for inference and evaluation
WARMUP_STEPS=10
# ------------------------------------------------------------------------------

# Path Configuration
INSTRUCT_FLAMINGO_ROOT="/app/baseline_models/instruct_flamingo"
DATA_PATH="/app/baseline_models/sample_data/llama_mri_cot/instruct_flamingo"
MODEL_PATH="/app/baseline_models/models/med-flamingo/model.pt"
LM_PATH="anas-awadalla/mpt-7b"
VISION_ENCODER="ViT-L-14-336"
TUNING_CONFIG="${INSTRUCT_FLAMINGO_ROOT}/open_flamingo/instruction_tuning/tuning_config/lora[lm+xqttn]+perceiver.json"
RESULTS_DIR="predictions_validation"

# Derived Variables (Do not edit)
RUN_DIR="${INSTRUCT_FLAMINGO_ROOT}/runs"
RUN_NAME="0904-clever_flamingo_v2_3b-batch${BATCH_SIZE}-lr${LR}-epochs${EPOCHS}-lrsched${LR_SCHEDULER}-resume-from-mpt7b"
CHECKPOINT_PATH="${INSTRUCT_FLAMINGO_ROOT}/${RUN_NAME}/checkpoint_${CHECKPOINT_EPOCH}.pt"
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
    conda activate instruct_flamingo
    which python

    export PYTHONPATH="$PYTHONPATH:open_flamingo"
    export NCCL_NVLS_ENABLE=0

    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' torchrun --nnodes=1 --nproc_per_node=8 --master_port=29502 open_flamingo/instruction_tuning/train.py \
        --instruction_data "${DATA_PATH}/dataset_config.json" \
        --instruction_prompt_templete 'guanaco-no-prompt' \
        --run_name "${RUN_DIR}/${RUN_NAME}" \
        --seed 42 \
        --vision_encoder_path "${VISION_ENCODER}" \
        --lm_path "${LM_PATH}" \
        --tokenizer_path "${LM_PATH}" \
        --cross_attn_every_n_layers 1 \
        --freeze_lm_embeddings \
        --tuning_config "${TUNING_CONFIG}" \
        --resume_from_checkpoint "${MODEL_PATH}" \
        --continue_training \
        --max_length 512 \
        --multiturn_augmentation 0 \
        --max_img 16 \
        --skip_check_overlength \
        --train_num_samples 10000 \
        --epoch_num_samples 50 \
        --batch_size "${BATCH_SIZE}" \
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
    conda activate instruct_flamingo
    which python
    
    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' torchrun --nnodes=1 --nproc_per_node=8 --master_port=29502 open_flamingo/instruction_tuning/instruction_dataset_inference_llava_cot_distributed.py \
        --lm_path "${LM_PATH}" \
        --vision_encoder_path "${VISION_ENCODER}" \
        --vision_encoder_pretrained "openai" \
        --tuning_config "${TUNING_CONFIG}" \
        --checkpoint_paths "${CHECKPOINT_PATH}" \
        --cross_attn_every_n_layers 4 \
        --instruction_path "${DATA_PATH}/eval_dataset_config.json" \
        --instruction_prompt_templete 'guanaco-no-prompt' \
        --num_samples -1 \
        --max_new_token 512 \
        --no_repeat_ngram_size 3 \
        --num_beams 1 \
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
    conda activate instruct_flamingo
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
