cd /app/baseline_models/instruct_flamingo
export HF_DATASETS_CACHE="/app/.cache/"
export HF_HOME="/app/.cache/"
echo 'activating virtual environment'
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate instruct_flamingo
which python

export PYTHONPATH="$PYTHONPATH:open_flamingo"
# export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' torchrun --nnodes=1 --nproc_per_node=8 --master_port=29502 open_flamingo/instruction_tuning/train.py \
    --instruction_data '/app/baseline_models/sample_data/llama_mri_cot/instruct_flamingo/dataset_config.json' \
    --instruction_prompt_templete 'guanaco-no-prompt' \
    --run_name 'runs/0821-clever_flamingo_v2_3b-2k_context-40G-resume-from-mpt7b-checkpoint-03' \
    --seed 42 \
    --vision_encoder_path 'ViT-L-14-336' \
    --lm_path 'anas-awadalla/mpt-7b' \
    --tokenizer_path 'anas-awadalla/mpt-7b' \
    --cross_attn_every_n_layers 1 \
    --freeze_lm_embeddings \
    --tuning_config '/app/baseline_models/instruct_flamingo/open_flamingo/instruction_tuning/tuning_config/lora[lm+xqttn]+perceiver.json' \
    --resume_from_checkpoint '/app/baseline_models/models/med-flamingo/model.pt' \
    --continue_training \
    --max_length 512 \
    --multiturn_augmentation 0 \
    --max_img 16 \
    --skip_check_overlength \
    --train_num_samples 10000 \
    --epoch_num_samples 50 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4 \
    --precision 'bf16' \
    --workers 8 \
    --num_epochs 20 \
    --lr_scheduler constant \
    --warmup_steps 10 \
    --logging_steps 1 \

