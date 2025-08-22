cd /cpfs/user/chendelong/open_flamingo_v2
export HF_DATASETS_CACHE="/cpfs/user/chendelong/.cache/"
export TRANSFORMERS_CACHE="/cpfs/user/chendelong/.cache/"
echo 'activating virtual environment'
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate polite_flamingo
which python

export PYTHONPATH="$PYTHONPATH:open_flamingo"
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' torchrun --nnodes=1 --nproc_per_node=8 --master_port=29502 open_flamingo/instruction_tuning/train.py \
    --instruction_data '/app/baseline_models/sample_data/llama_mri_cot/instruct_flamingo/dataset_config.json' \
    --instruction_prompt_templete 'guanaco-no-prompt' \
    --run_name 'runs/0713-clever_flamingo_v2_9b-2k_context-80G' \
    --seed 42 \
    --vision_encoder_path 'ViT-L-14-336' \
    --lm_path 'anas-awadalla/mpt-7b' \
    --tokenizer_path 'anas-awadalla/mpt-7b' \
    --cross_attn_every_n_layers 4 \
    --freeze_lm_embeddings \
    --tuning_config '/app/baseline_models/instruct_flamingo/open_flamingo/instruction_tuning/tuning_config/lora[lm+xqttn]+perceiver.json' \
    --resume_from_checkpoint '/cpfs/user/chendelong/cache/OpenFlamingo-9B-vitl-mpt7b.pt' \
    --max_length 2048 \
    --multiturn_augmentation 0 \
    --max_img 16 \
    --skip_check_overlength \
    --train_num_samples 10000 \
    --epoch_num_samples 50 \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4 \
    --precision 'bf16' \
    --workers 32 \
    --num_epochs 20 \
    --lr_scheduler constant \
    --warmup_steps 1000 \
    --logging_steps 500


