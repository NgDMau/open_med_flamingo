export HF_HOME="/app/.cache/"

echo 'activating virtual environment'
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate instruct_flamingo
which python

# --checkpoint_paths and --instruction_path need to be modified

CUDA_VISIBLE_DEVICES='1' python open_flamingo/instruction_tuning/instruction_dataset_inference_llava_cot.py \
    --lm_path "anas-awadalla/mpt-7b" \
    --vision_encoder_path "ViT-L-14-336" \
    --vision_encoder_pretrained "openai" \
    --tuning_config '/app/baseline_models/instruct_flamingo/open_flamingo/instruction_tuning/tuning_config/lora[lm+xqttn]+perceiver.json' \
    --checkpoint_paths '/app/baseline_models/instruct_flamingo/runs/0821-clever_flamingo_v2_3b-2k_context-40G-resume-from-mpt7b-checkpoint-03/checkpoint_15.pt'  \
    --cross_attn_every_n_layers 4 \
    --instruction_path '/app/baseline_models/sample_data/llama_mri_cot/instruct_flamingo/eval_dataset_config.json' \
    --instruction_prompt_templete 'guanaco-no-prompt' \
    --num_samples -1 \
    --max_new_token 512 \
    --no_repeat_ngram_size 3 \
    --num_beams 1 \
    --seed 42 \
    --results_dir "predictions_validation/"
