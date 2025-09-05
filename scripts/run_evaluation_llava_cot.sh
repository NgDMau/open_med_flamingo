export HF_HOME="/app/.cache/"
# export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0

echo 'activating virtual environment'
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate instruct_flamingo
which python

python open_flamingo/instruction_tuning/evaluation.py \
    --inference_result "/app/baseline_models/instruct_flamingo/predictions_validation/0829-clever_flamingo_v2_3b-batch256-lr2e-4-epochs30-lrschedconsine-resume-from-mpt7b-checkpoint_15/eval_dataset_config_-1/llava-mri-cot-1k-test_all.json"