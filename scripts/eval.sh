# SocialIQA, CommonsenseQA, and Minerva Math500
RELATIVE_PATH="../../.."
MODEL_PATH="outputs/OrthoMerge_Llama-3.1-8B-fisher/merged_model"

set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lm-eval; cd OrthoMerge/eval/lm-evaluation-harness
lm_eval --model hf \
    --tasks social_iqa,commonsense_qa,minerva_math500 \
    --model_args pretrained=${RELATIVE_PATH}/${MODEL_PATH} \
    --device cuda:0 \
    --batch_size 64 \
    --confirm_run_unsafe_code \
    --trust_remote_code