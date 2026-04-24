# SocialIQA, CommonsenseQA, and Minerva Math500
RELATIVE_PATH="../../.."
MODELS=(
    "outputs/models/Llama-3.1-8B-merged/merged_model"
    # "outputs/models/Qwen-2.5-3B-merged-std/merged_model"
)

set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lm-eval
cd OrthoMerge/eval/lm-evaluation-harness

for MODEL_PATH in "${MODELS[@]}"; do
    echo "Evaluating: $MODEL_PATH"
    
    lm_eval --model hf \
        --tasks social_iqa,commonsense_qa,minerva_math500 \
        --model_args pretrained=${RELATIVE_PATH}/${MODEL_PATH} \
        --device cuda:0 \
        --batch_size 64 \
        --confirm_run_unsafe_code \
        --trust_remote_code
done

# # ScienceQA
# conda activate OrthoMerge
# python eval/eval_scienceqa.py --merged_dir "outputs/models/Llama-3.1-8B-merged-fisher/merged_model" \
#     "data/models/Llama-3.1-8B/"
