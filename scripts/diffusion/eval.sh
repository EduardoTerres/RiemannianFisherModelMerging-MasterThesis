#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=diffusion_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=outputs/diffusion/slurms/eval_%A.out

set -e

METHODS=(
  "diagonal_fisher_rescaled"
  "diagonal_fisher"
  "standard_rescaled"
  "orthofuse"
)

REPO_ROOT="/gpfs/home6/eterres/MasterThesis"
OUTPUT_DIR="${REPO_ROOT}/outputs/diffusion"
EVAL_ROOT="${OUTPUT_DIR}/eval_runs"
CHECKPOINT_IDX=0
NUM_INFERENCE_STEPS=50
GUIDANCE_SCALE=5.0
VERSION=0
GPU="${CUDA_VISIBLE_DEVICES:-0}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate orthofuse_env

export HF_HOME=/scratch-shared/eterres/huggingface-cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${HF_HOME}/torch"
EVAL_OFFLINE="${EVAL_OFFLINE:-0}"
export HF_HUB_OFFLINE="${EVAL_OFFLINE}"
export TRANSFORMERS_OFFLINE="${EVAL_OFFLINE}"
export DIFFUSERS_OFFLINE="${EVAL_OFFLINE}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${USER}}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${REPO_ROOT}/OrthoFuse:${REPO_ROOT}:${PYTHONPATH:-}"

mapfile -t PAIRS < <(
  python - <<'PY'
from src.diffusion.dataset_1 import DIFFUSION_MERGE_PAIRS

for pair in DIFFUSION_MERGE_PAIRS:
    print(pair["name"])
PY
)

echo "[eval] staging ${#PAIRS[@]} dataset pairs"

ORTHOFUSE_CLIP_MODEL="${ORTHOFUSE_CLIP_MODEL:-ViT-B/32}"
ORTHOFUSE_CLIP_PRETRAINED="${ORTHOFUSE_CLIP_PRETRAINED:-openai}"
ORTHOFUSE_DINO_MODEL="${ORTHOFUSE_DINO_MODEL:-dinov2_vits14}"
ORTHOFUSE_DINO_REPO="${ORTHOFUSE_DINO_REPO:-facebookresearch/dinov2}"
ORTHOFUSE_DINO_SOURCE="${ORTHOFUSE_DINO_SOURCE:-github}"

mkdir -p "${EVAL_ROOT}" "${OUTPUT_DIR}/slurms"

exp_names=()
exp_idx=1

for method in "${METHODS[@]}"; do
  for pair in "${PAIRS[@]}"; do
    concept="${pair%%__*}"
    placeholder_token="<${concept}>"

    if [[ "${concept}" == cat* ]]; then
      class_name="cat"
    elif [[ "${concept}" == dog* ]]; then
      class_name="dog"
    else
      class_name="${concept}"
    fi

    if [[ "${method}" == "orthofuse" ]]; then
      sample_name="ns${NUM_INFERENCE_STEPS}_gs${GUIDANCE_SCALE}_orthofuse_t0.6_method_curve_over_id_${pair}"
    else
      sample_name="ns${NUM_INFERENCE_STEPS}_gs${GUIDANCE_SCALE}_gradients_${method}_${pair}"
    fi

    sample_src="${OUTPUT_DIR}/samples/${sample_name}"

    if [[ ! -d "${sample_src}/version_${VERSION}" ]]; then
      echo "[skip] missing samples: ${sample_src}/version_${VERSION}" >&2
      continue
    fi

    exp_name="$(printf "%05d" "${exp_idx}")-eval-${method}-${pair}"
    exp_dir="${EVAL_ROOT}/${exp_name}"
    logs_dir="${exp_dir}/logs"
    samples_parent="${exp_dir}/checkpoint-${CHECKPOINT_IDX}/samples"
    sample_dst="${samples_parent}/ns${NUM_INFERENCE_STEPS}_gs${GUIDANCE_SCALE}"
    train_data_dir="${OUTPUT_DIR}/d1_images/${concept}"

    if [[ ! -d "${train_data_dir}" ]]; then
      echo "[skip] missing reference images: ${train_data_dir}" >&2
      continue
    fi

    mkdir -p "${logs_dir}" "${samples_parent}"

    cat > "${logs_dir}/hparams.yml" <<YAML
class_name: '${class_name}'
exp_name: '${exp_name}'
output_dir: '${exp_dir}'
placeholder_token: '${placeholder_token}'
placeholder_token_concept: '${placeholder_token}'
placeholder_token_style: '<style>'
pretrained_model_name_or_path: 'stabilityai/stable-diffusion-xl-base-1.0'
resolution: 1024
revision: null
test_data_dir: '${train_data_dir}'
train_data_dir: '${train_data_dir}'
YAML

    if [[ -L "${sample_dst}" ]]; then
      ln -sfn "${sample_src}" "${sample_dst}"
    elif [[ -e "${sample_dst}" ]]; then
      echo "[keep] existing non-symlink sample path: ${sample_dst}" >&2
    else
      ln -s "${sample_src}" "${sample_dst}"
    fi

    exp_names+=("${exp_name}")
    exp_idx=$((exp_idx + 1))
  done
done

if [[ "${#exp_names[@]}" -eq 0 ]]; then
  echo "No evaluable runs were staged. Check METHODS, PAIRS, and generated sample folders." >&2
  exit 1
fi

eval_args=(
  --gpu 0
  --base_path "${EVAL_ROOT}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --guidance_scale "${GUIDANCE_SCALE}"
  --checkpoints_idxs "${CHECKPOINT_IDX}"
  --clip_model "${ORTHOFUSE_CLIP_MODEL}"
  --clip_pretrained "${ORTHOFUSE_CLIP_PRETRAINED}"
  --dino_model "${ORTHOFUSE_DINO_MODEL}"
  --dino_repo "${ORTHOFUSE_DINO_REPO}"
  --dino_source "${ORTHOFUSE_DINO_SOURCE}"
  --exp_names "${exp_names[@]}"
)

python -m nb_utils.evaluate "${eval_args[@]}"

python - "${EVAL_ROOT}/eval_summary.json" "${exp_names[@]}" <<'PY'
import json
import math
import sys
from pathlib import Path


summary_path = Path(sys.argv[1])
exp_names = sys.argv[2:]

metric_specs = [
    ("image_similarities_mx", "CLIP image similarity"),
    ("dino_image_similarities_mx", "DINO image similarity"),
    ("text_similarities_mx", "CLIP text similarity"),
    ("text_similarities_mx_with_class", "CLIP text similarity with class"),
    ("real_image_similarity_mx", "CLIP reference-image similarity"),
]


def flatten(values):
    if not isinstance(values, list):
        return [float(values)]
    flat = []
    for value in values:
        flat.extend(flatten(value))
    return flat


def mean_std(values):
    values = flatten(values)
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def format_cell(values):
    mean, std = mean_std(values)
    return f"{mean:.4f} +/- {std:.4f}"


def method_from_exp_name(exp_name):
    if "-eval-" in exp_name:
        exp_name = exp_name.split("-eval-", 1)[1]
    return exp_name.rsplit("-", 1)[0]


summary = json.loads(summary_path.read_text())
records = {
    value.get("config", {}).get("exp_name"): value
    for value in summary.values()
    if isinstance(value, dict)
}

aggregates = {}
method_order = []
prompts = None
for exp_name in exp_names:
    record = records.get(exp_name)
    if record is None:
        continue
    current_prompts = list(record.get("image_similarities_mx", {}))
    if prompts is None:
        prompts = current_prompts[:2]
    method = method_from_exp_name(exp_name)
    if method not in aggregates:
        aggregates[method] = {
            prompt: {metric_key: [] for metric_key, _ in metric_specs}
            for prompt in prompts
        }
        method_order.append(method)
    for prompt in prompts:
        for metric_key, _ in metric_specs:
            metric_values = record.get(metric_key)
            if isinstance(metric_values, dict):
                values = metric_values.get(prompt, [])
            else:
                values = metric_values or []
            aggregates[method][prompt][metric_key].extend(flatten(values))

rows = []
for method in method_order:
    row = [method]
    for prompt in prompts:
        for metric_key, _ in metric_specs:
            row.append(format_cell(aggregates[method][prompt][metric_key]))
    rows.append(row)

if not rows:
    raise SystemExit("No evaluated records found in eval_summary.json for this run.")

headers = ["method"]
for prompt_idx in range(1, len(prompts) + 1):
    headers.extend(f"({prompt_idx}.{metric_idx})" for metric_idx in range(1, len(metric_specs) + 1))

widths = [
    max(len(str(row[col_idx])) for row in [headers, *rows])
    for col_idx in range(len(headers))
]

lines = [
    "",
    "Evaluation summary table",
    " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)),
    "-+-".join("-" * width for width in widths),
]
for row in rows:
    lines.append(" | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)))

lines.extend(["", "Column correspondence"])
for prompt_idx, prompt in enumerate(prompts, start=1):
    for metric_idx, (_, label) in enumerate(metric_specs, start=1):
        if label == "CLIP reference-image similarity":
            lines.append(f"({prompt_idx}.{metric_idx}) {label}")
        else:
            lines.append(f"({prompt_idx}.{metric_idx}) {label}; prompt {prompt_idx}: {prompt}")

table_text = "\n".join(lines)
print(table_text)

table_path = summary_path.with_suffix(".txt")
table_path.write_text(table_text + "\n")
print(f"Saved evaluation summary table to {table_path}", flush=True)
PY
