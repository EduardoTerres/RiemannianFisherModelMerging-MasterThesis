"""Build OrthoFuse-style diffusion evaluation tables."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "OrthoFuse"))
sys.path.insert(0, str(REPO_ROOT))

from src.diffusion.dataset_1 import DIFFUSION_MERGE_PAIRS, get_pair


DEFAULT_METHODS = (
    "diagonal_fisher_rescaled",
    "diagonal_fisher",
    "standard_rescaled",
    "orthofuse",
)

METRICS = (
    ("image_similarities_mx", "CLIP image similarity"),
    ("dino_image_similarities_mx", "DINO image similarity"),
    ("text_similarities_mx", "CLIP text similarity"),
    ("text_similarities_mx_with_class", "CLIP text similarity with class"),
    ("real_image_similarity_mx", "CLIP reference-image similarity"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--pairs", nargs="+", default=["all_dataset_pairs"])
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs/diffusion")
    parser.add_argument("--samples_dir", type=Path, default=None)
    parser.add_argument("--eval_root", type=Path, default=None)
    parser.add_argument("--tables_dir", type=Path, default=None)
    parser.add_argument("--checkpoint_idx", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=str, default="50")
    parser.add_argument("--guidance_scale", type=str, default="5.0")
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--processes", type=int, default=6)
    parser.add_argument("--diagonal_fisher_mu", type=float, default=None)
    parser.add_argument("--orthofuse_t", type=str, default="0.6")
    parser.add_argument("--orthofuse_postprocessing", type=str, default="curve_over_id")
    parser.add_argument("--clip_model", default=os.environ.get("ORTHOFUSE_CLIP_MODEL", "ViT-B/32"))
    parser.add_argument("--clip_pretrained", default=os.environ.get("ORTHOFUSE_CLIP_PRETRAINED", "openai"))
    parser.add_argument("--dino_model", default=os.environ.get("ORTHOFUSE_DINO_MODEL", "dinov2_vits14"))
    parser.add_argument("--dino_repo", default=os.environ.get("ORTHOFUSE_DINO_REPO", "facebookresearch/dinov2"))
    parser.add_argument(
        "--dino_source",
        choices=("github", "local"),
        default=os.environ.get("ORTHOFUSE_DINO_SOURCE", "github"),
    )
    return parser.parse_args()


def selected_pairs(pair_names):
    pairs = []
    for name in pair_names:
        if name == "all_dataset_pairs":
            pairs.extend(DIFFUSION_MERGE_PAIRS)
        elif ":" in name:
            pairs.append(get_pair(*name.split(":", 1)))
        else:
            pairs.extend(pair for pair in DIFFUSION_MERGE_PAIRS if pair["name"] == name)
    if not pairs:
        raise ValueError(f"No dataset pairs matched: {pair_names}")
    return pairs


def class_name_for(concept):
    if concept.startswith("cat"):
        return "cat"
    if concept.startswith("dog"):
        return "dog"
    return concept


def sample_name(args, method, pair_name):
    prefix = f"ns{args.num_inference_steps}_gs{args.guidance_scale}"
    if method == "orthofuse":
        return f"{prefix}_orthofuse_t{args.orthofuse_t}_method_{args.orthofuse_postprocessing}_{pair_name}"
    if method == "diagonal_fisher" and args.diagonal_fisher_mu is not None:
        return f"{prefix}_gradients_{method}_mu{args.diagonal_fisher_mu:g}_{pair_name}"
    return f"{prefix}_gradients_{method}_{pair_name}"


def write_hparams(path, exp_name, exp_dir, train_data_dir, concept, class_name):
    text = f"""class_name: '{class_name}'
exp_name: '{exp_name}'
output_dir: '{exp_dir}'
placeholder_token: '<{concept}>'
placeholder_token_concept: '<{concept}>'
placeholder_token_style: '<style>'
pretrained_model_name_or_path: 'stabilityai/stable-diffusion-xl-base-1.0'
resolution: 1024
revision: null
test_data_dir: '{train_data_dir}'
train_data_dir: '{train_data_dir}'
"""
    path.write_text(text, encoding="utf-8")


def link_samples(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        dst.unlink()
    if dst.exists():
        print(f"[keep] existing non-symlink sample path: {dst}", file=sys.stderr)
        return
    dst.symlink_to(src, target_is_directory=True)


def stage_runs(args):
    eval_root = args.eval_root or args.output_dir / "eval_runs"
    samples_dir = args.samples_dir or args.output_dir / "samples"
    eval_root.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "slurms").mkdir(parents=True, exist_ok=True)

    exp_names = []
    exp_idx = 1
    for method in args.methods:
        for pair in selected_pairs(args.pairs):
            pair_name = pair["name"]
            concept = pair["concept"]["name"]
            src = samples_dir / sample_name(args, method, pair_name)
            if not (src / f"version_{args.version}").is_dir():
                print(f"[skip] missing samples: {src}/version_{args.version}", file=sys.stderr)
                continue

            train_data_dir = args.output_dir / "d1_images" / concept
            if not train_data_dir.is_dir():
                print(f"[skip] missing reference images: {train_data_dir}", file=sys.stderr)
                continue

            exp_name = f"{exp_idx:05d}-eval-{method}-{pair_name}"
            exp_dir = eval_root / exp_name
            logs_dir = exp_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            write_hparams(
                logs_dir / "hparams.yml",
                exp_name,
                exp_dir,
                train_data_dir,
                concept,
                class_name_for(concept),
            )

            samples_parent = exp_dir / f"checkpoint-{args.checkpoint_idx}" / "samples"
            dst = samples_parent / f"ns{args.num_inference_steps}_gs{args.guidance_scale}"
            link_samples(src, dst)

            exp_names.append(exp_name)
            exp_idx += 1

    if not exp_names:
        raise SystemExit("No evaluable runs were staged. Check methods, pairs, and generated sample folders.")
    return eval_root, exp_names


def run_orthofuse_eval(args, eval_root, exp_names):
    import torch
    from nb_utils.cache import Cache, DistributedCache
    from nb_utils.clip_eval import ExpEvaluator
    from nb_utils.experiments_viewer import ExpsViewer

    device = torch.device("cuda", args.gpu)
    evaluator = ExpEvaluator(
        device,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        dino_model=args.dino_model,
        dino_repo=args.dino_repo,
        dino_source=args.dino_source,
    )
    cache = DistributedCache("./diffusers/examples/*/training-runs/*/evaluate.cache").get()
    viewer = ExpsViewer(
        base_path=str(eval_root),
        exp_filter_fn=lambda name: name in exp_names,
        ncolumns=6,
        lazy_load=True,
        evaluator=evaluator,
    )
    stats = viewer.evaluate(
        exps_names=exp_names,
        checkpoint_idx=str(args.checkpoint_idx),
        inference_specs=(args.num_inference_steps, args.guidance_scale),
        cache=cache,
        processes=args.processes,
    )

    summary = {}
    for key, value in stats.items():
        if "config" not in value:
            continue
        exp_cache = Cache(os.path.join(value["config"]["output_dir"], "evaluate.cache"))
        exp_cache.update({key: value})
        summary[str(key)] = value

    tables_dir = args.tables_dir or eval_root
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_path = tables_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved evaluation summary to {summary_path}", flush=True)
    return summary_path


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
    return exp_name.split("-eval-", 1)[1].rsplit("-", 1)[0]


def display_method_name(method, args):
    if method == "diagonal_fisher" and args.diagonal_fisher_mu is not None:
        return f"diagonal_fisher_mu{args.diagonal_fisher_mu:g}"
    if method == "orthofuse":
        return f"orthofuse_{args.orthofuse_postprocessing}_t{args.orthofuse_t}"
    return method


def display_prompt(prompt, record):
    config = record.get("config", {})
    for token in (
        config.get("placeholder_token"),
        config.get("placeholder_token_concept"),
    ):
        if token:
            prompt = prompt.replace(token, "<concept>")
    style_token = config.get("placeholder_token_style")
    if style_token:
        prompt = prompt.replace(style_token, "<style>")
    return prompt


def write_summary_table(summary_path, exp_names, args):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = {
        value.get("config", {}).get("exp_name"): value
        for value in summary.values()
        if isinstance(value, dict)
    }

    aggregates = {}
    method_order = []
    prompts = None
    prompt_labels = None
    for exp_name in exp_names:
        record = records.get(exp_name)
        if record is None:
            continue
        current_prompts = list(record.get("image_similarities_mx", {}))
        if prompts is None:
            prompts = current_prompts[:2]
            prompt_labels = [display_prompt(prompt, record) for prompt in prompts]
        method = display_method_name(method_from_exp_name(exp_name), args)
        if method not in aggregates:
            aggregates[method] = {prompt: {metric: [] for metric, _ in METRICS} for prompt in prompts}
            method_order.append(method)
        for prompt in prompts:
            for metric, _ in METRICS:
                values = record.get(metric, {})
                if isinstance(values, dict):
                    values = values.get(prompt, [])
                aggregates[method][prompt][metric].extend(flatten(values))

    rows = []
    for method in method_order:
        row = [method]
        for prompt in prompts:
            for metric, _ in METRICS:
                row.append(format_cell(aggregates[method][prompt][metric]))
        rows.append(row)
    if not rows:
        raise SystemExit("No evaluated records found in eval_summary.json for this run.")

    headers = ["method"]
    for prompt_idx in range(1, len(prompts) + 1):
        headers.extend(f"({prompt_idx}.{metric_idx})" for metric_idx in range(1, len(METRICS) + 1))
    widths = [max(len(str(row[idx])) for row in [headers, *rows]) for idx in range(len(headers))]

    lines = [
        "",
        "Evaluation summary table",
        " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)),
        "-+-".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append(" | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)))

    lines.extend(["", "Column correspondence"])
    for prompt_idx, prompt in enumerate(prompt_labels, start=1):
        for metric_idx, (_, label) in enumerate(METRICS, start=1):
            suffix = "" if label == "CLIP reference-image similarity" else f"; prompt {prompt_idx}: {prompt}"
            lines.append(f"({prompt_idx}.{metric_idx}) {label}{suffix}")

    table_text = "\n".join(lines)
    print(table_text)
    table_path = summary_path.with_suffix(".txt")
    table_path.write_text(table_text + "\n", encoding="utf-8")
    print(f"Saved evaluation summary table to {table_path}", flush=True)


def main():
    args = parse_args()
    eval_root, exp_names = stage_runs(args)
    summary_path = run_orthofuse_eval(args, eval_root, exp_names)
    write_summary_table(summary_path, exp_names, args)


if __name__ == "__main__":
    main()
