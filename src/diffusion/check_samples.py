"""Compare diffusion eval_summary methods against Orthofuse per pair."""

import argparse
import json
from pathlib import Path

from src.diffusion.dataset_1 import DIFFUSION_MERGE_PAIRS


PAIR_NAMES = sorted((pair["name"] for pair in DIFFUSION_MERGE_PAIRS), key=len, reverse=True)
DEFAULT_METRICS = ("dino_image_similarities_mx", "style_dino_image_similarities_mx")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", nargs="?", type=Path, default=Path("outputs/diffusion/tables/eval_summary.json"))
    parser.add_argument("--metric", action="append", dest="metrics", help="Metric to report. Repeat for multiple.")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def flatten(value):
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (int, float)):
        return [float(value)]
    return [item for child in value for item in flatten(child)] if isinstance(value, list) else []


def mean(value):
    values = flatten(value)
    return sum(values) / len(values) if values else None


def prompt_scores(value):
    if not isinstance(value, dict):
        return {}
    return {prompt: score for prompt, values in value.items() if (score := mean(values)) is not None}


def split_exp_name(exp_name):
    _, rest = exp_name.split("-eval-", 1)
    for pair_name in PAIR_NAMES:
        suffix = f"-{pair_name}"
        if rest.endswith(suffix):
            return rest[: -len(suffix)], pair_name
    return None, None


def report_metric(records_json, metric_name, top_k):
    records = {}
    prompt_records = {}
    for record in records_json.values():
        exp_name = record.get("config", {}).get("exp_name")
        metric = record.get(metric_name)
        score = mean(metric)
        if not exp_name or score is None:
            continue
        method, pair_name = split_exp_name(exp_name)
        if method and pair_name:
            records[(method, pair_name)] = score
            for prompt, prompt_score in prompt_scores(metric).items():
                prompt_records[(method, pair_name, prompt)] = prompt_score

    orthofuse = {pair: score for (method, pair), score in records.items() if method == "orthofuse"}
    orthofuse_prompts = {
        (pair, prompt): score
        for (method, pair, prompt), score in prompt_records.items()
        if method == "orthofuse"
    }
    methods = sorted({method for method, _ in records if method != "orthofuse"})
    print(f"\nmetric: {metric_name}")
    for method in methods:
        deltas = [
            (score - orthofuse[pair], pair, score, orthofuse[pair])
            for (record_method, pair), score in records.items()
            if record_method == method and pair in orthofuse
        ]
        if not deltas:
            continue
        print(f"\n{method}")
        for title, rows in (("best pairs", sorted(deltas, reverse=True)[:top_k]), ("worst pairs", sorted(deltas)[:top_k])):
            print(f"  {title}:")
            for delta, pair, score, base in rows:
                print(f"    {pair}: delta={delta:+.4f} method={score:.4f} orthofuse={base:.4f}")
        prompt_deltas = [
            (score - orthofuse_prompts[(pair, prompt)], pair, prompt, score, orthofuse_prompts[(pair, prompt)])
            for (record_method, pair, prompt), score in prompt_records.items()
            if record_method == method and (pair, prompt) in orthofuse_prompts
        ]
        for title, rows in (
            ("best prompts", sorted(prompt_deltas, reverse=True)[:top_k]),
            ("worst prompts", sorted(prompt_deltas)[:top_k]),
        ):
            print(f"  {title}:")
            for delta, pair, prompt, score, base in rows:
                print(f"    {pair} | {prompt}: delta={delta:+.4f} method={score:.4f} orthofuse={base:.4f}")


def main():
    args = parse_args()
    records_json = json.loads(args.summary.read_text(encoding="utf-8"))
    for metric_name in args.metrics or DEFAULT_METRICS:
        report_metric(records_json, metric_name, args.top_k)


if __name__ == "__main__":
    main()
