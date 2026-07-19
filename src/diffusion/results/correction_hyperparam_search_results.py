import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.diffusion.correction_hyperparam_search import (
    PROMPTS,
    image_path,
    interpolation_grid,
    mus,
    root,
    text_prompt,
)
from src.diffusion.results.pareto import (
    METRIC_KEYS,
    add_evaluator_args,
    add_pareto_flags,
    aggregate_rows,
    make_evaluator,
    output_prefix,
    plot_style_concept_curves,
    score_records,
    selected_pairs,
    SimilarityScorer,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs/diffusion")
    parser.add_argument(
        "--samples",
        nargs="+",
        default=["cat:01_01", "dog2:dolina", "dog6:gondoliers", "cat2:pots", "dog:03_04"],
    )
    parser.add_argument("--num_points", type=int, default=9)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--version_start", type=int, default=0)
    parser.add_argument("--image_index", type=int, default=0)
    parser.add_argument("--force_recompute", action="store_true")
    add_evaluator_args(parser)
    parser.set_defaults(reference_root=REPO_ROOT / "outputs/diffusion/d1_images")
    return parser.parse_args()


def records(args, pairs):
    for pair in pairs:
        for mu in mus():
            alphas = interpolation_grid(mu, args.num_points)
            for point_idx, (t, alpha) in enumerate(alphas):
                for prompt_name, template in PROMPTS.items():
                    yield {
                        "pair": pair,
                        "pair_name": pair["name"],
                        "prompt": prompt_name,
                        "mu": mu,
                        "step": point_idx,
                        "t": t,
                        "alpha_1": alpha[0],
                        "alpha_2": alpha[1],
                        "image_path": image_path(args, pair, mu, point_idx, template),
                        "text_prompt": text_prompt(pair, template),
                    }


def plot_text_similarity(rows, save_path, title):
    import matplotlib.pyplot as plt

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    mu_values = mus()
    cmap = plt.get_cmap("viridis")
    denom = max(len(mu_values) - 1, 1)
    for idx, mu in enumerate(mu_values):
        curve = [row for row in rows if row["mu"] == mu]
        ax.plot(
            [row["t"] for row in curve],
            [row["clip_text"] for row in curve],
            marker="o",
            linewidth=1.5,
            markersize=3,
            color=cmap(idx / denom),
            label=f"mu={mu:g}",
        )
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("CLIP similarity to text prompt")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def parse_csv_value(key, value):
    if key in {"mu", "t", "alpha_1", "alpha_2", *METRIC_KEYS, "clip_text"}:
        return float(value)
    if key in {"step", "n"}:
        return int(value)
    if key in {"clip_pareto", "dino_pareto"}:
        return value == "True"
    return value


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            {key: parse_csv_value(key, value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def run(args):
    pairs = selected_pairs(args.samples)
    group_keys = ("prompt", "mu", "step", "t", "alpha_1", "alpha_2")
    prefix = output_prefix(pairs)
    csv_path = root(args) / f"{prefix}_pareto_frontier.csv"
    fieldnames = [
        *group_keys,
        "n",
        "clip_concept",
        "clip_style",
        "clip_text",
        "dino_concept",
        "dino_style",
        "clip_pareto",
        "dino_pareto",
    ]
    if csv_path.exists() and not args.force_recompute:
        aggregate = read_csv(csv_path)
        print(f"[correction-results] using existing {csv_path}", flush=True)
    else:
        reference_root = args.reference_root or args.output_dir / "d1_images"
        scorer = SimilarityScorer(make_evaluator(args), reference_root)
        rows = score_records(records(args, pairs), scorer)
        if not rows:
            raise RuntimeError("No correction search images were found for the requested pairs.")

        metric_keys = (*METRIC_KEYS, "clip_text")
        aggregate = aggregate_rows(rows, group_keys, metric_keys)
        aggregate = sorted(aggregate, key=lambda row: (row["prompt"], row["mu"], row["step"]))
        add_pareto_flags(aggregate, group_keys=("prompt",))
        write_csv(csv_path, aggregate, fieldnames)
        print(f"[correction-results] wrote {csv_path}", flush=True)

    for prompt_name in PROMPTS:
        prompt_rows = [row for row in aggregate if row["prompt"] == prompt_name]
        png_path = root(args) / f"{prefix}_pareto_frontier_{prompt_name}.png"
        plot_style_concept_curves(
            prompt_rows,
            png_path,
            f"Diagonal Fisher correction search ({prompt_name})",
            series_key="mu",
            label_key="t",
        )
        print(f"[correction-results] wrote {png_path}", flush=True)
        text_png_path = root(args) / f"{prefix}_text_similarity_{prompt_name}.png"
        plot_text_similarity(prompt_rows, text_png_path, f"CLIP text similarity ({prompt_name})")
        print(f"[correction-results] wrote {text_png_path}", flush=True)
        if prompt_name == "normal":
            default_png_path = root(args) / f"{prefix}_pareto_frontier.png"
            plot_style_concept_curves(
                prompt_rows,
                default_png_path,
                "Diagonal Fisher correction search",
                series_key="mu",
                label_key="t",
            )
            print(f"[correction-results] wrote {default_png_path}", flush=True)


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
