"""PCA of OFT task vectors (layerwise): one plot + GIF per module type."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import re
from collections import OrderedDict, defaultdict
from typing import Dict, List

import numpy as np
import torch

from src.analysis.plot_utils import plot_pca_task_vectors, save_pca_gif
from src.merging import OFTKarcherMerging, OFTMerging
from src.paths import MODEL_FAMILIES, ROOTDIR
from src.utils import parse_device

_TASK_LABELS = ["social_iqa", "commonsense", "numinamath", "magicoder", "science_qa"]


def _flatten(t: torch.Tensor) -> np.ndarray:
    return t.float().cpu().flatten().numpy()


def _group_by_module(weights: Dict[str, torch.Tensor]) -> Dict[str, Dict[int, str]]:
    """Returns {module_type: {layer_idx: key}}."""
    groups: Dict[str, Dict[int, str]] = defaultdict(dict)
    for k in weights:
        m = re.search(r"layers?\.(\d+)\.(.+?)(?:\.oft_\w+)?$", k)
        if m:
            groups[m.group(2)][int(m.group(1))] = k
    return groups


def _run(family_name: str, args: argparse.Namespace) -> None:
    family = MODEL_FAMILIES[family_name]
    device = args.device
    task_names = args.task_names or _TASK_LABELS[: len(family.adapter_paths)]

    merger_avg = OFTMerging(device=device)
    merger_karcher = OFTKarcherMerging(n_steps=args.n_steps, lr=args.lr, device=device)

    task_weights = merger_avg.load_weights(family.adapter_paths)
    if not task_weights:
        raise RuntimeError(f"No adapter weights found for {family_name}")

    with_fisher = bool(family.fisher_paths) and all(Path(p).exists() for p in family.fisher_paths)
    if not with_fisher:
        print(f"[INFO] No Fisher files for {family_name}, skipping Fisher modes.")

    # Collect all named weight dicts (task + pretrained + merged)
    named: OrderedDict[str, Dict[str, torch.Tensor]] = OrderedDict()
    for name, w in zip(task_names, task_weights):
        named[name] = w
    named["Pretrained"] = {
        key: torch.zeros_like(value)
        for key, value in task_weights[0].items()
    }
    named["Avg (std)"] = merger_avg.merge(adapter_paths=family.adapter_paths, mode="standard")
    named["Karcher (std)"] = merger_karcher.merge(adapter_paths=family.adapter_paths, mode="standard")
    if with_fisher:
        named["Avg (Fisher)"] = merger_avg.merge(
            adapter_paths=family.adapter_paths, fisher_paths=family.fisher_paths,
            mode="diagonal_fisher",
        )
        named["Karcher (Fisher)"] = merger_karcher.merge(
            adapter_paths=family.adapter_paths, fisher_paths=family.fisher_paths,
            mode="diagonal_fisher",
        )

    n_tasks = len(task_names)
    save_dir = Path(args.save_path) / family_name

    # Group by module type using the first weight dict
    module_layer_keys = _group_by_module(next(iter(named.values())))

    for module, layer_keys in module_layer_keys.items():
        sorted_layers = sorted(layer_keys)

        # Build one vectors-dict per layer
        frames: List[OrderedDict] = []
        for l in sorted_layers:
            key = layer_keys[l]
            frames.append(OrderedDict(
                (name, _flatten(wdict[key])) for name, wdict in named.items()
            ))

        # Average across layers
        avg_vecs: OrderedDict[str, np.ndarray] = OrderedDict(
            (name, np.mean([f[name] for f in frames], axis=0))
            for name in named
        )

        out_dir = save_dir / module.replace(".", "_")
        out_dir.mkdir(parents=True, exist_ok=True)

        plot_pca_task_vectors(
            vectors=avg_vecs,
            n_tasks=n_tasks,
            title=f"{module} — avg layers ({family_name})",
            save_path=str(out_dir / "pca_avg.png"),
        )
        print(f"Saved -> {out_dir / 'pca_avg.png'}")

        save_pca_gif(
            frames=frames,
            frame_titles=[f"Layer {l}" for l in sorted_layers],
            n_tasks=n_tasks,
            title=f"{module} ({family_name})",
            save_path=str(out_dir / "pca_layers.gif"),
            fps=args.fps,
        )
        print(f"Saved -> {out_dir / 'pca_layers.gif'}")


def main(args: argparse.Namespace) -> None:
    for family_name in args.model_family:
        _run(family_name, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layerwise PCA of OFT task vectors.")
    parser.add_argument(
        "--model-family", nargs="+", default=list(MODEL_FAMILIES), choices=list(MODEL_FAMILIES),
        dest="model_family",
    )
    parser.add_argument("--task-names", nargs="+", default=None, dest="task_names")
    parser.add_argument(
        "--save-path", type=str, default=f"{ROOTDIR}/outputs/visualization_merge", dest="save_path",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-steps", type=int, default=200, dest="n_steps")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--fps", type=int, default=2)
    args = parser.parse_args()
    args.device = parse_device(args.device)
    main(args)
