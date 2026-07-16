#!/usr/bin/env python3
"""Write short OFT summaries for Llama, Qwen, and one diffusion adapter."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

from safetensors import safe_open


DEFAULT_QWEN = Path(
    "/scratch-shared/eterres/models/Qwen-2.5-3B_OFT_dataset3_adapters/"
    "qwen2.5_3b_finetune_gsm8k"
)
DEFAULT_LLAMA = Path(
    "/scratch-shared/eterres/models/Llama-3.1-8B_OFT_dataset3_adapters/"
    "llama3-1_8b_finetune_wikitext"
)
DEFAULT_DIFFUSION = Path(
    "/scratch-shared/eterres/SDXL/concepts/adapters/pytorch_lora_weights_cat.safetensors"
)
DEFAULT_OUTPUT = Path("outputs/oft_info/oft_info.txt")
DIFFUSION_PROJECTIONS = ("to_q_moft", "to_k_moft", "to_v_moft", "to_out_moft")


def safetensors_path(path: Path) -> Path:
    if path.is_dir():
        path = path / "adapter_model.safetensors"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def keys_and_shapes(path: Path) -> list[tuple[str, tuple[int, ...]]]:
    path = safetensors_path(path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        return [(key, tuple(handle.get_slice(key).get_shape())) for key in sorted(handle.keys())]


def numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def compact(counter: Counter, limit: int = 20) -> str:
    items = [f"{key}: {value}" for key, value in counter.most_common(limit)]
    if len(counter) > limit:
        items.append(f"... {len(counter) - limit} more")
    return ", ".join(items) if items else "none"


def shape_text(shape: tuple[int, ...]) -> str:
    return str(shape)


def infer_so_dim(num_skew_params: int) -> int | None:
    # n(n - 1) / 2 = num_skew_params
    n = int((1 + math.sqrt(1 + 8 * num_skew_params)) / 2)
    return n if n * (n - 1) // 2 == num_skew_params else None


def llm_module(name: str) -> str:
    for module in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if f".{module}." in name:
            return module
    return "other"


def llm_layer(name: str) -> str:
    marker = ".layers."
    if marker not in name:
        return "unknown"
    return name.split(marker, 1)[1].split(".", 1)[0]


def diffusion_projection(name: str) -> str:
    for projection in DIFFUSION_PROJECTIONS:
        if projection in name:
            return projection
    return "scale" if name.endswith(("q_scale", "k_scale", "v_scale", "out_scale")) else "other"


def diffusion_block(name: str) -> str:
    return name.split(".", 1)[0]


def summarize_llm(label: str, path: Path) -> list[str]:
    rows = [(name, shape) for name, shape in keys_and_shapes(path) if ".oft_" in name]
    modules = Counter(llm_module(name) for name, _ in rows)
    layers = {llm_layer(name) for name, _ in rows}
    by_module_shape = Counter()
    stored_values = 0
    expanded_values = 0
    for name, shape in rows:
        stored_values += numel(shape)
        if len(shape) == 2:
            blocks, skew_params = shape
            so_dim = infer_so_dim(skew_params)
            full_shape = (blocks, so_dim, so_dim) if so_dim is not None else shape
            expanded_values += numel(full_shape)
            by_module_shape[f"{llm_module(name)} {shape_text(full_shape)}"] += 1
        else:
            expanded_values += numel(shape)
            by_module_shape[f"{llm_module(name)} {shape_text(shape)}"] += 1

    return [
        f"{label} adapter",
        f"  path: {safetensors_path(path)}",
        f"  OFT tensors: {len(rows)}",
        f"  transformer layers: {len(layers - {'unknown'})}",
        f"  stored skew-coordinate values: {stored_values:,}",
        f"  expanded OFT matrix entries: {expanded_values:,}",
        f"  modules: {compact(modules)}",
        f"  module shapes: {compact(by_module_shape)}",
    ]


def summarize_diffusion(path: Path) -> list[str]:
    rows = keys_and_shapes(path)
    matrices = [(name, shape) for name, shape in rows if name.endswith((".L", ".R"))]
    scales = [(name, shape) for name, shape in rows if name.endswith(("q_scale", "k_scale", "v_scale", "out_scale"))]
    projections = Counter(diffusion_projection(name) for name, _ in matrices)
    blocks = Counter(diffusion_block(name) for name, _ in matrices)
    by_projection_shape = Counter(
        f"{diffusion_projection(name)} {shape_text(shape)}" for name, shape in matrices
    )
    scale_shapes = Counter(shape_text(shape) for _, shape in scales)

    return [
        "Diffusion adapter",
        f"  path: {safetensors_path(path)}",
        f"  OFT matrix tensors: {len(matrices)}",
        f"  scale tensors: {len(scales)}",
        f"  expanded OFT matrix entries: {sum(numel(shape) for _, shape in matrices):,}",
        f"  scale values: {sum(numel(shape) for _, shape in scales):,}",
        f"  projections: {compact(projections)}",
        f"  SDXL blocks: {compact(blocks)}",
        f"  projection shapes: {compact(by_projection_shape)}",
        f"  scale shapes: {compact(scale_shapes)}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-adapter", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--llama-adapter", type=Path, default=DEFAULT_LLAMA)
    parser.add_argument("--diffusion-adapter", type=Path, default=DEFAULT_DIFFUSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    lines = (
        summarize_llm("Qwen LLM", args.qwen_adapter)
        + [""]
        + summarize_llm("Llama LLM", args.llama_adapter)
        + [""]
        + summarize_diffusion(args.diffusion_adapter)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
