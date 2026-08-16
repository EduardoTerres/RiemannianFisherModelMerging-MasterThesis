"""Print approximate FIM sizes for the default LLM OFT adapters.

This intentionally reads only safetensors headers. It does not import
transformers/peft or load the base models, so it can run from a lightweight
environment and still answer the storage-size question.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open


MODELS_DIR = Path("/scratch-shared/eterres/models")
DEFAULT_MODELS = {
    "Llama-3.1-8B": (
        MODELS_DIR
        / "Llama-3.1-8B_OFT_dataset3_adapters"
        / "llama3-1_8b_finetune_wikitext"
    ),
    "Qwen-2.5-3B": (
        MODELS_DIR
        / "Qwen-2.5-3B_OFT_dataset3_adapters"
        / "qwen2.5_3b_finetune_gsm8k"
    ),
}


def safetensors_path(path: Path) -> Path:
    if path.is_dir():
        path = path / "adapter_model.safetensors"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def adapter_shapes(path: Path) -> list[tuple[str, tuple[int, ...]]]:
    path = safetensors_path(path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        return [
            (key, tuple(handle.get_slice(key).get_shape()))
            for key in sorted(handle.keys())
            if ".oft_" in key.lower()
        ]


def numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def is_block(shape: tuple[int, ...]) -> bool:
    return len(shape) == 3 and shape[-1] == shape[-2]


def logical_blocks(shape: tuple[int, ...]) -> int:
    return shape[0] if len(shape) >= 2 else 1


def block_parameter_dim(shape: tuple[int, ...]) -> int:
    if len(shape) < 2:
        return numel(shape)
    return numel(shape[1:])


def layer_index(name: str) -> int:
    """Extract transformer layer index from a parameter name."""
    marker = ".layers."
    if marker in name:
        suffix = name.split(marker, 1)[1]
        first = suffix.split(".", 1)[0]
        if first.isdigit():
            return int(first)
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return -1


def fim_counts(params: list[tuple[str, tuple[int, ...]]]) -> tuple[int, int, int, int]:
    diagonal = sum(numel(shape) for _, shape in params)
    full = diagonal * diagonal

    by_layer: dict[int, int] = defaultdict(int)
    for name, shape in params:
        by_layer[layer_index(name)] += numel(shape)
    layerwise = sum(n_l * n_l for n_l in by_layer.values())

    blockwise = sum(
        logical_blocks(shape) * block_parameter_dim(shape) ** 2
        for _, shape in params
    )
    return layerwise, full, diagonal, blockwise


def print_info(model_name: str, adapter_path: Path) -> None:
    params = adapter_shapes(adapter_path)
    if not params:
        raise ValueError(f"No OFT tensors found in {safetensors_path(adapter_path)}")

    opt1, opt2, opt3, opt4 = fim_counts(params)
    print(f"--- {model_name} ---")
    print(f"Adapter            : {safetensors_path(adapter_path)}")
    print(
        f"Trainable tensors  : {len(params)} "
        f"({sum(1 for _, shape in params if is_block(shape))} expanded block)"
    )
    print(f"Logical blocks     : {sum(logical_blocks(shape) for _, shape in params):,}")
    print(f"Total params       : {opt3:,}")
    print(f"Option 1 diagonal  : {opt3:,}")
    print(f"Option 2 blockwise : {opt4:,}")
    print(f"Option 3 layerwise : {opt1:,}")
    print(f"Option 4 full      : {opt2:,}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llama-adapter",
        type=Path,
        default=DEFAULT_MODELS["Llama-3.1-8B"],
        help="Llama adapter directory or adapter_model.safetensors path.",
    )
    parser.add_argument(
        "--qwen-adapter",
        type=Path,
        default=DEFAULT_MODELS["Qwen-2.5-3B"],
        help="Qwen adapter directory or adapter_model.safetensors path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_info("Llama-3.1-8B", args.llama_adapter)
    print_info("Qwen-2.5-3B", args.qwen_adapter)


if __name__ == "__main__":
    main()
