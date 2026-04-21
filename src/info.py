import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.path import (
    LLAMA_BASE_MODEL_PATH, LLAMA_ADAPTERS_FOLDER,
    QWEN_BASE_MODEL_PATH, QWEN_ADAPTERS_FOLDER,
)

MODELS = {
    "Llama-3.1-8B": (LLAMA_BASE_MODEL_PATH, f"{LLAMA_ADAPTERS_FOLDER}/llama3-1_8b_finetune_commonsense"),
    "Qwen-2.5-3B":  (QWEN_BASE_MODEL_PATH,  f"{QWEN_ADAPTERS_FOLDER}/qwen2.5_3b_finetune_commonsense"),
}

device = "cuda" if torch.cuda.is_available() else "cpu"


def is_block(p):
    return p.dim() == 3 and p.shape[1] == p.shape[2]


def layer_index(name):
    """Extract transformer layer index from a parameter name."""
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return -1


def fim_counts(params):
    n = sum(p.numel() for _, p in params)

    # layerwise: group all tensors of the same transformer layer into one block
    from collections import defaultdict
    by_layer = defaultdict(int)
    for name, p in params:
        by_layer[layer_index(name)] += p.numel()
    layerwise = sum(n_l**2 for n_l in by_layer.values())

    # blockwise: OFT tensors [B, r, r] -> B independent (r^2 x r^2) blocks
    blockwise = sum(p.shape[0] * p.shape[1]**4 if is_block(p) else p.numel()**2
                    for _, p in params)

    return layerwise, n * n, n, blockwise


for model_name, (base_path, adapter_path) in MODELS.items():
    base = AutoModelForCausalLM.from_pretrained(base_path, cache_dir=None,
               torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
               device_map=None)
    base.to(device)
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    model.enable_adapter_layers()

    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    opt1, opt2, opt3, opt4 = fim_counts(params)

    print(f"--- {model_name} ---")
    print(f"Trainable tensors  : {len(params)} ({sum(1 for _,p in params if is_block(p))} block)")
    print(f"Total params       : {sum(p.numel() for _,p in params):,}")
    print(f"Option 1 diagonal  : {opt3:,}")
    print(f"Option 2 blockwise : {opt4:,}")
    print(f"Option 3 layerwise : {opt1:,}")
    print(f"Option 4 full      : {opt2:,}")
    print()

    del model, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
