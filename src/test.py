import torch
from transformers import AutoModelForCausalLM
from src.constants import OFT_LLAMA_MODELS_DIR
from peft import PeftModel


def print_oft_parameters(model):
    """Print all parameters whose name contains 'oft_r' or 'oft_R'."""
    found = [
        (n, p)
        for n, p in model.named_parameters()
        # if "oft_r" in n or "oft_R" in n
    ]
    print(f"\n=== OFT parameters ({len(found)} tensors) ===")
    for n, p in found:
        print(f"  {n}: shape={p.shape}, requires_grad={p.requires_grad}, dtype={p.dtype}")
    if not found:
        print("  (none found)")
    return found

ADAPTER_PATH = f"{OFT_LLAMA_MODELS_DIR}/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_commonsense/"
BASE_MODEL_NAME = f"{OFT_LLAMA_MODELS_DIR}/Llama-3.1-8B/"
device = "cuda" if torch.cuda.is_available() else "cpu"

base_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=BASE_MODEL_NAME,
    cache_dir=None,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    device_map=None,
)
base_model.to(device)
print(f"Base model loaded on {device}")

# Load adapter WITHOUT merging - must stay as PeftModel for Fisher computation
peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, is_trainable=True)
peft_model.enable_adapter_layers()  # force requires_grad=True on adapter params

print_oft_parameters(peft_model)

peft_model.print_trainable_parameters()
