from dataclasses import dataclass
from pathlib import Path

RELATIVE_ROOTDIR = Path("../..")
ROOTDIR = (Path(__file__) / RELATIVE_ROOTDIR).resolve()


@dataclass(frozen=True)
class ModelFamily:
    name: str
    base_model_path: str
    adapter_paths: list
    fisher_paths: list

    def __str__(self):
        return self.name

MODELS_DIR = ROOTDIR / "data" / "models"
FISHERS_DIR = ROOTDIR / "data" / "empirical_fishers"

# Llama 3.1 8B and its OFT adapters fine-tuned on 5 tasks
# (social_iqa, commonsense_qa, numinamath, magicoder, science_qa).
LLAMA_MODEL_NAME = "Llama-3.1-8B"
LLAMA_BASE_MODEL_PATH = f"{MODELS_DIR}/Llama-3.1-8B"
LLAMA_ADAPTERS_FOLDER = f"{MODELS_DIR}/Llama-3.1-8B_OFT_adapters"
LLAMA_ADAPTER_PATHS = [
    f"{LLAMA_ADAPTERS_FOLDER}/llama3-1_8b_finetune_socialiqa",
    f"{LLAMA_ADAPTERS_FOLDER}/llama3-1_8b_finetune_commonsense",
    f"{LLAMA_ADAPTERS_FOLDER}/llama3-1_8b_finetune_numinamath",
    f"{LLAMA_ADAPTERS_FOLDER}/llama3-1_8b_finetune_magicoder",
    f"{LLAMA_ADAPTERS_FOLDER}/llama3-1_8b_finetune_scienceqa",
    # "outputs/models/Llama-3.1-8B-merged-std/merged_adapter",
]
LLAMA_FISHER_PATHS = [
    f"{FISHERS_DIR}/llama3-1_8b_finetune_magicoder.safetensors",
    f"{FISHERS_DIR}/llama3-1_8b_finetune_numinamath.safetensors",
    f"{FISHERS_DIR}/llama3-1_8b_finetune_commonsense.safetensors",
    f"{FISHERS_DIR}/llama3-1_8b_finetune_socialiqa.safetensors",
    f"{FISHERS_DIR}/llama3-1_8b_finetune_scienceqa.safetensors",
]

# Qwen 2.5 3B and its OFT adapters fine-tuned on 5 tasks
# (social_iqa, commonsense_qa, numinamath, magicoder, science
QWEN_MODEL_NAME = "Qwen-2.5-3B"
QWEN_BASE_MODEL_PATH = f"{MODELS_DIR}/Qwen-2.5-3B"
QWEN_ADAPTERS_FOLDER = f"{MODELS_DIR}/Qwen-2.5-3B_OFT_adapters"
QWEN_ADAPTER_PATHS = [
    f"{QWEN_ADAPTERS_FOLDER}/qwen2.5_3b_finetune_socialiqa",
    f"{QWEN_ADAPTERS_FOLDER}/qwen2.5_3b_finetune_commonsense",
    f"{QWEN_ADAPTERS_FOLDER}/qwen2.5_3b_finetune_numinamath",
    f"{QWEN_ADAPTERS_FOLDER}/qwen2.5_3b_finetune_magicoder",
    f"{QWEN_ADAPTERS_FOLDER}/qwen2.5_3b_finetune_scienceqa",
    # "outputs/models/Qwen-2.5-3B-merged-std/merged_adapter",
]
QWEN_FISHER_PATHS = [
    f"{FISHERS_DIR}/qwen2.5_3b_finetune_magicoder.safetensors",
    f"{FISHERS_DIR}/qwen2.5_3b_finetune_numinamath.safetensors",
    f"{FISHERS_DIR}/qwen2.5_3b_finetune_commonsense.safetensors",
    f"{FISHERS_DIR}/qwen2.5_3b_finetune_socialiqa.safetensors",
    f"{FISHERS_DIR}/qwen2.5_3b_finetune_scienceqa.safetensors",
]

MODEL_FAMILIES = {
    "llama3.1": ModelFamily(
        name="llama3.1",
        base_model_path=LLAMA_BASE_MODEL_PATH,
        adapter_paths=LLAMA_ADAPTER_PATHS,
        fisher_paths=LLAMA_FISHER_PATHS,
    ),
    "qwen2.5": ModelFamily(
        name="qwen2.5",
        base_model_path=QWEN_BASE_MODEL_PATH,
        adapter_paths=QWEN_ADAPTER_PATHS,
        fisher_paths=QWEN_FISHER_PATHS,
    ),
}
