from pathlib import Path

RELATIVE_ROOTDIR = Path("../..")
ROOTDIR = (Path(__file__) / RELATIVE_ROOTDIR).resolve()

MODELS_DIR = ROOTDIR / "OrthoMerge/models"

# Llama 3.1 8B and its OFT adapters fine-tuned on 5 tasks
# (social_iqa, commonsense_qa, numinamath, magicoder, science_qa).
LLAMA_BASE_MODEL_PATH = f"{MODELS_DIR}/Llama-3.1-8B"
LLAMA_ADAPTERS_FOLDER = f"{MODELS_DIR}/Llama-3.1-8B_OFT_adapters"
LLAMA_ADAPTER_PATHS = [
    f"{LLAMA_ADAPTERS_FOLDER}/llama3-1_8b_finetune_socialiqa",
    f"{LLAMA_ADAPTERS_FOLDER}/llama3-1_8b_finetune_commonsense",
    f"{LLAMA_ADAPTERS_FOLDER}/llama3-1_8b_finetune_numinamath",
    f"{LLAMA_ADAPTERS_FOLDER}/llama3-1_8b_finetune_magicoder",
    f"{LLAMA_ADAPTERS_FOLDER}/llama3-1_8b_finetune_scienceqa",
]
