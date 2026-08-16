from dataclasses import dataclass
from pathlib import Path

from src.dataset.dataset_1 import DATASET_1_TRAIN
from src.dataset.dataset_3 import DATASET_3_TRAIN

RELATIVE_ROOTDIR = Path("../..")
ROOTDIR = (Path(__file__) / RELATIVE_ROOTDIR).resolve()

WANDB_PROJECT = "fisher-orthomerge"

@dataclass(frozen=True)
class ModelFamily:
    name: str
    base_model_path: str
    adapter_paths: list
    fisher_finetuned_paths: list
    fisher_pretrained_paths: list
    default_fisher: str = "finetuned"

    @property
    def fisher_paths(self):
        if self.default_fisher == "pretrained":
            return self.fisher_pretrained_paths
        return self.fisher_finetuned_paths

    def __str__(self):
        return self.name

# MODELS_DIR = ROOTDIR / "data" / "models"
MODELS_DIR = Path("/path/to/models")
FISHERS_DIR = Path("/path/to/fishers")

ADAPTER_TASK_NAMES = {
    "social_iqa": "socialiqa",
    "commonsense_qa": "commonsense",
    "science_qa": "scienceqa",
}


def adapter_task_name(task: str) -> str:
    return ADAPTER_TASK_NAMES.get(task, task)


def fisher_path(model_family: str, task: str, adapter_tag: str, model_state: str) -> str:
    return f"{FISHERS_DIR}/{model_family}/{task}/{adapter_tag}_{model_state}.safetensors"


def with_default_fisher(families: dict[str, ModelFamily], default_fisher: str) -> dict[str, ModelFamily]:
    return {
        name: ModelFamily(
            name=family.name,
            base_model_path=family.base_model_path,
            adapter_paths=family.adapter_paths,
            fisher_finetuned_paths=family.fisher_finetuned_paths,
            fisher_pretrained_paths=family.fisher_pretrained_paths,
            default_fisher=default_fisher,
        )
        for name, family in families.items()
    }

# LLAMA WITH DATASET 1
LLAMA_MODEL_NAME = "Llama-3.1-8B"
LLAMA_BASE_MODEL_PATH = f"{MODELS_DIR}/Llama-3.1-8B"
LLAMA_D1_TASKS = [tag for tag, *_ in DATASET_1_TRAIN]
LLAMA_D1_ADAPTERS_FOLDER = f"{MODELS_DIR}/Llama-3.1-8B_OFT_dataset1_adapters"
LLAMA_D1_ADAPTER_PATHS = [
    f"{LLAMA_D1_ADAPTERS_FOLDER}/llama3-1_8b_finetune_{adapter_task_name(task)}"
    for task in LLAMA_D1_TASKS
]
LLAMA_D1_FISHER_PATHS = [
    fisher_path(
        "llama3.1", task, f"llama3-1_8b_finetune_{adapter_task_name(task)}", "finetuned"
    )
    for task in LLAMA_D1_TASKS
]
LLAMA_D1_PRETRAINED_FISHER_PATHS = [
    fisher_path(
        "llama3.1", task, f"llama3-1_8b_finetune_{adapter_task_name(task)}", "pretrained"
    )
    for task in LLAMA_D1_TASKS
]

# LLAMA WITH DATASET 3
LLAMA_D3_ADAPTERS_FOLDER = f"{MODELS_DIR}/Llama-3.1-8B_OFT_dataset3_adapters"
LLAMA_D3_TASKS = [tag for tag, *_ in DATASET_3_TRAIN]
LLAMA_D3_ADAPTER_PATHS = [
    f"{LLAMA_D3_ADAPTERS_FOLDER}/llama3-1_8b_finetune_{adapter_task_name(task)}"
    for task in LLAMA_D3_TASKS
]
LLAMA_D3_FISHER_PATHS = [
    fisher_path(
        "llama3.1", task, f"llama3-1_8b_finetune_{adapter_task_name(task)}", "finetuned"
    )
    for task in LLAMA_D3_TASKS
]
LLAMA_D3_PRETRAINED_FISHER_PATHS = [
    fisher_path(
        "llama3.1", task, f"llama3-1_8b_finetune_{adapter_task_name(task)}", "pretrained"
    )
    for task in LLAMA_D3_TASKS
]

# QWEN WITH DATASET 1
QWEN_MODEL_NAME = "Qwen-2.5-3B"
QWEN_BASE_MODEL_PATH = f"{MODELS_DIR}/Qwen-2.5-3B"
QWEN_D1_TASKS = [tag for tag, *_ in DATASET_1_TRAIN]
QWEN_D1_ADAPTERS_FOLDER = f"{MODELS_DIR}/Qwen-2.5-3B_OFT_adapters"
QWEN_D1_ADAPTER_PATHS = [
    f"{QWEN_D1_ADAPTERS_FOLDER}/qwen2.5_3b_finetune_{adapter_task_name(task)}"
    for task in QWEN_D1_TASKS
]
QWEN_D1_FISHER_PATHS = [
    fisher_path(
        "qwen2.5", task, f"qwen2.5_3b_finetune_{adapter_task_name(task)}", "finetuned"
    )
    for task in QWEN_D1_TASKS
]
QWEN_D1_PRETRAINED_FISHER_PATHS = [
    fisher_path(
        "qwen2.5", task, f"qwen2.5_3b_finetune_{adapter_task_name(task)}", "pretrained"
    )
    for task in QWEN_D1_TASKS
]

# QWEN WITH DATASET 3
QWEN_D3_ADAPTERS_FOLDER = f"{MODELS_DIR}/Qwen-2.5-3B_OFT_dataset3_adapters"
QWEN_D3_TASKS = [tag for tag, *_ in DATASET_3_TRAIN]
QWEN_D3_ADAPTER_PATHS = [
    f"{QWEN_D3_ADAPTERS_FOLDER}/qwen2.5_3b_finetune_{adapter_task_name(task)}"
    for task in QWEN_D3_TASKS
]
QWEN_D3_FISHER_PATHS = [
    fisher_path(
        "qwen2.5", task, f"qwen2.5_3b_finetune_{adapter_task_name(task)}", "finetuned"
    )
    for task in QWEN_D3_TASKS
]
QWEN_D3_PRETRAINED_FISHER_PATHS = [
    fisher_path(
        "qwen2.5", task, f"qwen2.5_3b_finetune_{adapter_task_name(task)}", "pretrained"
    )
    for task in QWEN_D3_TASKS
]

# MODEL FAMILIES for importing
MODEL_FAMILIES_D1 = {
    "llama3.1": ModelFamily(
        name="llama3.1",
        base_model_path=LLAMA_BASE_MODEL_PATH,
        adapter_paths=LLAMA_D1_ADAPTER_PATHS,
        fisher_finetuned_paths=LLAMA_D1_FISHER_PATHS,
        fisher_pretrained_paths=LLAMA_D1_PRETRAINED_FISHER_PATHS,
    ),
    "qwen2.5": ModelFamily(
        name="qwen2.5",
        base_model_path=QWEN_BASE_MODEL_PATH,
        adapter_paths=QWEN_D1_ADAPTER_PATHS,
        fisher_finetuned_paths=QWEN_D1_FISHER_PATHS,
        fisher_pretrained_paths=QWEN_D1_PRETRAINED_FISHER_PATHS,
    ),
}

MODEL_FAMILIES_D3 = {
    "llama3.1": ModelFamily(
        name="llama3.1",
        base_model_path=LLAMA_BASE_MODEL_PATH,
        adapter_paths=LLAMA_D3_ADAPTER_PATHS,
        fisher_finetuned_paths=LLAMA_D3_FISHER_PATHS,
        fisher_pretrained_paths=LLAMA_D3_PRETRAINED_FISHER_PATHS,
    ),
    "qwen2.5": ModelFamily(
        name="qwen2.5",
        base_model_path=QWEN_BASE_MODEL_PATH,
        adapter_paths=QWEN_D3_ADAPTER_PATHS,
        fisher_finetuned_paths=QWEN_D3_FISHER_PATHS,
        fisher_pretrained_paths=QWEN_D3_PRETRAINED_FISHER_PATHS,
    ),
}

MODEL_FAMILIES_D3_FISHER_FINETUNES = with_default_fisher(MODEL_FAMILIES_D3, "finetuned")
MODEL_FAMILIES_D3_FISHER_PRETRAINED = with_default_fisher(MODEL_FAMILIES_D3, "pretrained")
MODEL_FAMILIES = MODEL_FAMILIES_D3_FISHER_FINETUNES
