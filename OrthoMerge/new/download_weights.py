from huggingface_hub import snapshot_download
import os

# ADAPTERS_URLS = [
#     "https://huggingface.co/SphereLab/OrthoMerge_Llama-3.1-8B_OFT_5_task",
#     "https://huggingface.co/SphereLab/OrthoMerge-C-TSVM_Llama-3.2-3B",
#     "https://huggingface.co/SphereLab/OrthoMerge-C-TIES_Qwen2-5-VL-7B-Instruct",
#     "https://huggingface.co/SphereLab/OrthoMerge-C-TA_Llama-3.2-3B",
#     "https://huggingface.co/SphereLab/OrthoMerge-G-TA_Llama-3.2-3B",
#     "https://huggingface.co/SphereLab/OrthoMerge-G-TSVM_Llama-3.2-3B",
#     "https://huggingface.co/SphereLab/OrthoMerge-G-TIES_Llama-3.2-3B",
#     "https://huggingface.co/SphereLab/Llama-3.1-8B_OFT_adapters",
#     "https://huggingface.co/SphereLab/OrthoMerge-C-TIES_Llama-3.2-3B",
#     "https://huggingface.co/SphereLab/OrthoMerge-C-TA_Qwen2-5-VL-7B-Instruct",
#     "https://huggingface.co/SphereLab/OrthoMerge-C-TSVM_Qwen2-5-VL-7B-Instruct",
#     "https://huggingface.co/SphereLab/OrthoMerge-G-TA_Qwen2-5-VL-7B-Instruct",
#     "https://huggingface.co/SphereLab/OrthoMerge-G-TIES_Qwen2-5-VL-7B-Instruct",
#     "https://huggingface.co/SphereLab/OrthoMerge-G-TSVM_Qwen2-5-VL-7B-Instruct",
# ]

ADAPTERS_URLS = [
    # "https://huggingface.co/SphereLab/OrthoMerge_Llama-3.1-8B_OFT_5_task",
    # "https://huggingface.co/SphereLab/Llama-3.1-8B_OFT_adapters",
]

LLAMA_MODEL = "https://huggingface.co/meta-llama/Llama-3.1-8B"
# LLAMA_MODEL = "https://huggingface.co/VityaVitalich/Llama3.1-8b"


os.makedirs("models", exist_ok=True)

# Download adapters
for url in ADAPTERS_URLS:
    repo_id = url.split("huggingface.co/")[1]
    local_dir = f"models/{repo_id.split('/')[-1]}"

    if os.path.exists(local_dir):
        print(f"Skipping {repo_id}, already downloaded")
        continue

    snapshot_download(repo_id=repo_id, local_dir=local_dir)

    snapshot_download(repo_id=repo_id, local_dir=f"models/{repo_id.split('/')[-1]}")

# Download Llama-3.1-8B model
repo_id = LLAMA_MODEL.split("huggingface.co/")[1]
snapshot_download(repo_id=repo_id, local_dir="models/Llama-3.1-8B")
