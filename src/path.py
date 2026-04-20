from pathlib import Path

RELATIVE_ROOTDIR = Path("../../..")
ROOTDIR = (Path(__file__) / RELATIVE_ROOTDIR).resolve()

OFT_LLAMA_MODELS_DIR = ROOTDIR / "OrthoMerge/models"
