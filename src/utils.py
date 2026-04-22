import torch

def parse_device(device_str: str) -> str:
    device_str = device_str.lower().strip()

    if device_str in ["gpu", "cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available")
        return "cuda"

    if device_str == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS requested but not available")
        return "mps"

    if device_str == "cpu":
        return "cpu"

    raise ValueError(f"Invalid device: '{device_str}'. Choose from 'cpu', 'cuda', 'gpu', or 'mps'")
