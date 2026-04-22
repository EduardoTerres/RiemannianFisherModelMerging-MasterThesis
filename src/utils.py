import torch

def parse_device(device_str: str) -> torch.device:
    """Parse a device string and return a torch.device object.

    Args:
        device_str: Device specification ('cpu', 'cuda', 'gpu', or 'mps')

    Returns:
        A torch.device object

    Raises:
        ValueError: If device_str is invalid or no suitable device is available
    """
    device_str = device_str.lower().strip()

    if device_str in ["gpu", "cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available")
        return torch.device("cuda:0")

    if device_str == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS requested but not available")
        return torch.device("mps")

    if device_str == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Invalid device: '{device_str}'. Choose from 'cpu', 'cuda', 'gpu', or 'mps'")
