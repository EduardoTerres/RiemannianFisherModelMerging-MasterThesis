import argparse
from collections import Counter

import torch
from safetensors.torch import load_file


OFT_LAYER_MARKERS = (
    "to_q_moft",
    "to_k_moft",
    "to_v_moft",
    "to_out_moft",
)
OFT_MATRIX_SUFFIXES = (".L", ".R")
OFT_SCALE_SUFFIXES = ("q_scale", "k_scale", "v_scale", "out_scale")


def _numel(parameters):
    return sum(parameter.numel() for parameter in parameters)


def is_oft_matrix_parameter(name):
    return any(marker in name for marker in OFT_LAYER_MARKERS) and name.endswith(
        OFT_MATRIX_SUFFIXES
    )


def is_oft_parameter(name):
    return is_oft_matrix_parameter(name) or any(
        name.endswith(scale_name) for scale_name in OFT_SCALE_SUFFIXES
    )


def print_parameter_loading_summary(module, label):
    named_parameters = list(module.named_parameters())
    trainable = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
    trainable_oft_matrices = [
        (name, parameter)
        for name, parameter in trainable
        if is_oft_matrix_parameter(name)
    ]
    trainable_oft = [
        (name, parameter)
        for name, parameter in trainable
        if is_oft_parameter(name)
    ]
    trainable_non_oft = [
        (name, parameter)
        for name, parameter in trainable
        if not is_oft_parameter(name)
    ]

    print(f"[OFT load] {label}")
    print(
        "[OFT load] parameters: "
        f"total={_numel(parameter for _, parameter in named_parameters):,}, "
        f"active={_numel(parameter for _, parameter in trainable):,}"
    )
    print(
        "[OFT load] active OFT parameters: "
        f"all={_numel(parameter for _, parameter in trainable_oft):,}, "
        f"L/R matrices={_numel(parameter for _, parameter in trainable_oft_matrices):,}, "
        f"non-OFT={_numel(parameter for _, parameter in trainable_non_oft):,}"
    )

    if trainable_non_oft:
        print("[OFT load] WARNING: active non-OFT parameters detected:")
        for name, parameter in trainable_non_oft[:20]:
            print(f"[OFT load]   {name}: shape={tuple(parameter.shape)}")
        if len(trainable_non_oft) > 20:
            print(f"[OFT load]   ... {len(trainable_non_oft) - 20} more")
    else:
        print("[OFT load] all active parameters belong to the OFT/MOFT adapter.")


def print_state_dict_summary(state_dict, label, path=None):
    keys = list(state_dict.keys())
    key_types = Counter()
    total_values = 0

    for key, tensor in state_dict.items():
        if key.endswith(".L"):
            key_types["L"] += 1
        elif key.endswith(".R"):
            key_types["R"] += 1
        elif key.endswith(OFT_SCALE_SUFFIXES):
            key_types["scale"] += 1
        else:
            key_types["other"] += 1
        if torch.is_tensor(tensor):
            total_values += tensor.numel()

    print(f"[OFT load] loading {label}")
    if path is not None:
        print(f"[OFT load] path: {path}")
    print(
        "[OFT load] checkpoint tensors: "
        f"total_keys={len(keys)}, values={total_values:,}, "
        f"L={key_types['L']}, R={key_types['R']}, "
        f"scale={key_types['scale']}, other={key_types['other']}"
    )

    if key_types["L"] == 0 or key_types["R"] == 0:
        print("[OFT load] WARNING: checkpoint does not look like an OFT/MOFT adapter.")


def print_unet_oft_summary(unet, label):
    processor_type_counts = Counter(
        type(processor).__name__ for processor in unet.attn_processors.values()
    )
    oft_processor_count = sum(
        1
        for processor in unet.attn_processors.values()
        if any(hasattr(processor, marker) for marker in OFT_LAYER_MARKERS)
    )

    print(f"[OFT load] {label}: attention processors={len(unet.attn_processors)}")
    print(f"[OFT load] processor types: {dict(processor_type_counts)}")
    print(f"[OFT load] processors with OFT/MOFT layers={oft_processor_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Print information about saved OFT/MOFT adapter checkpoints."
    )
    parser.add_argument(
        "adapter_paths",
        nargs="+",
        help="Path(s) to .safetensors OFT/MOFT adapter checkpoints.",
    )
    args = parser.parse_args()

    for adapter_path in args.adapter_paths:
        state_dict = load_file(adapter_path)
        print_state_dict_summary(state_dict, "adapter checkpoint", adapter_path)

        active_oft_values = 0
        active_oft_matrix_values = 0
        non_oft_values = 0
        for name, tensor in state_dict.items():
            if not torch.is_tensor(tensor):
                continue
            if is_oft_parameter(name):
                active_oft_values += tensor.numel()
            else:
                non_oft_values += tensor.numel()
            if is_oft_matrix_parameter(name):
                active_oft_matrix_values += tensor.numel()

        print(
            "[OFT load] checkpoint OFT values: "
            f"all={active_oft_values:,}, "
            f"L/R matrices={active_oft_matrix_values:,}, "
            f"non-OFT={non_oft_values:,}"
        )


if __name__ == "__main__":
    main()
