import argparse
from collections import Counter

import torch
from safetensors.torch import load_file

try:
    from src.diffusion.dataset_1 import CONCEPT_ADAPTERS, STYLE_ADAPTERS
except ModuleNotFoundError:
    from dataset_1 import CONCEPT_ADAPTERS, STYLE_ADAPTERS


OFT_LAYER_MARKERS = (
    "to_q_moft",
    "to_k_moft",
    "to_v_moft",
    "to_out_moft",
)
OFT_MATRIX_SUFFIXES = (".L", ".R")
OFT_SCALE_SUFFIXES = ("q_scale", "k_scale", "v_scale", "out_scale")
OFT_SCALE_TO_PROJECTION = {
    "q_scale": "to_q_moft",
    "k_scale": "to_k_moft",
    "v_scale": "to_v_moft",
    "out_scale": "to_out_moft",
}


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


def oft_adapter_name(name):
    for suffix in OFT_MATRIX_SUFFIXES + OFT_SCALE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)].rstrip(".")
    return name


def oft_layer_name(name):
    for marker in OFT_LAYER_MARKERS:
        marker_start = name.find(marker)
        if marker_start >= 0:
            return name[:marker_start].rstrip(".")
    return oft_adapter_name(name)


def oft_projection_name(name):
    for marker in OFT_LAYER_MARKERS:
        if marker in name:
            return marker
    for scale_name, projection_name in OFT_SCALE_TO_PROJECTION.items():
        if name.endswith(scale_name):
            return projection_name
    return "unknown"


def oft_matrix_side(name):
    if name.endswith(".L"):
        return "L"
    if name.endswith(".R"):
        return "R"
    return "other"


def _format_counts(counts, limit=8):
    parts = [f"{key}={value}" for key, value in counts.most_common(limit)]
    if len(counts) > limit:
        parts.append(f"... {len(counts) - limit} more")
    return ", ".join(parts) if parts else "none"


def _format_oft_geometry(counts, limit=8):
    parts = []
    for (blocks, rows, cols), count in counts.most_common(limit):
        if rows == cols:
            parts.append(f"{blocks} blocks of SO({rows})={count} tensors")
        else:
            parts.append(f"{blocks} blocks of {rows}x{cols}={count} tensors")
    if len(counts) > limit:
        parts.append(f"... {len(counts) - limit} more")
    return ", ".join(parts) if parts else "none"


def module_family(module_name):
    if "transformer_blocks" in module_name:
        return "transformer"
    if ".attn" in module_name or ".attentions." in module_name:
        return "attention"
    return "other"


def sdxl_block_name(module_name):
    return module_name.split(".", maxsplit=1)[0]


def attention_name(module_name):
    for part in module_name.split("."):
        if part in {"attn1", "attn2"}:
            return part
    return "unknown"


def print_oft_projection_details(named_tensors, prefix="[OFT load]"):
    named_tensors = [
        (name, tensor)
        for name, tensor in named_tensors
        if torch.is_tensor(tensor) and is_oft_parameter(name)
    ]
    print(f"{prefix} OFT projection details:")

    for projection_name in OFT_LAYER_MARKERS:
        projection_tensors = [
            (name, tensor)
            for name, tensor in named_tensors
            if oft_projection_name(name) == projection_name
        ]
        matrices = [
            (name, tensor)
            for name, tensor in projection_tensors
            if is_oft_matrix_parameter(name)
        ]
        scales = [
            (name, tensor)
            for name, tensor in projection_tensors
            if name.endswith(OFT_SCALE_SUFFIXES)
        ]
        host_modules = sorted({oft_layer_name(name) for name, _ in matrices})
        sdxl_blocks = Counter(sdxl_block_name(module_name) for module_name in host_modules)
        attention_types = Counter(attention_name(module_name) for module_name in host_modules)
        matrix_sides = Counter(oft_matrix_side(name) for name, _ in matrices)
        geometry = Counter(
            (tensor.shape[0], tensor.shape[-2], tensor.shape[-1])
            for _, tensor in matrices
            if tensor.ndim >= 3
        )
        scale_shapes = Counter(tuple(tensor.shape) for _, tensor in scales)
        values = sum(tensor.numel() for _, tensor in projection_tensors)

        print(
            f"{prefix}   {projection_name}: "
            f"host_modules={len(host_modules)}, "
            f"L={matrix_sides['L']}, R={matrix_sides['R']}, "
            f"scales={len(scales)}, values={values:,}"
        )
        print(f"{prefix}     geometry: {_format_oft_geometry(geometry)}")
        print(f"{prefix}     scale shapes: {_format_counts(scale_shapes)}")
        print(
            f"{prefix}     SDXL blocks: {_format_counts(sdxl_blocks)}; "
            f"attention: {_format_counts(attention_types)}"
        )


def print_oft_structure_summary(named_tensors, prefix="[OFT load]"):
    oft_matrices = [
        (name, tensor)
        for name, tensor in named_tensors
        if torch.is_tensor(tensor) and is_oft_matrix_parameter(name)
    ]
    host_modules = sorted({oft_layer_name(name) for name, _ in oft_matrices})
    adapters = {oft_adapter_name(name) for name, _ in oft_matrices}
    families = Counter(module_family(module_name) for module_name in host_modules)
    sdxl_blocks = Counter(sdxl_block_name(module_name) for module_name in host_modules)
    attention_types = Counter(attention_name(module_name) for module_name in host_modules)
    matrix_shapes = Counter(tuple(tensor.shape) for _, tensor in oft_matrices)
    matrix_dims = Counter(tuple(tensor.shape[-2:]) for _, tensor in oft_matrices if tensor.ndim >= 2)
    geometry = Counter(
        (tensor.shape[0], tensor.shape[-2], tensor.shape[-1])
        for _, tensor in oft_matrices
        if tensor.ndim >= 3
    )
    projections = Counter(oft_projection_name(name) for name, _ in oft_matrices)

    print(
        f"{prefix} OFT structure: "
        f"host_modules={len(host_modules)}, projections={len(adapters)}, "
        f"matrix_tensors={len(oft_matrices)}"
    )
    print(f"{prefix} OFT geometry: {_format_oft_geometry(geometry)}")
    print(f"{prefix} matrix dims: {_format_counts(matrix_dims)}")
    print(f"{prefix} tensor shapes: {_format_counts(matrix_shapes)}")
    print(f"{prefix} projections: {_format_counts(projections)}")
    print(f"{prefix} SDXL UNet blocks: {_format_counts(sdxl_blocks)}")
    print(f"{prefix} attention blocks: {_format_counts(attention_types)}")
    print(
        f"{prefix} module types: {_format_counts(families)}; "
        f"only_transformer={set(families) == {'transformer'}}"
    )
    print(
        f"{prefix} no input/conv-only blocks="
        f"{set(sdxl_blocks).issubset({'down_blocks', 'mid_block', 'up_blocks'})}"
    )
    print_oft_projection_details(named_tensors, prefix)
    print(f"{prefix} OFT host modules by SDXL block:")
    for block_name in ("down_blocks", "mid_block", "up_blocks"):
        block_modules = [name for name in host_modules if sdxl_block_name(name) == block_name]
        if not block_modules:
            continue
        print(f"{prefix}   {block_name} ({len(block_modules)}):")
        for module_name in block_modules:
            print(f"{prefix}     {module_name}")


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
    trainable_oft_adapters = sorted({oft_adapter_name(name) for name, _ in trainable_oft})
    print(f"[OFT load] training OFT adapters ({len(trainable_oft_adapters)}):")
    for adapter_name in trainable_oft_adapters:
        print(f"[OFT load]   {adapter_name}")
    print_oft_structure_summary(trainable_oft)

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
    print_oft_structure_summary(state_dict.items())

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


def default_adapter_path():
    return CONCEPT_ADAPTERS[0]["adapter_path"]


def main():
    parser = argparse.ArgumentParser(
        description="Print information about saved OFT/MOFT adapter checkpoints."
    )
    parser.add_argument(
        "adapter_path",
        nargs="?",
        default=default_adapter_path(),
        help="Path to one .safetensors OFT/MOFT adapter checkpoint.",
    )
    args = parser.parse_args()

    state_dict = load_file(args.adapter_path)
    print_state_dict_summary(state_dict, "adapter checkpoint", args.adapter_path)

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
