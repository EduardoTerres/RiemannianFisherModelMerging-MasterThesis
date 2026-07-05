from pathlib import Path

import yaml


def orthofuse_inference_folder_name(args):
    pair_name = getattr(args, "dataset_pair_name", None)
    pair_suffix = f"_{pair_name}" if pair_name else ""
    return (
        f"ns{args.num_inference_steps}_gs{args.guidance_scale}"
        f"_orthofuse_t{args.t}_method_{args.postprocessing_method}{pair_suffix}"
    )


def gradients_merge_mode(args):
    explicit_mode = getattr(args, "merge_mode", None)
    if explicit_mode is not None:
        return explicit_mode

    concept_fisher_path = getattr(args, "concept_fisher_path", None)
    style_fisher_path = getattr(args, "style_fisher_path", None)
    if concept_fisher_path is None and style_fisher_path is None:
        return "standard_rescaled" if getattr(args, "rescale", False) else "standard"
    if concept_fisher_path is None or style_fisher_path is None:
        raise ValueError("Both concept_fisher_path and style_fisher_path are required.")
    return "diagonal_fisher_rescaled" if getattr(args, "rescale", False) else "diagonal_fisher"


def gradients_inference_folder_name(args):
    mode = gradients_merge_mode(args)
    backend_suffix = ""
    if mode == "geodesic":
        backend_suffix = f"_{getattr(args, 'geodesic_backend', 'cayley')}"
        if getattr(args, "geodesic_use_fishers", False):
            backend_suffix += "_fisher"
    fisher_backend = getattr(args, "fisher_backend", "diagonal")
    if fisher_backend != "diagonal" and "fisher" in mode:
        backend_suffix += f"_{fisher_backend}"
    if getattr(args, "diagonal_fisher_correction_mu", None) is not None:
        backend_suffix += f"_mu{args.diagonal_fisher_correction_mu:g}"

    pair_name = getattr(args, "dataset_pair_name", None)
    pair_suffix = f"_{pair_name}" if pair_name else ""
    return (
        f"ns{args.num_inference_steps}_gs{args.guidance_scale}"
        f"_gradients_{mode}{backend_suffix}{pair_suffix}"
    )


def output_root(args):
    if args.output_dir is not None:
        return Path(args.output_dir)

    with open(args.config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if config.get("output_dir") is None:
        raise ValueError("output_dir is required either as an argument or in the config.")
    return Path(config["output_dir"])


def existing_output_path(args, inference_folder_name):
    root = output_root(args)
    if args.checkpoint_idx is not None:
        root = root / f"checkpoint-{args.checkpoint_idx}"

    folder = inference_folder_name(args)
    candidates = [
        root / folder,
        root / "samples" / folder,
    ]
    if args.version is not None:
        candidates.append(root / "samples" / folder / f"version_{args.version}")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
