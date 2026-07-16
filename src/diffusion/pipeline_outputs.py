from pathlib import Path

import yaml


def canonical_gradients_mode(mode):
    if mode == "diagonal_fisher":
        return "fisher"
    if mode == "diagonal_fisher_rescaled":
        return "fisher_rescaled"
    return mode


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
        return canonical_gradients_mode(explicit_mode)

    concept_fisher_path = getattr(args, "concept_fisher_path", None)
    style_fisher_path = getattr(args, "style_fisher_path", None)
    if concept_fisher_path is None and style_fisher_path is None:
        return "standard_rescaled" if getattr(args, "rescale", False) else "standard"
    if concept_fisher_path is None or style_fisher_path is None:
        raise ValueError("Both concept_fisher_path and style_fisher_path are required.")
    return "fisher_rescaled" if getattr(args, "rescale", False) else "fisher"


def gradients_inference_folder_name(args):
    mode = gradients_merge_mode(args)
    backend_suffix = ""
    fisher_backend = getattr(args, "fisher_backend", "diagonal")
    if mode == "geodesic":
        backend_suffix = f"_{getattr(args, 'geodesic_backend', 'cayley')}"
        if getattr(args, "geodesic_use_fishers", False):
            backend_suffix += "_fisher"
            if fisher_backend != "diagonal":
                backend_suffix += f"_{fisher_backend}"
    elif fisher_backend != "diagonal" and "fisher" in mode:
        backend_suffix += f"_{fisher_backend}"
    correction_mu = getattr(args, "fisher_correction_mu", None)
    if correction_mu is None:
        correction_mu = getattr(args, "diagonal_fisher_correction_mu", None)
    if correction_mu is not None:
        backend_suffix += f"_mu{correction_mu:g}"
    geodesic_correction_mu = getattr(args, "correction_mu", None)
    if mode == "geodesic" and geodesic_correction_mu is not None:
        backend_suffix += f"_corr{geodesic_correction_mu:g}"
    fim_normalization = getattr(args, "fim_normalization", None)
    if mode == "geodesic" and getattr(args, "geodesic_use_fishers", False):
        if fim_normalization is not None:
            backend_suffix += f"_fim_{fim_normalization}"
    elif "fisher" in mode and fim_normalization not in {None, "none"}:
        backend_suffix += f"_fim_{fim_normalization}"

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
