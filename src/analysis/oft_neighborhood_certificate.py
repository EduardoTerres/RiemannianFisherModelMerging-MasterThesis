from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from safetensors.torch import load_file
from tqdm import tqdm

from src.dataset.dataset_3 import DATASET_3_PLOT_LABELS
from src.merging import OFTMerging
from src.paths import MODEL_FAMILIES, ROOTDIR


RADIUS_THRESHOLD = math.pi / 2.0
RADIUS_THRESHOLD_LABEL = r"$\pi/2$"
PAIRWISE_THRESHOLD = math.pi
ANGLE_TOL = 1e-10
FONT_SCALE = 1.5


def scaled_fontsize(size: float) -> float:
    return size * FONT_SCALE


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.weight": "bold",
    "font.size": scaled_fontsize(10),
    "axes.labelsize": scaled_fontsize(10),
    "axes.labelweight": "bold",
    "axes.titlesize": scaled_fontsize(12),
    "axes.titleweight": "bold",
    "xtick.labelsize": scaled_fontsize(10),
    "ytick.labelsize": scaled_fontsize(10),
    "legend.fontsize": scaled_fontsize(10),
})


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def performance_task_name(task: str) -> str:
    return {"numinamath": "math500"}.get(task, task)


def performance_plot_label(task: str) -> str:
    task_name = performance_task_name(task)
    return DATASET_3_PLOT_LABELS.get(task_name, task_name)


def is_oft_key(key: str) -> bool:
    lowered = key.lower()
    return ("oft_r" in lowered or "oft_" in lowered) and "classifier" not in lowered


def task_from_adapter_path(path: str) -> str:
    return Path(path).name.split("_finetune_", maxsplit=1)[-1]


def block_size_from_son_dimension(son_dimension: int) -> int:
    n = int((1 + math.sqrt(1 + 8 * son_dimension)) / 2)
    if n * (n - 1) // 2 != son_dimension:
        raise ValueError(f"{son_dimension=} is not n * (n - 1) / 2 for an integer n")
    return n


def rotation_plane_angles_from_phases(phases: torch.Tensor) -> list[float]:
    phases = torch.sort(torch.abs(phases[torch.abs(phases) > ANGLE_TOL])).values
    if phases.numel() == 0:
        return []
    if phases.numel() % 2 != 0:
        raise ValueError(f"Expected paired SO(n) eigenvalue phases, got {phases.numel()} phases")
    return phases.reshape(-1, 2).mean(dim=1).cpu().tolist()


def rotation_plane_angles(Q: torch.Tensor) -> list[float]:
    """Return principal rotation-plane angles from eigenvalue phases.

    For an SO(n) matrix, nontrivial eigenvalues occur as conjugate pairs
    exp(+/- i theta). Taking half of the squared phase sum therefore gives
    sum_j theta_j^2, including the -1 case where two phases equal pi.
    """
    return rotation_plane_angles_from_phases(torch.angle(torch.linalg.eigvals(Q)))


def block_radius_sq_via_trace(Q: torch.Tensor) -> float:
    """Return sum_j theta_j^2 for an SO(n) matrix Q via -0.5 * trace(log(Q) @ log(Q)).

    log(Q) = V diag(log eigvals) V^-1 is the (complex) skew generator of Q, and
    trace is invariant under the similarity transform, so trace(log(Q) @ log(Q))
    equals sum of squared eigenvalue logs directly -- no need to sort/pair phases.
    Each conjugate pair exp(+/- i theta) contributes log eigvals +/- i*theta, whose
    squares sum to -2*theta^2, so -0.5 * trace(...) gives sum_j theta_j^2.
    """
    log_eigvals = torch.log(torch.linalg.eigvals(Q))
    return float((-0.5 * (log_eigvals * log_eigvals).sum()).real.item())


def radius_from_angles(angles: list[float]) -> float:
    return float(math.sqrt(sum(angle * angle for angle in angles)))


def mean_block_distance_to_identity(rotations: dict[str, torch.Tensor]) -> float:
    radius_sq = 0.0
    num_blocks = 0
    for blocks in rotations.values():
        radius_sq += sum(block_radius_sq_via_trace(block) for block in blocks)
        num_blocks += blocks.shape[0]
    if num_blocks == 0:
        raise ValueError("Cannot compute mean block distance for an adapter with zero OFT blocks")
    return math.sqrt(radius_sq / num_blocks)


def mean_block_distance_between(
    rotations_a: dict[str, torch.Tensor],
    rotations_b: dict[str, torch.Tensor],
) -> float:
    radius_sq = 0.0
    num_blocks = 0
    shared_keys = sorted(set(rotations_a) & set(rotations_b))
    if set(rotations_a) != set(rotations_b):
        missing_a = sorted(set(rotations_b) - set(rotations_a))
        missing_b = sorted(set(rotations_a) - set(rotations_b))
        raise ValueError(f"OFT key mismatch: missing_a={missing_a[:3]} missing_b={missing_b[:3]}")
    for key in shared_keys:
        A_blocks = rotations_a[key]
        B_blocks = rotations_b[key]
        if A_blocks.shape != B_blocks.shape:
            raise ValueError(f"Shape mismatch for {key}: {A_blocks.shape} vs {B_blocks.shape}")
        relatives = A_blocks.transpose(-1, -2) @ B_blocks
        radius_sq += sum(block_radius_sq_via_trace(relative) for relative in relatives)
        num_blocks += A_blocks.shape[0]
    if num_blocks == 0:
        raise ValueError("Cannot compute mean block distance for adapters with zero shared OFT blocks")
    return math.sqrt(radius_sq / num_blocks)


def load_oft_rotations(adapter_path: str, merger: OFTMerging) -> dict[str, torch.Tensor]:
    state_path = Path(adapter_path) / "adapter_model.safetensors"
    if not state_path.exists():
        raise FileNotFoundError(state_path)

    state = load_file(str(state_path), device="cpu")
    rotations: dict[str, torch.Tensor] = {}
    for key, params in state.items():
        if not is_oft_key(key):
            continue
        params64 = params.detach().cpu().to(torch.float64)
        son_dimension = params64.shape[-1]
        skew = merger.oft_params_to_skew_matrix(params64, son_dimension)
        rotations[key] = torch.matrix_exp(skew).cpu()
    if not rotations:
        raise ValueError(f"No OFT tensors found in {state_path}")
    return rotations


def summarize_block(
    model_family: str,
    task: str,
    adapter_path: str,
    tensor_key: str,
    block_index: int,
    Q: torch.Tensor,
) -> dict[str, str | int | float]:
    angles = rotation_plane_angles(Q)
    radius = radius_from_angles(angles)
    margin = RADIUS_THRESHOLD - radius
    identity = torch.eye(Q.shape[0], dtype=Q.dtype)
    ortho_error = float(torch.linalg.matrix_norm(Q.T @ Q - identity, ord="fro").item())
    determinant = float(torch.linalg.det(Q).item())
    max_angle = float(max(angles)) if len(angles) else 0.0
    return {
        "model_family": model_family,
        "task": task,
        "adapter_path": adapter_path,
        "tensor_key": tensor_key,
        "block_index": block_index,
        "block_size": Q.shape[0],
        "orthogonality_error_fro": ortho_error,
        "determinant": determinant,
        "num_rotation_planes": len(angles),
        "schur_angles_json": json.dumps([float(x) for x in angles]),
        "max_angle": max_angle,
        "radius": radius,
        "margin_to_pi_over_2": margin,
        "status": "pass" if radius < RADIUS_THRESHOLD else "fail",
    }


def summarize_tensor_blocks(
    model_family: str,
    task: str,
    adapter_path: str,
    tensor_key: str,
    blocks: torch.Tensor,
) -> list[dict[str, str | int | float]]:
    identity = torch.eye(blocks.shape[-1], dtype=blocks.dtype)
    orthogonality_errors = torch.linalg.matrix_norm(
        blocks.transpose(-1, -2) @ blocks - identity, ord="fro"
    ).cpu()
    determinants = torch.linalg.det(blocks).cpu()
    phases_by_block = torch.angle(torch.linalg.eigvals(blocks)).cpu()

    rows: list[dict[str, str | int | float]] = []
    for block_index, phases in enumerate(phases_by_block):
        angles = rotation_plane_angles_from_phases(phases)
        radius = radius_from_angles(angles)
        margin = RADIUS_THRESHOLD - radius
        max_angle = float(max(angles)) if len(angles) else 0.0
        rows.append(
            {
                "model_family": model_family,
                "task": task,
                "adapter_path": adapter_path,
                "tensor_key": tensor_key,
                "block_index": block_index,
                "block_size": blocks.shape[-1],
                "orthogonality_error_fro": float(orthogonality_errors[block_index].item()),
                "determinant": float(determinants[block_index].item()),
                "num_rotation_planes": len(angles),
                "schur_angles_json": json.dumps([float(x) for x in angles]),
                "max_angle": max_angle,
                "radius": radius,
                "margin_to_pi_over_2": margin,
                "status": "pass" if radius < RADIUS_THRESHOLD else "fail",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def plot_margins(rows: list[dict[str, object]], output_dir: Path, family_name: str) -> None:
    radii = [float(row["radius"]) for row in rows]
    positive_radii = [radius for radius in radii if radius > 0.0]

    plt.figure(figsize=(7, 4))
    if positive_radii:
        bins = torch.logspace(
            math.log10(min(positive_radii)),
            math.log10(max(max(positive_radii), RADIUS_THRESHOLD)),
            steps=51,
            dtype=torch.float64,
        ).tolist()
        plt.hist(positive_radii, bins=bins)
        plt.xscale("log")
    else:
        plt.hist(radii, bins=50)
    plt.axvline(RADIUS_THRESHOLD, color="red", linestyle="--", linewidth=1, label=RADIUS_THRESHOLD_LABEL)
    plt.xlabel("Blockwise distance r")
    plt.ylabel("Count")
    plt.title(f"{family_name}: identity-to-finetune OFT block radii")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{family_name}_block_radii_hist.png", dpi=200)
    plt.close()


def classical_mds(distance_matrix: torch.Tensor, dimensions: int = 2) -> torch.Tensor:
    n = distance_matrix.shape[0]
    squared = distance_matrix.square()
    centering = torch.eye(n, dtype=torch.float64) - torch.full((n, n), 1.0 / n, dtype=torch.float64)
    gram = -0.5 * centering @ squared @ centering
    eigvals, eigvecs = torch.linalg.eigh(gram)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order][:dimensions].clamp_min(0.0)
    eigvecs = eigvecs[:, order][:, :dimensions]
    return eigvecs * torch.sqrt(eigvals)


def log_radial_transform(coords: torch.Tensor, center_index: int, reference_radius: float) -> tuple[torch.Tensor, float, float]:
    centered = coords - coords[center_index]
    radii = torch.linalg.vector_norm(centered, dim=1)
    positive = radii[radii > 0]
    if positive.numel() == 0:
        return centered, reference_radius, 1.0

    floor = max(float(positive.min().item()) / 10.0, 1e-12)
    transformed_radii = torch.log1p(radii / floor)
    scale = torch.zeros_like(radii)
    nonzero = radii > 0
    scale[nonzero] = transformed_radii[nonzero] / radii[nonzero]
    transformed = centered * scale[:, None]
    transformed_reference = math.log1p(reference_radius / floor)
    return transformed, transformed_reference, floor


def format_power_of_ten(exponent: int) -> str:
    return rf"$10^{{{exponent}}}$"


def radial_tick_specs(min_radius: float, max_radius: float) -> list[tuple[float, str | None, bool]]:
    if max_radius <= 0:
        return [(RADIUS_THRESHOLD, None, True)]

    lo = max(min_radius / 10.0, 1e-12)
    hi = max(max_radius, RADIUS_THRESHOLD)
    min_exponent = math.floor(math.log10(lo))
    max_exponent = math.ceil(math.log10(hi))

    ticks: list[tuple[float, str | None, bool]] = []
    for exponent in range(min_exponent, max_exponent + 1):
        major = 10.0**exponent
        if lo <= major <= hi:
            ticks.append((major, format_power_of_ten(exponent) if exponent < 0 else None, True))
        for mantissa in (2.0, 3.0, 5.0, 7.0):
            minor = mantissa * 10.0**exponent
            if lo <= minor <= hi:
                ticks.append((minor, None, False))
    ticks.append((RADIUS_THRESHOLD, None, True))

    deduped: dict[float, tuple[float, str | None, bool]] = {}
    for radius, label, is_major in ticks:
        key = round(radius, 12)
        current = deduped.get(key)
        if current is None or (is_major and not current[2]):
            deduped[key] = (radius, label, is_major)
    return [deduped[key] for key in sorted(deduped)]


def add_radial_marks(ax, center: tuple[float, float], min_radius: float, max_radius: float, transform_floor: float) -> None:
    ticks = radial_tick_specs(min_radius, max_radius)
    cx, cy = center
    transformed_max_radius = math.log1p(max(max_radius, RADIUS_THRESHOLD) / transform_floor)
    for radius, label, is_major in ticks:
        plot_radius = math.log1p(radius / transform_floor)
        ax.add_patch(
            plt.Circle(
                center,
                plot_radius,
                fill=False,
                color="#777777" if label is not None else "#aaaaaa",
                linestyle="--" if label is not None else ":",
                linewidth=0.8 if label is not None else 0.55,
                alpha=0.58 if label is not None else 0.28,
                zorder=0,
            )
        )
        tick_height = transformed_max_radius * 0.018
        if label is not None:
            label_angle = math.radians(-20.0)
            label_radius = plot_radius + 0.9 * tick_height
            ax.text(
                cx + label_radius * math.cos(label_angle),
                cy + label_radius * math.sin(label_angle),
                label,
                fontsize=scaled_fontsize(14),
                fontweight="bold",
                color="#333333",
                ha="left",
                va="center",
            )


def hide_cartesian_axes(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(
        axis="both",
        which="both",
        top=False,
        right=False,
        bottom=False,
        left=False,
        length=0,
        labelbottom=False,
        labelleft=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")


def distance_matrix_from_long_csv(path: Path) -> tuple[list[str], torch.Tensor]:
    rows = read_csv(path)
    labels: list[str] = []
    for row in rows:
        for key in ("source", "target"):
            label = row[key]
            if label not in labels:
                labels.append(label)

    index = {label: idx for idx, label in enumerate(labels)}
    distances = torch.zeros((len(labels), len(labels)), dtype=torch.float64)
    for row in rows:
        distances[index[row["source"]], index[row["target"]]] = float(row["distance"])
    return labels, distances


def write_distance_matrix_long(path: Path, labels: list[str], distances: torch.Tensor) -> None:
    rows = []
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            rows.append({"source": label_i, "target": label_j, "distance": float(distances[i, j])})
    write_csv(path, rows, ["source", "target", "distance"])


def render_distance_mds(
    family_name: str,
    labels: list[str],
    distances: torch.Tensor,
    output_dir: Path,
) -> dict[str, object]:
    if not labels:
        return None

    n = len(labels)
    pretrained_index = labels.index("pretrained") if "pretrained" in labels else 0
    coords = classical_mds(distances)
    coords = coords - coords[pretrained_index]
    radii = torch.linalg.vector_norm(coords - coords[pretrained_index], dim=1)
    positive_radii = radii[radii > 0]
    min_radius = float(positive_radii.min().item()) if positive_radii.numel() else RADIUS_THRESHOLD
    max_radius = float(max(radii.max().item(), RADIUS_THRESHOLD))
    plot_coords, plot_threshold, transform_floor = log_radial_transform(coords, pretrained_index, RADIUS_THRESHOLD)

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    add_radial_marks(ax, (0.0, 0.0), min_radius, max_radius, transform_floor)
    circle = plt.Circle(
        (0.0, 0.0),
        plot_threshold,
        fill=False,
        color="red",
        linestyle="--",
        linewidth=1.5,
        zorder=1,
    )
    ax.add_patch(circle)
    ax.scatter(
        [float(plot_coords[pretrained_index, 0])],
        [float(plot_coords[pretrained_index, 1])],
        s=120,
        color="black",
        marker="*",
        label="Pretrained",
        zorder=3,
    )

    finetune_indices = [idx for idx, label in enumerate(labels) if label != "pretrained"]
    finite_coords = plot_coords[finetune_indices].cpu().numpy()
    if len(finite_coords):
        ax.scatter(finite_coords[:, 0], finite_coords[:, 1], s=70, color="#2f6fbb", label="Finetunes", zorder=3)
    label_offsets = itertools.cycle(
        [(5, 5), (5, -9), (-5, 5), (-5, -9), (8, 0), (-8, 0)]
    )
    for idx in finetune_indices:
        task = labels[idx]
        x = float(plot_coords[idx, 0])
        y = float(plot_coords[idx, 1])
        x0 = float(plot_coords[pretrained_index, 0])
        y0 = float(plot_coords[pretrained_index, 1])
        ax.plot([x0, x], [y0, y], color="#777777", linewidth=0.8, alpha=0.75)
        offset_x, offset_y = next(label_offsets)
        ax.annotate(
            performance_plot_label(task),
            xy=(x, y),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=scaled_fontsize(12),
            fontweight="bold",
            ha="left" if offset_x >= 0 else "right",
            va="bottom" if offset_y >= 0 else "top",
        )

    limit = plot_threshold * 1.14 if plot_threshold else 1.0
    ax.text(
        plot_threshold + 0.018 * limit,
        0.035 * limit,
        RADIUS_THRESHOLD_LABEL,
        color="red",
        fontsize=scaled_fontsize(14),
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    hide_cartesian_axes(ax)
    ax.legend(loc="best", prop={"size": scaled_fontsize(13), "weight": "bold"})
    plt.tight_layout(pad=0.25)
    png_path = output_dir / f"{family_name}_mean_block_distance_mds.png"
    pdf_path = output_dir / f"{family_name}_mean_block_distance_mds.pdf"
    plt.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    embedded = []
    for i, label in enumerate(labels):
        embedded.append(
            {
                "label": label,
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "plot_x_log_radial": float(plot_coords[i, 0]),
                "plot_y_log_radial": float(plot_coords[i, 1]),
                "mean_block_distance_to_pretrained": float(distances[pretrained_index, i]),
            }
        )
    write_csv(
        output_dir / "mean_block_distance_mds_coordinates.csv",
        embedded,
        ["label", "x", "y", "plot_x_log_radial", "plot_y_log_radial", "mean_block_distance_to_pretrained"],
    )

    reconstructed = torch.cdist(coords, coords)
    stress_num = float(((reconstructed - distances) ** 2).sum().item())
    stress_den = float((distances.square()).sum().item())
    return {
        "distance_plot": str(png_path),
        "distance_plot_pdf": str(pdf_path),
        "num_mds_points": n,
        "mds_stress": math.sqrt(stress_num / stress_den) if stress_den else 0.0,
        "max_mean_block_distance_to_pretrained": float(distances[pretrained_index].max().item()),
        "distance_kind": "root_mean_square_block_geodesic_distance",
        "display_transform": "log_radial_about_pretrained",
        "block_radius_threshold": RADIUS_THRESHOLD,
    }


def plot_distance_mds(
    family_name: str,
    rotations_by_task: dict[str, dict[str, torch.Tensor]],
    output_dir: Path,
) -> dict[str, object] | None:
    distance_csv = output_dir / "mean_block_distance_matrix_long.csv"
    if distance_csv.exists():
        print(f"[{family_name}] loading cached MDS distance matrix: {distance_csv}", flush=True)
        labels, distances = distance_matrix_from_long_csv(distance_csv)
        return render_distance_mds(family_name, labels, distances, output_dir)

    tasks = sorted(rotations_by_task)
    if not tasks:
        return None

    labels = ["pretrained"] + tasks
    n = len(labels)
    distances = torch.zeros((n, n), dtype=torch.float64)

    print(f"[{family_name}] computing mean block distance matrix for MDS plot", flush=True)
    for i, task in tqdm(
        list(enumerate(tasks, start=1)),
        desc=f"[{family_name}] pretrained distances",
    ):
        distances[0, i] = distances[i, 0] = mean_block_distance_to_identity(
            rotations_by_task[task]
        )

    task_pairs = list(itertools.combinations(enumerate(tasks, start=1), 2))
    for (i, task_a), (j, task_b) in tqdm(
        task_pairs,
        desc=f"[{family_name}] finetune pair distances",
    ):
        dist = mean_block_distance_between(rotations_by_task[task_a], rotations_by_task[task_b])
        distances[i, j] = distances[j, i] = dist

    write_distance_matrix_long(distance_csv, labels, distances)
    return render_distance_mds(family_name, labels, distances, output_dir)

    plt.figure(figsize=(9, 4))
    plt.scatter(range(len(radii)), radii, s=4)
    plt.axhline(RADIUS_THRESHOLD, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Block index across finetune tensors")
    plt.ylabel("r")
    plt.title(f"{family_name}: blockwise distance to pretrained identity")
    plt.tight_layout()
    plt.savefig(output_dir / f"{family_name}_radii.png", dpi=200)
    plt.close()


def run_positive_control(block_size: int) -> dict[str, object]:
    theta = RADIUS_THRESHOLD + 1e-3
    Q = torch.eye(block_size, dtype=torch.float64)
    Q[0, 0] = math.cos(theta)
    Q[0, 1] = -math.sin(theta)
    Q[1, 0] = math.sin(theta)
    Q[1, 1] = math.cos(theta)
    angles = rotation_plane_angles(Q)
    radius = radius_from_angles(angles)
    return {
        "block_size": block_size,
        "target_angle": theta,
        "measured_radius": radius,
        "margin_to_pi_over_2": RADIUS_THRESHOLD - radius,
        "status": "pass" if radius < RADIUS_THRESHOLD else "expected_fail",
    }


def pairwise_rows(
    model_family: str,
    rotations_by_task: dict[str, dict[str, torch.Tensor]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task_a, task_b in itertools.combinations(sorted(rotations_by_task), 2):
        rotations_a = rotations_by_task[task_a]
        rotations_b = rotations_by_task[task_b]
        shared_keys = sorted(set(rotations_a) & set(rotations_b))
        for key in shared_keys:
            A_blocks = rotations_a[key]
            B_blocks = rotations_b[key]
            if A_blocks.shape != B_blocks.shape:
                raise ValueError(f"Shape mismatch for {task_a}, {task_b}, {key}")
            relatives = A_blocks.transpose(-1, -2) @ B_blocks
            phases_by_block = torch.angle(torch.linalg.eigvals(relatives)).cpu()
            for block_index, phases in enumerate(phases_by_block):
                angles = rotation_plane_angles_from_phases(phases)
                theta_max = float(max(angles)) if len(angles) else 0.0
                rows.append(
                    {
                        "model_family": model_family,
                        "task_a": task_a,
                        "task_b": task_b,
                        "tensor_key": key,
                        "block_index": block_index,
                        "theta_max": theta_max,
                        "margin_to_pi": PAIRWISE_THRESHOLD - theta_max,
                        "status": "pass" if theta_max < PAIRWISE_THRESHOLD else "fail",
                    }
                )
    return rows


def certify_family(family_name: str, args: argparse.Namespace) -> dict[str, object]:
    family = MODEL_FAMILIES[family_name]
    output_dir = Path(args.output_dir) / family_name
    output_dir.mkdir(parents=True, exist_ok=True)
    merger = OFTMerging(device="cpu")

    block_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    rotations_by_task: dict[str, dict[str, torch.Tensor]] = {}

    block_csv = output_dir / "blockwise_identity_certificate.csv"
    missing_csv = output_dir / "missing_adapters.csv"
    pairwise_csv = output_dir / "pairwise_finetune_uniqueness.csv"
    distance_csv = output_dir / "mean_block_distance_matrix_long.csv"
    block_fields = [
        "model_family",
        "task",
        "adapter_path",
        "tensor_key",
        "block_index",
        "block_size",
        "orthogonality_error_fro",
        "determinant",
        "num_rotation_planes",
        "schur_angles_json",
        "max_angle",
        "radius",
        "margin_to_pi_over_2",
        "status",
    ]

    adapter_paths = family.adapter_paths
    if args.max_adapters is not None:
        adapter_paths = adapter_paths[: args.max_adapters]

    needs_block_csv = not block_csv.exists()
    needs_pairwise_csv = args.pairwise and not pairwise_csv.exists()
    needs_distance_csv = not args.skip_distance_plot and not distance_csv.exists()
    needs_rotations = needs_block_csv or needs_pairwise_csv or needs_distance_csv

    if block_csv.exists():
        print(f"[{family_name}] loading cached blockwise certificate: {block_csv}", flush=True)
        block_rows = read_csv(block_csv)

    if missing_csv.exists():
        missing_rows = read_csv(missing_csv)

    if args.pairwise and pairwise_csv.exists():
        print(f"[{family_name}] loading cached pairwise uniqueness table: {pairwise_csv}", flush=True)
        pair_rows = read_csv(pairwise_csv)

    if needs_rotations:
        for adapter_path in adapter_paths:
            task = task_from_adapter_path(adapter_path)
            print(f"[{family_name}] loading {task}: {adapter_path}", flush=True)
            try:
                rotations = load_oft_rotations(adapter_path, merger)
            except FileNotFoundError as exc:
                missing_rows.append(
                    {
                        "model_family": family_name,
                        "task": task,
                        "adapter_path": adapter_path,
                        "error": str(exc),
                    }
                )
                continue

            rotations_by_task[task] = rotations
            if needs_block_csv:
                task_rows: list[dict[str, object]] = []
                for tensor_key, blocks in rotations.items():
                    task_rows.extend(
                        summarize_tensor_blocks(
                            model_family=family_name,
                            task=task,
                            adapter_path=adapter_path,
                            tensor_key=tensor_key,
                            blocks=blocks,
                        )
                    )
                block_rows.extend(task_rows)
                task_delta_min = min(float(row["margin_to_pi_over_2"]) for row in task_rows)
                print(
                    f"[{family_name}] finished {task}: "
                    f"tensors={len(rotations)} blocks={len(task_rows)} "
                    f"delta_min={task_delta_min:.6g}",
                    flush=True,
                )

    if needs_block_csv:
        write_csv(block_csv, block_rows, block_fields)

    if missing_rows and (needs_rotations or not missing_csv.exists()):
        write_csv(missing_csv, missing_rows, ["model_family", "task", "adapter_path", "error"])

    if needs_pairwise_csv:
        pair_rows = pairwise_rows(family_name, rotations_by_task)
    if pair_rows and needs_pairwise_csv:
        write_csv(
            pairwise_csv,
            pair_rows,
            ["model_family", "task_a", "task_b", "tensor_key", "block_index", "theta_max", "margin_to_pi", "status"],
        )

    if block_rows:
        plot_margins(block_rows, output_dir, family_name)

    distance_plot_summary = None
    if not args.skip_distance_plot:
        distance_plot_summary = plot_distance_mds(family_name, rotations_by_task, output_dir)

    positive_control = run_positive_control(args.positive_control_block_size)
    write_csv(
        output_dir / "positive_control.csv",
        [positive_control],
        ["block_size", "target_angle", "measured_radius", "margin_to_pi_over_2", "status"],
    )

    margins = [float(row["margin_to_pi_over_2"]) for row in block_rows]
    pair_margins = [float(row["margin_to_pi"]) for row in pair_rows]
    failed_blocks = [row for row in block_rows if row["status"] == "fail"]
    failed_pairs = [row for row in pair_rows if row["status"] == "fail"]

    decision = "certified"
    if missing_rows and not args.allow_missing:
        decision = "inconclusive_missing_adapters"
    if failed_blocks or failed_pairs:
        decision = "falsified"

    summary = {
        "model_family": family_name,
        "decision": decision,
        "num_loaded_finetunes": len(rotations_by_task),
        "num_missing_finetunes": len(missing_rows),
        "num_blocks": len(block_rows),
        "delta_min": min(margins) if margins else None,
        "worst_radius": max(float(row["radius"]) for row in block_rows) if block_rows else None,
        "num_block_violations": len(failed_blocks),
        "pairwise_checked": bool(args.pairwise),
        "num_pairwise_blocks": len(pair_rows),
        "pairwise_margin_min": min(pair_margins) if pair_margins else None,
        "num_pairwise_violations": len(failed_pairs),
        "positive_control_status": positive_control["status"],
        "distance_plot": distance_plot_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [certify_family(family_name, args) for family_name in args.model_family]
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")

    for summary in summaries:
        print(
            f"{summary['model_family']}: {summary['decision']} "
            f"delta_min={summary['delta_min']} "
            f"blocks={summary['num_blocks']} "
            f"missing={summary['num_missing_finetunes']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Certify that OFT finetunes lie in the blockwise convex normal neighborhood of the pretrained identity."
    )
    parser.add_argument(
        "--model-family",
        nargs="+",
        default=list(MODEL_FAMILIES),
        choices=list(MODEL_FAMILIES),
        help="Model families to check.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOTDIR / "outputs" / "oft_neighborhood_certificate",
        help="Directory for CSV tables, plots, and summaries.",
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Also verify finetune-to-finetune uniqueness via theta_max(A^T B) < pi.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Do not mark the family inconclusive when configured adapters are missing.",
    )
    parser.add_argument(
        "--positive-control-block-size",
        type=int,
        default=32,
        help="SO(n) size for the deliberate pi/2 violation positive control.",
    )
    parser.add_argument(
        "--max-adapters",
        type=int,
        default=None,
        help="Debug option: only check the first N configured adapters per family.",
    )
    parser.add_argument(
        "--skip-distance-plot",
        action="store_true",
        help="Skip the product-distance MDS plot and distance matrix.",
    )
    main(parser.parse_args())
