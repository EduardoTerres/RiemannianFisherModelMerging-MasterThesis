from __future__ import annotations

import os
from typing import List, Optional

import matplotlib.pyplot as plt


def plot_interpolation_curve(
    alphas: List[float],
    losses: List[float],
    title: str = "Loss interpolation",
    save_path: Optional[str] = None,
    xlabel: str = "α (0 = pretrained, 1 = fine-tuned)",
    ylabel: str = "Cross-entropy loss",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(alphas, losses, marker="o", linewidth=2, markersize=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
