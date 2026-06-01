"""
Compute mean gradients of fine-tuned OFT adapters on their training datasets
and compare them to the task vectors (adapter parameters).

From the optimality condition (Eq. 8 in Daheim et al. 2024):
    H_0 * xi_t = -grad_l_t(theta_t)    where xi_t = adapter params (task vector)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.merging import OFTMerging
from src.paths import MODEL_FAMILIES
from src.dataset.dataset_1 import DATASET_1_TRAIN, build_loader
from src.analysis.plot_utils import plot_gradient_analysis

TASK_NAMES = ["socialiqa", "commonsense", "numinamath", "magicoder", "scienceqa"]


def task_of(p: str) -> str:
    for n in TASK_NAMES:
        if n in p:
            return n
    return os.path.basename(p)


def compute_mean_gradient(model, loader, device, max_samples):
    """Accumulate batch-level gradients and average -- sufficient for the mean gradient."""
    model.eval()
    named_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    accum = {n: torch.zeros_like(p, dtype=torch.float32, device="cpu")
             for n, p in named_params.items()}
    n_batches, n_samples = 0, 0

    for batch in tqdm(loader, desc="Computing mean gradient"):
        if n_samples >= max_samples:
            break
        model.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        shift_logits = out.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        label_mask = (shift_labels != -100).float()
        shift_labels_safe = shift_labels.clone()
        shift_labels_safe[shift_labels_safe == -100] = 0

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_nll = F.nll_loss(
            log_probs.view(-1, log_probs.size(-1)),
            shift_labels_safe.view(-1),
            reduction="none",
        ).view(shift_labels.shape)
        loss = (token_nll * label_mask).sum() / label_mask.sum().clamp(min=1)
        loss.backward()

        with torch.no_grad():
            for n, p in named_params.items():
                if p.grad is not None:
                    accum[n] += p.grad.detach().float().cpu()
        n_batches += 1
        n_samples += input_ids.shape[0]

    if n_batches == 0:
        raise ValueError("No batches processed.")
    for n in accum:
        accum[n] /= n_batches
    return accum


def build_h0(fisher_paths, device="cpu"):
    """H_0 = element-wise mean of all pre-computed diagonal transported Fishers."""
    merger = OFTMerging(device=device)
    fishers = merger.load_fishers(fisher_paths)
    keys = fishers[0].keys()
    return {k: torch.stack([f[k].float() for f in fishers]).mean(0) for k in keys}


# -- per-parameter metrics ------------------------------------------------------

def _unit(t: torch.Tensor) -> torch.Tensor:
    t = t.float().flatten()
    return t / (t.norm() + 1e-12)


def stats(xi: torch.Tensor, v: torch.Tensor) -> dict:
    """Compute comparison metrics between task vector xi and direction candidate v.

    v should be the *preconditioned negative gradient* (-H0_inv * g), i.e. the
    quantity that should equal xi at the optimum.

    Returns
    -------
    cosine         : cos(xi, v)  -- scale-invariant direction
    abs_diff       : ||xi - v||   -- absolute difference (raw scale)
    rel_diff       : ||xi - v|| / ||xi||
    normed_diff    : ||x_hati - v_hat||  -- direction mismatch on unit sphere (fair scale)
    norm_v         : ||v||
    """
    xi_f, v_f = xi.float().flatten(), v.float().flatten()
    cos = float((xi_f * v_f).sum() / (xi_f.norm() * v_f.norm() + 1e-12))
    diff = xi_f - v_f
    abs_diff = float(diff.norm())
    rel_diff = abs_diff / (xi_f.norm().item() + 1e-12)
    normed_diff = float((_unit(xi_f) - _unit(v_f)).norm())
    return dict(cosine=cos, abs_diff=abs_diff, rel_diff=rel_diff,
                normed_diff=normed_diff, norm_v=float(v_f.norm()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family_name", default="qwen2.5", choices=["llama3.1", "qwen2.5"])
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--save_path", default="outputs/gradient_analysis")
    args = parser.parse_args()

    out_dir = Path(args.save_path) / args.family_name
    out_dir.mkdir(parents=True, exist_ok=True)

    family = MODEL_FAMILIES[args.family_name]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(family.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building H_0 = mean of all Fishers...")
    h0 = build_h0(family.fisher_paths, device="cpu")

    results: dict[str, dict] = {}

    for (tag, ds_path, ds_name, split, doc_fn), adapter_path in zip(
        DATASET_1_TRAIN, family.adapter_paths
    ):
        print(f"\n{'='*60}\nTask: {tag}  |  {os.path.basename(adapter_path)}\n{'='*60}")

        base = AutoModelForCausalLM.from_pretrained(
            family.base_model_path, torch_dtype=torch.float32, device_map=None
        )
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
        model.enable_adapter_layers()
        model.to(device)

        task_vectors = {
            n: p.detach().float().cpu()
            for n, p in model.named_parameters()
            if p.requires_grad and "oft" in n.lower()
        }

        loader = build_loader(
            ds_path, ds_name, split, doc_fn,
            tokenizer, args.num_samples, args.batch_size, args.max_length,
        )
        grads = compute_mean_gradient(model, loader, device, args.num_samples)
        oft_grads = {n: g for n, g in grads.items() if "oft" in n.lower()}

        per_key: dict[str, dict] = {}
        for n, xi in task_vectors.items():
            g = oft_grads.get(n)
            if g is None:
                continue

            # H_0 = I  ->  -H0_inv*g = -g
            s_raw = stats(xi, -g)

            # H_0 = avg(F)  ->  -H0_inv*g = -g / h0  (element-wise, diagonal)
            h0_key = n.replace(".default", "")
            if h0_key in h0:
                h0_diag = h0[h0_key].float()
                neg_precond = -g / (h0_diag + 1e-10)
            else:
                neg_precond = -g
            s_fisher = stats(xi, neg_precond)

            per_key[n] = dict(
                norm_tv=float(xi.float().norm()),
                norm_grad=float(g.float().norm()),
                # raw (H_0=I)
                cos_raw=s_raw["cosine"],
                abs_diff_raw=s_raw["abs_diff"],
                rel_diff_raw=s_raw["rel_diff"],
                normed_diff_raw=s_raw["normed_diff"],
                # fisher (H_0=avg F)
                cos_fisher=s_fisher["cosine"],
                abs_diff_fisher=s_fisher["abs_diff"],
                rel_diff_fisher=s_fisher["rel_diff"],
                normed_diff_fisher=s_fisher["normed_diff"],
                norm_precond=s_fisher["norm_v"],
            )

        results[tag] = per_key

        def _mean(key): return np.mean([v[key] for v in per_key.values()])
        print(f"  Layers: {len(per_key)}")
        print(f"  {'Metric':<22} {'H0=I':>8} {'H0=F_avg':>10}")
        for label, kr, kf in [
            ("cosine",       "cos_raw",         "cos_fisher"),
            ("rel diff",     "rel_diff_raw",     "rel_diff_fisher"),
            ("normed diff",  "normed_diff_raw",  "normed_diff_fisher"),
        ]:
            print(f"  {label:<22} {_mean(kr):>8.4f} {_mean(kf):>10.4f}")

        del model, base
        torch.cuda.empty_cache()

    # -- Save ------------------------------------------------------------------
    save_data = {}
    all_metrics = ["norm_tv", "norm_grad", "norm_precond",
                   "cos_raw", "cos_fisher",
                   "abs_diff_raw", "abs_diff_fisher",
                   "rel_diff_raw", "rel_diff_fisher",
                   "normed_diff_raw", "normed_diff_fisher"]
    for task, per_key in results.items():
        for m in all_metrics:
            save_data[f"{task}/{m}"] = np.array([v[m] for v in per_key.values()])
    np.savez(str(out_dir / "gradient_vs_taskvec.npz"), **save_data)
    print(f"\nResults saved to {out_dir}/gradient_vs_taskvec.npz")

    # -- Plot ------------------------------------------------------------------
    save_png = str(out_dir / "gradient_vs_taskvec.png")
    try:
        plot_gradient_analysis(results, args.family_name, save_path=save_png)
        print(f"Plot saved to {save_png}")
    except Exception as e:
        print(f"[WARNING] Plotting failed: {e}")


if __name__ == "__main__":
    main()
