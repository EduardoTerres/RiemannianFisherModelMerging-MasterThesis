# MasterThesis — Fisher-Weighted Riemannian Merging of Orthogonal Adapters

This README documents every experiment reported in the thesis and the exact commands used to run it.

## Repository layout

| Path | Contents |
|---|---|
| `src/`, `scripts/` | This project's own code: Fisher computation, merging, evaluation, plots. |
| `OrthoFuse/` | Vendored baseline/dependency: [OrthoFuse](https://github.com/ControlGenAI/OrthoFuse) (diffusion adapter merging baseline, also provides the GSOFT/`moft` training and inference code used for the SDXL experiments). |
| `OrthoMerge/` | Vendored baseline/dependency: [OrthoMerge](https://github.com/Sphere-AI-Lab/OrthoMerge) (LLM merging baseline, and the `OrthoMerge/eval/` folder that hosts the evaluation harnesses used by every LLM experiment: `lm-evaluation-harness`, `bigcode-evaluation-harness`). |
| `TFM-4/` | LaTeX source and compiled PDF of the thesis manuscript. |
| `tests/` | Unit tests for the merging math and dataset formatting (`pytest`). |
| `utils/` | One-off conversion/download helpers (e.g. converting a raw Llama checkpoint to the HF format). |

Everything the scripts write (`outputs/`, `_outputs_d2/`, `data/`, `wandb/`, `*.out` SLURM logs, caches) is
git-ignored and not part of this delivery — only the code that produces it is.

## Before you start: fill in the path placeholders

None of the machine-specific paths from the original development machine are checked in. Instead, every
script and every default in `src/` uses one of the placeholders below. **Search-and-replace them with
real paths for your setup before running anything.**

```bash
grep -rn "/path/to/" --include="*.py" --include="*.sh" .
```

| Placeholder | What it should point to | Used for |
|---|---|---|
| `/path/to/MasterThesis` | Absolute path to this repository checkout | `REPO_ROOT`/`SCRIPT_DIR` in almost every script, and SLURM `--output`/`--chdir` directives |
| `/path/to/models` | Base model weights + finetuned OFT adapters | `MODELS_DIR` in [`src/paths.py`](src/paths.py), `MODELS_ROOT` in `scripts/eval/*.sh` |
| `/path/to/fishers` | Computed Fisher/FIM `.safetensors` files | `FISHERS_DIR` in [`src/paths.py`](src/paths.py) and `src/analysis/*.py` |
| `/path/to/scratch` | Scratch/shared storage for large intermediate merge outputs | `SAVE_DIR`/`PIPE_MERGE_ROOT` in `scripts/eval/pipe_fishermerge.sh`, `pipe_orthomerge.sh`, `scripts/analysis/compare_kl_geodist*.sh` |
| `/path/to/materialized_adapters` | Scratch cache for materialized (dense) adapters | `scripts/compute_fim.sh`, `src/compute_fim.py` |
| `/path/to/hf_cache` | Hugging Face cache (`HF_HOME`) | all `scripts/diffusion/*.sh` |
| `/path/to/sdxl_data` | Root of the SDXL/GSOFT concept & style adapters and their training datasets | [`src/diffusion/dataset_1.py`](src/diffusion/dataset_1.py), `scripts/oft_info.sh` |
| `/path/to/llama_checkpoints` | Raw (non-HF) Llama 3.1 checkpoint, only needed if you convert weights yourself | [`utils/convert_llama_weights/convert_llama_to_hf_weiths.sh`](utils/convert_llama_weights/convert_llama_to_hf_weiths.sh) |

`src/paths.py` is the single source of truth for `MODELS_DIR`/`FISHERS_DIR` used by all Python entry points;
most `.sh` scripts additionally hardcode their own `REPO_ROOT`/`MODELS_ROOT` copies for SLURM convenience, so
the `grep` above is the reliable way to catch every occurrence.

## Setup

1. **Base repo environment** (`merge` / `OrthoMerge` conda env name used throughout the scripts):
   ```bash
   conda create -n merge python=3.11 -y
   conda activate merge
   pip install -e .
   ```
2. **Base models**: download `meta-llama/Llama-3.1-8B` and `Qwen/Qwen2.5-3B` into `<models_dir>` (see
   `utils/download_weights.py` for a starting point).
3. **Finetuned OFT adapters**: either finetune them yourself (see "Merging LLM orthogonal adapters" below)
   or download the ones released with the thesis:
   [`eduardo-terres2/OFT-Llama-Qwen-Finetunes`](https://huggingface.co/eduardo-terres2/OFT-Llama-Qwen-Finetunes)
   into `<models_dir>/Llama-3.1-8B_OFT_dataset3_adapters` / `<models_dir>/Qwen-2.5-3B_OFT_dataset3_adapters`.
4. **Evaluation harnesses**: `OrthoMerge/eval/lm-evaluation-harness` and `OrthoMerge/eval/bigcode-evaluation-harness`
   are vendored copies; install each into its own conda env (`lm-eval`, `bigcode`) following their own
   `pip install -e .` instructions — every `scripts/eval/*.sh` script `conda activate`s these by name.
5. **Diffusion (SDXL) environment**: `conda env create -f OrthoFuse/environment.yml` creates `orthofuse_env`,
   used by every `scripts/diffusion/*.sh` script. GSOFT concept/style adapters and datasets for the
   diffusion experiments are available from the
   [OrthoFuse data/adapters Google Drive folder](https://drive.google.com/drive/folders/1ncJSpozSDvamOBAW7uw7PTTKzpIj1zUb?usp=sharing)
   — place them under `<sdxl_data>`.

Every pipeline below is a SLURM `sbatch` script (some can also be run directly with `bash` if you are not
on a SLURM cluster; strip the `#SBATCH` header and set `CUDA_VISIBLE_DEVICES` yourself in that case).

---

## Experiments

### 1. Loss & weight-space analysis of the finetuned LLMs

Sanity checks on the 12-task-per-backbone finetunes before merging them.

**Same loss basin** (geodesic loss interpolation between pretrained and each finetune):
```bash
sbatch scripts/analysis/loss_interpolation.sh
```
This already produces the plots (`--plots` flag); the underlying plotting code is
[`src/analysis/loss_interpolation.py`](src/analysis/loss_interpolation.py).

**Convex normal neighborhood** (per-block geodesic distances + MDS embedding):
```bash
sbatch scripts/analysis/oft_neighborhood_certificate.sh
```
Runs [`src/analysis/oft_neighborhood_certificate.py`](src/analysis/oft_neighborhood_certificate.py).

### 2. Merging LLM orthogonal adapters (main experiment)

Merges the 12 finetuned OFT adapters of Llama 3.1 8B / Qwen 2.5 3B with the diagonal Fisher method against
five baselines (Lie sum, OrthoMerge, TIES, TSVM, Wudi), then evaluates every merge on all 12 tasks.

**Step 0 — finetune the 12 per-task OFT adapters** (only needed if you are not using the released
checkpoints):
```bash
sbatch scripts/finetune_parallel.sh          # array job, one task per array index; src/finetune/finetune.py
```

**Step 1 — compute the diagonal Fisher information for each finetune**:
```bash
sbatch scripts/compute_fim.sh                # src/compute_fim.py
```

**Step 2 — merge and evaluate.** Each pipeline merges the adapters with one method and submits the
12-task evaluation array (`scripts/eval/eval.sh`, `lm-evaluation-harness`/`bigcode-evaluation-harness`):
```bash
sbatch scripts/eval/pipe_fishermerge.sh       # ours: diagonal Fisher merge     (src/scripts/perform_merging.py --merge_mode diagonal_fisher)
sbatch scripts/eval/pipe_orthomerge.sh        # OrthoMerge baseline             (--merge_mode standard_rescaled)
sbatch scripts/eval/pipe_baselines.sh         # TIES / TSVM / Wudi baselines    (--merge_method orthomerge_c_ties|orthomerge_c_tsvm|wudi)
sbatch scripts/eval/pipe_finetunes.sh         # per-task finetuned reference models
sbatch scripts/eval/pipe_pretrained.sh        # zero-shot pretrained reference
```
The "Lie sum" baseline is `--merge_mode standard` of the same `perform_merging.py` entry point (an ablation
of the diagonal Fisher formula with $\widetilde{\mathcal I}_t \equiv I$).

**Step 3 — evaluation loss** (used for the cobweb loss plots):
```bash
sbatch scripts/loss/compute_eval_loss.sh      # src/loss/compute_eval_loss.py
```

**Step 4 — build the results table**:
```bash
bash scripts/llm_table_results.sh             # src/analysis/performance_table.py
```

**Step 5 — cobweb plots**:
```bash
bash scripts/plots/plot_cowebs_all.sh         # performance + loss, both backbones
bash scripts/plots/plot_cowebs_performance.sh # performance only
bash scripts/plots/plot_cowebs_loss.sh        # loss only
```
All three call [`src/plots/plot_cowebs.py`](src/plots/plot_cowebs.py).

**Step 6 — KL divergence / geodesic distance comparison against OrthoMerge** (and the KL/performance-ratio
correlation table):
```bash
sbatch scripts/analysis/compare_kl_geodist.sh        # approximate KL via the diagonal FIM
sbatch scripts/analysis/compare_kl_geodist_exact.sh  # exact KL (more expensive, used to sanity-check the approximation)
```
Both run [`src/analysis/compare_kl_geodist.py`](src/analysis/compare_kl_geodist.py).

#### Robustness to the number of merged tasks (subsets experiment)

Merges nested task subsets of size 2, 4, 6, 8, 10, 12 with the diagonal Fisher method and OrthoMerge:
```bash
sbatch scripts/subsets/subsets_pipe.sh        # merges the subsets, submits subsets_eval.sh
sbatch scripts/subsets/subsets_eval.sh SEED MODEL_FAMILY MODELS_DIR   # (submitted automatically above)
sbatch scripts/subsets/subsets_analysis.sh    # src/subsets/subsets_analysis.py
bash scripts/subsets/subsets_plots.sh         # src/subsets/subsets_merge.py --phase results
```

### 3. Merging diffusion model orthogonal adapters (SDXL)

**Step 0 — compute the FIM for each GSOFT concept/style adapter**:
```bash
sbatch scripts/diffusion/compute_fim.sh       # diagonal FIM, src/diffusion/compute_fim.py
sbatch scripts/diffusion/compute_fim_kfac.sh  # K-FAC FIM variant
```

#### Multiple prompts benchmark

Merges each style/concept pair at $t=0.6$ with the anchored Fisher formulation (three FIM normalizations)
and the geodesic formulation, against the OrthoFuse baseline, then scores CLIP/DINO similarity over 10
prompts × 5 images:
```bash
sbatch scripts/diffusion/pipe_fisher.sh           # anchored Fisher merge, array over styles
sbatch scripts/diffusion/pipe_orthofuse.sh        # OrthoFuse baseline, array over concepts
sbatch scripts/diffusion/pipe_all.sh              # everything above end-to-end for every concept/style pair
sbatch scripts/diffusion/eval_10_prompts.sh       # CLIP/DINO scoring -> src/diffusion/eval/table.py
```
FIM-normalization dominance analysis:
```bash
sbatch scripts/diffusion/analysis/fim_contributions.sh
sbatch scripts/diffusion/analysis/fim_orthofuse_correlation.sh
sbatch scripts/diffusion/analysis/correction_tangent_visualization.sh
```
Similarity/statistics table:
```bash
sbatch scripts/diffusion/analysis/stats_sdxl.sh   # src/diffusion/analysis/sdxl_merge_stats.py
```

Preliminary ablation of the correction parameter $\mu$ in $1+\mu t(1-t)$ (confirms $\mu=4$):
```bash
sbatch scripts/diffusion/correction_hyperparam_search.sh
bash scripts/diffusion/plots/plot_correction_hyperparam_search.sh
```

#### Pareto frontier

Sweeps the interpolation parameter $t \in [0,1]$ for every method, with and without the correction, and
plots style similarity against concept similarity:
```bash
sbatch scripts/diffusion/geodesic_interpolation.sh          # generate images across the t grid
sbatch scripts/diffusion/plots/plot_geodesic_interpolation.sh
sbatch scripts/diffusion/interpolation_hyperparam_search.sh
bash scripts/diffusion/plots/plot_interpolation_hyperparam_search.sh
bash scripts/diffusion/plots/plot_pareto.sh                 # src/diffusion/eval/pareto_curves.py
```

### 4. Adapter/FIM size report

Prints approximate parameter and FIM-storage sizes for a Llama, Qwen, and diffusion OFT adapter, used to
motivate the diagonal (rather than full/K-FAC) Fisher approximation:
```bash
sbatch scripts/oft_info.sh          # src/scripts/oft_info.py -> outputs/oft_info/oft_info.txt
```

---

## Additional / exploratory scripts

These were used during development and for extra sanity checks; they are not tied to a specific numbered
result in the thesis but are kept for completeness.

| Script | Purpose |
|---|---|
| `scripts/analysis/gradient_analysis.sh`, `scripts/gradient_analysis.sh` | Compares true gradients against Fisher-approximated task vectors (`src/analysis/gradients.py`). |
| `scripts/fisher_analysis.sh` | Visualizes per-task Fisher vectors (`src/analysis/fisher_vectors.py`). |
| `scripts/interpolation_merge.sh` | Interpolates between the pretrained model and a merged model (`src/analysis/interpolation_merge.py`). |
| `scripts/merging.sh`, `scripts/merging_all.sh` | Ad hoc single-config merge/eval runs (`src/scripts/perform_merging.py`), including AdaMerging-optimized alphas. |
| `scripts/plots/plot_mergeschema.sh`, `plot_mergeschema_fim.sh` | 2D loss-landscape grid over a pair of task vectors (`src/plots/plot_mergeschema*.py`). |
| `scripts/visualization.sh` | Visualization for the earlier 5-task pilot merge (`src/dataset/dataset_1.py`, distinct from the 12-task experiment above). |
| `utils/convert_llama_weights/` | One-off script to convert a raw Llama checkpoint to the HF format. |

## Tests

```bash
pytest tests/
```
