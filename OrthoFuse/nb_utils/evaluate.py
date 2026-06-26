import json
import os
from argparse import ArgumentParser

import torch

from .clip_eval import ExpEvaluator
from .experiments_viewer import ExpsViewer
from .cache import Cache, DistributedCache


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--gpu",
        type=int,
        required=True,
        help='GPU device'
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=6,
        required=False,
        help='Number of parallel threads to perform evaluation'
    )
    parser.add_argument(
        '--exp_names',
        type=str,
        nargs='+',
        required=True,
        help='Target experiments'
    )
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help='Path to the folder with experiments'
    )
    parser.add_argument(
        "--with_segmentation",
        action="store_true",
        default=False,
        help='Whether to calculate IS/TS with masking'
    )
    parser.add_argument(
        "--checkpoints_idxs",
        type=int,
        nargs='+',
        required=True,
        help='Target checkpoint idxs'
    )
    parser.add_argument(
        "--num_inference_steps",
        type=str,
        default="50",
        required=False,
        help="Inference step count used in the samples folder name",
    )
    parser.add_argument(
        "--guidance_scale",
        type=str,
        default="6.0",
        required=False,
        help="Guidance scale used in the samples folder name",
    )
    parser.add_argument(
        "--cache_files_template",
        type=str,
        default='./diffusers/examples/*/training-runs/*/evaluate.cache',
        required=False,
        help='Template for existing cache files'
    )
    parser.add_argument(
        "--summary_output_path",
        type=str,
        default=None,
        required=False,
        help="Optional JSON summary path. Defaults to <base_path>/eval_summary.json",
    )
    parser.add_argument(
        "--beta",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--tau",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default=os.environ.get("ORTHOFUSE_CLIP_MODEL", "ViT-B/32"),
        help="CLIP model name used for image/text similarity",
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default=os.environ.get("ORTHOFUSE_CLIP_PRETRAINED", "openai"),
        help="open_clip pretrained tag or a local checkpoint path",
    )
    parser.add_argument(
        "--dino_model",
        type=str,
        default=os.environ.get("ORTHOFUSE_DINO_MODEL", "dinov2_vits14"),
        help="DINOv2 torch.hub model name",
    )
    parser.add_argument(
        "--dino_repo",
        type=str,
        default=os.environ.get("ORTHOFUSE_DINO_REPO", "facebookresearch/dinov2"),
        help="DINOv2 torch.hub repo or local repo path",
    )
    parser.add_argument(
        "--dino_source",
        type=str,
        choices=("github", "local"),
        default=os.environ.get("ORTHOFUSE_DINO_SOURCE", "github"),
        help="Use 'local' when --dino_repo points to a local torch.hub repo checkout",
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    device = torch.device('cuda', args.gpu)
    print(f'Evaluating for {args.base_path}')
    print(f'Run evaluation for names: {args.exp_names} for checkpoints: {args.checkpoints_idxs}')

    evaluator = ExpEvaluator(
        device,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        dino_model=args.dino_model,
        dino_repo=args.dino_repo,
        dino_source=args.dino_source,
    )
    all_cache = DistributedCache(args.cache_files_template).get()

    exps_viewer = ExpsViewer(
        base_path=args.base_path,
        exp_filter_fn=lambda x: x in args.exp_names,
        ncolumns=6, lazy_load=True, evaluator=evaluator
    )
    summary = {}
    for checkpoint_idx in args.checkpoints_idxs:
        if args.beta is not None:
            inference_specs = (args.num_inference_steps, args.guidance_scale, args.beta, args.tau)
        else:
            inference_specs = (args.num_inference_steps, args.guidance_scale)
        stats = exps_viewer.evaluate(
            exps_names=args.exp_names,
            checkpoint_idx=str(checkpoint_idx),
            inference_specs=inference_specs,
            cache=all_cache,
            processes=args.processes,
        )
        for key, value in stats.items():
            if 'config' not in value:
                continue
            exp_cache = Cache(os.path.join(value['config']['output_dir'], 'evaluate.cache'))
            exp_cache.update({key: value})
            summary[str(key)] = value

    summary_output_path = args.summary_output_path or os.path.join(args.base_path, 'eval_summary.json')
    os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)
    with open(summary_output_path, 'w') as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(f'Saved evaluation summary to {summary_output_path}', flush=True)
