import os
import argparse
import json
import subprocess
from datetime import datetime

import torch
import torch.distributed as dist

from src.engine.clg_trainer import CLGTrainer
from src.configs.paths import ensure_outputs_logs, get_by_dotted_key, load_simple_yaml, resolve_repo_path


def init_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
    return rank, world_size, local_rank


def _git_info(repo_root: str) -> dict:
    def _run(args: list[str]) -> str:
        try:
            out = subprocess.check_output(args, cwd=repo_root, stderr=subprocess.DEVNULL)
            return out.decode("utf-8").strip()
        except Exception:
            return ""

    sha = _run(["git", "rev-parse", "HEAD"])
    dirty = bool(_run(["git", "status", "--porcelain"]))
    return {"sha": sha, "dirty": dirty}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_simple_yaml(os.path.join(project_root, "configs", "paths.yaml"))
    parser.add_argument(
        "--sc_dir",
        type=str,
        default=resolve_repo_path(get_by_dotted_key(cfg, "local.data.sc_connectome_schaefer400")),
    )
    parser.add_argument(
        "--morph_root",
        type=str,
        default=resolve_repo_path(get_by_dotted_key(cfg, "local.data.morphology")),
    )
    parser.add_argument(
        "--subject_info_csv",
        type=str,
        default=resolve_repo_path(get_by_dotted_key(cfg, "local.data.subject_info_sc")),
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=resolve_repo_path(get_by_dotted_key(cfg, "local.outputs.clg_ode")),
    )
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--lambda_kl", type=float, default=1e-4)
    parser.add_argument("--lambda_weight", type=float, default=1.0)
    parser.add_argument(
        "--edge_loss",
        type=str,
        default="bce",
        choices=["bce", "focal"],
        help="Edge existence loss type for p_hat supervision.",
    )
    parser.add_argument(
        "--edge_pos_weight",
        type=float,
        default=5.0,
        help="Positive class weight for BCE (ignored for focal).",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma (only for edge_loss=focal).",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=-1.0,
        help="Focal loss alpha in [0,1]; use -1 to disable (only for edge_loss=focal).",
    )
    parser.add_argument(
        "--lambda_full_log_mse",
        type=float,
        default=0.0,
        help="Weight for full-edge log-domain MSE to align with eval metrics.",
    )
    parser.add_argument(
        "--lambda_zero_log",
        type=float,
        default=0.0,
        help="Weight for zero-edge log penalty to suppress false positives.",
    )
    parser.add_argument(
        "--zero_log_warmup_epochs",
        type=int,
        default=0,
        help="Warmup epochs before enabling the zero-edge log penalty.",
    )
    parser.add_argument(
        "--zero_log_ramp_epochs",
        type=int,
        default=0,
        help="Linear ramp epochs for the zero-edge log penalty after warmup.",
    )
    parser.add_argument(
        "--lambda_delta_log",
        type=float,
        default=0.0,
        help="Weight for log-space residual shrinkage toward identity.",
    )
    parser.add_argument(
        "--lambda_density",
        type=float,
        default=0.0,
        help="Weight for expected edge-count constraint on p_hat (soft mask).",
    )
    parser.add_argument(
        "--density_warmup_epochs",
        type=int,
        default=0,
        help="Warmup epochs before enabling the density/edge-count constraint.",
    )
    parser.add_argument(
        "--density_ramp_epochs",
        type=int,
        default=0,
        help="Linear ramp epochs for the density/edge-count constraint after warmup.",
    )
    parser.add_argument(
        "--compute_mask_auprc",
        action="store_true",
        help="Compute mask AUPRC (average precision) for p_hat diagnostics (slower).",
    )
    parser.add_argument("--lambda_manifold", type=float, default=1.0)
    parser.add_argument("--lambda_vel", type=float, default=0.2)
    parser.add_argument("--lambda_acc", type=float, default=0.1)
    parser.add_argument("--warmup_manifold_epochs", type=int, default=10)
    parser.add_argument("--warmup_vel_epochs", type=int, default=20)
    parser.add_argument("--morph_noise_sigma", type=float, default=0.05)
    parser.add_argument("--sc_pos_edge_drop_prob", type=float, default=0.02)
    parser.add_argument("--lambda_topo", type=float, default=1e-3)
    parser.add_argument("--topo_loss_bins", type=int, default=8)
    parser.add_argument("--betti_sharpness", type=float, default=20.0)
    parser.add_argument("--betti_t", type=float, default=10.0)
    parser.add_argument("--betti_taylor_order", type=int, default=20)
    parser.add_argument("--betti_probes", type=int, default=2)
    parser.add_argument("--topo_scale_quantile", type=float, default=0.9)
    parser.add_argument(
        "--disable_topo_log_compress",
        action="store_true",
        help="Disable log1p compression for topo loss scaling.",
    )
    parser.add_argument(
        "--topo_warmup_frac",
        type=float,
        default=0.2,
        help="Cosine warmup fraction for topo loss weight (0-1).",
    )
    parser.add_argument(
        "--gradnorm_scope",
        type=str,
        default="manifold_topo",
        choices=["manifold_topo", "none"],
        help="GradNorm scope for balancing losses.",
    )
    parser.add_argument("--topo_bins", type=int, default=32)
    parser.add_argument("--adjacent_pair_prob", type=float, default=0.7)
    parser.add_argument(
        "--residual_skip",
        action="store_true",
        help="Enable log-space residual skip for SC prediction.",
    )
    parser.add_argument(
        "--residual_tau",
        type=float,
        default=1.0,
        help="Time-scale for residual skip (dt/(dt+tau)).",
    )
    parser.add_argument(
        "--residual_cap",
        type=float,
        default=0.5,
        help="Max magnitude for log-space residual (tanh cap).",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Optional run directory name under results_dir/runs (default: <timestamp>_job<id>).",
    )
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument(
        "--cv_fold",
        type=int,
        default=-1,
        help="Train a single CV fold (0-based). Use -1 to run all folds.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="Optional checkpoint path to resume a single fold training.",
    )
    parser.add_argument(
        "--disable_s_mean",
        action="store_true",
        help="Disable the default s_mean strength covariate.",
    )
    parser.add_argument("--solver_steps", type=int, default=12)
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        default="val_loss",
        choices=[
            "val_loss",
            "val_sc_log_mse",
            "val_sc_log_pearson_topk",
            "val_sc_log_pearson_sparse",
            "monitor_mse_plus_density",
        ],
        help="Early stopping monitor metric. Non-loss options require computing val SC metrics.",
    )
    parser.add_argument(
        "--early_stop_density_weight",
        type=float,
        default=0.0,
        help="Weight for density term in monitor_mse_plus_density.",
    )
    parser.add_argument(
        "--val_sc_eval_every",
        type=int,
        default=0,
        help="Compute val SC metrics every N epochs (0 disables). Required for early_stop_metric based on SC metrics.",
    )
    return parser


def main() -> None:
    ensure_outputs_logs()
    parser = build_arg_parser()
    args = parser.parse_args()
    rank, world_size, local_rank = init_distributed()
    base_results_dir = args.results_dir
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID") or "local"
    run_name = args.run_name.strip()
    if not run_name and rank == 0:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{ts}_job{job_id}"
    if world_size > 1:
        obj = [run_name]
        dist.broadcast_object_list(obj, src=0)
        run_name = obj[0]
    run_dir = os.path.join(base_results_dir, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    args.results_dir = run_dir
    if rank == 0:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        meta = {
            "run_name": run_name,
            "run_dir": run_dir,
            "base_results_dir": base_results_dir,
            "job_id": job_id,
            "world_size": world_size,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "git": _git_info(project_root),
            "argv": os.sys.argv,
        }
        with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)
    trainer = CLGTrainer(
        sc_dir=args.sc_dir,
        morph_root=args.morph_root,
        subject_info_csv=args.subject_info_csv,
        results_dir=args.results_dir,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
        lambda_kl=args.lambda_kl,
        lambda_weight=args.lambda_weight,
        edge_loss=args.edge_loss,
        edge_pos_weight=args.edge_pos_weight,
        focal_gamma=args.focal_gamma,
        focal_alpha=None if args.focal_alpha < 0 else args.focal_alpha,
        lambda_full_log_mse=args.lambda_full_log_mse,
        lambda_zero_log=args.lambda_zero_log,
        zero_log_warmup_epochs=args.zero_log_warmup_epochs,
        zero_log_ramp_epochs=args.zero_log_ramp_epochs,
        lambda_delta_log=args.lambda_delta_log,
        lambda_density=args.lambda_density,
        density_warmup_epochs=args.density_warmup_epochs,
        density_ramp_epochs=args.density_ramp_epochs,
        compute_mask_auprc=args.compute_mask_auprc,
        lambda_manifold=args.lambda_manifold,
        lambda_vel=args.lambda_vel,
        lambda_acc=args.lambda_acc,
        warmup_manifold_epochs=args.warmup_manifold_epochs,
        warmup_vel_epochs=args.warmup_vel_epochs,
        morph_noise_sigma=args.morph_noise_sigma,
        sc_pos_edge_drop_prob=args.sc_pos_edge_drop_prob,
        lambda_topo=args.lambda_topo,
        topo_loss_bins=args.topo_loss_bins,
        betti_sharpness=args.betti_sharpness,
        betti_t=args.betti_t,
        betti_taylor_order=args.betti_taylor_order,
        betti_probes=args.betti_probes,
        topo_log_compress=not args.disable_topo_log_compress,
        topo_scale_quantile=args.topo_scale_quantile,
        topo_warmup_frac=args.topo_warmup_frac,
        gradnorm_scope=args.gradnorm_scope,
        use_s_mean=not args.disable_s_mean,
        topo_bins=args.topo_bins,
        adjacent_pair_prob=args.adjacent_pair_prob,
        residual_skip=args.residual_skip,
        residual_tau=args.residual_tau,
        residual_cap=args.residual_cap,
        solver_steps=args.solver_steps,
        cv_folds=args.cv_folds,
        cv_fold=None if args.cv_fold < 0 else args.cv_fold,
        resume_from=args.resume_from.strip() or None,
        early_stop_metric=args.early_stop_metric,
        early_stop_density_weight=args.early_stop_density_weight,
        val_sc_eval_every=args.val_sc_eval_every,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )
    trainer.run()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
