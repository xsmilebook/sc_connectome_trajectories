import argparse
import json
import os
from datetime import datetime

import numpy as np
from sklearn.model_selection import GroupKFold

import torch

from src.data.utils import compute_triu_indices, ensure_dir
from src.engine.clg_trainer import CLGTrainer


def _load_args_json(run_dir: str) -> dict:
    path = os.path.join(run_dir, "args.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_checkpoint(run_dir: str, cv_fold: int) -> str:
    candidate = os.path.join(run_dir, f"clg_ode_fold{cv_fold}_best.pt")
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Checkpoint not found: {candidate}")


def _build_trainer_from_args(args: dict) -> CLGTrainer:
    focal_alpha = args.get("focal_alpha", -1.0)
    focal_alpha = None if focal_alpha is None or float(focal_alpha) < 0 else float(focal_alpha)
    cv_fold = int(args.get("cv_fold", -1))
    cv_fold = None if cv_fold < 0 else cv_fold
    trainer = CLGTrainer(
        sc_dir=args["sc_dir"],
        morph_root=args["morph_root"],
        subject_info_csv=args["subject_info_csv"],
        results_dir=args["results_dir"],
        latent_dim=int(args.get("latent_dim", 64)),
        hidden_dim=int(args.get("hidden_dim", 128)),
        batch_size=int(args.get("batch_size", 2)),
        max_epochs=int(args.get("max_epochs", 0)),
        patience=int(args.get("patience", 0)),
        learning_rate=float(args.get("learning_rate", 1e-4)),
        random_state=int(args.get("random_state", 42)),
        lambda_kl=float(args.get("lambda_kl", 0.0)),
        lambda_weight=float(args.get("lambda_weight", 1.0)),
        edge_loss=str(args.get("edge_loss", "bce")),
        edge_pos_weight=float(args.get("edge_pos_weight", 5.0)),
        focal_gamma=float(args.get("focal_gamma", 2.0)),
        focal_alpha=focal_alpha,
        lambda_full_log_mse=float(args.get("lambda_full_log_mse", 0.0)),
        lambda_zero_log=float(args.get("lambda_zero_log", 0.0)),
        zero_log_warmup_epochs=int(args.get("zero_log_warmup_epochs", 0)),
        zero_log_ramp_epochs=int(args.get("zero_log_ramp_epochs", 0)),
        lambda_delta_log=float(args.get("lambda_delta_log", 0.0)),
        lambda_density=float(args.get("lambda_density", 0.0)),
        density_warmup_epochs=int(args.get("density_warmup_epochs", 0)),
        density_ramp_epochs=int(args.get("density_ramp_epochs", 0)),
        compute_mask_auprc=bool(args.get("compute_mask_auprc", False)),
        lambda_manifold=float(args.get("lambda_manifold", 1.0)),
        lambda_vel=float(args.get("lambda_vel", 0.0)),
        lambda_acc=float(args.get("lambda_acc", 0.0)),
        warmup_manifold_epochs=int(args.get("warmup_manifold_epochs", 10)),
        warmup_vel_epochs=int(args.get("warmup_vel_epochs", 20)),
        morph_noise_sigma=float(args.get("morph_noise_sigma", 0.0)),
        sc_pos_edge_drop_prob=float(args.get("sc_pos_edge_drop_prob", 0.0)),
        lambda_topo=float(args.get("lambda_topo", 0.0)),
        topo_loss_bins=int(args.get("topo_loss_bins", 8)),
        betti_sharpness=float(args.get("betti_sharpness", 20.0)),
        betti_t=float(args.get("betti_t", 10.0)),
        betti_taylor_order=int(args.get("betti_taylor_order", 20)),
        betti_probes=int(args.get("betti_probes", 2)),
        topo_log_compress=not bool(args.get("disable_topo_log_compress", False)),
        topo_scale_quantile=float(args.get("topo_scale_quantile", 0.9)),
        topo_warmup_frac=float(args.get("topo_warmup_frac", 0.2)),
        gradnorm_scope=str(args.get("gradnorm_scope", "none")),
        use_s_mean=not bool(args.get("disable_s_mean", False)),
        topo_bins=int(args.get("topo_bins", 32)),
        adjacent_pair_prob=float(args.get("adjacent_pair_prob", 0.7)),
        solver_steps=int(args.get("solver_steps", 8)),
        residual_skip=bool(args.get("residual_skip", False)),
        residual_tau=float(args.get("residual_tau", 1.0)),
        residual_no_dt_gate=bool(args.get("residual_no_dt_gate", False)),
        residual_cap=float(args.get("residual_cap", 0.5)),
        fixed_support=bool(args.get("fixed_support", False)),
        innovation_enabled=bool(args.get("innovation_enabled", False)),
        innovation_topm=int(args.get("innovation_topm", 400)),
        innovation_k_new=int(args.get("innovation_k_new", 80)),
        innovation_tau=float(args.get("innovation_tau", 0.10)),
        innovation_delta_quantile=float(args.get("innovation_delta_quantile", 0.95)),
        innovation_dt_scale_years=float(args.get("innovation_dt_scale_years", 1.0)),
        innovation_dt_offset_months=float(args.get("innovation_dt_offset_months", 0.0)),
        innovation_dt_ramp_months=float(args.get("innovation_dt_ramp_months", 12.0)),
        innovation_focal_gamma=float(args.get("innovation_focal_gamma", 2.0)),
        innovation_focal_alpha=float(args.get("innovation_focal_alpha", 0.25)),
        lambda_new_sparse=float(args.get("lambda_new_sparse", 0.10)),
        new_sparse_warmup_epochs=int(args.get("new_sparse_warmup_epochs", 10)),
        new_sparse_ramp_epochs=int(args.get("new_sparse_ramp_epochs", 10)),
        lambda_new_reg=float(args.get("lambda_new_reg", 0.0)),
        innovation_freeze_backbone_after=int(args.get("innovation_freeze_backbone_after", -1)),
        cv_folds=int(args.get("cv_folds", 5)),
        cv_fold=cv_fold,
        resume_from=(str(args.get("resume_from", "")).strip() or None),
        early_stop_metric=str(args.get("early_stop_metric", "val_loss")),
        early_stop_density_weight=float(args.get("early_stop_density_weight", 0.0)),
        val_sc_eval_every=int(args.get("val_sc_eval_every", 0)),
        rank=0,
        world_size=1,
        local_rank=0,
    )
    return trainer


def _compute_fold_train_indices(trainer: CLGTrainer, dataset, trainval_indices: list[int], subjects: list[str]) -> list[int]:
    groups = np.array([subjects[i] for i in trainval_indices])
    indices = np.array(trainval_indices)
    n_splits = int(trainer.cv_folds)
    cv_fold = trainer.cv_fold
    if cv_fold is None:
        raise ValueError("args.json must contain a single cv_fold to evaluate deterministically.")
    gkf = GroupKFold(n_splits=n_splits)
    for fold_idx, (train_idx_rel, _val_idx_rel) in enumerate(gkf.split(indices, groups=groups)):
        if fold_idx != cv_fold:
            continue
        return indices[train_idx_rel].tolist()
    raise ValueError(f"Could not locate cv_fold={cv_fold} within cv_folds={n_splits}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Existing training run directory (e.g., outputs/results/clg_ode/runs/<run>/fold0). Must contain args.json and checkpoint.",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="test_sc_metrics_ext.json",
        help="Output filename written under run_dir.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional explicit checkpoint path; defaults to clg_ode_fold{cv_fold}_best.pt under run_dir.",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    args_json = _load_args_json(run_dir)
    trainer = _build_trainer_from_args(args_json)

    dataset = trainer._build_dataset()
    subjects = [sid for sid, _ in dataset.sequences]
    trainval_indices, test_indices = trainer._split_outer(subjects)
    train_idx = _compute_fold_train_indices(trainer, dataset, trainval_indices, subjects)
    stats = trainer._compute_norm_stats(dataset, train_idx)

    checkpoint = args.checkpoint.strip() or _resolve_checkpoint(run_dir, int(trainer.cv_fold))
    model = trainer._build_model(dataset).to(trainer.device)
    state = torch.load(checkpoint, map_location=trainer.device)
    model.load_state_dict(state, strict=False)

    triu_idx = compute_triu_indices(dataset.max_nodes)
    test_loader = trainer._get_loader(dataset, test_indices, shuffle=False)
    sc_metrics = trainer._evaluate_sc_metrics(model, test_loader, triu_idx, stats, dataset.volume_metric_indices)

    out_path = os.path.join(run_dir, args.out_file)
    ensure_dir(run_dir)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sc_metrics, f, indent=2)

    meta = {
        "run_dir": run_dir,
        "checkpoint": checkpoint,
        "out_file": out_path,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cv_fold": trainer.cv_fold,
        "cv_folds": trainer.cv_folds,
        "n_subjects": len(dataset),
        "n_test": len(test_indices),
    }
    with open(os.path.join(run_dir, "test_sc_metrics_ext_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()

