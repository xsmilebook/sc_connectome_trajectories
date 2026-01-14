import os
import argparse

from src.engine.clg_trainer import CLGTrainer
from src.configs.paths import ensure_outputs_logs, get_by_dotted_key, load_simple_yaml, resolve_repo_path


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
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--lambda_kl", type=float, default=0.0)
    parser.add_argument("--lambda_weight", type=float, default=1.0)
    parser.add_argument("--topo_bins", type=int, default=32)
    parser.add_argument("--adjacent_pair_prob", type=float, default=0.7)
    parser.add_argument(
        "--disable_s_mean",
        action="store_true",
        help="Disable the default s_mean strength covariate.",
    )
    parser.add_argument("--solver_steps", type=int, default=8)
    return parser


def main() -> None:
    ensure_outputs_logs()
    parser = build_arg_parser()
    args = parser.parse_args()
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
        use_s_mean=not args.disable_s_mean,
        topo_bins=args.topo_bins,
        adjacent_pair_prob=args.adjacent_pair_prob,
        solver_steps=args.solver_steps,
    )
    trainer.run()


if __name__ == "__main__":
    main()
