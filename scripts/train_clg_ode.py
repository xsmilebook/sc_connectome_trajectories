import os
import argparse

from src.engine.clg_trainer import CLGTrainer
from src.configs.paths import ensure_outputs_logs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument(
        "--sc_dir",
        type=str,
        default=os.path.join(
            project_root,
            "data",
            "processed",
            "sc_connectome",
            "schaefer400",
        ),
    )
    parser.add_argument(
        "--morph_root",
        type=str,
        default=os.path.join(
            project_root,
            "data",
            "processed",
            "morphology",
        ),
    )
    parser.add_argument(
        "--subject_info_csv",
        type=str,
        default=os.path.join(
            project_root,
            "data",
            "processed",
            "table",
            "subject_info_sc.csv",
        ),
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(
            project_root,
            "outputs",
            "results",
            "clg_ode",
        ),
    )
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--lambda_kl", type=float, default=1e-3)
    parser.add_argument("--lambda_topo", type=float, default=0.0)
    parser.add_argument("--lambda_smooth", type=float, default=1e-2)
    parser.add_argument("--topk", type=int, default=20)
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
        lambda_topo=args.lambda_topo,
        lambda_smooth=args.lambda_smooth,
        topk=args.topk,
        solver_steps=args.solver_steps,
    )
    trainer.run()


if __name__ == "__main__":
    main()
