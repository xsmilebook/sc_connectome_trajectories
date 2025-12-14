import os
import argparse

from models.vector_lstm import VectorLSTM
from engine.trainer import Trainer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument(
        "--sc_dir",
        type=str,
        default=os.path.join(
            project_root,
            "data",
            "ABCD",
            "sc_connectome",
            "schaefer400",
        ),
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(
            project_root,
            "data",
            "ABCD",
            "results",
        ),
    )
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--random_state", type=int, default=42)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    trainer = Trainer(
        sc_dir=args.sc_dir,
        results_dir=args.results_dir,
        model_class=VectorLSTM,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
    )
    trainer.run()


if __name__ == "__main__":
    main()

