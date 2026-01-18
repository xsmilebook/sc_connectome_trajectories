import argparse

from src.configs.paths import get_by_dotted_key, load_simple_yaml, resolve_repo_path
from src.engine.trainer import Trainer
from src.models.vector_lstm import VectorLSTM


def _resolve(cfg: dict, key: str) -> str:
    return resolve_repo_path(str(get_by_dotted_key(cfg, key)))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument("--sc_dir", default=None)
    parser.add_argument("--morph_root", default=None)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--topo_bins", type=int, default=32)
    parser.add_argument("--max_nodes", type=int, default=400)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = load_simple_yaml(args.config)
    sc_dir = args.sc_dir or _resolve(cfg, "local.data.sc_connectome_schaefer400")
    morph_root = args.morph_root or _resolve(cfg, "local.data.morphology")
    results_dir = args.results_dir or _resolve(cfg, "local.outputs.vector_lstm_baseline")
    trainer = Trainer(
        sc_dir=sc_dir,
        results_dir=results_dir,
        model_class=VectorLSTM,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
        morph_root=morph_root,
        topo_bins=args.topo_bins,
        max_nodes=args.max_nodes,
    )
    trainer.run()


if __name__ == "__main__":
    main()
