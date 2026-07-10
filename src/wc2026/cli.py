"""Command-line entry points for WC 2026 predictor."""

from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

load_dotenv()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def download_data(args=None) -> None:
    parser = argparse.ArgumentParser(description="Download project data")
    parser.add_argument("--force", action="store_true", help="Re-download cached files")
    parser.add_argument("-v", "--verbose", action="store_true")
    parsed = parser.parse_args(args)
    _setup_logging(parsed.verbose)

    from wc2026.data_loading import download_results
    download_results(force=parsed.force)
    print("Data download complete.")


def train_models(args=None) -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost models")
    parser.add_argument("--no-xg", action="store_true", help="Skip xG features")
    parser.add_argument("-v", "--verbose", action="store_true")
    parsed = parser.parse_args(args)
    _setup_logging(parsed.verbose)

    from wc2026.data_loading import load_results
    from wc2026.features import build_modeling_dataset
    from wc2026.model_train import train_models as _train

    results = load_results()
    matches = build_modeling_dataset(results, include_xg=not parsed.no_xg)
    _train(matches)
    print("Training complete. See models/training_meta.json for metrics.")


def summary(args=None) -> None:
    parser = argparse.ArgumentParser(description="Show tournament summary")
    parser.add_argument("-v", "--verbose", action="store_true")
    parsed = parser.parse_args(args)
    _setup_logging(parsed.verbose)

    from wc2026.summary import print_tournament_summary, print_standings
    print_tournament_summary()
    print()
    print_standings()


def predict_fixtures(args=None) -> None:
    parser = argparse.ArgumentParser(description="Predict WC 2026 fixtures")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: outputs/wc2026_predictions.csv)")
    parser.add_argument("--from-csv", type=str, default=None,
                        help="Path to fixtures CSV (default: data/fixtures.csv)")
    parser.add_argument("--source", choices=["csv", "kaggle", "github"], default="csv",
                        help="Fixture source: csv (local), kaggle (download), or github")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed predictions with group stage and stakes")
    parser.add_argument("--show-all", action="store_true",
                        help="Show all matches including group stage")

    parser.add_argument("--kaggle-path", type=str, default=None,
                        help="Direct path to a Kaggle matches CSV")
    parser.add_argument("--force-kaggle", action="store_true",
                        help="Re-download the Kaggle dataset even if cached")

    parsed = parser.parse_args(args)
    _setup_logging(parsed.verbose)

    from pathlib import Path
    from wc2026.predict_world_cup import predict_fixtures as _predict
    from wc2026.visualization import display_predictions

    out = Path(parsed.output) if parsed.output else None
    csv = Path(parsed.from_csv) if parsed.from_csv else None
    df = _predict(
        output_path=out,
        fixtures_csv=csv,
        print_top_outcome=True,
        source=parsed.source,
        kaggle_path=Path(parsed.kaggle_path) if parsed.kaggle_path else None,
        force_kaggle_download=parsed.force_kaggle,
    )

    if parsed.verbose or parsed.show_all:
        display_predictions(df, verbose=True)
    else:
        print(df.to_string(index=False))


def main() -> None:
    """Main function for World Cup 2026 predictor."""
    import sys
    logging.getLogger(__name__).debug("Executing wc2026.cli from: %s", __file__)
    logging.getLogger(__name__).debug("argv: %s", sys.argv)

    parser = argparse.ArgumentParser(prog="wc2026", description="World Cup 2026 predictor")
    parser.add_argument("command", choices=["download", "train", "predict", "summary", "update"])

    # Global flags
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-xg", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--from-csv", type=str, default=None,
                        help="Fixtures CSV path for predict")
    parser.add_argument("--source", choices=["csv", "kaggle", "github"], default="csv")
    parser.add_argument("--kaggle-path", type=str, default=None)
    parser.add_argument("--force-kaggle", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--show-all", action="store_true")

    # update-specific
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--fixtures-min-stage-id", type=int, default=4)

    args = parser.parse_args()

    command_args = []
    if args.verbose:
        command_args.append("-v")
    if args.force:
        command_args.append("--force")
    if args.no_xg:
        command_args.append("--no-xg")
    if args.output:
        command_args.extend(["--output", args.output])
    if args.from_csv:
        command_args.extend(["--from-csv", args.from_csv])
    if args.source != "csv":
        command_args.extend(["--source", args.source])
    if args.kaggle_path:
        command_args.extend(["--kaggle-path", args.kaggle_path])
    if args.force_kaggle:
        command_args.append("--force-kaggle")
    if args.show_all:
        command_args.append("--show-all")

    if args.command == "download":
        download_data(command_args)
    elif args.command == "train":
        train_models(command_args)
    elif args.command == "predict":
        predict_fixtures(command_args)
    elif args.command == "summary":
        summary(command_args)
    else:
        # update
        from wc2026.wc2026_scheduler import main as scheduler_main
        if args.watch:
            scheduler_main()
        else:
            from wc2026.wc2026_scheduler import run_update
            run_update(
                force_download=args.force,
                fixtures_min_stage_id=args.fixtures_min_stage_id,
            )


if __name__ == "__main__":
    main()