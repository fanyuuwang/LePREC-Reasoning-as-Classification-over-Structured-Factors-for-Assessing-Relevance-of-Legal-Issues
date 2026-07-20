"""Command-line entry points for preparing and evaluating a LEPREC release."""

import argparse
import json
from pathlib import Path

from .evaluation import evaluate_linear_models
from .features import load_question_features
from .release_data import prepare_release_dataset
from .validation import validate_release_dataset


def _add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LEPREC public-release tools")
    subcommands = parser.add_subparsers(dest="command", required=True)

    prepare = subcommands.add_parser(
        "prepare-data", help="stage public case, annotation, and feature files"
    )
    prepare.add_argument("--source-root", required=True, type=Path)
    prepare.add_argument("--release-root", required=True, type=Path)

    evaluate = subcommands.add_parser(
        "evaluate", help="run nested cross-validation on probability features"
    )
    evaluate.add_argument("--features", required=True, type=Path)
    _add_evaluation_arguments(evaluate)

    validate = subcommands.add_parser(
        "validate-data", help="validate canonical public release artifacts"
    )
    validate.add_argument("--release-root", required=True, type=Path)

    evaluate_release = subcommands.add_parser(
        "evaluate-release", help="validate and evaluate the canonical public release"
    )
    evaluate_release.add_argument("--release-root", required=True, type=Path)
    _add_evaluation_arguments(evaluate_release)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-data":
        report = prepare_release_dataset(args.source_root, args.release_root)
        print(json.dumps(report, indent=2))
        return

    if args.command == "validate-data":
        print(json.dumps(validate_release_dataset(args.release_root), indent=2))
        return

    if args.command == "evaluate-release":
        validate_release_dataset(args.release_root)
        feature_path = (
            args.release_root
            / "expert_annotations"
            / "phi4_question_scores.json.gz"
        )
    else:
        feature_path = args.features

    features, labels, _ = load_question_features(feature_path)
    metrics = evaluate_linear_models(features, labels, folds=args.folds, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
