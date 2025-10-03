from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

from src.config import AppConfig, load_config
from src.data.loaders import load_raw_frames
from src.features.preprocessing import preprocess
from src.models import catboost_model, linear, xgboost_model
from src.utils.io import write_submission


MODEL_TO_FILENAME = {
    "linear": "house_price_lr.csv",
    "xgboost": "house_price_xgb.csv",
    "catboost": "house_price_cat.csv",
    "catboost_grid": "house_price_cat_GS.csv",
}


CATBOOST_GRID_PARAMS = {
    "depth": [1, 5, 10],
    "learning_rate": [0.05, 0.1, 0.15],
    "l2_leaf_reg": [1, 4, 9],
    "iterations": [500],
}


def _select_models(requested: Iterable[str] | None, config: AppConfig) -> List[str]:
    if requested is None:
        return [config.models.default]
    requested = list(requested)
    if "all" in requested:
        return ["linear", "xgboost", "catboost"]
    unknown = set(requested) - MODEL_TO_FILENAME.keys()
    if unknown:
        raise ValueError(f"Unknown models requested: {', '.join(sorted(unknown))}")
    return requested


def _run_linear(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> Tuple[pd.Series, str]:
    _, preds = linear.train_and_predict(X_train, y_train, X_test)
    return pd.Series(preds), MODEL_TO_FILENAME["linear"]


def _run_xgboost(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> Tuple[pd.Series, str]:
    _, preds = xgboost_model.train_and_predict(X_train, y_train, X_test)
    return pd.Series(preds), MODEL_TO_FILENAME["xgboost"]


def _run_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    use_grid: bool = False,
) -> Tuple[pd.Series, str]:
    if use_grid:
        grid = catboost_model.grid_search(X_train, y_train, CATBOOST_GRID_PARAMS)
        best_model = grid.best_estimator_
        transformed_test = catboost_model.transform_features(X_test)
        predictions = best_model.predict(transformed_test)
        filename = MODEL_TO_FILENAME["catboost_grid"]
    else:
        _, predictions = catboost_model.train_and_predict(X_train, y_train, X_test)
        filename = MODEL_TO_FILENAME["catboost"]
    return pd.Series(predictions), filename


def run_pipeline(config_path: Path, models: Iterable[str] | None = None) -> List[Path]:
    config = load_config(config_path)
    train_df, test_df, submission_df = load_raw_frames(config.data)
    X_train, y_train, X_test = preprocess(train_df, test_df)

    selected_models = _select_models(models, config)
    outputs: List[Path] = []

    for model_name in selected_models:
        if model_name == "linear":
            predictions, filename = _run_linear(X_train, y_train, X_test)
        elif model_name == "xgboost":
            predictions, filename = _run_xgboost(X_train, y_train, X_test)
        elif model_name == "catboost":
            predictions, filename = _run_catboost(X_train, y_train, X_test)
        elif model_name == "catboost_grid":
            predictions, filename = _run_catboost(X_train, y_train, X_test, use_grid=True)
        else:
            raise ValueError(f"Unsupported model '{model_name}'")

        output_path = config.data.output_dir / filename
        outputs.append(write_submission(predictions, submission_df, output_path))

    return outputs


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the House Prices training pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "default.yml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="List of models to train (linear, xgboost, catboost, catboost_grid, all).",
    )
    return parser


def main() -> List[Path]:
    parser = build_argument_parser()
    args = parser.parse_args()
    return run_pipeline(args.config, args.models)


if __name__ == "__main__":
    main()
