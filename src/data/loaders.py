from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config import DataConfig


def _validate_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    return path


def load_raw_frames(data_config: DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the raw Kaggle House Prices datasets."""
    train_path = _validate_file(data_config.train_path)
    test_path = _validate_file(data_config.test_path)
    submission_path = _validate_file(data_config.submission_path)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    submission_df = pd.read_csv(submission_path)
    return train_df, test_df, submission_df
