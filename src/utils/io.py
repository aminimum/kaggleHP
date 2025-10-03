from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_submission(
    predictions: Iterable[float],
    sample_submission: pd.DataFrame,
    output_path: Path,
    target_column: str = "SalePrice",
) -> Path:
    ensure_directory(output_path.parent)
    submission = sample_submission.copy()
    submission[target_column] = np.asarray(predictions)
    submission.to_csv(output_path, index=False)
    return output_path
