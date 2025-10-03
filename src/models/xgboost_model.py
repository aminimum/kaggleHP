from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


DEFAULT_PARAMS: Dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "objective": "reg:squarederror",
}


def train_and_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    **kwargs: Any,
) -> Tuple[XGBRegressor, np.ndarray]:
    params = {**DEFAULT_PARAMS, **kwargs}
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions
