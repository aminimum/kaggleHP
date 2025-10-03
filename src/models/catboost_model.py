from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV


DEFAULT_PARAMS: Dict[str, Any] = {
    "verbose": False,
}


def transform_features(features: pd.DataFrame, log_features: bool = True) -> pd.DataFrame:
    if not log_features:
        return features
    return pd.DataFrame(np.log1p(features), columns=features.columns, index=features.index)


def train_and_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    log_features: bool = True,
    **kwargs: Any,
) -> Tuple[CatBoostRegressor, np.ndarray]:
    params = {**DEFAULT_PARAMS, **kwargs}

    X_train_transformed = transform_features(X_train, log_features=log_features)
    X_test_transformed = transform_features(X_test, log_features=log_features)

    model = CatBoostRegressor(**params)
    model.fit(X_train_transformed, y_train)
    predictions = model.predict(X_test_transformed)
    return model, predictions


def grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, Any],
    log_features: bool = True,
    scoring: str = "neg_root_mean_squared_error",
    cv: int = 3,
    **kwargs: Any,
) -> GridSearchCV:
    X_train_transformed = transform_features(X_train, log_features=log_features)

    estimator = CatBoostRegressor(**{**DEFAULT_PARAMS, **kwargs})
    grid = GridSearchCV(
        estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=2,
        n_jobs=1,
    )
    grid.fit(X_train_transformed, y_train)
    return grid
