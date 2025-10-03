from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def train_and_predict(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
) -> Tuple[LinearRegression, np.ndarray]:
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions
