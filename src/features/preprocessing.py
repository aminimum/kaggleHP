from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def lowercase_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.lower()
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    if "lotfrontage" in df.columns:
        df["lotfrontage"] = df["lotfrontage"].fillna(df["lotfrontage"].mode().iloc[0])
    if "fireplacequ" in df.columns:
        df["fireplacequ"] = df["fireplacequ"].fillna(df["fireplacequ"].mode().iloc[0])

    few_null_columns = df.columns[df.isnull().sum() < 200]
    if len(few_null_columns) > 0:
        mode_values = df[few_null_columns].mode().iloc[0]
        df[few_null_columns] = df[few_null_columns].fillna(mode_values)

        numeric_columns = df[few_null_columns].select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            median_values = df[numeric_columns].median()
            df[numeric_columns] = df[numeric_columns].fillna(median_values)

    for column in ("alley", "fence"):
        if column in df.columns:
            df[column] = df[column].fillna("None")

    df = df.drop(columns=[col for col in ["id", "miscfeature", "poolqc"] if col in df.columns])
    return df


def drop_low_information_columns(df: pd.DataFrame) -> pd.DataFrame:
    object_summary = df.describe(include=["object"])
    if not object_summary.empty and "freq" in object_summary.index:
        freq = object_summary.loc["freq"]
        to_drop = freq[freq > 2860].index.tolist()
        df = df.drop(columns=to_drop, errors="ignore")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["exists_pool"] = df.get("poolarea", 0).apply(lambda x: 1 if x > 0 else 0)
    df["exists_garage"] = df.get("garagearea", 0).apply(lambda x: 1 if x > 0 else 0)
    df["exists_fireplace"] = df.get("fireplaces", 0).apply(lambda x: 1 if x > 0 else 0)
    df["exists_bsmt"] = df.get("totalbsmtsf", 0).apply(lambda x: 1 if x > 0 else 0)
    df["exists_2ndflr"] = df.get("2ndflrsf", 0).apply(lambda x: 1 if x > 0 else 0)
    df["exists_half"] = df.get("halfbath", 0).apply(lambda x: 1 if x > 0 else 0)
    df["new_house"] = df.get("yearbuilt", 0).apply(lambda x: 1 if x > 2000 else 0)

    df["totalhousesf"] = (
        df.get("totalbsmtsf", 0)
        + df.get("1stflrsf", 0)
        + df.get("2ndflrsf", 0)
        + df.get("bsmtfinsf1", 0)
        + df.get("bsmtfinsf2", 0)
        + df.get("bsmtunfsf", 0)
        + df.get("openporchsf", 0)
        + df.get("wooddecksf", 0)
    )

    df["total_bath"] = (
        df.get("fullbath", 0)
        + df.get("halfbath", 0)
        + df.get("bsmtfullbath", 0)
        + df.get("bsmthalfbath", 0)
    )

    df["total_porch"] = (
        df.get("3ssnporch", 0)
        + df.get("enclosedporch", 0)
        + df.get("screenporch", 0)
    )

    df = df.drop(columns=[col for col in ["yearremodadd", "yrsold", "bsmthalfbath"] if col in df.columns])
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = df.select_dtypes(include=["object"]).columns
    if len(categorical_columns) == 0:
        return df
    df = pd.get_dummies(df, columns=list(categorical_columns), drop_first=False)
    return df


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()

    combined = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    combined = lowercase_columns(combined)
    combined = fill_missing_values(combined)
    combined = drop_low_information_columns(combined)
    combined = engineer_features(combined)
    combined = encode_categoricals(combined)

    if "saleprice" not in combined.columns:
        raise KeyError("Expected 'saleprice' column in the combined dataset.")

    feature_matrix = combined.drop(columns=["saleprice"], errors="ignore")
    target = combined["saleprice"].iloc[: len(train_df)]

    X_train = feature_matrix.iloc[: len(train_df)].reset_index(drop=True)
    X_test = feature_matrix.iloc[len(train_df) :].reset_index(drop=True)

    return X_train, target.reset_index(drop=True), X_test
