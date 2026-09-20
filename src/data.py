"""
Data ingestion, validation, and preparation module for the Wine dataset.
"""
from typing import Tuple
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

EXPECTED_FEATURES = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines",
    "proline",
]

TARGET_CLASSES = [0, 1, 2]
TARGET_NAMES = ["class_0", "class_1", "class_2"]


def load_raw_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the Wine dataset from scikit-learn as pandas DataFrame and Series.
    """
    wine = load_wine(as_frame=True)
    X: pd.DataFrame = wine.data
    y: pd.Series = wine.target
    return X, y


def validate_data(X: pd.DataFrame, y: pd.Series) -> bool:
    """
    Validate that data matches schema requirements:
    - Expected columns are present
    - No missing (NaN) values
    - Target classes contain only expected class labels [0, 1, 2]
    - Sample count is positive
    """
    if X.empty or y.empty:
        raise ValueError("Dataset is empty.")

    if len(X) != len(y):
        raise ValueError(f"Feature count ({len(X)}) does not match target count ({len(y)}).")

    missing_cols = set(EXPECTED_FEATURES) - set(X.columns)
    if missing_cols:
        raise ValueError(f"Missing expected feature columns: {missing_cols}")

    if X.isnull().sum().sum() > 0:
        raise ValueError("Feature dataset contains NaN/null values.")

    if y.isnull().sum() > 0:
        raise ValueError("Target dataset contains NaN/null values.")

    unique_classes = set(y.unique())
    invalid_classes = unique_classes - set(TARGET_CLASSES)
    if invalid_classes:
        raise ValueError(f"Target contains unexpected classes: {invalid_classes}")

    return True


def prepare_splits(
    test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load, validate, and perform a stratified train-test split.
    """
    X, y = load_raw_data()
    validate_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
