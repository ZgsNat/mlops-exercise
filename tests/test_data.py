"""
Unit tests for data extraction and validation pipeline.
"""
import pytest
import pandas as pd
from src.data import load_raw_data, validate_data, prepare_splits, EXPECTED_FEATURES


def test_load_raw_data():
    X, y = load_raw_data()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == 178
    assert len(y) == 178
    assert list(X.columns) == EXPECTED_FEATURES


def test_validate_data_success():
    X, y = load_raw_data()
    assert validate_data(X, y) is True


def test_validate_data_missing_columns():
    X, y = load_raw_data()
    X_corrupted = X.drop(columns=["alcohol"])
    with pytest.raises(ValueError, match="Missing expected feature columns"):
        validate_data(X_corrupted, y)


def test_validate_data_null_values():
    X, y = load_raw_data()
    X_corrupted = X.copy()
    X_corrupted.iloc[0, 0] = None
    with pytest.raises(ValueError, match="contains NaN/null"):
        validate_data(X_corrupted, y)


def test_prepare_splits():
    X_train, X_test, y_train, y_test = prepare_splits(test_size=0.25, random_state=42)
    assert len(X_train) == 133
    assert len(X_test) == 45
    assert len(y_train) == 133
    assert len(y_test) == 45
