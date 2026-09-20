"""
Unit tests for model training module.
"""
import tempfile
from src.train import train_and_log


def test_train_and_log():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracking_uri = f"sqlite:///{tmp_dir}/mlflow_test.db"
        run_id, metrics = train_and_log(
            experiment_name="test-experiment",
            n_estimators=10,
            max_depth=3,
            tracking_uri=tracking_uri,
        )

        assert isinstance(run_id, str)
        assert len(run_id) > 0
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        # Wine dataset with 10 trees should achieve at least 80% accuracy
        assert metrics["accuracy"] >= 0.80
