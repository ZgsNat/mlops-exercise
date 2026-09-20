"""
Model training module with MLflow experiment tracking.
"""
import os
from typing import Dict, Tuple
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import prepare_splits


def train_and_log(
    experiment_name: str = "wine-classification-pipeline",
    n_estimators: int = 150,
    max_depth: int = 5,
    random_state: int = 42,
    tracking_uri: str | None = None,
) -> Tuple[str, Dict[str, float]]:
    """
    Train a Random Forest pipeline and log parameters, metrics, signature,
    and model artifacts to MLflow.

    Returns:
        Tuple of (run_id, metrics_dict)
    """
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)

    X_train, X_test, y_train, y_test = prepare_splits(random_state=random_state)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
            ),
        ),
    ])

    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        acc = float(accuracy_score(y_test, predictions))
        f1 = float(f1_score(y_test, predictions, average="macro"))
        prec = float(precision_score(y_test, predictions, average="macro"))
        rec = float(recall_score(y_test, predictions, average="macro"))

        metrics = {
            "accuracy": acc,
            "f1_macro": f1,
            "precision_macro": prec,
            "recall_macro": rec,
        }

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))

        # Log metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        # Log model with signature and input example
        signature = infer_signature(X_train, predictions)
        input_example = X_train.head(2)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        run_id = run.info.run_id
        print(f"Successfully finished run {run_id}")
        print(f"Metrics: Accuracy={acc:.4f}, F1-macro={f1:.4f}")

    return run_id, metrics


def main() -> None:
    exp_name = os.getenv("EXPERIMENT_NAME", "wine-classification-pipeline")
    n_est = int(os.getenv("N_ESTIMATORS", "150"))
    depth = int(os.getenv("MAX_DEPTH", "5"))
    run_id, metrics = train_and_log(
        experiment_name=exp_name,
        n_estimators=n_est,
        max_depth=depth,
    )
    print(f"Completed run: {run_id}, Accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
