"""
Model evaluation, quality gating, and registration module for MLflow Model Registry.
"""
import os
from typing import Dict, Any
import mlflow
from mlflow.tracking import MlflowClient


def evaluate_and_register(
    run_id: str,
    model_name: str = "wine-classifier",
    min_accuracy: float = 0.90,
    alias: str = "champion",
    tracking_uri: str | None = None,
) -> Dict[str, Any]:
    """
    Check if the model meets performance thresholds, register it to MLflow Model Registry,
    and tag / alias it for serving.
    """
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()

    # Retrieve run information
    run = client.get_run(run_id)
    acc = run.data.metrics.get("accuracy", 0.0)
    print(f"Checking quality gate for run {run_id}: accuracy={acc:.4f}, required={min_accuracy:.4f}")

    if acc < min_accuracy:
        raise ValueError(
            f"Quality gate failed! Accuracy {acc:.4f} is lower than required threshold {min_accuracy:.4f}"
        )

    # Register model to MLflow Model Registry
    model_uri = f"runs:/{run_id}/model"
    print(f"Registering model from {model_uri} as '{model_name}'...")
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Set alias and tags
    client.set_registered_model_alias(name=model_name, alias=alias, version=mv.version)
    client.set_model_version_tag(
        name=model_name,
        version=mv.version,
        key="validation_status",
        value="passed_quality_gate",
    )
    client.set_model_version_tag(
        name=model_name,
        version=mv.version,
        key="accuracy",
        value=f"{acc:.4f}",
    )

    print(
        f"Successfully registered '{model_name}' version {mv.version} with alias '@{alias}'"
    )

    return {
        "status": "success",
        "model_name": model_name,
        "version": mv.version,
        "accuracy": acc,
        "alias": alias,
    }


def main() -> None:
    run_id = os.getenv("RUN_ID")
    if not run_id:
        print("No RUN_ID specified, skipping evaluation.")
        return
    model_name = os.getenv("MODEL_NAME", "wine-classifier")
    threshold = float(os.getenv("MIN_ACCURACY", "0.90"))
    evaluate_and_register(run_id=run_id, model_name=model_name, min_accuracy=threshold)


if __name__ == "__main__":
    main()
