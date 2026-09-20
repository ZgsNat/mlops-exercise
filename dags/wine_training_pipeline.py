"""
Airflow DAG: Automated Wine Classification Training & Registration Pipeline.
Orchestrates:
1. Data Ingestion & Schema Validation
2. Model Training & Parameter/Metric Logging to MLflow
3. Performance Threshold Evaluation & Model Registry Promotion
4. Deployment Verification
"""
from datetime import datetime, timedelta
import os
import sys

# Ensure src modules are discoverable by Airflow
sys.path.append("/opt/airflow")
sys.path.append("/work")

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.data import load_raw_data, validate_data
from src.train import train_and_log
from src.evaluate import evaluate_and_register
import mlflow
from mlflow.tracking import MlflowClient

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_validate_dataset():
    """Extract Wine dataset and validate schema, types, and values."""
    print("Executing Task 1: Data Ingestion & Schema Validation")
    X, y = load_raw_data()
    validate_data(X, y)
    print(f"Data validated successfully: {X.shape[0]} samples, {X.shape[1]} features.")
    return {"samples": len(X), "features": X.shape[1]}


def task_train_model(**context):
    """Train Random Forest classifier and log to MLflow."""
    print("Executing Task 2: Model Training & Experiment Tracking")
    exp_name = os.getenv("EXPERIMENT_NAME", "wine-classification-pipeline")
    n_estimators = int(os.getenv("N_ESTIMATORS", "150"))
    max_depth = int(os.getenv("MAX_DEPTH", "5"))

    run_id, metrics = train_and_log(
        experiment_name=exp_name,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    # Push run_id to XCom for downstream tasks
    context["ti"].xcom_push(key="run_id", value=run_id)
    context["ti"].xcom_push(key="metrics", value=metrics)
    return run_id


def task_evaluate_and_register(**context):
    """Evaluate model metrics against quality gates and register to MLflow."""
    print("Executing Task 3: Quality Gating & Model Registry")
    ti = context["ti"]
    run_id = ti.xcom_pull(task_ids="train_model", key="run_id")
    if not run_id:
        raise ValueError("Could not retrieve run_id from upstream training task.")

    model_name = os.getenv("MODEL_NAME", "wine-classifier")
    min_acc = float(os.getenv("MIN_ACCURACY", "0.90"))
    alias = os.getenv("MODEL_ALIAS", "champion")

    result = evaluate_and_register(
        run_id=run_id,
        model_name=model_name,
        min_accuracy=min_acc,
        alias=alias,
    )
    ti.xcom_push(key="registration_result", value=result)
    return result


def task_verify_pipeline(**context):
    """Verify the registered model is ready and accessible in the registry."""
    print("Executing Task 4: Pipeline Verification")
    ti = context["ti"]
    result = ti.xcom_pull(task_ids="evaluate_and_register", key="registration_result")
    model_name = os.getenv("MODEL_NAME", "wine-classifier")
    alias = os.getenv("MODEL_ALIAS", "champion")

    client = MlflowClient()
    model_version = client.get_model_version_by_alias(name=model_name, alias=alias)
    print("=" * 60)
    print("PIPELINE VERIFICATION SUCCESSFUL")
    print(f"Registered Model: {model_name}")
    print(f"Alias '@{alias}' points to Version: {model_version.version}")
    print(f"Artifact Source: {model_version.source}")
    print("=" * 60)


with DAG(
    dag_id="wine_model_training_pipeline",
    default_args=default_args,
    description="End-to-End Wine Classification Pipeline: Validate, Train, Evaluate, and Register",
    schedule=None,  # Triggered manually or by external event
    catchup=False,
    tags=["mlops", "wine", "mlflow", "registry"],
) as dag:

    validate_task = PythonOperator(
        task_id="validate_dataset",
        python_callable=task_validate_dataset,
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model,
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_and_register",
        python_callable=task_evaluate_and_register,
    )

    verify_task = PythonOperator(
        task_id="verify_pipeline",
        python_callable=task_verify_pipeline,
    )

    validate_task >> train_task >> evaluate_task >> verify_task
