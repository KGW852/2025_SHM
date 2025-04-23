# utils/logger.py

import os
import json

import mlflow
from datetime import datetime
from typing import Dict, Optional


class MLFlowLogger:
    """
    MLflow-based logging utility class.
    log parameters, metrics, artifacts, etc., during the training process
    organize experiment information.
    """
    def __init__(self, tracking_uri, experiment_name):
        """
        Args:
            tracking_uri (str, optional): MLflow Tracking server URI. e.g. 'file:./logs/mlruns' for local logging.
            experiment_name (str): MLflow experiment name. (ex1, ex2 ... from config.py)
        """
        self.run_id = None
        self.experiment_id = None
        self.experiment_name = experiment_name

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri("file:./mlruns")
            
        self.set_experiment(experiment_name)

    def set_experiment(self, experiment_name):
        mlflow.set_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id

    def start_run(self, run_name: Optional[str] = None):
        """
        Start MLflow run
        Args:
            run_name (str, optional): run_name on MLflow UI
        """
        if not mlflow.active_run():
            if run_name is None:  # time, if not set run_name
                run_name = f"Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            run = mlflow.start_run(run_name=run_name, experiment_id=self.experiment_id)
            self.run_id = run.info.run_id

    def resume_run(self, run_id: str):
        """
        Resume an existing MLflow run by run_id.
        Useful for continuing to log metrics/artifacts in a separate process/time.
        """
        if mlflow.active_run():
            mlflow.end_run()

        run = mlflow.start_run(run_id=run_id)
        self.run_id = run.info.run_id

    def log_params(self, params: Dict):
        if mlflow.active_run():
            mlflow.log_params(params)
        else:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """
        Log metrics
        Args:
            metrics (dict): {metric_name: metric_value, ...}
            step (int, optional): iteration step, epoch, ...
        """
        if mlflow.active_run():
            mlflow.log_metrics(metrics, step=step)
        else:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
    def log_artifact(self, file_path: str, artifact_path: Optional[str] = None):
        """
        Artifact(file) log. Upload model params, image, csv, etc.
        Args:
            file_path (str): local path(ex. './result.png')
            artifact_path (str, optional): The artifact folder path to be displayed in the UI
        """
        if mlflow.active_run():
            mlflow.log_artifact(local_path=file_path, artifact_path=artifact_path)
        else:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
    def log_artifacts(self, dir_path: str, artifact_path: Optional[str] = None):
        """
        Artifact(directory) log. Upload all files within the directory
        Args:
            dir_path (str): local directory path
            artifact_path (str, optional): The artifact folder path to be displayed in the UI
        """
        if mlflow.active_run():
            mlflow.log_artifacts(local_dir=dir_path, artifact_path=artifact_path)
        else:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
    def end_run(self):
        if mlflow.active_run():
            mlflow.end_run()
            self.run_id = None

    def set_tag(self, key: str, value: str):
        if mlflow.active_run():
            mlflow.set_tag(key, value)
        else:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
    def set_tags(self, tags: Dict[str, str]):
        if mlflow.active_run():
            mlflow.set_tags(tags)
        else:
            raise RuntimeError("No active MLflow run. Call start_run() first.")

    def register_model(self, model_name: str, model_path: str):
        """
        Register the trained model in the MLflow Model Registry
        Args:
            model_name (str): The name of the model to be registered in the MLflow Model Registry.
            model_path (str): The path to the model directory saved via log_artifact.
        """
        from mlflow.tracking import MlflowClient
        
        # need current run_id
        if not mlflow.active_run():
            raise RuntimeError("No active MLflow run. Call start_run() first.")

        client = MlflowClient()
        # register model
        model_uri = f"runs:/{self.run_id}/{model_path}"
        client.create_registered_model(model_name)
        client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=self.run_id
        )
        print(f"[INFO] Model '{model_name}' registered with run ID: {self.run_id}")

    def save_run_id(self, file_path: str):
        """
        Save the current run_id and experiment_id to a JSON file.
        """
        if self.run_id is None:
            raise RuntimeError("No run_id to save. Make sure you've called start_run() or resume_run().")

        data = {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] run_id saved to {file_path}")

    @staticmethod
    def load_run_id(file_path: str):
        """
        Load a saved run_id and experiment info from JSON file.
        """
        with open(file_path, "r") as f:
            data = json.load(f)
        run_id = data.get("run_id")
        exp_id = data.get("experiment_id")
        exp_name = data.get("experiment_name")
        print(f"[INFO] run_id loaded from {file_path}")
        return run_id, exp_id, exp_name