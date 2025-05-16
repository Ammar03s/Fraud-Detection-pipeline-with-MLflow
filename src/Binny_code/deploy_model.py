import os
import sys
import logging
import mlflow
import mlflow.sklearn
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "model_deployment_V1"
mlflow.set_experiment(experiment_name)

def get_best_model_run(experiment_name="fraud-detection-hyperparameter-tuning"):
    """
    Find the best model run from the hyperparameter tuning experiment
    Returns the run_id of the best model based on ROC AUC
    """
    logger.info(f"Searching for best model in experiment: {experiment_name}")
    
    # Get the experiment ID
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.error(f"Experiment '{experiment_name}' not found")
        return None
    
    experiment_id = experiment.experiment_id
    
    # Get all runs for the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    
    if runs.empty:
        logger.error(f"No runs found for experiment '{experiment_name}'")
        return None
    
    # Find the run with the best ROC AUC
    best_run_idx = runs['metrics.roc_auc'].idxmax()
    best_run = runs.iloc[best_run_idx]
    best_run_id = best_run['run_id']
    best_auc = best_run['metrics.roc_auc']
    model_type = best_run['params.model_type'] if 'params.model_type' in best_run else "unknown"
    
    logger.info(f"Found best model: {model_type} with ROC AUC: {best_auc:.4f}, run_id: {best_run_id}")
    
    return best_run_id

def promote_model_to_registry(run_id, model_name="FraudDetectionModel", stage="Staging"):
    """
    Register the model to the MLflow Model Registry and set its stage
    """
    logger.info(f"Registering model from run: {run_id} as {model_name}")
    
    # Check if the model already exists in the registry
    try:
        latest_version = mlflow.register_model(f"runs:/{run_id}/model", model_name)
        version = latest_version.version
        logger.info(f"Model registered as {model_name} version {version}")
    except mlflow.exceptions.RestException as e:
        if "already exists with this name" in str(e):
            # Get the latest version of the model
            latest_versions = mlflow.tracking.MlflowClient().get_latest_versions(model_name)
            version = max([int(v.version) for v in latest_versions])
            
            # Register a new version
            latest_version = mlflow.register_model(f"runs:/{run_id}/model", model_name)
            version = latest_version.version
            logger.info(f"Model already existed, registered new version: {version}")
        else:
            logger.error(f"Error registering model: {str(e)}")
            return None
    
    # Transition the model to the specified stage
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )
    
    logger.info(f"Model {model_name} version {version} transitioned to {stage}")
    
    return version

def load_and_evaluate_registered_model(model_name="FraudDetectionModel", stage="Staging"):
    """
    Load a model from the MLflow Model Registry and evaluate it on test data
    """
    logger.info(f"Loading {model_name} model (stage: {stage}) from registry")
    
    # Load the model
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Load test data
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }
    
    logger.info(f"Model evaluation results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    
    return metrics

def prepare_model_for_deployment(model_name="FraudDetectionModel", stage="Production"):
    """
    Prepare the model for deployment by saving necessary files
    """
    logger.info(f"Preparing {model_name} model (stage: {stage}) for deployment")
    
    # Create deployment directory
    deployment_dir = 'models/deployment'
    os.makedirs(deployment_dir, exist_ok=True)
    
    # Load the model
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Load scaler
    scaler = joblib.load('data/processed/scaler.pkl')
    
    # Load feature names
    feature_names = joblib.load('data/processed/feature_names.pkl')
    
    # Save model, scaler and feature names
    joblib.dump(model, os.path.join(deployment_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(deployment_dir, 'scaler.pkl'))
    joblib.dump(feature_names, os.path.join(deployment_dir, 'feature_names.pkl'))
    
    # Create model info
    model_info = {
        "model_name": model_name,
        "stage": stage,
        "deployment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": feature_names
    }
    
    with open(os.path.join(deployment_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=4)
    
    logger.info(f"Model prepared for deployment in {deployment_dir}")
    
    return deployment_dir

def deploy_model(deployment_dir='models/deployment'):
    """
    Deploy the model (in a real scenario, this might involve Kubernetes, Docker, etc.)
    Here we'll just copy the model to a 'deployed' directory
    """
    logger.info("Simulating model deployment")
    
    # Create deployed directory
    deployed_dir = 'models/deployed'
    os.makedirs(deployed_dir, exist_ok=True)
    
    # Copy model, scaler and feature names to deployed directory
    import shutil
    for filename in os.listdir(deployment_dir):
        src = os.path.join(deployment_dir, filename)
        dst = os.path.join(deployed_dir, filename)
        shutil.copy2(src, dst)
    
    # Create deployment log
    deployment_log = {
        "deployment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "deployment_status": "success"
    }
    
    with open(os.path.join(deployed_dir, 'deployment_log.json'), 'w') as f:
        json.dump(deployment_log, f, indent=4)
    
    logger.info(f"Model deployed to {deployed_dir}")
    
    return deployed_dir

def main():
    """Main function to deploy the best model"""
    with mlflow.start_run():
        try:
            # Find the best model from hyperparameter tuning
            best_run_id = get_best_model_run()
            if best_run_id is None:
                logger.error("Could not find best model run")
                return
            
            # Register model and promote to Staging
            staged_version = promote_model_to_registry(best_run_id, stage="Staging")
            
            # Evaluate registered model
            with mlflow.start_run(nested=True, run_name="model_evaluation"):
                metrics = load_and_evaluate_registered_model(stage="Staging")
                mlflow.log_metrics(metrics)
            
            # If model performs well, promote to Production
            if metrics['roc_auc'] > 0.8:  # Threshold for production promotion
                production_version = promote_model_to_registry(best_run_id, stage="Production")
                
                # Prepare model for deployment
                deployment_dir = prepare_model_for_deployment(stage="Production")
                
                # Deploy model
                deployed_dir = deploy_model(deployment_dir)
                
                logger.info(f"Model successfully promoted to Production (version {production_version}) and deployed")
            else:
                logger.warning(f"Model performance below threshold for Production promotion: {metrics['roc_auc']:.4f} < 0.8")
                logger.info("Model will remain in Staging stage for further evaluation")
        
        except Exception as e:
            logger.error(f"Error during model deployment: {str(e)}")
            mlflow.log_param("error", str(e))
            raise

if __name__ == "__main__":
    main()