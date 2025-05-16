import os
import shutil
import json
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Deployment_and_Monitoring_pptZ")

# ---- Deployment functions ----

def get_best_run(experiment_name, metric='f1_score'):
    """Return run_id of best run by given metric."""
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.error(f"Experiment '{experiment_name}' not found")
        return None
    runs = mlflow.search_runs(exp.experiment_id)
    if runs.empty:
        logger.error(f"No runs in experiment '{experiment_name}'")
        return None
    best = runs.loc[runs[f"metrics.{metric}"].idxmax()]
    run_id = best['run_id']
    logger.info(f"Best run {run_id} by {metric}={best[f'metrics.{metric}']:.4f}")
    return run_id


def register_and_stage(run_id, model_name, stage):
    """Register model from run_id and transition to stage."""
    uri = f"runs:/{run_id}/model"
    client = mlflow.tracking.MlflowClient()
    try:
        mv = mlflow.register_model(uri, model_name)
    except mlflow.exceptions.RestException:
        mv = client.create_model_version(model_name, uri, exp.experiment_id)
    client.transition_model_version_stage(model_name, mv.version, stage)
    logger.info(f"Model {model_name} v{mv.version} -> {stage}")
    return mv.version


def evaluate_registered(model_name, stage, data_dir='data/processed'):
    """Load model from registry and evaluate on test set."""
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    metrics = dict(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1_score=f1_score(y_test, y_pred)
    )
    logger.info(f"Evaluation at {stage}: {metrics}")
    return metrics


def prepare_deployment(model_name, stage, deploy_dir='models/deployed', data_dir='data/processed'):
    """Save model, scaler, features locally for deployment."""
    os.makedirs(deploy_dir, exist_ok=True)
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")
    scaler = joblib.load(os.path.join(data_dir, 'scaler.pkl'))
    features = joblib.load(os.path.join(data_dir, 'feature_names.pkl'))
    joblib.dump(model, os.path.join(deploy_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(deploy_dir, 'scaler.pkl'))
    joblib.dump(features, os.path.join(deploy_dir, 'features.pkl'))
    info = dict(model_name=model_name, stage=stage, deployed=datetime.now().isoformat())
    with open(os.path.join(deploy_dir, 'info.json'), 'w') as f:
        json.dump(info, f)
    logger.info(f"Prepared deployment in {deploy_dir}")
    return deploy_dir

# ---- Monitoring functions ----

def generate_data_sample(n=1000, drift=0, data_dir='data/processed'):
    """Sample data and optionally apply drift"""
    X = np.load(os.path.join(data_dir,'X_test.npy'))
    y = np.load(os.path.join(data_dir,'y_test.npy'))
    idx = np.random.choice(len(X), n, replace=True)
    Xs, ys = X[idx], y[idx]
    if drift>0:
        shift = 1 + drift/100
        Xs = Xs * shift
        logger.info(f"Applied {drift}% drift")
    return Xs, ys


def monitor_model(deploy_dir='models/deployed', cycles=[0,5,10], out='models/monitoring'):
    """Run monitoring cycles and log F1-score drift to MLflow."""
    os.makedirs(out, exist_ok=True)
    model = joblib.load(os.path.join(deploy_dir,'model.pkl'))
    scaler = joblib.load(os.path.join(deploy_dir,'scaler.pkl'))
    history=[]
    for d in cycles:
        Xs, ys = generate_data_sample(n=1000, drift=d) #fixed d issue
        Xt = scaler.transform(Xs)
        yp = model.predict(Xt)
        f1 = f1_score(ys, yp)
        m = dict(
            timestamp=datetime.now().isoformat(),
            drift=d,
            f1_score=f1
        )
        history.append(m)
        mlflow.log_metric('f1_score', f1, step=d)
    # plot F1-score vs drift
    df=pd.DataFrame(history)
    plt.figure()
    plt.plot(df['drift'], df['f1_score'], marker='o')
    plt.xlabel('Drift %'); plt.ylabel('F1 Score'); plt.title('Drift Impact on F1')
    plt.savefig(os.path.join(out,'drift_f1_score.png'))
    plt.close()
    mlflow.log_artifact(os.path.join(out,'drift_f1_score.png'))
    logger.info('Monitoring complete')
    return history

# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deploy', action='store_true')
    parser.add_argument('--monitor', action='store_true')
    args = parser.parse_args()
    model_name = 'FraudDetectionModel'

    if args.deploy:
        run_id = get_best_run('Hyp_Tune_V11','f1_score')
        if run_id:
            register_and_stage(run_id, model_name, 'Staging')
            metrics = evaluate_registered(model_name, 'Staging')
            if metrics['f1_score']>0.5:
                register_and_stage(run_id, model_name, 'Production')
                prepare_deployment(model_name, 'Production')

    if args.monitor:
        monitor_model()

if __name__=='__main__':
    main()