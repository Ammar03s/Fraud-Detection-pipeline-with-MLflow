import os
import sys
import json
import logging
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "model_monitoring_V1"
mlflow.set_experiment(experiment_name)

def load_deployed_model(model_dir='models/deployed'):
    """Load the deployed model, scaler, and feature names"""
    logger.info(f"Loading deployed model from {model_dir}")
    
    try:
        model = joblib.load(os.path.join(model_dir, 'model.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
        
        with open(os.path.join(model_dir, 'model_info.json'), 'r') as f:
            model_info = json.load(f)
        
        logger.info(f"Loaded model: {model_info['model_name']} (stage: {model_info['stage']})")
        
        return model, scaler, feature_names, model_info
    except Exception as e:
        logger.error(f"Error loading deployed model: {str(e)}")
        return None, None, None, None

def generate_simulated_data(n_samples=1000, drift_percent=0, data_dir='data/processed'):
    """
    Generate simulated data for monitoring
    Optionally introduce data drift by modifying feature distributions
    """
    logger.info(f"Generating simulated data for monitoring with {drift_percent}% drift")
    
    # Load original test data
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # Sample from test data
    indices = np.random.choice(range(len(X_test)), n_samples, replace=True)
    X_sample = X_test[indices]
    y_sample = y_test[indices]
    
    # Introduce drift if requested
    if drift_percent > 0:
        # Randomly select features to drift
        n_features = X_sample.shape[1]
        n_drift_features = int(n_features * drift_percent / 100)
        drift_features = np.random.choice(range(n_features), n_drift_features, replace=False)
        
        # Add drift to selected features
        for feature in drift_features:
            # Shift the feature by some amount
            shift = np.random.uniform(0.5, 2.0)
            X_sample[:, feature] = X_sample[:, feature] * shift
            
        logger.info(f"Introduced drift to {n_drift_features} features")
    
    return X_sample, y_sample

def evaluate_model_on_data(model, X, y):
    """Evaluate the model on the given data and return metrics"""
    # Make predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_proba)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }
    
    return metrics, y_pred, y_pred_proba

def plot_model_drift(metrics_over_time, output_dir='models/monitoring'):
    """Plot model performance metrics over time"""
    logger.info("Plotting model performance over time")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame for easier plotting
    metrics_df = pd.DataFrame(metrics_over_time)
    
    # Set the timestamp as index
    metrics_df.set_index('timestamp', inplace=True)
    
    # Plot each metric over time
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    plt.figure(figsize=(12, 8))
    for metric in metrics_to_plot:
        plt.plot(metrics_df.index, metrics_df[metric], marker='o', label=metric)
    
    plt.xlabel('Timestamp')
    plt.ylabel('Score')
    plt.title('Model Performance Metrics Over Time')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'model_drift_metrics.png')
    plt.savefig(filename)
    plt.close()
    
    logger.info(f"Saved performance plot to {filename}")
    
    return filename

def plot_prediction_distribution(y_pred_proba, y_true, output_dir='models/monitoring'):
    """Plot the distribution of prediction probabilities"""
    logger.info("Plotting prediction probability distribution")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a DataFrame for the data
    df = pd.DataFrame({
        'probability': y_pred_proba,
        'true_label': y_true
    })
    
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='probability', hue='true_label', bins=50, element='step')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Probabilities')
    plt.grid(True)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'prediction_distribution.png')
    plt.savefig(filename)
    plt.close()
    
    logger.info(f"Saved prediction distribution plot to {filename}")
    
    return filename

def run_monitoring_cycle(model, scaler, feature_names, drift_percent=0):
    """Run a single monitoring cycle"""
    logger.info("Running monitoring cycle")
    
    # Generate simulated data (in a real scenario, this would be incoming production data)
    X_monitor, y_monitor = generate_simulated_data(n_samples=1000, drift_percent=drift_percent)
    
    # Evaluate model on data
    metrics, y_pred, y_pred_proba = evaluate_model_on_data(model, X_monitor, y_monitor)
    
    # Add timestamp to metrics
    metrics['timestamp'] = datetime.now()
    
    logger.info(f"Monitoring cycle metrics:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    return metrics, y_pred, y_pred_proba, y_monitor

def main():
    """Main function to run model monitoring"""
    # Load deployed model
    model, scaler, feature_names, model_info = load_deployed_model()
    if model is None:
        logger.error("Failed to load deployed model")
        return
    
    # Create monitoring directory
    monitoring_dir = 'models/monitoring'
    os.makedirs(monitoring_dir, exist_ok=True)
    
    # Load existing monitoring data if available
    monitoring_file = os.path.join(monitoring_dir, 'monitoring_metrics.json')
    if os.path.exists(monitoring_file):
        with open(monitoring_file, 'r') as f:
            monitoring_data = json.load(f)
            metrics_over_time = monitoring_data['metrics_over_time']
    else:
        metrics_over_time = []
    
    # Run multiple monitoring cycles with increasing drift to simulate time passing
    with mlflow.start_run():
        drift_levels = [0, 5, 10, 15, 20]  # Percentages of drift
        
        for drift in drift_levels:
            # Run monitoring cycle
            metrics, y_pred, y_pred_proba, y_true = run_monitoring_cycle(
                model, scaler, feature_names, drift_percent=drift
            )
            
            # Add metrics to history
            metrics_dict = {key: float(value) for key, value in metrics.items() if key != 'timestamp'}
            metrics_dict['timestamp'] = metrics['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            metrics_dict['drift_percent'] = drift
            metrics_over_time.append(metrics_dict)
            
            # Log metrics to MLflow
            mlflow_metrics = {key: value for key, value in metrics_dict.items() 
                              if key not in ['timestamp', 'drift_percent']}
            mlflow.log_metrics(mlflow_metrics)
            
            # If this is the last cycle, generate plots
            if drift == drift_levels[-1]:
                # Plot prediction distribution
                pred_plot = plot_prediction_distribution(y_pred_proba, y_true)
                mlflow.log_artifact(pred_plot)
        
        # Plot metrics over time
        metrics_plot = plot_model_drift(metrics_over_time)
        mlflow.log_artifact(metrics_plot)
        
        # Save monitoring data
        monitoring_data = {
            'metrics_over_time': metrics_over_time,
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(monitoring_file, 'w') as f:
            json.dump(monitoring_data, f, indent=4)
        
        logger.info(f"Monitoring data saved to {monitoring_file}")
        
        # Check for significant drift
        latest_metrics = metrics_over_time[-1]
        baseline_metrics = metrics_over_time[0]
        
        # Calculate drift magnitude
        drift_magnitude = abs(latest_metrics['roc_auc'] - baseline_metrics['roc_auc'])
        mlflow.log_metric("drift_magnitude", drift_magnitude)
        
        # Alert if drift exceeds threshold
        if drift_magnitude > 0.05:  # 5% drop in ROC AUC
            alert_message = f"ALERT: Model performance has drifted by {drift_magnitude:.4f} in ROC AUC"
            logger.warning(alert_message)
            
            # In a real scenario, this could trigger an email, Slack notification, etc.
            alert_file = os.path.join(monitoring_dir, 'drift_alert.txt')
            with open(alert_file, 'w') as f:
                f.write(f"{alert_message}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Baseline ROC AUC: {baseline_metrics['roc_auc']:.4f}\n")
                f.write(f"Current ROC AUC: {latest_metrics['roc_auc']:.4f}\n")
                f.write("Recommendation: Retrain model with new data\n")
            
            mlflow.log_artifact(alert_file)
        else:
            logger.info(f"No significant drift detected. Drift magnitude: {drift_magnitude:.4f}")

if __name__ == "__main__":
    main()