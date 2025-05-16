import os
import numpy as np
import pandas as pd
import joblib
import logging
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "Train_model_Version11"
mlflow.set_experiment(experiment_name)

def load_processed_data(data_dir='data/processed'):
    """Load the processed data from the data directory"""
    logger.info(f"Loading processed data from {data_dir}")
    
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    feature_names = joblib.load(os.path.join(data_dir, 'feature_names.pkl'))
    
    logger.info(f"Loaded data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_names

def save_confusion_matrix_plot(y_true, y_pred, filename):
    """Generate and save a confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close()
    return filename

def evaluate_model(model, X_test, y_test, feature_names=None):
    """Evaluate a model and calculate performance metrics"""
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Generate confusion matrix
    cm_filename = f"confusion_matrix_{model.__class__.__name__}.png"
    save_confusion_matrix_plot(y_test, y_pred, cm_filename)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Log results
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    
    # Feature importance if available
    feature_importance = None
    
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        feature_importance = dict(zip(feature_names, model.feature_importances_))
    elif hasattr(model, 'coef_') and feature_names is not None:
        # For linear models like logistic regression
        feature_importance = dict(zip(feature_names, model.coef_[0]))
    
    metrics = {"accuracy": accuracy, "precision": precision , "recall": recall , "f1_score": f1 , "roc_auc": roc_auc}
    return metrics, cm_filename, feature_importance, report

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test, feature_names=None, params=None):
    """Train a model, evaluate it, and log results to MLflow"""
    logger.info(f"Training {model_name} model")
    
    with mlflow.start_run(run_name=model_name):
        # Log model parameters
        if params:
            mlflow.log_params(params)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate the model
        metrics, cm_filename, feature_importance, report = evaluate_model(
            model, X_test, y_test, feature_names
        )
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log confusion matrix
        mlflow.log_artifact(cm_filename)
        os.remove(cm_filename)  # Remove local file after logging
        
        # Log feature importance if available
        if feature_importance:
            # Convert to DataFrame for better visualization
            fi_df = pd.DataFrame({
                'Feature': list(feature_importance.keys()),
                'Importance': list(feature_importance.values())
            }).sort_values('Importance', ascending=False)
            
            # Save and log feature importance plot
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=fi_df.head(20))
            plt.title(f'Top 20 Feature Importance - {model_name}')
            plt.tight_layout()
            fi_filename = f"feature_importance_{model_name}.png"
            plt.savefig(fi_filename)
            plt.close()
            mlflow.log_artifact(fi_filename)
            os.remove(fi_filename)  # Remove local file after logging
        
        # Log classification report
        report_df = pd.DataFrame(report).transpose()
        report_filename = f"classification_report_{model_name}.csv"
        report_df.to_csv(report_filename)
        mlflow.log_artifact(report_filename)
        os.remove(report_filename)  # Remove local file after logging
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        # Save the model locally
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, f"models/{model_name}.pkl")
        
        logger.info(f"{model_name} model training completed and logged to MLflow")
        
        return metrics

def main():
    """Main function to train multiple models"""
    # Load processed data
    X_train, X_test, y_train, y_test, feature_names = load_processed_data()
    
    # Define models to train
    models = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            "params": {"C": 1.0, "penalty": "l2", "solver": "lbfgs"}
        },
        "RandomForest": {
            "model": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
            "params": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2}
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(class_weight='balanced', random_state=42),
            "params": {"max_depth": 10, "min_samples_split": 2, "criterion": "gini"}
        },
        "XGBoost": {
            "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            "params": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
        },
        "LightGBM": {
            "model": LGBMClassifier(random_state=42),
            "params": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
        }
    }
    
    # Train and evaluate each model
    results = {}
    for model_name, model_config in models.items():
        try:
            metrics = train_and_log_model(
                model_config["model"],
                model_name,
                X_train, X_test, y_train, y_test,
                feature_names,
                model_config["params"]
            )
            results[model_name] = metrics
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            continue
    
    # Determine the best model based on ROC AUC score
    if results:
        best_model = max(results.items(), key=lambda x: x[1]["roc_auc"])
        logger.info(f"Best model: {best_model[0]} with ROC AUC: {best_model[1]['roc_auc']:.4f}")
    else:
        logger.warning("No models were trained successfully")




if __name__ == "__main__":
    main()