import os
import numpy as np
import joblib
import logging
import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "hyperparameter_tuning_V1"
mlflow.set_experiment(experiment_name)

def load_processed_data(data_dir='data/processed'):
    """Load the processed data"""
    logger.info(f"Loading processed data from {data_dir}")
    
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    logger.info(f"Loaded data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def get_space_rf():
    """Define the search space for Random Forest"""
    space = {
        'n_estimators': scope.int(hp.quniform('n_estimators', 50, 300, 10)),
        'max_depth': scope.int(hp.quniform('max_depth', 5, 30, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 10, 1)),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
        'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample', None])
    }
    return space

def get_space_xgb():
    """Define the search space for XGBoost"""
    space = {
        'n_estimators': scope.int(hp.quniform('n_estimators', 50, 300, 10)),
        'max_depth': scope.int(hp.quniform('max_depth', 3, 12, 1)),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
        'gamma': hp.loguniform('gamma', np.log(0.0001), np.log(1.0)),
        'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 10, 1)),
        'scale_pos_weight': hp.loguniform('scale_pos_weight', np.log(1), np.log(100))
    }
    return space

def create_model_rf(params):
    """Create a Random Forest model with the given parameters"""
    # Convert parameters to appropriate types
    params = {
        'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth']),
        'min_samples_split': int(params['min_samples_split']),
        'min_samples_leaf': int(params['min_samples_leaf']),
        'max_features': params['max_features'],
        'class_weight': params['class_weight'],
        'random_state': 42
    }
    
    return RandomForestClassifier(**params)

def create_model_xgb(params):
    """Create an XGBoost model with the given parameters"""
    # Convert parameters to appropriate types
    params = {
        'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth']),
        'learning_rate': float(params['learning_rate']),
        'subsample': float(params['subsample']),
        'colsample_bytree': float(params['colsample_bytree']),
        'gamma': float(params['gamma']),
        'min_child_weight': int(params['min_child_weight']),
        'scale_pos_weight': float(params['scale_pos_weight']),
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42
    }
    
    return XGBClassifier(**params)

def objective_rf(params, X_train, y_train, X_test, y_test):
    """Objective function for Hyperopt to minimize for Random Forest"""
    with mlflow.start_run(nested=True):
        # Create model
        model = create_model_rf(params)
        
        # Log parameters
        params_to_log = {key: val for key, val in params.items()}
        mlflow.log_params(params_to_log)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        })
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        logger.info(f"RandomForest - ROC AUC: {roc_auc:.4f}")
        
        # We want to maximize ROC AUC, but Hyperopt tries to minimize
        return {'loss': -roc_auc, 'status': STATUS_OK, 'model': model}

def objective_xgb(params, X_train, y_train, X_test, y_test):
    """Objective function for Hyperopt to minimize for XGBoost"""
    with mlflow.start_run(nested=True):
        # Create model
        model = create_model_xgb(params)
        
        # Log parameters
        params_to_log = {key: val for key, val in params.items()}
        mlflow.log_params(params_to_log)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        })
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        logger.info(f"XGBoost - ROC AUC: {roc_auc:.4f}")
        
        # We want to maximize ROC AUC, but Hyperopt tries to minimize
        return {'loss': -roc_auc, 'status': STATUS_OK, 'model': model}

def optimize_rf(X_train, y_train, X_test, y_test, max_evals=50):
    """Optimize Random Forest hyperparameters"""
    logger.info("Starting Random Forest hyperparameter tuning")
    
    with mlflow.start_run(run_name="RandomForest_Hyperopt"):
        mlflow.log_param("max_evals", max_evals)
        mlflow.log_param("model_type", "RandomForest")
        
        # Define the objective function
        def objective(params):
            return objective_rf(params, X_train, y_train, X_test, y_test)
        
        # Run optimization
        trials = Trials()
        best = fmin(
            fn=objective,
            space=get_space_rf(),
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials
        )
        
        # Get the best model from the trial with the lowest loss
        best_trial = min(trials.trials, key=lambda trial: trial['result']['loss'])
        best_model = best_trial['result']['model']
        best_roc_auc = -best_trial['result']['loss']  # Convert back to positive
        
        # Log best parameters and metrics
        mlflow.log_metric("best_roc_auc", best_roc_auc)
        
        # Save the best model
        os.makedirs('models', exist_ok=True)
        joblib.dump(best_model, 'models/RF_best_hyperopt.pkl')
        
        logger.info(f"Best RandomForest model - ROC AUC: {best_roc_auc:.4f}")
        logger.info(f"Best parameters: {best}")
        
        return best_model, best_roc_auc

def optimize_xgb(X_train, y_train, X_test, y_test, max_evals=50):
    """Optimize XGBoost hyperparameters"""
    logger.info("Starting XGBoost hyperparameter tuning")
    
    with mlflow.start_run(run_name="XGBoost_Hyperopt"):
        mlflow.log_param("max_evals", max_evals)
        mlflow.log_param("model_type", "XGBoost")
        
        # Define the objective function
        def objective(params):
            return objective_xgb(params, X_train, y_train, X_test, y_test)
        
        # Run optimization
        trials = Trials()
        best = fmin(
            fn=objective,
            space=get_space_xgb(),
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials
        )
        
        # Get the best model from the trial with the lowest loss
        best_trial = min(trials.trials, key=lambda trial: trial['result']['loss'])
        best_model = best_trial['result']['model']
        best_roc_auc = -best_trial['result']['loss']  # Convert back to positive
        
        # Log best parameters and metrics
        mlflow.log_metric("best_roc_auc", best_roc_auc)
        
        # Save the best model
        os.makedirs('models', exist_ok=True)
        joblib.dump(best_model, 'models/XGB_best_hyperopt.pkl')
        
        logger.info(f"Best XGBoost model - ROC AUC: {best_roc_auc:.4f}")
        logger.info(f"Best parameters: {best}")
        
        return best_model, best_roc_auc

def main():
    """Main function to run hyperparameter tuning"""
    # Load processed data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Configuration
    max_evals = 10 #was 30
    
    # Optimize Random Forest hyperparameters
    rf_best_model, rf_best_roc_auc = optimize_rf(X_train, y_train, X_test, y_test, max_evals)
    
    # Optimize XGBoost hyperparameters
    xgb_best_model, xgb_best_roc_auc = optimize_xgb(X_train, y_train, X_test, y_test, max_evals)
    
    # Compare best models
    logger.info("Hyperparameter tuning completed")
    logger.info(f"Best RandomForest model - ROC AUC: {rf_best_roc_auc:.4f}")
    logger.info(f"Best XGBoost model - ROC AUC: {xgb_best_roc_auc:.4f}")
    
    # Determine the overall best model
    best_model_name = "RandomForest" if rf_best_roc_auc > xgb_best_roc_auc else "XGBoost"
    best_roc_auc = max(rf_best_roc_auc, xgb_best_roc_auc)
    
    logger.info(f"Overall best model: {best_model_name} with ROC AUC: {best_roc_auc:.4f}")

if __name__ == "__main__":
    main()