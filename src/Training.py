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
) #had to add this
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns




logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "Train_Model_pptZ"
mlflow.set_experiment(experiment_name)



def load_processed_data(data_dir = 'data/processed'):
    logger.info(f"Loading processed data from {data_dir}")
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    feature_names = joblib.load(os.path.join(data_dir, 'feature_names.pkl'))
    logger.info(f"Loaded data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_names



def save_confusion_matrix_plot(y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize = (10, 8))
    sns.heatmap(cm, annot=True, fmt = 'd', cmap = 'Blues')
    plt.xlabel('predicted')
    plt.ylabel('true')
    plt.title('confusion matrix')
    plt.savefig(filename)
    plt.close()
    return filename



def evaluate_model(model, X_test, y_test, feature_names = None):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm_filename = f"confusion_matrix_{model.__class__.__name__}.png"
    save_confusion_matrix_plot(y_test, y_pred, cm_filename)


    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info(f"model: {model.__class__.__name__}")
    logger.info(f"accuracy: {accuracy:.2f}")
    logger.info(f"precision: {precision:.2f}")
    logger.info(f"recall: {recall:.2f}")
    logger.info(f"F1 score: {f1:.2f}")
    logger.info(f"ROC AUC: {roc_auc:.2f}")
    feature_importance = None


    if hasattr(model, 'feature_importances_') and feature_names is not None:
        feature_importance = dict(zip(feature_names, model.feature_importances_))
    elif hasattr(model, 'coef_') and feature_names is not None:
        feature_importance = dict(zip(feature_names, model.coef_[0]))
    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "roc_auc": roc_auc}
    return metrics, cm_filename, feature_importance, report


def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test, feature_names = None, params = None):
    logger.info(f"training {model_name} model")
    with mlflow.start_run(run_name=model_name):
        if params:
            mlflow.log_params(params)
        #cross-val
        cv_scores = cross_val_score(model, X_train, y_train, cv = 5, scoring = 'roc_auc')
        mlflow.log_metric('cv_roc_auc_mean', cv_scores.mean())
        mlflow.log_metric('cv_roc_auc_std', cv_scores.std())
        #train model
        model.fit(X_train, y_train)
        #eval model
        metrics, cm_filename, feature_importance, report = evaluate_model(model, X_test, y_test, feature_names)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(cm_filename)
        os.remove(cm_filename)

        #top 3 feat.
        if feature_importance:
            fi_df = pd.DataFrame({'Feature': list(feature_importance.keys()), 'Importance': list(feature_importance.values())}).sort_values('Importance', ascending=False)
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=fi_df.head(3))
            plt.title(f'top 3 feat. importance - {model_name}')
            plt.tight_layout()
            fi_filename = f"feature_importance_{model_name}.png"
            plt.savefig(fi_filename)
            plt.close()
            mlflow.log_artifact(fi_filename)
            os.remove(fi_filename)
            
        #log classification report
        report_df = pd.DataFrame(report).transpose()
        report_filename = f"classification_report_{model_name}.csv"
        report_df.to_csv(report_filename)
        mlflow.log_artifact(report_filename)
        os.remove(report_filename)
        #log the model
        mlflow.sklearn.log_model(model, "model")
        #save locally
        os.makedirs('models', exist_ok = True)
        joblib.dump(model, f"models/{model_name}.pkl")
        logger.info(f"{model_name} model training completed and logged to MLflow")

        return metrics



def main():
    X_train, X_test, y_train, y_test, feature_names = load_processed_data()
    #handle imbalance since this is a fraud detection model
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_weight = (n_neg / n_pos) if n_pos > 0 else 1
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
            "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=scale_weight),
            "params": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "scale_pos_weight": scale_weight}
        },
        "LightGBM": {
            "model": LGBMClassifier(random_state=42, scale_pos_weight=scale_weight),
            "params": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "scale_pos_weight": scale_weight}
        }
    }
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
            logger.error(f"error training {model_name}: {str(e)}")
            continue
    if results:
        best_model = max(results.items(), key=lambda x: x[1]["roc_auc"])
        logger.info(f"best model: {best_model[0]} with ROC AUC: {best_model[1]['roc_auc']:.2f}")
    else:
        logger.warning("no models were trained correctly")




if __name__ == "__main__":
    main()
