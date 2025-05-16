import os
import numpy as np
import pandas as pd
import joblib
import logging
import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns






logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "Hyperparameter_Tune_pptZ"
mlflow.set_experiment(experiment_name)


def load_processed_data(data_dir = 'data/processed'):
    logger.info(f"loading processed data from = {data_dir}")
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    logger.info(f"Loaded data shapes == X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test



def get_space_rf():
    return {
        'n_estimators': scope.int(hp.quniform('rf_n_estimators', 50, 300, 10)),
        'max_depth': scope.int(hp.quniform('rf_max_depth', 5, 30, 1)),
        'min_samples_split': scope.int(hp.quniform('rf_min_samples_split', 2, 20, 1)),
        'min_samples_leaf': scope.int(hp.quniform('rf_min_samples_leaf', 1, 10, 1)),
        'max_features': hp.choice('rf_max_features', ['sqrt', 'log2', None]),
        'class_weight': hp.choice('rf_class_weight', ['balanced', 'balanced_subsample', None])
    }


def get_space_lgbm():
    return {
        'n_estimators': scope.int(hp.quniform('lgbm_n_estimators', 50, 300, 10)),
        'max_depth': scope.int(hp.quniform('lgbm_max_depth', 5, 30, 1)),
        'learning_rate': hp.loguniform('lgbm_learning_rate', np.log(0.01), np.log(0.3)),
        'num_leaves': scope.int(hp.quniform('lgbm_num_leaves', 20, 100, 1)),
        'subsample': hp.uniform('lgbm_subsample', 0.6, 1.0),
        'colsample_bytree': hp.uniform('lgbm_colsample_bytree', 0.6, 1.0),
        'min_child_samples': scope.int(hp.quniform('lgbm_min_child_samples', 5, 50, 1)),
        'scale_pos_weight': hp.loguniform('lgbm_scale_pos_weight', np.log(1), np.log(100))
    }


def create_model_rf(params):
    return RandomForestClassifier(
        n_estimators = int(params['n_estimators']),
        max_depth = int(params['max_depth']),
        min_samples_split = int(params['min_samples_split']),
        min_samples_leaf = int(params['min_samples_leaf']),
        max_features = params['max_features'],
        class_weight = params['class_weight'],
        random_state=42
    )


def create_model_lgbm(params):
    return LGBMClassifier(
        n_estimators = int(params['n_estimators']),
        max_depth = int(params['max_depth']),
        learning_rate = float(params['learning_rate']),
        num_leaves = int(params['num_leaves']),
        subsample = float(params['subsample']),
        colsample_bytree = float(params['colsample_bytree']),
        min_child_samples = int(params['min_child_samples']),
        scale_pos_weight = float(params['scale_pos_weight']),
        random_state=42
    )





def objective_rf(params, X_train, y_train, X_test, y_test):
    with mlflow.start_run(nested=True):
        model = create_model_rf(params)
        mlflow.log_params(params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_metrics({'roc_auc': roc_auc, 'F1_score': f1})
        mlflow.sklearn.log_model(model, 'model')
        logger.info(f"Random Forest trial ROC AUC: {roc_auc:.2f}, F1: {f1:.2f}")
        return {'loss': -roc_auc, 'status': STATUS_OK, 'model': model}


def objective_lgbm(params, X_train, y_train, X_test, y_test):
    with mlflow.start_run(nested = True):
        model = create_model_lgbm(params)
        mlflow.log_params(params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_metrics({'roc_auc': roc_auc, 'f1_score': f1})
        mlflow.sklearn.log_model(model, 'model')
        logger.info(f"LGBM trial ROC AUC: {roc_auc:.2f}, F1: {f1:.2f}")
        return {'loss': -roc_auc, 'status': STATUS_OK, 'model': model}




def optimize_rf(X_train, y_train, X_test, y_test, max_evals = 10): #was 20
    logger.info("Tuning RF hyperparams...")
    with mlflow.start_run(run_name='RF_Hyperopt'):
        mlflow.log_param('max_evals', max_evals)
        trials = Trials()
        best = fmin(
            fn=lambda p: objective_rf(p, X_train, y_train, X_test, y_test),
            space=get_space_rf(),
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials
        )

        best_trial = min(trials.trials, key=lambda t: t['result']['loss'])
        best_model = best_trial['result']['model']
        best_auc = -best_trial['result']['loss']
        y_pred = best_model.predict(X_test)
        best_f1 = f1_score(y_test, y_pred)

        mlflow.log_metrics({'best_rf_f1': best_f1})
        joblib.dump(best_model, 'models/RF_best_hyperopt.pkl')
        logger.info(f"best RF ROC AUC: {best_auc:.2f}, F1: {best_f1:.2f}")

        return best_model, best_auc, best_f1, trials

def optimize_lgbm(X_train, y_train, X_test, y_test, max_evals=10): #was 20
    logger.info("tuning LGBM hyperparamters.....")
    with mlflow.start_run(run_name = 'LGBM_Hyperopt'):
        mlflow.log_param('max_evals', max_evals)
        trials = Trials()
        best = fmin(
            fn=lambda p: objective_lgbm(p, X_train, y_train, X_test, y_test),
            space=get_space_lgbm(),
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials
        )

        best_trial = min(trials.trials, key=lambda t: t['result']['loss'])
        best_model = best_trial['result']['model']
        best_auc = -best_trial['result']['loss']
        y_pred = best_model.predict(X_test)
        best_f1 = f1_score(y_test, y_pred)

        mlflow.log_metrics({'best_lgbm_f1': best_f1})
        joblib.dump(best_model, 'models/LGBM_best_hyperopt.pkl')
        logger.info(f"best LightGBM ROC AUC: {best_auc:.2f}, F1: {best_f1:.2f}")


        return best_model, best_auc, best_f1, trials






def main():
    X_train, X_test, y_train, y_test = load_processed_data()
    os.makedirs('models', exist_ok=True)
    rf_model, _rf_auc, rf_f1, _rf_trials = optimize_rf(X_train, y_train, X_test, y_test)
    lgbm_model, _lgbm_auc, lgbm_f1, _lgbm_trials = optimize_lgbm(X_train, y_train, X_test, y_test) #_ 
    #based on F1 score
    if rf_f1 > lgbm_f1:
        best_model, best_score, name = rf_model, rf_f1, 'RandomForest'
    else:
        best_model, best_score, name = lgbm_model, lgbm_f1, 'LightGBM'
    logger.info(f"overall best: {name} with F1-score = {best_score:.2f}")
    
    # Generate confusion matrix and save it
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'Confusion Matrix - {name}')
    cm_path = f'confusion_{name}.png'
    plt.savefig(cm_path)
    plt.close()

    # Log artifacts to MLflow
    mlflow.log_artifact(cm_path)
    
    # Generate and log classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    rpt_path = f'report_{name}.csv'
    report_df.to_csv(rpt_path)
    mlflow.log_artifact(rpt_path)
    
    # Log best model info
    mlflow.set_tag('best_model', name)
    mlflow.log_metric('best_model_f1', best_score)
    logger.info("hyperparameter tuning is done sir")




if __name__ == '__main__':
    main()
