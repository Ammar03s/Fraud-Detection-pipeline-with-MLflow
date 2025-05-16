import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns


logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Pre-Processing_pptZ")


def load_data(filepath, sample_size = 10_000, fraud_frac = 0.2, target_col = None): #data is too large so i took a sample
    logger.info(f"loading data from {filepath}")
    df = pd.read_csv(filepath)
    if target_col is None:
        orig = df.columns[-1]
        df = df.rename(columns = {orig: 'target'})
        target_col = 'target'
    logger.info(f"Using target column: {target_col} (data shape {df.shape})")
    if len(df) > sample_size:
        n_pos = min(int(sample_size * fraud_frac), int(df[target_col].sum()))
        n_neg = sample_size - n_pos
        pos = df[df[target_col] == 1].sample(n_pos, random_state=42)
        neg = df[df[target_col] == 0].sample(n_neg, random_state=42)
        df = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(f"Sampled to {len(df)} rows (~{fraud_frac*100:.0f}% positives)")
    return df, target_col




def preprocess_data(df, target_col):
    logger.info("Starting data preprocessing")
    thresh = len(df) * 0.3
    df = df.dropna(axis=1, thresh=thresh)
    logger.info(f"Dropped high-missing cols, remaining shape {df.shape}")
    num_cols = df.select_dtypes(include=['int64','float64']).columns.drop(target_col)
    cat_cols = [c for c in df.select_dtypes(include=['object']).columns if c != target_col]
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])
    for c in cat_cols:
        topk = df[c].value_counts().nlargest(10).index
        df[c] = df[c].where(df[c].isin(topk), other='Other')
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    logger.info(f"Preprocessed data shape: {df.shape}")
    return df


def split_and_scale_data(df, target_col='target', test_size = 0.2, random_state=42): #target_col is adjustable
    logger.info("splitting and scaling data")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    strat = y if y.nunique() == 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat
    )
    scaler = StandardScaler() #minmax scaler
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    logger.info(f"Split: train {X_train_scaled.shape}, test {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()





def save_processed_data(X_train, X_test, y_train, y_test, scaler, feature_names, output_dir='data/processed'):
    logger.info(f"Saving processed data to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'),  X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train.values)
    np.save(os.path.join(output_dir, 'y_test.npy'),  y_test.values)
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(feature_names, os.path.join(output_dir, 'feature_names.pkl'))
    logger.info("Data & artifacts saved")





def main():
    input_filepath = 'data/Fraud_Detection_Dataset.csv'
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)

    with mlflow.start_run():
        mlflow.log_param("input_filepath", input_filepath)
        mlflow.log_param("output_dir", output_dir)

        #1 Load & sample
        df, target_col = load_data(input_filepath, sample_size=500_000, fraud_frac=0.2)
        mlflow.log_param("target_col", target_col)
        #2 preprocess
        df_processed = preprocess_data(df, target_col)
        #3 artifacts: head of data
        head_path = os.path.join(output_dir, "df_head.csv")
        df_processed.head().to_csv(head_path, index=False)
        mlflow.log_artifact(head_path)
        #4 artifacts
        describe_path = os.path.join(output_dir, "df_describe.csv")
        df_processed.describe().T.to_csv(describe_path)
        mlflow.log_artifact(describe_path)
        #5 artifact2.0
        num_df = df_processed.select_dtypes(include=['int64', 'float64'])
        corr = num_df.corr()
        plt.figure(figsize=(12,10))
        sns.heatmap(corr, cmap='vlag', center=0, cbar_kws={'shrink': .5})
        plt.title("Numeric Feature Correlation Heatmap")
        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
        mlflow.log_artifact(heatmap_path)

        #6 log processed col. count
        mlflow.log_metric("processed_columns", df_processed.shape[1])

        #7 split & scale
        X_train, X_test, y_train, y_test, scaler, feats = split_and_scale_data(df_processed, target_col)
        mlflow.log_metric("fraud_pct_train", y_train.mean() * 100)
        mlflow.log_metric("fraud_pct_test",  y_test.mean() * 100)

        #8 save processed arrays & scaler
        save_processed_data(X_train, X_test, y_train, y_test, scaler, feats, output_dir)
        logger.info("Pre-Processing completed successfully")




if __name__ == "__main__":
    main()
