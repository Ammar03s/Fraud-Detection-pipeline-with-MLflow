import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "data_preprocessing_Version1"
mlflow.set_experiment(experiment_name)

def load_data(filepath, sample_size = 500_000):  #i took a sample cuz the dataset is 2 large
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Data loaded successfully with shape: {df.shape}")
    
    # Sample the data to a manageable size
    if len(df) > sample_size:
        # Ensure we get a balanced sample of fraud and non-fraud cases
        fraud = df[df['isFraud'] == 1].sample(min(sample_size // 10, sum(df['isFraud'] == 1)))
        non_fraud = df[df['isFraud'] == 0].sample(sample_size - len(fraud))
        df = pd.concat([fraud, non_fraud]).sample(frac=1).reset_index(drop=True)
        logger.info(f"Sampled data to {len(df)} rows")
    return df

def preprocess_data(df):
    logger.info("Starting data preprocessing")
    missing_values = df.isnull().sum() #check for missing values
    logger.info(f"missing values before the preprocessing is = {missing_values.sum()}")
    
    #drop columns with too many missing values over 30% (adjustable)
    threshold = len(df) * 0.3 #30%
    df = df.dropna(axis=1, thresh=threshold)
    #for the remaining columns, handle missing values
    numeric_cols = df.select_dtypes(include = ['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include = ['object']).columns
    
    #fill it with median
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    #fill categ. missing values with mode (in case of categorical data)
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    # Check if 'isFraud' column exists
    if 'isFraud' not in df.columns: #the target variable
        logger.error("Target variable not found in dataset")
        raise ValueError("Target variable not found in dataset")
    
    for col in categorical_cols:
        #keep top 10 frequent categories for each col
        top_categories = df[col].value_counts().nlargest(10).index
        df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')
    
    # Now use get_dummies with a more limited set of categories
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    logger.info(f"data preprocessed successfully with shape = {df.shape}")
    return df

def split_and_scale_data(df, target_col = 'isFraud', test_size = 0.2, random_state=42): #target_col is adjustable
    logger.info("Splitting data into train and test sets")

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    #split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    logger.info(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
    
    #scale it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("scaling completed")
    

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()




def save_processed_data(X_train, X_test, y_train, y_test, scaler, feature_names, output_dir='data/processed'):
    """Save the processed data and scaler"""
    logger.info(f"Saving processed data to {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train.values)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test.values)
    
    # Save scaler and feature names
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(feature_names, os.path.join(output_dir, 'feature_names.pkl'))
    
    logger.info("Processed data saved successfully")

def main():
    # Define paths
    input_filepath = 'data/Fraud_Detection_Dataset.csv'
    output_dir = 'data/processed'
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("input_filepath", input_filepath)
        mlflow.log_param("output_dir", output_dir)
        
        try:
            # Load data
            df = load_data(input_filepath)
            
            # Preprocess data
            df_processed = preprocess_data(df)
            
            # Log dataset stats
            mlflow.log_metric("dataset_rows", df.shape[0])
            mlflow.log_metric("dataset_columns", df.shape[1])
            mlflow.log_metric("processed_columns", df_processed.shape[1])
            
            # Split and scale data
            X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale_data(df_processed)
            
            # Calculate class distribution
            fraud_percent_train = y_train.mean() * 100
            fraud_percent_test = y_test.mean() * 100
            
            # Log class distribution
            mlflow.log_metric("fraud_percent_train", fraud_percent_train)
            mlflow.log_metric("fraud_percent_test", fraud_percent_test)
            
            # Save processed data
            save_processed_data(X_train, X_test, y_train, y_test, scaler, feature_names, output_dir)
            
            logger.info("Data preprocessing completed successfully")
        
        except Exception as e:
            logger.error(f"Error during data preprocessing: {str(e)}")
            mlflow.log_param("error", str(e))
            raise

if __name__ == "__main__":
    main()