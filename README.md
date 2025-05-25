# Fraud Hunter - MLOps Project

A  fraud detection system built with MLflow for model tracking, hyperparameter tuning, deployment, and monitoring.

## Project Overview

This project implements a complete MLOps pipeline for fraud detection:
- Data preprocessing and feature engineering
- Model training with multiple algorithms
- Hyperparameter tuning with MLflow tracking
- Model deployment and serving
- Real-time monitoring dashboard for model performance

## Dataset:
https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset

### Setup

```

1. Create a virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the Project

### 1. Start MLflow Server

Start the MLflow tracking server:
```bash
mlflow server --host 0.0.0.0 --port 5000
```

(Keep this terminal window open and running)

### 2. Data Preprocessing

In a new terminal, navigate to the project directory and run:
```bash
# Activate the virtual environment first

# Run preprocessing
python src/preprocess.py
```

This will:
- Load the raw transaction data
- Create features for fraud detection
- Split data into training and testing sets
- Save processed data for model training
- Note: this preprocessing part was designed in a way to work on an binray classification dataset so in other words other dataset can be used

### 3. Hyperparameter Tuning

Run the hyperparameter tuning script:
```bash
python src/Hyp_tune.py
```

This will:
- Train Random Forest & LightGBM models with different hyperparameters
- Log all experiments to MLflow
- Select the best model based on F1 score
- Save the best model to the models directory

### 4. Model Deployment

Deploy the best model:
```bash
python src/Monitor_and_Deploy.py --deploy 
python src/Monitor_and_Deploy.py --monitor
```

This will:
- Select the best performing model from the experiments
- Create a deployment package with model artifacts
- Register the model in MLflow Model Registry
- Transition the model to production stage

## License

MIT 
