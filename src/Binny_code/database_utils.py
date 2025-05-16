import os
import psycopg2
import pandas as pd
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database connection parameters
DB_PARAMS = {
    'dbname': 'postgres',  # Default database name
    'user': 'postgres',    # Default PostgreSQL user
    'password': 'ammar2003',
    'host': 'localhost',
    'port': '5432'
}

def get_db_connection():
    """Establish and return a connection to the PostgreSQL database"""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

def initialize_database():
    """Initialize the database by creating necessary tables if they don't exist"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create models table to track model metadata
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            model_id SERIAL PRIMARY KEY,
            model_name VARCHAR(100) NOT NULL,
            model_type VARCHAR(50) NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            mlflow_run_id VARCHAR(50),
            metrics JSONB,
            parameters JSONB,
            artifact_path VARCHAR(255),
            status VARCHAR(20) DEFAULT 'training'
        )
        ''')
        
        # Create predictions table to track model predictions for monitoring
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id SERIAL PRIMARY KEY,
            model_id INTEGER REFERENCES models(model_id),
            prediction_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            input_data JSONB,
            prediction FLOAT,
            actual_label FLOAT NULL,
            prediction_proba FLOAT
        )
        ''')
        
        # Create feature importance table to track feature importance
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feature_importance (
            id SERIAL PRIMARY KEY,
            model_id INTEGER REFERENCES models(model_id),
            feature_name VARCHAR(100) NOT NULL,
            importance FLOAT NOT NULL
        )
        ''')
        
        # Create model_performance table to track ongoing model performance
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_performance (
            id SERIAL PRIMARY KEY,
            model_id INTEGER REFERENCES models(model_id),
            evaluation_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            metric_name VARCHAR(50) NOT NULL,
            metric_value FLOAT NOT NULL,
            evaluation_set VARCHAR(20) NOT NULL
        )
        ''')
        
        conn.commit()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def register_model(model_name, model_type, mlflow_run_id, metrics, parameters, artifact_path):
    """Register a model in the database"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insert model metadata
        cursor.execute('''
        INSERT INTO models (model_name, model_type, mlflow_run_id, metrics, parameters, artifact_path)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING model_id
        ''', (model_name, model_type, mlflow_run_id, json.dumps(metrics), json.dumps(parameters), artifact_path))
        
        model_id = cursor.fetchone()[0]
        conn.commit()
        logger.info(f"Model registered with ID: {model_id}")
        return model_id
    except Exception as e:
        logger.error(f"Failed to register model: {str(e)}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def store_feature_importance(model_id, feature_importances):
    """Store feature importance values for a model"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insert feature importance values
        for feature, importance in feature_importances.items():
            cursor.execute('''
            INSERT INTO feature_importance (model_id, feature_name, importance)
            VALUES (%s, %s, %s)
            ''', (model_id, feature, float(importance)))
        
        conn.commit()
        logger.info(f"Feature importance stored for model ID: {model_id}")
    except Exception as e:
        logger.error(f"Failed to store feature importance: {str(e)}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def log_prediction(model_id, input_data, prediction, prediction_proba, actual_label=None):
    """Log a prediction to the database for monitoring"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insert prediction
        cursor.execute('''
        INSERT INTO predictions (model_id, input_data, prediction, actual_label, prediction_proba)
        VALUES (%s, %s, %s, %s, %s)
        ''', (model_id, json.dumps(input_data), float(prediction), actual_label, float(prediction_proba)))
        
        conn.commit()
        logger.debug(f"Prediction logged for model ID: {model_id}")
    except Exception as e:
        logger.error(f"Failed to log prediction: {str(e)}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

def update_model_performance(model_id, metrics, evaluation_set='test'):
    """Update model performance metrics in the database"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insert each metric as a separate record
        for metric_name, metric_value in metrics.items():
            cursor.execute('''
            INSERT INTO model_performance (model_id, metric_name, metric_value, evaluation_set)
            VALUES (%s, %s, %s, %s)
            ''', (model_id, metric_name, float(metric_value), evaluation_set))
        
        conn.commit()
        logger.info(f"Performance metrics updated for model ID: {model_id}")
    except Exception as e:
        logger.error(f"Failed to update model performance: {str(e)}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def get_best_model_id(metric_name='f2_score', evaluation_set='test'):
    """Get the ID of the best performing model based on a specific metric"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT m.model_id, m.model_name, m.model_type, mp.metric_value 
        FROM models m
        JOIN model_performance mp ON m.model_id = mp.model_id
        WHERE mp.metric_name = %s AND mp.evaluation_set = %s
        ORDER BY mp.metric_value DESC
        LIMIT 1
        ''', (metric_name, evaluation_set))
        
        result = cursor.fetchone()
        if result:
            logger.info(f"Best model found with ID: {result[0]}, {metric_name}: {result[3]}")
            return {
                'model_id': result[0],
                'model_name': result[1],
                'model_type': result[2],
                'metric_value': result[3]
            }
        else:
            logger.warning(f"No models found with metric: {metric_name}")
            return None
    except Exception as e:
        logger.error(f"Failed to get best model: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

def get_model_details(model_id):
    """Get detailed information about a specific model"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get model metadata
        cursor.execute('''
        SELECT model_id, model_name, model_type, created_at, mlflow_run_id, 
               metrics, parameters, artifact_path, status
        FROM models
        WHERE model_id = %s
        ''', (model_id,))
        
        model_info = cursor.fetchone()
        if not model_info:
            logger.warning(f"No model found with ID: {model_id}")
            return None
        
        # Get feature importance
        cursor.execute('''
        SELECT feature_name, importance
        FROM feature_importance
        WHERE model_id = %s
        ORDER BY importance DESC
        ''', (model_id,))
        
        feature_importance = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Get latest performance metrics
        cursor.execute('''
        SELECT metric_name, metric_value, evaluation_set
        FROM model_performance
        WHERE model_id = %s
        ''', (model_id,))
        
        performance = {}
        for row in cursor.fetchall():
            metric_name, metric_value, eval_set = row
            if eval_set not in performance:
                performance[eval_set] = {}
            performance[eval_set][metric_name] = metric_value
        
        # Compile the results
        model_details = {
            'model_id': model_info[0],
            'model_name': model_info[1],
            'model_type': model_info[2],
            'created_at': model_info[3],
            'mlflow_run_id': model_info[4],
            'metrics': model_info[5],
            'parameters': model_info[6],
            'artifact_path': model_info[7],
            'status': model_info[8],
            'feature_importance': feature_importance,
            'performance': performance
        }
        
        return model_details
    except Exception as e:
        logger.error(f"Failed to get model details: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

def export_predictions_to_csv(model_id, output_path):
    """Export model predictions to a CSV file for analysis"""
    conn = None
    try:
        conn = get_db_connection()
        
        # Query the predictions
        query = '''
        SELECT prediction_id, prediction_time, input_data, prediction, actual_label, prediction_proba
        FROM predictions
        WHERE model_id = %s
        ORDER BY prediction_time
        '''
        
        # Use pandas to handle the data
        df = pd.read_sql_query(query, conn, params=(model_id,))
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Predictions exported to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to export predictions: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Initialize the database when the module is run directly
    initialize_database()
    logger.info("Database initialization complete") 