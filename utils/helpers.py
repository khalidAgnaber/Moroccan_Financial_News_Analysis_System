"""
Helper functions for the project.
"""
import os
import re
import pandas as pd
from datetime import datetime
import pickle
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, MODELS_DIR
from utils.logger import setup_logger

logger = setup_logger("helpers")

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data from {file_path}: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return None

def save_model_artifacts(model, preprocessor, metrics, model_name=None):
    """
    Save model and related artifacts.
    
    Args:
        model: Trained model
        preprocessor: Text preprocessor used for the model
        metrics: Evaluation metrics
        model_name: Name for the model artifacts
    
    Returns:
        str: Path to the saved model directory
    """
    if model_name is None:
        model_name = model.name if hasattr(model, 'name') else 'sentiment_model'
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = MODELS_DIR / f"{model_name}_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = model_dir / f"{model_name}.h5"
    if hasattr(model, 'save_model'):
        model.save_model(model_path)
    elif hasattr(model, 'save'):
        model.save(model_path)
    
    # Save preprocessor
    preprocessor_path = model_dir / "preprocessor.pkl"
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # Save metrics
    metrics_path = model_dir / "metrics.json"
    if hasattr(metrics, 'save_metrics'):
        metrics.save_metrics(metrics_path)
    elif isinstance(metrics, dict):
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
    
    logger.info(f"Model artifacts saved to {model_dir}")
    
    return model_dir

def load_model_artifacts(model_dir):
    """
    Load model and related artifacts.
    
    Args:
        model_dir: Directory containing model artifacts
    
    Returns:
        tuple: (model, preprocessor, metrics)
    """
    model_dir = Path(model_dir)
    
    # Find model file
    model_files = list(model_dir.glob("*.h5"))
    if not model_files:
        logger.error(f"No model file found in {model_dir}")
        return None, None, None
    
    model_path = model_files[0]
    
    # Load model
    try:
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        model = None
    
    # Load preprocessor
    preprocessor_path = model_dir / "preprocessor.pkl"
    try:
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        logger.info(f"Loaded preprocessor from {preprocessor_path}")
    except Exception as e:
        logger.error(f"Error loading preprocessor from {preprocessor_path}: {e}")
        preprocessor = None
    
    # Load metrics
    metrics_path = model_dir / "metrics.json"
    try:
        import json
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        logger.info(f"Loaded metrics from {metrics_path}")
    except Exception as e:
        logger.error(f"Error loading metrics from {metrics_path}: {e}")
        metrics = None
    
    return model, preprocessor, metrics

def extract_company_from_text(text, company_keywords):
    """
    Extract mentioned companies from text.
    
    Args:
        text: Text to search for company mentions
        company_keywords: Dictionary mapping company names to keyword lists
    
    Returns:
        list: Names of mentioned companies
    """
    mentioned_companies = []
    text_lower = text.lower()
    
    for company_name, keywords in company_keywords.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                mentioned_companies.append(company_name)
                break
    
    return mentioned_companies

def find_latest_file(directory, pattern):
    """
    Find the most recent file matching a pattern in a directory.
    
    Args:
        directory: Directory to search
        pattern: File pattern to match
    
    Returns:
        Path: Path to the latest file
    """
    files = list(Path(directory).glob(pattern))
    
    if not files:
        return None
    
    # Sort by modification time, newest first
    return max(files, key=lambda x: x.stat().st_mtime)