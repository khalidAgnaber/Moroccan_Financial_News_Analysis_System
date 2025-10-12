import os
import pandas as pd
import datetime
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.logger import setup_logger

logger = setup_logger("data_storage")

def load_raw_news_data():
    file_path = RAW_DATA_DIR / "moroccan_financial_news.csv"
    
    if not os.path.exists(file_path):
        logger.warning("Raw news data file not found.")
        return None
    
    logger.info(f"Loading raw news data from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        logger.error(f"Error loading raw news data: {e}")
        return None

def save_processed_data(df, filename="processed_news_data.csv"):
    if df is None or df.empty:
        logger.warning("No data to save.")
        return
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{filename.split('.')[0]}_{timestamp}.csv"
    file_path = PROCESSED_DATA_DIR / file_name
    
    try:
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        df.to_csv(file_path, index=False)
        logger.info(f"Saved processed data to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        return None

def get_data_by_date_range(df, start_date=None, end_date=None):
    """Filter data by date range."""
    if df is None or df.empty:
        return None
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        logger.error("No 'date' column in the data")
        return df
    
    if start_date:
        try:
            start_date = pd.to_datetime(start_date)
            df = df[df['date'] >= start_date]
        except Exception as e:
            logger.error(f"Error parsing start_date: {e}")
    
    if end_date:
        try:
            end_date = pd.to_datetime(end_date)
            df = df[df['date'] <= end_date]
        except Exception as e:
            logger.error(f"Error parsing end_date: {e}")
    
    return df

def get_data_statistics(df):
    if df is None or df.empty:
        return None
    
    stats = {}
    
    # Total number of articles
    stats['total_articles'] = len(df)
    
    # Articles by source
    if 'source' in df.columns:
        stats['articles_by_source'] = df['source'].value_counts().to_dict()
    
    # Date range
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        stats['date_range'] = {
            'min_date': df['date'].min().strftime('%Y-%m-%d') if not pd.isna(df['date'].min()) else None,
            'max_date': df['date'].max().strftime('%Y-%m-%d') if not pd.isna(df['date'].max()) else None
        }
    return stats

def backup_raw_data():
    file_path = RAW_DATA_DIR / "moroccan_financial_news.csv"
    
    if not os.path.exists(file_path):
        logger.warning("No raw data file to backup.")
        return False
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = RAW_DATA_DIR / f"moroccan_financial_news_backup_{timestamp}.csv"
    
    try:
        df = pd.read_csv(file_path)
        df.to_csv(backup_path, index=False)
        logger.info(f"Created backup at {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    df = load_raw_news_data()
    if df is not None:
        stats = get_data_statistics(df)
        print("Dataset Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for subkey, subvalue in value.items():
                    print(f"  {subkey}: {subvalue}")
            else:
                print(f"{key}: {value}")