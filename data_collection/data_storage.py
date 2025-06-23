"""
Functions for storing and managing collected news data.
"""
import os
import pandas as pd
import datetime
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.logger import setup_logger

logger = setup_logger("data_storage")

def load_latest_raw_data():
    """Load the most recent raw data file."""
    files = list(RAW_DATA_DIR.glob("news_articles_*.csv"))
    
    if not files:
        logger.warning("No raw data files found.")
        return None
    
    # Sort by file modification time, newest first
    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading raw data from {latest_file}")
    
    try:
        df = pd.read_csv(latest_file)
        return df
    except Exception as e:
        logger.error(f"Error loading raw data: {e}")
        return None

def merge_raw_data_files():
    """Merge all raw data files into a single dataframe."""
    files = list(RAW_DATA_DIR.glob("news_articles_*.csv"))
    
    if not files:
        logger.warning("No raw data files found for merging.")
        return None
    
    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading file {file}: {e}")
    
    if not dfs:
        return None
    
    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates
    merged_df = merged_df.drop_duplicates(subset=['url'])
    
    logger.info(f"Merged {len(files)} files, resulting in {len(merged_df)} unique articles")
    return merged_df

def save_processed_data(df, suffix=""):
    """Save processed data to the processed data directory."""
    if df is None or df.empty:
        logger.warning("No data to save.")
        return
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"processed_data_{suffix}_{timestamp}.csv" if suffix else f"processed_data_{timestamp}.csv"
    file_path = PROCESSED_DATA_DIR / file_name
    
    try:
        df.to_csv(file_path, index=False)
        logger.info(f"Saved processed data to {file_path}")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")

def get_company_data(company_name):
    """Get all news data for a specific company."""
    merged_df = merge_raw_data_files()
    
    if merged_df is None:
        return None
    
    # Filter articles mentioning the company
    company_df = merged_df[merged_df['companies'].str.contains(company_name, na=False)]
    
    logger.info(f"Found {len(company_df)} articles for {company_name}")
    return company_df

def get_all_companies_data():
    """Get data grouped by company."""
    merged_df = merge_raw_data_files()
    
    if merged_df is None:
        return None
    
    # Split the companies column into a list
    merged_df['companies_list'] = merged_df['companies'].str.split(',')
    
    # Explode the dataframe so each company gets its own row
    exploded_df = merged_df.explode('companies_list')
    
    # Group by company
    grouped = exploded_df.groupby('companies_list')
    
    result = {}
    for company, group in grouped:
        if company and not pd.isna(company):
            result[company] = group.drop('companies_list', axis=1).reset_index(drop=True)
    
    return result

def daily_data_maintenance():
    """Perform daily maintenance on data files."""
    # Merge all raw files into a consolidated file
    merged_df = merge_raw_data_files()
    
    if merged_df is not None:
        # Save consolidated file
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        file_path = RAW_DATA_DIR / f"consolidated_news_{timestamp}.csv"
        merged_df.to_csv(file_path, index=False)
        
        # Optionally, remove individual files to save space
        # This is commented out to prevent accidental data loss
        # for file in RAW_DATA_DIR.glob("news_articles_*.csv"):
        #     if "consolidated" not in file.name:
        #         file.unlink()
        
        logger.info(f"Performed daily data maintenance, saved consolidated file to {file_path}")

if __name__ == "__main__":
    daily_data_maintenance()