"""
Preprocessing module for Moroccan financial news data.
Handles company detection and text preprocessing.
"""
import pandas as pd
import re
import unicodedata
import sys
import os
from pathlib import Path
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COMPANIES, PROCESSED_DATA_DIR, RAW_DATA_DIR
from utils.logger import setup_logger
from data_collection.data_storage import load_raw_news_data, save_processed_data

logger = setup_logger("preprocess")

# French stop words to remove before matching
FRENCH_STOP_WORDS = {
    'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'ce', 'cette', 'ces',
    'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses', 'notre', 'nos',
    'votre', 'vos', 'leur', 'leurs', 'à', 'au', 'aux', 'avec', 'ce', 'ces',
    'dans', 'de', 'des', 'du', 'en', 'entre', 'et', 'est', 'il', 'ils', 'je',
    'j\'', 'l\'', 'd\'', 'la', 'le', 'leur', 'lui', 'ma', 'mais', 'me', 'même',
    'mes', 'moi', 'mon', 'ni', 'notre', 'nous', 'ou', 'où', 'par', 'pas', 'pour',
    'qu\'', 'que', 'qui', 'sa', 'se', 'si', 'son', 'sur', 'ta', 'te', 'tes', 'toi',
    'ton', 'tu', 'un', 'une', 'votre', 'vous', 'c\'', 's\'', 'n\'', 'été', 'etait',
    'étaient', 'étions', 'été', 'être', 'sont', 'suis', 'est'
}

def normalize_text(text):
    """Normalize text by removing accents"""
    if not text:
        return ""
    
    # Normalize unicode and remove accents
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    return text.lower()

def preprocess_text(text):
    """
    Preprocess text by:
    1. Normalizing (removing accents, lowercasing)
    2. Removing punctuation
    3. Removing stop words
    """
    if not text:
        return ""
    
    # Normalize
    text = normalize_text(text)
    
    # Replace punctuation with spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Split into words
    words = text.split()
    
    # Remove stop words
    words = [word for word in words if word not in FRENCH_STOP_WORDS]
    
    # Rejoin into text
    return " ".join(words)

def detect_companies(title, text, debug=False):
    """
    Detect companies mentioned in an article's title and text
    """
    mentioned_companies = set()
    company_details = {}  # For debugging
    
    # Combine title and text
    title = title or ""
    text = text or ""
    combined_text = f"{title} {text}"
    
    # Preprocess text - this removes stop words!
    preprocessed_text = preprocess_text(combined_text)
    
    for company in COMPANIES:
        company_name = company["name"]
        matching_keywords = []
        
        # First try to match ticker (exact match)
        if "ticker" in company and company["ticker"]:
            ticker = company["ticker"]
            if re.search(r'\b' + re.escape(ticker.lower()) + r'\b', preprocessed_text):
                mentioned_companies.add(company_name)
                matching_keywords.append(f"ticker:{ticker}")
                
        # Then try keywords
        if not matching_keywords:  # Skip if ticker already matched
            for keyword in company["keywords"]:
                # Normalize and preprocess the keyword too
                processed_keyword = normalize_text(keyword).lower()
                
                # Skip very short keywords unless they're ticker symbols
                if len(processed_keyword) < 3 and processed_keyword != company.get("ticker", "").lower():
                    continue
                
                # Look for the keyword as a whole word
                if re.search(r'\b' + re.escape(processed_keyword) + r'\b', preprocessed_text):
                    mentioned_companies.add(company_name)
                    matching_keywords.append(keyword)
                    break  # Found one keyword, no need to check others
        
        # Store details for debugging
        if matching_keywords and debug:
            company_details[company_name] = matching_keywords
    
    # Return different formats based on debug flag
    if debug:
        return sorted(mentioned_companies), company_details
    return sorted(mentioned_companies)

def process_articles_for_companies(df, debug=False):
    """
    Process all articles to detect mentioned companies
    
    Args:
        df (DataFrame): DataFrame containing news articles
        debug (bool): Whether to return detailed match information
        
    Returns:
        DataFrame: The input DataFrame with an added 'companies' column
    """
    if df is None or df.empty:
        logger.error("No data to process")
        return None
    
    # Make sure required columns exist
    required_cols = ['title', 'text']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Data missing required columns: {required_cols}")
        return df
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Initialize debug info if needed
    debug_info = []
    
    # Process each article
    logger.info(f"Processing {len(df_copy)} articles for company detection...")
    
    company_lists = []
    for i, row in df_copy.iterrows():
        title = row['title'] if not pd.isna(row['title']) else ""
        text = row['text'] if not pd.isna(row['text']) else ""
        
        if debug:
            companies, details = detect_companies(title, text, debug=True)
            company_lists.append(companies)
            
            if companies:
                debug_info.append({
                    'title': title[:50] + "..." if len(title) > 50 else title,
                    'url': row['url'] if 'url' in row else "N/A", 
                    'detected': companies,
                    'details': details
                })
        else:
            companies = detect_companies(title, text)
            company_lists.append(companies)
        
        if (i+1) % 100 == 0 or i+1 == len(df_copy):
            logger.info(f"Processed {i+1}/{len(df_copy)} articles")
    
    # Add companies as a comma-separated string
    df_copy['companies'] = [','.join(companies) for companies in company_lists]
    
    # Write debug information if requested
    if debug and debug_info:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_file = PROCESSED_DATA_DIR / f"company_detection_debug_{timestamp}.txt"
        
        # Ensure directory exists
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write("COMPANY DETECTION RESULTS\n")
            f.write("========================\n\n")
            
            for item in debug_info:
                f.write(f"ARTICLE: {item['title']}\n")
                f.write(f"URL: {item['url']}\n")
                f.write(f"DETECTED: {', '.join(item['detected'])}\n")
                f.write("MATCHING DETAILS:\n")
                
                for company, keywords in item['details'].items():
                    f.write(f"  - {company}: matched by [{', '.join(keywords)}]\n")
                
                f.write("\n" + "-"*50 + "\n\n")
        
        logger.info(f"Debug information written to {debug_file}")
    
    return df_copy

def preprocess_all_news():
    """
    Main function to preprocess all news data:
    1. Load raw news data
    2. Process and detect companies
    3. Save the processed data
    """
    # Load raw data
    raw_df = load_raw_news_data()
    
    if raw_df is None:
        logger.error("Failed to load raw news data")
        return
    
    logger.info(f"Loaded {len(raw_df)} articles from raw data")
    
    # Process data to detect companies
    processed_df = process_articles_for_companies(raw_df, debug=True)
    
    if processed_df is None:
        logger.error("Failed to process articles")
        return
    
    # Save processed data
    file_path = save_processed_data(processed_df, "news_with_companies.csv")
    
    if file_path:
        logger.info(f"Successfully processed {len(processed_df)} articles and saved to {file_path}")
    else:
        logger.error("Failed to save processed data")

if __name__ == "__main__":
    preprocess_all_news()