"""
Text cleaning functions for preprocessing news articles.
"""
import re
import unicodedata
import pandas as pd
import nltk
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing.french_stopwords import COMBINED_STOPWORDS
from utils.logger import setup_logger

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = setup_logger("cleaner")

def normalize_text(text):
    """Normalize text by removing accents and converting to lowercase."""
    if not isinstance(text, str):
        return ""
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    # Convert to lowercase
    text = text.lower()
    return text

def remove_urls(text):
    """Remove URLs from text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)

def remove_html_tags(text):
    """Remove HTML tags from text."""
    html_pattern = re.compile(r'<.*?>')
    return html_pattern.sub('', text)

def remove_mentions_and_hashtags(text):
    """Remove mentions (@user) and hashtags (#topic) from text."""
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    return text

def remove_punctuation(text):
    """Remove punctuation and special characters from text."""
    return re.sub(r'[^\w\s]', ' ', text)

def remove_extra_spaces(text):
    """Remove extra spaces, including newlines and tabs."""
    return re.sub(r'\s+', ' ', text).strip()

def remove_stopwords(text, stopwords=COMBINED_STOPWORDS):
    """Remove stopwords from text."""
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    return ' '.join(filtered_words)

def expand_contractions(text):
    """Expand common French contractions."""
    # French contractions mapping
    contractions = {
        "c'est": "ce est",
        "j'ai": "je ai",
        "j'": "je ",
        "n'": "ne ",
        "d'": "de ",
        "l'": "le ",
        "qu'": "que ",
        "s'": "se ",
        "t'": "te ",
        "m'": "me ",
    }
    
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    return text

def clean_text(text):
    """Apply all cleaning steps to the text."""
    if not text or not isinstance(text, str):
        return ""
    
    text = normalize_text(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_mentions_and_hashtags(text)
    text = expand_contractions(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = remove_extra_spaces(text)
    
    return text

def clean_dataframe(df, text_column='text', title_column='title'):
    """Clean text data in a dataframe."""
    if df is None or df.empty:
        logger.warning("Empty dataframe passed to clean_dataframe.")
        return df
    
    df_copy = df.copy()
    
    # Clean title and text columns
    if title_column in df_copy.columns:
        df_copy['clean_title'] = df_copy[title_column].apply(clean_text)
    
    if text_column in df_copy.columns:
        df_copy['clean_text'] = df_copy[text_column].apply(clean_text)
    
    # Combine cleaned title and text
    if 'clean_title' in df_copy.columns and 'clean_text' in df_copy.columns:
        df_copy['clean_combined'] = df_copy['clean_title'] + ' ' + df_copy['clean_text']
    
    return df_copy

if __name__ == "__main__":
    # Test the cleaning functions
    sample_text = "C'est un article sur l'économie marocaine. #économie @media24 https://example.com"
    cleaned_text = clean_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Cleaned: {cleaned_text}")