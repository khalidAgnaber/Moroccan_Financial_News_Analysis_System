"""
Data preprocessing module for sentiment analysis of financial news.
"""

import re
import string
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from config import PROCESSED_DATA_DIR
from utils.logger import setup_logger

# Set up logger
logger = setup_logger("preprocessor")

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

class TextPreprocessor:
    """Preprocess financial news data for sentiment analysis."""
    
    def __init__(self, max_features=10000, max_len=100):
        """Initialize preprocessor with parameters."""
        self.max_features = max_features
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_features)
        self.label_encoder = LabelEncoder()
        
        try:
            self.french_stopwords = set(stopwords.words('french'))
            logger.info("Loaded French stopwords")
        except Exception as e:
            logger.warning(f"Failed to load stopwords: {e}")
            self.french_stopwords = set()
    
    def clean_text(self, text):
        """Clean and normalize text."""
        if not isinstance(text, str) or not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove stopwords
        if self.french_stopwords:
            try:
                tokens = word_tokenize(text)
                tokens = [word for word in tokens if word not in self.french_stopwords]
                text = ' '.join(tokens)
            except Exception as e:
                logger.warning(f"Failed to remove stopwords: {e}")
        
        return text
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using keyword matching."""
        if not isinstance(text, str) or len(text.strip()) < 10:
            return 'neutral'  # Default for empty or very short text
            
        try:
            # French financial sentiment keywords
            negative_words = ['baisse', 'chute', 'perte', 'déficit', 'faillite', 'échec', 
                             'déception', 'risque', 'crise', 'problème', 'difficulté',
                             'recul', 'diminution', 'ralentissement', 'danger', 'inquiétude']
            
            positive_words = ['hausse', 'croissance', 'profit', 'bénéfice', 'succès', 'réussite', 
                             'amélioration', 'opportunité', 'performance', 'innovation',
                             'progression', 'augmentation', 'avantage', 'gain', 'optimisme']
            
            text_lower = text.lower()
            neg_count = sum(1 for word in negative_words if word in text_lower)
            pos_count = sum(1 for word in positive_words if word in text_lower)
            
            if neg_count > pos_count:
                return 'negative'
            elif pos_count > neg_count:
                return 'positive'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 'neutral'  # Default if analysis fails
    
    def find_best_text_column(self, df):
        """Find the best column to use as the main text content."""
        text_columns = {}
        
        # Get all string columns and their average length
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_len = df[col].fillna('').astype(str).str.len().mean()
                text_columns[col] = avg_len
        
        if not text_columns:
            logger.warning("No text columns found in dataframe")
            return None
        
        # Get the column with the longest average text
        main_text_col = max(text_columns, key=text_columns.get)
        logger.info(f"Using '{main_text_col}' as main text content (avg length: {text_columns[main_text_col]:.1f})")
        
        return main_text_col
    
    def find_title_column(self, df):
        """Find a column that likely contains the title."""
        # Look for columns with title-like names
        title_cols = [col for col in df.columns if 'title' in col.lower() or 'headline' in col.lower() or 'header' in col.lower()]
        
        if title_cols:
            logger.info(f"Using '{title_cols[0]}' as title column")
            return title_cols[0]
        
        # If no obvious title column, look for the second longest text column
        text_columns = {}
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_len = df[col].fillna('').astype(str).str.len().mean()
                text_columns[col] = avg_len
        
        if len(text_columns) > 1:
            # Sort by length and take the second longest
            sorted_cols = sorted(text_columns.items(), key=lambda x: x[1], reverse=True)
            second_col = sorted_cols[1][0]
            logger.info(f"Using '{second_col}' as title column")
            return second_col
        
        return None
    
    def preprocess(self, df):
        """Preprocess dataframe with financial news."""
        logger.info("Cleaning text data...")
        
        # Find best columns to use
        main_text_col = self.find_best_text_column(df)
        title_col = self.find_title_column(df)
        
        if not main_text_col:
            # Create a dummy text column from all string columns
            logger.warning("Creating composite text column from all available text")
            df['content'] = df.select_dtypes(include=['object']).fillna('').apply(
                lambda x: ' '.join(x.astype(str)), axis=1
            )
            main_text_col = 'content'
        
        # For compatibility, ensure we have content column
        df['content'] = df[main_text_col]
        
        # Clean text
        df['clean_content'] = df['content'].fillna('').apply(self.clean_text)
        
        if title_col:
            # For compatibility, ensure we have title column
            df['title'] = df[title_col]
            df['clean_title'] = df['title'].fillna('').apply(self.clean_text)
            df['clean_combined'] = df['clean_title'] + ' ' + df['clean_content']
        else:
            df['title'] = ''
            df['clean_title'] = ''
            df['clean_combined'] = df['clean_content']
        
        # Analyze sentiment
        logger.info("Analyzing sentiment of articles...")
        df['sentiment'] = df['clean_combined'].apply(self.analyze_sentiment)
        sentiment_counts = df['sentiment'].value_counts().to_dict()
        logger.info(f"Sentiment distribution: {sentiment_counts}")
        
        # Fit tokenizer
        logger.info("Fitting tokenizer...")
        self.tokenizer.fit_on_texts(df['clean_combined'])
        logger.info(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        
        # Encode text to sequences
        sequences = self.tokenizer.texts_to_sequences(df['clean_combined'])
        df['sequence'] = [json.dumps(seq) for seq in sequences]
        
        # Fit label encoder
        logger.info("Fitting label encoder...")
        self.label_encoder.fit(df['sentiment'])
        logger.info(f"Classes: {self.label_encoder.classes_}")
        
        # Encode labels
        df['sentiment_encoded'] = self.label_encoder.transform(df['sentiment'])
        
        return df
    
    def save_processed_data(self, df, output_dir=None):
        """Save preprocessed data and artifacts."""
        if output_dir is None:
            output_dir = PROCESSED_DATA_DIR
            
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save preprocessed data
        output_file = os.path.join(output_dir, f"preprocessed_data_{timestamp}.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Preprocessed data saved to {output_file}")
        
        # Save tokenizer
        tokenizer_file = os.path.join(output_dir, f"tokenizer_{timestamp}.json")
        tokenizer_json = {
            'word_index': self.tokenizer.word_index,
            'num_words': self.max_features
        }
        with open(tokenizer_file, 'w') as f:
            json.dump(tokenizer_json, f)
        logger.info(f"Tokenizer saved to {tokenizer_file}")
        
        return output_file, tokenizer_file
    
    def prepare_data(self, df):
        """Prepare inputs for model training."""
        # Get sequences from dataframe
        sequences = [json.loads(seq) for seq in df['sequence']]
        
        # Pad sequences
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len)
        
        # Get labels
        labels = df['sentiment_encoded'].values
        
        # Split data
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            padded_sequences, labels, test_size=0.2, random_state=42
        )
        
        # Split training data into training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Validation set size: {len(X_val)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_for_training(raw_data):
    """Preprocess data for model training."""
    # Check if raw_data is already a DataFrame
    if isinstance(raw_data, pd.DataFrame):
        df = raw_data.copy()
    # Check if it's a string path
    elif isinstance(raw_data, (str, Path)) and os.path.exists(raw_data):
        df = pd.read_csv(raw_data)
    else:
        # Try to convert to DataFrame as a last resort
        try:
            df = pd.DataFrame(raw_data)
        except:
            raise TypeError(f"Cannot process raw_data of type {type(raw_data)}. Expected DataFrame or file path.")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocess data
    df_processed = preprocessor.preprocess(df)
    
    # Save processed data
    preprocessor.save_processed_data(df_processed)
    
    # Return the processed dataframe and preprocessor
    return df_processed, preprocessor