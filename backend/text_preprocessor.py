import re
import pandas as pd
import unicodedata
import string
import logging
from typing import List, Dict, Any, Set, Optional, Union, Tuple
import spacy
import nltk
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from french_stopwords import FRENCH_STOPWORDS

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Set up logging
logger = logging.getLogger('financial_news_preprocessing')

class FinancialNewsPreprocessor:

    def __init__(self, 
                 use_lemmatization: bool = True, 
                 use_stemming: bool = False,
                 additional_stopwords: Set[str] = None,
                 min_token_length: int = 2,
                 keep_pos_tags: List[str] = None):

        self.use_lemmatization = use_lemmatization
        self.use_stemming = use_stemming
        self.min_token_length = min_token_length
        self.keep_pos_tags = keep_pos_tags if keep_pos_tags else ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']
        
        # Initialize stemmer and lemmatizer
        self.stemmer = SnowballStemmer('french') if use_stemming else None
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        
        # Initialize stopwords set
        self.stopwords = set(stopwords.words('french'))
        self.stopwords.update(FRENCH_STOPWORDS)
        
        # Add custom financial stopwords
        financial_stopwords = {
            "le", "la", "les", "du", "de", "des", "au", "aux", "un", "une",
            "et", "en", "que", "qui", "pour", "dans", "par", "sur", "avec",
            "a", "se", "ce", "cette", "ces", "son", "sa", "ses", "nous", "vous",
            "ils", "elles", "leur", "leurs", "notre", "nos", "votre", "vos"
        }
        self.stopwords.update(financial_stopwords)
        
        # Add additional stopwords
        if additional_stopwords:
            self.stopwords.update(additional_stopwords)
            
        # Initialize spaCy model
        try:
            self.nlp = spacy.load('fr_core_news_md')
            logger.info("Loaded spaCy French model")
        except OSError:
            logger.warning("French spaCy model not found.")
            self.nlp = None
    
    def preprocess_text(self, text: str) -> Dict[str, Any]:

        if not isinstance(text, str) or not text.strip():
            return {"processed_text": "", "tokens": []}
        
        # Normalize text (remove accents, lowercase, etc.)
        normalized_text = self._normalize_text(text)
        
        # Tokenize and process with spaCy 
        if self.nlp:
            doc = self.nlp(normalized_text)
            tokens = []
            
            for token in doc:
                # Skip if token is stopword, punctuation, or too short
                if (token.text.lower() in self.stopwords or 
                    token.is_punct or 
                    token.is_space or 
                    len(token.text) < self.min_token_length):
                    continue
                
                # Skip if not in desired POS tags
                if self.keep_pos_tags and token.pos_ not in self.keep_pos_tags:
                    continue
                
                # Get base form of token
                if self.use_lemmatization:
                    token_text = token.lemma_
                else:
                    token_text = token.text.lower()
                
                # Apply stemming if requested
                if self.use_stemming:
                    token_text = self.stemmer.stem(token_text)
                
                tokens.append(token_text)
            
            # Join tokens back into processed text
            processed_text = " ".join(tokens)
            
        else:

            translator = str.maketrans('', '', string.punctuation)
            normalized_text = normalized_text.translate(translator)
            words = nltk.word_tokenize(normalized_text)
            
            # Filter and process tokens
            tokens = []
            for word in words:
                word = word.lower()
                
                # Skip stopwords and short words
                if word in self.stopwords or len(word) < self.min_token_length:
                    continue
                
                # Apply lemmatization or stemming
                if self.use_lemmatization:
                    word = self.lemmatizer.lemmatize(word)
                if self.use_stemming:
                    word = self.stemmer.stem(word)
                
                tokens.append(word)
            
            # Join tokens back into processed text
            processed_text = " ".join(tokens)
        
        return {
            "processed_text": processed_text,
            "tokens": tokens
        }
    
    def preprocess_df(self, 
                     df: pd.DataFrame, 
                     text_column: str = 'text', 
                     new_column: str = 'processed_text',
                     tokens_column: str = 'tokens') -> pd.DataFrame:

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        # Create a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Process each document
        results = []
        for text in processed_df[text_column]:
            results.append(self.preprocess_text(text))
        
        # Add processed text and tokens to DataFrame
        processed_df[new_column] = [r["processed_text"] for r in results]
        processed_df[tokens_column] = [r["tokens"] for r in results]
        
        return processed_df
    
    def _normalize_text(self, text: str) -> str:

        # Replace URLs with placeholder
        text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
        
        # Replace email addresses with placeholder
        text = re.sub(r'\S+@\S+', ' EMAIL ', text)
        
        # Replace numbers with placeholder
        text = re.sub(r'\d+[,.]?\d*%?', ' NUM ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove accents
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def create_tfidf_vectorizer(self, **kwargs):

        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        
        # Default parameters
        params = {
            'min_df': 2, 
            'max_df': 0.85,  
            'stop_words': list(self.stopwords),
            'ngram_range': (1, 2), 
            'tokenizer': lambda doc: doc.split(),  
            'sublinear_tf': True  
        }
        
        # Update with any additional parameters
        params.update(kwargs)
        
        return TfidfVectorizer(**params)