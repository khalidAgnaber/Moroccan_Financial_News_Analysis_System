"""
Automated sentiment labeling for financial news with improved financial context understanding
Only labels unlabeled data to support incremental processing
"""

import pandas as pd
import os
import logging
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sentiment_labeler')

class FinancialSentimentLabeler:
    """
    Class for automated sentiment labeling of financial news
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the sentiment labeler
        
        Args:
            model_path: Path to a pre-trained model (optional)
        """
        # Initialize lexicon-based analyzer
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            logger.info("Downloading VADER lexicon...")
            nltk.download('vader_lexicon', quiet=True)
        
        self.sia = SentimentIntensityAnalyzer()
        
        # Load French financial sentiment lexicon
        self.load_french_financial_lexicon()
        
        # Initialize model
        self.model = None
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading pre-trained model from {model_path}")
            try:
                self.model = pickle.load(open(model_path, 'rb'))
            except Exception as e:
                logger.warning(f"Error loading model: {e}")
    
    def load_french_financial_lexicon(self):
        """
        Load custom French financial sentiment lexicon with improved context understanding
        """
        # French financial terms with sentiment scores
        financial_lexicon = {
            # Positive financial indicators
            'croissance': 2.5,
            'progression': 2.0,
            'expansion': 2.0,
            'rebond': 2.0,
            'dynamique': 1.5,
            'profitable': 2.5,
            'rentable': 2.0,
            'positif': 1.5,
            'positive': 1.5,
            'record': 2.0,
            'excédent': 2.0,
            
            # Negative financial indicators
            'crise': -3.0,
            'déficit': -2.5,
            'dette': -1.5,
            'faillite': -3.5,
            'défaillance': -3.0,
            'risque': -1.5,
            'difficile': -1.5,
            'difficultés': -2.0,
            'négative': -1.5,
            'négatif': -1.5,
            'instabilité': -2.0,
            'volatilité': -1.5,
            'incertitude': -1.5,
            'ralentissement': -2.0,
            'tendu': -2.0,            # Added for "contexte économique tendu"
            'saturé': -1.5,           # Added for "marché saturé"
            'essoufflé': -2.0,        # Added for "dynamique essoufflée"
            'fragilise': -1.5,        # Added for "se fragilise"
            
            # Financial movement words - neutral without context
            'hausse': 0.0,  # Need context to determine if good/bad
            'baisse': 0.0,  # Need context to determine if good/bad
            'augmentation': 0.0,
            'diminution': 0.0,
            'recul': 0.0,
            'repli': 0.0,
            'chute': 0.0,
            'progression': 0.0,
            
            # Market-specific terms
            'séance négative': -2.5,
            'clôture en baisse': -2.5,
            'masi en baisse': -2.5,
            'masi en hausse': 2.5,
            
            # Neutral terms
            'maroc': 0.0,
            'casablanca': 0.0,
            'dirham': 0.0,
            'mdh': 0.0,
            'million': 0.0,
            'milliard': 0.0,
            'secteur': 0.0,
            'trimestre': 0.0,
        }
        
        # Add the French financial lexicon to VADER
        for word, score in financial_lexicon.items():
            self.sia.lexicon[word] = score
            # Add lowercase version as well for case-insensitivity
            self.sia.lexicon[word.lower()] = score
    
    def preprocess_text(self, text):
        """
        Preprocess text for sentiment analysis
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Replace URLs with placeholders
        text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
        
        # Mark percentages specially (preserve the sign which is important)
        text = re.sub(r'(-?\d+(?:\.\d+)?)\s*%', r'\1 PERCENT ', text)
        
        # Replace email addresses with placeholders
        text = re.sub(r'\S+@\S+', ' EMAIL ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_financial_context(self, text, title=None):
        """
        Analyze financial context to determine appropriate sentiment
        
        Args:
            text: Article text
            title: Article title (optional)
            
        Returns:
            Contextual sentiment override and confidence
        """
        # Combine title and text if available
        combined_text = ""
        if title:
            combined_text = title + ". "
        combined_text += text
        combined_text = combined_text.lower()
        
        # Context patterns with financial meaning
        context_patterns = [
            # INFLATION CONTEXTS - highest priority
            # Inflation slowing/decreasing (POSITIVE)
            (r'(inflation.{0,30}(ralentit|baisse|diminue|recul))', 'positive', 0.9),
            (r'((baisse|diminution|recul|ralentit).{0,30}inflation)', 'positive', 0.9),
            (r'(hausse des prix.{0,30}ralentit)', 'positive', 0.9),
            (r'(hausse.{0,5}prix.{0,15}ralentit)', 'positive', 0.9),
            (r'(hausse.{0,5}prix.{0,15}modérée)', 'positive', 0.9),
            
            # Inflation rising (NEGATIVE)
            (r'(inflation.{0,30}(hausse|augmente|accélère))', 'negative', 0.9),
            (r'((hausse|augmentation|accélère).{0,30}inflation)', 'negative', 0.9),
            (r'(inflation.{0,30}record)', 'negative', 0.9),
            
            # BUSINESS PERFORMANCE CONTEXTS
            # Negative performance language (NEGATIVE)
            (r'(baisse.{0,30}performances)', 'negative', 0.9),
            (r'(contexte.{0,10}(tendu|difficile))', 'negative', 0.9),
            (r'(marché.{0,10}saturé)', 'negative', 0.9),
            (r'(dynamique.{0,10}essoufflée)', 'negative', 0.9),
            (r'(difficultés.{0,20}secteur)', 'negative', 0.9),
            (r'(peinent.{0,30}objectifs)', 'negative', 0.9),
            (r'(fragilise|fragile|fragilité)', 'negative', 0.8),
            
            # MARKET PERFORMANCE CONTEXTS
            # Market decline patterns (NEGATIVE)
            (r'(masi|madex).{0,30}(baisse|recul|chute|repli)', 'negative', 0.9),
            (r'séance[s]? négative[s]?', 'negative', 0.9),
            (r'clôture en baisse', 'negative', 0.9),
            
            # Market increase patterns (POSITIVE)
            (r'(masi|madex).{0,30}(hausse|progression|augmentation)', 'positive', 0.9),
            (r'séance[s]? positive[s]?', 'positive', 0.9),
            (r'clôture en hausse', 'positive', 0.9),
            
            # PRICE CONTEXTS (NOT inflation)
            # General price increases (NEGATIVE for consumers)
            (r'(hausse|augmentation).{0,20}(prix|tarif|coût)', 'negative', 0.8),
            (r'(prix|tarifs).{0,20}(hausse|augmentent|augmentation)', 'negative', 0.8),
            
            # General price decreases (POSITIVE for consumers)
            (r'(baisse|diminution|recul|repli).{0,20}(prix|tarif|coût)', 'positive', 0.8),
            (r'(prix|tarifs).{0,20}(baisse|diminuent|recul)', 'positive', 0.8),
            
            # EMPLOYMENT CONTEXTS
            # Unemployment rising (NEGATIVE)
            (r'(hausse|augmentation).{0,20}(chômage|taux de chômage)', 'negative', 0.9),
            (r'(chômage|taux de chômage).{0,20}(hausse|augmentation|augmente)', 'negative', 0.9),
            
            # Unemployment falling (POSITIVE)
            (r'(baisse|diminution|recul).{0,20}(chômage|taux de chômage)', 'positive', 0.9),
            (r'(chômage|taux de chômage).{0,20}(baisse|diminue|recul)', 'positive', 0.9),
            
            # INTEREST RATES
            # Interest rates rising (generally NEGATIVE for borrowers/economy)
            (r'(hausse|augmentation|relèvement).{0,20}(taux d\'intérêt|taux directeur)', 'negative', 0.8),
            (r'(taux d\'intérêt|taux directeur).{0,20}(hausse|augmente|relèvement)', 'negative', 0.8),
            
            # Interest rates falling (generally POSITIVE for borrowers/economy)
            (r'(baisse|diminution|abaissement).{0,20}(taux d\'intérêt|taux directeur)', 'positive', 0.8),
            (r'(taux d\'intérêt|taux directeur).{0,20}(baisse|diminue|abaissement)', 'positive', 0.8),
            
            # COMPANY PERFORMANCE
            # Company performance declining (NEGATIVE)
            (r'(baisse|diminution|recul|chute).{0,20}(bénéfice|résultat|profit|revenu|chiffre d\'affaires)', 'negative', 0.9),
            (r'(perte|déficit).{0,40}(entreprise|société|groupe)', 'negative', 0.9),
            
            # Company performance improving (POSITIVE)
            (r'(hausse|augmentation|progression|croissance).{0,20}(bénéfice|résultat|profit|revenu|chiffre d\'affaires)', 'positive', 0.9),
            (r'(bénéfice|profit).{0,40}(entreprise|société|groupe)', 'positive', 0.9),
        ]
        
        # Check title first (title gives stronger context signal)
        if title:
            title_lower = title.lower()
            for pattern, sentiment, confidence in context_patterns:
                if re.search(pattern, title_lower):
                    return sentiment, confidence
        
        # Then check full text
        for pattern, sentiment, confidence in context_patterns:
            if re.search(pattern, combined_text):
                return sentiment, confidence
        
        # No definitive context found
        return None, 0
    
    def get_lexicon_sentiment(self, text):
        """
        Get sentiment using lexicon-based approach
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment label and score
        """
        # Preprocess text
        text = self.preprocess_text(text)
        
        # Get sentiment scores
        scores = self.sia.polarity_scores(text)
        compound = scores['compound']
        
        # More conservative thresholds for financial news
        if compound >= 0.1:
            return 'positive', abs(compound)
        elif compound <= -0.05:
            return 'negative', abs(compound)
        else:
            return 'neutral', 1 - abs(compound)
    
    def get_sentiment_score(self, text, title=None):
        """
        Get sentiment using contextual financial analysis and lexicon backup
        
        Args:
            text: Text to analyze
            title: Article title (optional)
            
        Returns:
            Sentiment label and confidence score
        """
        # First try contextual financial analysis
        context_sentiment, context_confidence = self.analyze_financial_context(text, title)
        
        # If we have a strong contextual match, use it
        if context_sentiment and context_confidence >= 0.8:
            return context_sentiment, context_confidence
            
        # Fall back to lexicon analysis
        lexicon_sentiment, lexicon_confidence = self.get_lexicon_sentiment(text)
        
        # If context gives a weak signal, combine with lexicon
        if context_sentiment and context_confidence > 0:
            if context_sentiment == lexicon_sentiment:
                # Agreement strengthens confidence
                return context_sentiment, max(context_confidence, lexicon_confidence)
            else:
                # Disagreement, prefer context but with reduced confidence
                return context_sentiment, context_confidence
        
        # If no context, use lexicon result
        return lexicon_sentiment, lexicon_confidence
    
    def label_financial_news(self, df, text_column='text', title_column='title', 
                            output_column='sentiment', save_scores=True, force_relabel=False):
        """
        Label financial news articles with sentiment (only unlabeled data unless forced)
        
        Args:
            df: DataFrame with news articles
            text_column: Column containing article text
            title_column: Column containing article title
            output_column: Column to store sentiment labels
            save_scores: Whether to save confidence scores
            force_relabel: Whether to force relabeling of already labeled data
            
        Returns:
            DataFrame with sentiment labels
        """
        # Create output columns if they don't exist
        if output_column not in df.columns:
            df[output_column] = None
        if save_scores and f"{output_column}_score" not in df.columns:
            df[f"{output_column}_score"] = None
        
        # Identify articles that need labeling
        if force_relabel:
            articles_to_label = df.index.tolist()
            logger.info(f"Force relabeling ALL {len(articles_to_label)} articles")
        else:
            # Only label articles that don't have a sentiment yet
            articles_to_label = df[df[output_column].isnull()].index.tolist()
            already_labeled = len(df) - len(articles_to_label)
            
            if already_labeled > 0:
                logger.info(f"Found {already_labeled} already labeled articles - skipping these")
            
            if not articles_to_label:
                logger.info("All articles are already labeled!")
                return df
            
            logger.info(f"Labeling {len(articles_to_label)} new/unlabeled articles")
        
        # Process only articles that need labeling
        processed_count = 0
        for i in articles_to_label:
            # Get title and text, ensuring they're strings
            title = str(df.loc[i, title_column]) if pd.notna(df.loc[i, title_column]) else ""
            text = str(df.loc[i, text_column]) if pd.notna(df.loc[i, text_column]) else ""
            
            # Skip empty content
            if not title and not text:
                logger.warning(f"Empty content for article {i}")
                continue
            
            # Get sentiment
            label, score = self.get_sentiment_score(text, title)
            
            # Store results
            df.at[i, output_column] = label
            if save_scores:
                df.at[i, f"{output_column}_score"] = score
            
            processed_count += 1
            
            # Log progress
            if processed_count % 10 == 0:
                logger.info(f"Labeled {processed_count}/{len(articles_to_label)} articles")
        
        logger.info(f"Completed labeling {processed_count} articles")
        
        # Print sentiment distribution
        distribution = df[output_column].value_counts()
        logger.info(f"Overall sentiment distribution: {distribution.to_dict()}")
        
        return df

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated sentiment labeling for financial news")
    parser.add_argument("input_file", help="Path to the CSV file with articles")
    parser.add_argument("-o", "--output_file", help="Path to save labeled articles")
    parser.add_argument("-t", "--text_column", default="text", help="Column containing article text")
    parser.add_argument("-i", "--title_column", default="title", help="Column containing article title")
    parser.add_argument("-s", "--sentiment_column", default="sentiment", help="Column to store sentiment labels")
    parser.add_argument("--no_scores", action="store_true", help="Don't save confidence scores")
    parser.add_argument("-m", "--model_path", help="Path to save/load model")
    parser.add_argument("-f", "--force", action="store_true", help="Force relabeling of all articles")
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    if args.output_file is None:
        base = os.path.splitext(args.input_file)[0]
        args.output_file = f"{base}_labeled.csv"
    
    # Set default model path if not provided
    if args.model_path is None:
        args.model_path = "models/sentiment_model.pkl"
    
    # Load data
    df = pd.read_csv(args.input_file)
    logger.info(f"Loaded {len(df)} articles from {args.input_file}")
    
    # Create labeler
    labeler = FinancialSentimentLabeler(model_path=args.model_path)
    
    # Label articles
    labeled_df = labeler.label_financial_news(
        df,
        text_column=args.text_column,
        title_column=args.title_column,
        output_column=args.sentiment_column,
        save_scores=not args.no_scores,
        force_relabel=args.force
    )
    
    # Save results
    labeled_df.to_csv(args.output_file, index=False)
    logger.info(f"Saved labeled articles to {args.output_file}")

if __name__ == "__main__":
    main()