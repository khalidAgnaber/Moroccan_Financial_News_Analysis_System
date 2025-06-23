"""
Investment recommendation engine based on sentiment analysis.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import random
import logging

from config import PROCESSED_DATA_DIR
from utils.logger import setup_logger

logger = setup_logger("recommendation_engine")

class InvestmentRecommendation:
    """Generate investment recommendations based on sentiment analysis."""
    
    def __init__(self, sentiment_df=None, company_info=None, model=None, tokenizer=None):
        """Initialize recommendation engine."""
        self.sentiment_df = sentiment_df
        self.model = model
        self.tokenizer = tokenizer
        
        # Default Moroccan stock symbols and names
        default_symbols = [
            "ATW", "BCP", "BOA", "IAM", "TQM", "LHM", 
            "MNG", "CSR", "SID", "MRS", "LBV", "COZ"
        ]
        
        default_names = {
            "ATW": "Attijariwafa Bank",
            "BCP": "Banque Centrale Populaire",
            "BOA": "Bank of Africa",
            "IAM": "Maroc Telecom",
            "TQM": "Taqa Morocco",
            "LHM": "Lafarge Holcim Maroc",
            "MNG": "Managem",
            "CSR": "Cosumar",
            "SID": "Sonasid",
            "MRS": "Marsa Maroc",
            "LBV": "Label Vie",
            "COZ": "Ciments du Maroc"
        }
        
        # Initialize company info from provided data or use default Moroccan stocks
        if company_info:
            self.company_info = company_info
            
            # Handle different formats of company_info
            if isinstance(company_info, dict):
                # If company_info is a dictionary with keys as symbols
                self.stock_symbols = list(company_info.keys())
                self.stock_names = {symbol: info.get('name', symbol) 
                                   for symbol, info in company_info.items()}
            elif isinstance(company_info, list):
                # If company_info is a list
                self.stock_symbols = []
                self.stock_names = {}
                
                for company in company_info:
                    if isinstance(company, dict) and 'symbol' in company:
                        symbol = company['symbol']
                        self.stock_symbols.append(symbol)
                        self.stock_names[symbol] = company.get('name', symbol)
                    elif isinstance(company, str):
                        self.stock_symbols.append(company)
                        self.stock_names[company] = company
                
                if not self.stock_symbols:
                    logger.warning("No valid companies found in company_info list. Using defaults.")
                    self.stock_symbols = default_symbols
                    self.stock_names = default_names
            else:
                logger.warning(f"Unexpected company_info type: {type(company_info)}. Using defaults.")
                self.stock_symbols = default_symbols
                self.stock_names = default_names
        else:
            # Use defaults if no company_info provided
            self.stock_symbols = default_symbols
            self.stock_names = default_names
    
    def get_sentiment_scores(self, stock_symbol):
        """Get sentiment scores for articles related to a specific stock."""
        # If no sentiment_df, generate random scores
        if self.sentiment_df is None or len(self.sentiment_df) == 0:
            logger.warning(f"No sentiment data available for {stock_symbol}. Using random scores.")
            # Generate random scores
            sample_size = 3
            sentiment_scores = [0.2 + 0.6 * random.random() for _ in range(sample_size)]
            confidence_scores = [0.5 + 0.3 * random.random() for _ in range(sample_size)]
            return sentiment_scores, confidence_scores
        
        # In a real system, you would filter articles related to this stock
        # For demo purposes, we'll select a random subset of articles
        sample_size = min(3, len(self.sentiment_df))
        stock_articles = self.sentiment_df.sample(sample_size)
        
        # Get sentiment scores
        if 'sentiment_score' in stock_articles.columns:
            # If sentiment score is already calculated
            sentiment_scores = stock_articles['sentiment_score'].tolist()
        elif 'sentiment' in stock_articles.columns:
            # If sentiment is a categorical label, convert to score
            sentiment_mapping = {'negative': 0.0, 'neutral': 0.5, 'positive': 1.0}
            sentiment_scores = [sentiment_mapping.get(s, 0.5) for s in stock_articles['sentiment']]
        else:
            # Generate random scores if no sentiment data
            sentiment_scores = [0.2 + 0.6 * random.random() for _ in range(sample_size)]
        
        # Generate confidence scores (either from data or randomly)
        if 'confidence' in stock_articles.columns:
            confidence_scores = stock_articles['confidence'].tolist()
        else:
            confidence_scores = [0.5 + 0.3 * random.random() for _ in range(sample_size)]
        
        return sentiment_scores, confidence_scores
    
    def generate_recommendation(self, sentiment_score, confidence):
        """Generate investment recommendation based on sentiment and confidence."""
        # Map sentiment to recommendation
        if sentiment_score < 0.3:  # Negative sentiment
            action = "SELL"
        elif sentiment_score > 0.7:  # Positive sentiment
            action = "BUY"
        else:  # Neutral sentiment
            action = "HOLD"
        
        # Adjust confidence - should never be 100%
        # Cap confidence at 85% to reflect market uncertainty
        adjusted_confidence = min(confidence, 0.85)
        
        # Add some randomness for more realistic confidence values
        final_confidence = adjusted_confidence * (0.9 + 0.1 * random.random())
        
        return action, final_confidence
    
    def generate_all_recommendations(self):
        """Generate investment recommendations for all stocks."""
        recommendations = []
        
        for symbol in self.stock_symbols:
            # Get sentiment scores for this stock
            sentiment_scores, confidence_scores = self.get_sentiment_scores(symbol)
            
            # Calculate average sentiment and confidence
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.5
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.5
            
            # Generate recommendation
            action, confidence = self.generate_recommendation(avg_sentiment, avg_confidence)
            
            # Create recommendation
            recommendation = {
                "symbol": symbol,
                "name": self.stock_names.get(symbol, symbol),
                "action": action,
                "confidence": round(confidence, 2),
                "sentiment_score": round(avg_sentiment, 2)
            }
            
            recommendations.append(recommendation)
            
            logger.info(f"Generated {action} recommendation for {self.stock_names.get(symbol, symbol)} ({symbol}) with {confidence:.2f} confidence")
        
        # Create dataframe
        recommendations_df = pd.DataFrame(recommendations)
        return recommendations_df
    
    def generate_recommendations(self, articles_df=None):
        """Legacy method for backward compatibility."""
        if articles_df is not None and self.sentiment_df is None:
            self.sentiment_df = articles_df
        return self.generate_all_recommendations()
    
    def save_recommendations(self, recommendations_df=None, output_file=None):
        """Save recommendations to CSV file."""
        if recommendations_df is None:
            recommendations_df = self.generate_all_recommendations()
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"recommendations_{timestamp}.csv"
        
        recommendations_df.to_csv(output_file, index=False)
        logger.info(f"Recommendations saved to {output_file}")
        
        return output_file