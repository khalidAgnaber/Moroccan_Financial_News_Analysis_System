"""
Main entry point for the Moroccan News Sentiment Analysis application.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import schedule
import time

from config import (
    COMPANIES, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
    COLLECTION_INTERVAL_HOURS
)
from data_collection.scraper import collect_news
from data_collection.data_storage import (
    load_latest_raw_data, merge_raw_data_files, save_processed_data,
    daily_data_maintenance
)
from data_preprocessing.cleaner import clean_dataframe
from data_preprocessing.preprocessor import TextPreprocessor, preprocess_for_training
from models.lstm_model import LSTMSentimentModel, SimpleLSTMModel
from models.bert_model import BERTSentimentModel
from models.transformer_model import TransformerSentimentModel
from evaluation.metrics import SentimentMetrics
from evaluation.visualization import (
    plot_training_history, plot_sentiment_distribution,
    plot_sentiment_by_company, plot_sentiment_over_time
)
from investment_recommendation.recommendation_engine import InvestmentRecommendation
from utils.logger import setup_logger
from utils.helpers import (
    load_data, save_model_artifacts, load_model_artifacts,
    find_latest_file
)

logger = setup_logger("main")

def collect_data():
    """Collect news data from sources."""
    logger.info("Starting data collection...")
    collect_news()
    logger.info("Data collection completed.")

def preprocess_data(input_file=None):
    """Preprocess collected data."""
    logger.info("Starting data preprocessing...")
    
    if input_file:
        raw_df = load_data(input_file)
    else:
        raw_df = load_latest_raw_data()
    
    if raw_df is None:
        logger.error("No raw data available for preprocessing.")
        return None
    
    # Clean and preprocess data
    clean_df, preprocessor = preprocess_for_training(raw_df)
    
    logger.info("Data preprocessing completed.")
    return clean_df, preprocessor

def train_model(X_train, y_train, X_val, y_val, model_type='lstm', vocab_size=10000, input_shape=512):
    """Train a sentiment analysis model."""
    logger.info(f"Starting model training with {model_type} architecture...")
    
    if model_type.lower() == 'lstm':
        model = LSTMSentimentModel(input_shape, vocab_size)
    elif model_type.lower() == 'simple_lstm':
        model = SimpleLSTMModel(input_shape, vocab_size)
    elif model_type.lower() == 'transformer':
        model = TransformerSentimentModel(input_shape, vocab_size)
    elif model_type.lower() == 'bert':
        model = BERTSentimentModel()
        # Special handling for BERT model
        return model
    else:
        logger.error(f"Unknown model type: {model_type}")
        return None
    
    # Build and compile model
    model.build_model()
    model.compile_model()
    
    # Train model
    history = model.train(X_train, y_train, X_val, y_val)
    
    logger.info("Model training completed.")
    return model, history

def evaluate_model(model, X_test, y_test):
    """Evaluate trained model."""
    logger.info("Starting model evaluation...")
    
    # Make predictions
    y_pred = model.predict_classes(X_test)
    
    # Calculate metrics
    metrics = SentimentMetrics(y_test, y_pred)
    
    # Print evaluation results
    metrics.print_metrics()
    
    logger.info("Model evaluation completed.")
    return metrics

def generate_recommendations(sentiment_df):
    """Generate investment recommendations based on sentiment analysis."""
    logger.info("Generating investment recommendations...")
    
    # Initialize recommendation engine
    recommender = InvestmentRecommendation(sentiment_df, company_info=COMPANIES)
    
    # Generate recommendations for all companies
    recommendations_df = recommender.generate_all_recommendations()
    
    # Save recommendations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"recommendations_{timestamp}.csv"
    recommendations_df.to_csv(file_path, index=False)
    
    logger.info(f"Investment recommendations saved to {file_path}")
    return recommendations_df

def schedule_daily_tasks():
    """Schedule daily data collection and processing tasks."""
    logger.info("Setting up daily scheduled tasks...")
    
    # Schedule data collection
    collection_interval = COLLECTION_INTERVAL_HOURS
    schedule.every(collection_interval).hours.do(collect_data)
    
    # Schedule data maintenance
    schedule.every().day.at("00:00").do(daily_data_maintenance)
    
    logger.info(f"Tasks scheduled: Data collection every {collection_interval} hours and daily maintenance at midnight.")
    
    # Run the scheduler
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

def run_full_pipeline():
    """Run the complete sentiment analysis pipeline."""
    logger.info("Starting full sentiment analysis pipeline...")
    
    # 1. Collect data
    collect_data()
    
    # 2. Preprocess data
    clean_df, preprocessor = preprocess_data()
    if clean_df is None:
        logger.error("Data preprocessing failed. Exiting pipeline.")
        return
    
    # 3. Prepare data for model training
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_data(clean_df)
    
    # 4. Train model
    model, history = train_model(
        X_train, y_train, X_val, y_val, 
        model_type='lstm',
        vocab_size=len(preprocessor.tokenizer.word_index),
        input_shape=X_train.shape[1]
    )
    
    # 5. Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # 6. Save model artifacts
    save_model_artifacts(model, preprocessor, metrics)
    
    # 7. Generate recommendations
    recommendations_df = generate_recommendations(clean_df)
    
    logger.info("Full sentiment analysis pipeline completed successfully.")
    return model, preprocessor, metrics, recommendations_df

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Moroccan News Sentiment Analysis for Investment Insights')
    
    parser.add_argument('--action', type=str, choices=['collect', 'preprocess', 'train', 'evaluate', 'recommend', 'schedule', 'pipeline'],
                        help='Action to perform')
    
    parser.add_argument('--input', type=str, help='Input file path')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--model-type', type=str, default='lstm', choices=['lstm', 'simple_lstm', 'transformer', 'bert'],
                        help='Model architecture to use')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.action == 'collect':
        collect_data()
    
    elif args.action == 'preprocess':
        preprocess_data(args.input)
    
    elif args.action == 'train':
        # This requires preprocessed data
        logger.info("Training requires preprocessed data. Run with --action preprocess first.")
    
    elif args.action == 'evaluate':
        logger.info("Evaluation requires a trained model. Run with --action train first.")
    
    elif args.action == 'recommend':
        # Load the latest processed data
        latest_file = find_latest_file(PROCESSED_DATA_DIR, "preprocessed_data_*.csv")
        
        if latest_file:
            df = load_data(latest_file)
            generate_recommendations(df)
        else:
            logger.error("No processed data found for generating recommendations.")
    
    elif args.action == 'schedule':
        schedule_daily_tasks()
    
    elif args.action == 'pipeline':
        run_full_pipeline()
    
    else:
        # Default: show help
        logger.info("No action specified. Use --action to specify an action.")
        logger.info("Available actions: collect, preprocess, train, evaluate, recommend, schedule, pipeline")