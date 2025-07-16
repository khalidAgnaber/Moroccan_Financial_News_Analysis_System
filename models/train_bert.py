"""
Training script for BERT sentiment analysis model with class balancing
"""

import os
import sys
import numpy as np
from sklearn.utils import class_weight

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Now we can import our modules
from models.bert_model import BERTSentimentModel
from sklearn.model_selection import train_test_split
from config import PROCESSED_DATA_DIR
from utils.logger import setup_logger

# Setup logger
logger = setup_logger("bert_training")

def main():
    # Current date for logging
    logger.info("Training started on 2025-07-03 by khalidAgnaber")
    
    # Path to your labeled data
    data_file = os.path.join(PROCESSED_DATA_DIR, "processed_financial_news_labeled_corrected.csv")
    
    logger.info(f"Starting BERT model training using data from {data_file}")
    
    # Initialize model
    bert_model = BERTSentimentModel(max_length=128)
    
    # Load your data
    X, y = bert_model.load_data(data_file)
    
    # Split data - use stratify to maintain class distribution
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, 
                                                        random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, 
                                                   random_state=42, stratify=y_temp)
    
    logger.info(f"Training set: {len(X_train)} articles")
    logger.info(f"Validation set: {len(X_val)} articles") 
    logger.info(f"Test set: {len(X_test)} articles")
    
    # Calculate class weights to handle imbalance
    # This gives higher weight to minority classes
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    logger.info(f"Class weights: {class_weight_dict}")
    
    # Add class weights to model
    bert_model.class_weights = class_weight_dict
    
    # Train model with more epochs
    bert_model.train(X_train, y_train, X_val, y_val, epochs=8, batch_size=8)
    
    # Plot training history
    bert_model.plot_training_history()
    
    # Evaluate on test set
    loss, accuracy, report = bert_model.evaluate(X_test, y_test)
    
    # Save model
    bert_model.save()
    
    logger.info(f"Training completed! Final accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()