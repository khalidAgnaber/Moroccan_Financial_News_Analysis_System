import pandas as pd
import os
import sys
import time
import logging
import ast
from datetime import datetime
from text_preprocessor import FinancialNewsPreprocessor
import pickle
import numpy as np
from embedding_utils import FinancialNewsEmbedder
import matplotlib
matplotlib.use('Agg')

# Setup logs
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('financial_news_preprocessing')

def main():
    start_time = time.time()
    
    input_file = "data/raw/moroccan_financial_news.csv"
    output_dir = "data/processed"
    models_dir = "models"
    visualizations_dir = "visualizations"
    
    # Create directories if they dont exist
    for directory in [output_dir, models_dir, visualizations_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {input_file}")
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} articles")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Remove unwanted columns
    columns_to_remove = ['content_hash', 'scraped_at']
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(col, axis=1)
            logger.info(f"Removed {col} column")
    
    # Initialize preprocessor
    logger.info("Initializing preprocessor")
    preprocessor = FinancialNewsPreprocessor(
        use_lemmatization=True,
        use_stemming=False,
        min_token_length=2,
        keep_pos_tags=['NOUN', 'VERB', 'ADJ', 'PROPN']
    )
    
    # Preprocess the data
    logger.info("Preprocessing data")
    try:
        processed_df = preprocessor.preprocess_df(
            df, 
            text_column='text', 
            new_column='processed_text', 
            tokens_column='tokens'
        )
        logger.info("Preprocessing completed")
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return
    
    # Save the processed data
    processed_path = os.path.join(output_dir, "processed_financial_news.csv")
    logger.info(f"Saving processed data to {processed_path}")
    processed_df.to_csv(processed_path, index=False)
    
    # Create word embeddings
    logger.info("Creating word embeddings")
    try:
        # Initialize embedder with french model
        embedder = FinancialNewsEmbedder(embedding_dim=300)
        
        # Get tokens for each document
        tokens_list = []
        logger.info("Converting tokens to list format")
        for token_data in processed_df['tokens']:
            if isinstance(token_data, list):
                tokens_list.append(token_data)
            elif isinstance(token_data, str):
                try:
                    tokens = ast.literal_eval(token_data)
                    if isinstance(tokens, list):
                        tokens_list.append(tokens)
                    else:
                        tokens_list.append([token_data])
                except (ValueError, SyntaxError):
                    tokens_list.append(token_data.split())
            else:
                tokens_list.append([str(token_data)])
                
        logger.info(f"Extracted {len(tokens_list)} token lists")
        
        if tokens_list:
            sample_idx = min(5, len(tokens_list) - 1)
            logger.info(f"Sample tokens (document {sample_idx}): {tokens_list[sample_idx][:10]}")
        
        logger.info("Training phraser for multi-word expressions")
        if not embedder.load_phraser():
            embedder.train_phraser(tokens_list)
        
        # Create document embeddings
        logger.info("Creating document embeddings")
        embeddings = embedder.create_embeddings_matrix(tokens_list)
        
        # Save embeddings
        embeddings_path = os.path.join(output_dir, "document_embeddings.npy")
        np.save(embeddings_path, embeddings)
        logger.info(f"Saved document embeddings to {embeddings_path}")
        
        # Visualize embeddings
        if 'category' in processed_df.columns:
            visualization_path = os.path.join(visualizations_dir, "embeddings_visualization.png")
            embedder.visualize_embeddings(
                embeddings, 
                processed_df['category'].tolist(),
                visualization_path
            )
        
        # Extract top features based on embeddings
        top_features = embedder.extract_top_features(tokens_list, k=100)
        
        # Log top 20 features
        logger.info("Top 20 features by importance:")
        for feature, score in top_features[:20]:
            logger.info(f"  {feature}: {score:.4f}")
        
        # Save top features to file
        features_path = os.path.join(output_dir, "top_features.csv")
        features_df = pd.DataFrame(top_features, columns=['feature', 'score'])
        features_df.to_csv(features_path, index=False)
        logger.info(f"Saved top 100 features to {features_path}")
        
    except Exception as e:
        logger.error(f"Error during embedding creation: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Report processing time
    execution_time = time.time() - start_time
    logger.info(f"Preprocessing pipeline completed successfully in {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()