import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from embedding_utils import FinancialNewsEmbedder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('embedding_analysis')

def find_optimal_clusters(embeddings, max_clusters=10):
    # Find optimal number of clusters using silhouette score
    silhouette_scores = []
    for n_clusters in range(2, min(max_clusters + 1, len(embeddings))):
        # Apply KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        # Calculate silhouette score
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        logger.info(f"Clusters: {n_clusters}, Silhouette Score: {silhouette_avg:.4f}")
    
    # Return optimal number of clusters
    optimal_clusters = np.argmax(silhouette_scores) + 2 
    return optimal_clusters

def cluster_documents(embeddings, n_clusters, processed_df):
    #Cluster documents using KMeans and analyze clusters
    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Add cluster labels to dataframe
    processed_df['cluster'] = cluster_labels
    
    # Analyze each cluster
    for cluster_id in range(n_clusters):
        # Get documents in this cluster
        cluster_docs = processed_df[processed_df['cluster'] == cluster_id]
        
        logger.info(f"\nCluster {cluster_id} ({len(cluster_docs)} documents):")
        
        # Get most common tokens in cluster
        all_tokens = []
        for tokens in cluster_docs['tokens'].apply(eval):
            all_tokens.extend(tokens)
        
        # Count token frequencies
        token_counts = {}
        for token in all_tokens:
            if token not in ["l'", "d'", "s'", "n'", "c'", "j'", "m'", "t'", "qu'"]:
                token_counts[token] = token_counts.get(token, 0) + 1
        
        # Get top tokens
        top_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info(f"Top tokens: {top_tokens}")
        
        # Print a few document titles
        logger.info("Sample documents:")
        for title in cluster_docs['title'].head(3):
            logger.info(f"  - {title}")
    
    # Save clustered data
    processed_df.to_csv('data/processed/clustered_financial_news.csv', index=False)
    logger.info(f"Saved clustered data to data/processed/clustered_financial_news.csv")
    
    return cluster_labels

def find_similar_documents(embeddings, processed_df, query_index, top_n=5):
    # Get query embedding
    query_embedding = embeddings[query_index]
    
    # Calculate cosine similarity to all documents
    similarities = []
    for i, embedding in enumerate(embeddings):
        if i != query_index:  # Skip the query document itself
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
            similarities.append((i, similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N similar documents
    logger.info(f"\nQuery document: {processed_df['title'].iloc[query_index]}")
    logger.info("Similar documents:")
    
    for i, similarity in similarities[:top_n]:
        logger.info(f"  - {processed_df['title'].iloc[i]} (Similarity: {similarity:.4f})")
    
    return similarities[:top_n]

def explore_financial_terms(embedder, terms=None):
    if terms is None:
        terms = [
            "banque", "finance", "investissement", "bourse", "marché", 
            "action", "obligation", "crédit", "capital", "économie"
        ]
    
    logger.info("\nExploring financial term relationships:")
    
    # Get embeddings for terms
    term_embeddings = {}
    for term in terms:
        term_embeddings[term] = embedder.model.get_word_vector(term)
    
    # Calculate cosine similarity between each pair of terms
    similarities = {}
    for i, term1 in enumerate(terms):
        for term2 in terms[i+1:]:
            similarity = np.dot(term_embeddings[term1], term_embeddings[term2]) / (
                np.linalg.norm(term_embeddings[term1]) * np.linalg.norm(term_embeddings[term2]))
            similarities[(term1, term2)] = similarity
    
    # Sort similarities
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    # Print top relationships
    logger.info("Top term relationships:")
    for (term1, term2), similarity in sorted_similarities[:10]:
        logger.info(f"  {term1} - {term2}: {similarity:.4f}")
    
    # Find similar terms to key financial terms
    logger.info("\nTerms similar to 'finance':")
    similar_terms = embedder.model.get_nearest_neighbors("finance", k=10)
    for similarity, term in similar_terms:
        logger.info(f"  {term}: {similarity:.4f}")
    
    return similarities

def main():
    # Load processed data
    processed_df = pd.read_csv('data/processed/processed_financial_news.csv')
    logger.info(f"Loaded {len(processed_df)} processed documents")
    
    # Load embeddings
    embeddings = np.load('data/processed/document_embeddings.npy')
    logger.info(f"Loaded document embeddings of shape {embeddings.shape}")
    
    # Load embedder
    embedder = FinancialNewsEmbedder(load_if_exists=True)
    
    # Find optimal number of clusters
    logger.info("Finding optimal number of clusters")
    optimal_clusters = find_optimal_clusters(embeddings)
    logger.info(f"Optimal number of clusters: {optimal_clusters}")
    
    # Cluster documents
    logger.info(f"Clustering documents into {optimal_clusters} clusters")
    cluster_labels = cluster_documents(embeddings, optimal_clusters, processed_df)
    
    # Find similar documents to a sample document
    logger.info("Finding similar documents")
    sample_index = 0
    similar_docs = find_similar_documents(embeddings, processed_df, sample_index)
    
    # Explore financial term relationships
    term_similarities = explore_financial_terms(embedder)
    
    logger.info("Analysis completed")

if __name__ == "__main__":
    main()