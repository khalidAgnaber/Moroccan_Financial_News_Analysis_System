"""
Utilities for creating word embeddings for financial news analysis
"""

import os
import numpy as np
import logging
import fasttext
import fasttext.util
from gensim.models.phrases import Phrases, Phraser
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import gzip
import shutil
import sys
import time

logger = logging.getLogger('financial_news_embeddings')

class FinancialNewsEmbedder:
    """
    Class to create and manage word embeddings for financial news
    """
    
    def __init__(self, model_path=None, embedding_dim=300, load_if_exists=True):
        """
        Initialize the embedder with a pre-trained model or download one
        
        Args:
            model_path: Path to a pre-trained fastText model file
            embedding_dim: Dimension of embeddings (default 300)
            load_if_exists: Whether to load existing model if found
        """
        self.embedding_dim = embedding_dim
        self.model = None
        self.phraser = None
        
        # Define model paths
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = os.path.join(self.models_dir, f"cc.fr.{embedding_dim}.bin")
        
        self.compressed_model_path = os.path.join(self.models_dir, f"cc.fr.{embedding_dim}.compressed.bin")
        self.gz_model_path = os.path.join(self.models_dir, f"cc.fr.{embedding_dim}.bin.gz")
        
        # Try to load existing model
        if load_if_exists:
            self._load_or_download_model()
    
    def _load_or_download_model(self):
        """Load existing model or download if needed"""
        # Try to load compressed model first (fastest)
        if os.path.exists(self.compressed_model_path):
            logger.info(f"Loading compressed fastText model from {self.compressed_model_path}")
            try:
                self.model = fasttext.load_model(self.compressed_model_path)
                return
            except Exception as e:
                logger.warning(f"Failed to load compressed model: {e}")
            
        # Try to load full model
        if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 1000000:  # Check if file exists and is large enough
            logger.info(f"Loading fastText model from {self.model_path}")
            try:
                self.model = fasttext.load_model(self.model_path)
                return
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
        else:
            # Check if we have the gzipped model already
            if os.path.exists(self.gz_model_path) and os.path.getsize(self.gz_model_path) > 1000000:
                logger.info(f"Found compressed model file {self.gz_model_path}, decompressing...")
                self._decompress_model()
            else:
                # Download and save model
                logger.info(f"Downloading fastText model for French (this may take some time)...")
                
                # Direct download using urllib instead of fasttext.util.download_model
                import urllib.request
                url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.{self.embedding_dim}.bin.gz"
                
                def reporthook(count, block_size, total_size):
                    global start_time
                    if count == 0:
                        start_time = time.time()
                        return
                    duration = time.time() - start_time
                    progress_size = int(count * block_size)
                    speed = int(progress_size / (1024 * duration)) if duration > 0 else 0
                    percent = min(int(count * block_size * 100 / total_size), 100)
                    sys.stdout.write(f"\r{percent}% | {progress_size / (1024 * 1024):.1f} MB | {speed} KB/s | {duration:.1f} sec")
                    sys.stdout.flush()
                
                logger.info(f"Downloading from {url} to {self.gz_model_path}")
                urllib.request.urlretrieve(url, self.gz_model_path, reporthook)
                print()  # New line after progress bar
                
                # Decompress the model
                self._decompress_model()
            
        # Now try to load the decompressed model
        if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 1000000:
            logger.info(f"Loading fastText model from {self.model_path}")
            try:
                self.model = fasttext.load_model(self.model_path)
            except Exception as e:
                logger.error(f"Failed to load model after decompression: {e}")
                raise ValueError(f"Could not load model {self.model_path}: {e}")
        else:
            raise ValueError(f"Model file {self.model_path} not found or is too small")
        
        # Compress model to save memory and disk space
        logger.info(f"Compressing model to {self.embedding_dim} dimensions...")
        fasttext.util.reduce_model(self.model, self.embedding_dim)
        
        # Save compressed model
        logger.info(f"Saving compressed model to {self.compressed_model_path}")
        self.model.save_model(self.compressed_model_path)
    
    def _decompress_model(self):
        """Decompress the gzipped model file"""
        logger.info(f"Decompressing {self.gz_model_path} to {self.model_path}...")
        try:
            with gzip.open(self.gz_model_path, 'rb') as f_in:
                with open(self.model_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.info(f"Decompression complete")
        except Exception as e:
            logger.error(f"Error during decompression: {e}")
            raise
    
    def create_document_embedding(self, tokens):
        """
        Create a document embedding from tokens
        
        Args:
            tokens: List of tokens for a document
            
        Returns:
            Document embedding vector (numpy array)
        """
        if not tokens:
            return np.zeros(self.embedding_dim)
        
        # Check if we have phraser and apply it
        if self.phraser:
            tokens = self.phraser[tokens]
        
        # Get vector for each token and average
        vectors = []
        for token in tokens:
            # Skip tokens with apostrophes like l', d', etc.
            if token in ["l'", "d'", "s'", "n'", "c'", "j'", "m'", "t'", "qu'"] or \
               token.startswith(("l'", "d'", "s'", "n'", "c'", "j'", "m'", "t'", "qu'")):
                continue
                
            # Get vector
            vectors.append(self.model.get_word_vector(token))
        
        if not vectors:
            return np.zeros(self.embedding_dim)
        
        # Return average of all token vectors
        return np.mean(vectors, axis=0)
    
    def train_phraser(self, tokenized_docs, min_count=5, threshold=10):
        """
        Train a phraser model to detect common bigrams and trigrams
        
        Args:
            tokenized_docs: List of tokenized documents
            min_count: Minimum count for a phrase to be considered
            threshold: Threshold for phrase detection
        """
        logger.info("Training phraser model for bigrams and trigrams")
        
        # Train bigram detector
        bigram = Phrases(tokenized_docs, min_count=min_count, threshold=threshold)
        bigram_phraser = Phraser(bigram)
        
        # Train trigram detector on bigrams
        trigram = Phrases(bigram_phraser[tokenized_docs], min_count=min_count, threshold=threshold)
        trigram_phraser = Phraser(trigram)
        
        # Save the final phraser
        self.phraser = trigram_phraser
        
        # Save phraser model
        phraser_path = os.path.join(self.models_dir, "financial_phraser.pkl")
        with open(phraser_path, 'wb') as f:
            pickle.dump(self.phraser, f)
            
        logger.info(f"Phraser model saved to {phraser_path}")
        
        # Log some example phrases
        phrases = set()
        for doc in tokenized_docs:
            phrases.update([p for p in self.phraser[doc] if '_' in p])
        
        top_phrases = sorted(list(phrases), key=len, reverse=True)[:20]
        logger.info(f"Top detected phrases: {top_phrases}")
        
        return self.phraser
    
    def load_phraser(self, path=None):
        """Load existing phraser model"""
        if path is None:
            path = os.path.join(self.models_dir, "financial_phraser.pkl")
            
        if os.path.exists(path):
            logger.info(f"Loading phraser model from {path}")
            with open(path, 'rb') as f:
                self.phraser = pickle.load(f)
            return True
        else:
            logger.warning(f"No phraser model found at {path}")
            return False
    
    def create_embeddings_matrix(self, tokenized_docs):
        """
        Create an embeddings matrix for a corpus of documents
        
        Args:
            tokenized_docs: List of tokenized documents
            
        Returns:
            Numpy array of document embeddings (n_docs x embedding_dim)
        """
        logger.info(f"Creating embeddings for {len(tokenized_docs)} documents")
        
        # Try to load phraser if we don't have one
        if self.phraser is None:
            self.load_phraser()
        
        # Create embeddings for each document
        embeddings = []
        for i, tokens in enumerate(tokenized_docs):
            embedding = self.create_document_embedding(tokens)
            embeddings.append(embedding)
            
            # Log progress
            if (i + 1) % 100 == 0:
                logger.info(f"Created embeddings for {i + 1} documents")
        
        return np.array(embeddings)
    
    def visualize_embeddings(self, embeddings, labels, output_file, n_components=2):
        """
        Create a visualization of document embeddings
        
        Args:
            embeddings: Matrix of document embeddings
            labels: Labels for each document (e.g., categories)
            output_file: Path to save the visualization
            n_components: Number of dimensions for visualization (2 or 3)
        """
        logger.info(f"Creating {n_components}D visualization of {len(embeddings)} embeddings")
        
        # Use t-SNE to reduce dimensions
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(embeddings)-1))
        reduced_embeddings = tsne.fit_transform(embeddings)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Get unique labels and colors
        unique_labels = list(set(labels))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        # Plot each category with a different color
        for i, label in enumerate(unique_labels):
            indices = [j for j, l in enumerate(labels) if l == label]
            plt.scatter(
                reduced_embeddings[indices, 0],
                reduced_embeddings[indices, 1],
                c=[colors[i]],
                label=label,
                alpha=0.7
            )
        
        plt.title(f"Document Embeddings Visualization")
        plt.legend()
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_file, dpi=300)
        logger.info(f"Visualization saved to {output_file}")
    
    def extract_top_features(self, tokens_list, k=100):
        """
        Extract top features based on embeddings similarity to corpus
        
        Args:
            tokens_list: List of tokenized documents
            k: Number of top features to extract
            
        Returns:
            List of (feature, score) tuples
        """
        logger.info(f"Extracting top {k} features based on embeddings")
        
        # Create a flattened list of all tokens
        all_tokens = []
        for tokens in tokens_list:
            all_tokens.extend(tokens)
        
        # Get unique tokens
        unique_tokens = list(set(all_tokens))
        
        # Create a corpus embedding (average of all documents)
        corpus_embedding = np.mean([self.create_document_embedding(tokens) for tokens in tokens_list], axis=0)
        
        # Calculate similarity of each token to the corpus
        token_scores = []
        for token in unique_tokens:
            # Skip short tokens and tokens with apostrophes
            if len(token) < 3 or token in ["l'", "d'", "s'", "n'", "c'", "j'", "m'", "t'", "qu'"]:
                continue
            
            # Get embedding for token
            token_embedding = self.model.get_word_vector(token)
            
            # Calculate cosine similarity
            similarity = np.dot(token_embedding, corpus_embedding) / (
                np.linalg.norm(token_embedding) * np.linalg.norm(corpus_embedding))
            
            # Calculate frequency
            frequency = all_tokens.count(token) / len(all_tokens)
            
            # Combine similarity and frequency for final score
            score = similarity * (1 + frequency)
            
            token_scores.append((token, score))
        
        # Sort by score and return top k
        return sorted(token_scores, key=lambda x: x[1], reverse=True)[:k]