"""
BERT model for sentiment analysis of French financial news.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import MODELS_DIR
from utils.logger import setup_logger

logger = setup_logger("bert_model")

class BERTSentimentModel:
    def __init__(self, max_length=128, model_name="camembert-base"):
        """Initialize the BERT model."""
        self.max_length = max_length
        self.model_name = model_name
        self.model = None
        self.history = None
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        
    def load_data(self, file_path):
        """Load and prepare your labeled data."""
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Combine title and text for better context
        df['combined_text'] = df['title'].fillna('') + '. ' + df['text'].fillna('')
        
        # Remove rows with missing sentiment
        df = df.dropna(subset=['sentiment'])
        
        # Encode sentiment labels
        sentiment_encoded = self.label_encoder.fit_transform(df['sentiment'])
        
        logger.info(f"Loaded {len(df)} articles")
        logger.info(f"Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
        
        return df['combined_text'].values, sentiment_encoded
        
    def build_model(self):
        """Build a BERT model for sentiment classification."""
        # Load pre-trained model and tokenizer using Auto classes
        logger.info(f"Loading {self.model_name} model and tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model with classification head
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=3
        )
        
        # Use legacy optimizer for better performance on M1/M2 Macs
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=2e-5)
        
        # Compile the model
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        logger.info(f"Built and compiled {self.model_name} model for sentiment classification")
        return self.model
    
    def prepare_inputs(self, texts):
        """Convert text to BERT inputs."""
        # Tokenize the texts
        encoded = self.tokenizer(
            list(texts),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='tf'
        )
        
        # Extract the tensors from the BatchEncoding object
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def train(self, X_train, y_train, X_val, y_val, epochs=13, batch_size=32):
        """Train the model with early stopping."""
        if self.model is None:
            self.build_model()
        
        # Prepare inputs
        train_inputs = self.prepare_inputs(X_train)
        val_inputs = self.prepare_inputs(X_val)
        
        # Convert labels to TensorFlow tensors
        y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.int64)
        y_val_tensor = tf.convert_to_tensor(y_val, dtype=tf.int64)
        
        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True,
            verbose=1
        )
        
        # Reduce learning rate on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=1,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train the model
        logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
        
        self.history = self.model.fit(
            train_inputs,
            y_train_tensor,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_inputs, y_val_tensor),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        logger.info("Model training completed")
        return self.history
    
    def predict(self, X):
        """Generate sentiment predictions."""
        if self.model is None:
            raise ValueError("Model is not trained yet")
            
        # Prepare inputs
        inputs = self.prepare_inputs(X)
        
        # Get predictions (logits)
        output = self.model.predict(inputs)
        logits = output.logits
        
        # Convert logits to probabilities
        return tf.nn.softmax(logits, axis=1).numpy()
    
    def predict_classes(self, X):
        """Get class predictions from probabilities."""
        predictions = self.predict(X)
        return np.argmax(predictions, axis=1)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data."""
        if self.model is None:
            raise ValueError("Model is not trained yet")
        
        # Prepare inputs
        test_inputs = self.prepare_inputs(X_test)
        
        # Convert labels to TensorFlow tensor
        y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.int64)
        
        # Evaluate
        loss, accuracy = self.model.evaluate(test_inputs, y_test_tensor)
        logger.info(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
        
        # Get predictions
        y_pred = self.predict_classes(X_test)
        
        # Calculate metrics
        report = classification_report(y_test, y_pred)
        logger.info(f"Classification Report:\n{report}")
        
        return loss, accuracy, report
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            logger.warning("No training history available")
            return
        
        # Create a figure with 2 subplots
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.grid(True)
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.grid(True)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, "bert_training_history.png"))
        plt.close()
        
        logger.info("Training history plot saved to model directory")
    
    def save(self, filepath=None):
        """Save the model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        if filepath is None:
            # Create directory for the model
            model_dir = os.path.join(MODELS_DIR, "bert_sentiment_model")
            os.makedirs(model_dir, exist_ok=True)
            
            # Save the model
            model_path = os.path.join(model_dir, "model")
            self.model.save_pretrained(model_path)
            
            # Save the tokenizer
            tokenizer_path = os.path.join(model_dir, "tokenizer")
            self.tokenizer.save_pretrained(tokenizer_path)
            
            # Save label encoder mapping
            label_mapping = {i: label for i, label in enumerate(self.label_encoder.classes_)}
            with open(os.path.join(model_dir, "label_mapping.txt"), "w") as f:
                for idx, label in label_mapping.items():
                    f.write(f"{idx}: {label}\n")
            
            filepath = model_path
            
        logger.info(f"Model saved to {filepath}")
        return filepath