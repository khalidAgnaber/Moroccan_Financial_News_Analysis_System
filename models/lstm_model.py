"""
LSTM model for sentiment analysis of financial news.
"""

import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from config import MODELS_DIR
from utils.logger import setup_logger

logger = setup_logger("lstm_model")

class LSTMSentimentModel:
    def __init__(self, input_shape=100, vocab_size=10000):
        """Initialize the LSTM model."""
        self.input_shape = input_shape
        self.vocab_size = vocab_size
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build a simplified LSTM model for sentiment analysis."""
        self.model = tf.keras.Sequential([
            # Embedding layer with smaller embedding dimensions
            tf.keras.layers.Embedding(self.vocab_size + 1, 32, input_length=self.input_shape),
            
            # Add dropout to prevent overfitting
            tf.keras.layers.SpatialDropout1D(0.3),
            
            # Smaller LSTM layer with fewer units
            tf.keras.layers.LSTM(16),
            
            # Simple dense layer with fewer neurons
            tf.keras.layers.Dense(16, activation='relu'),
            
            # More dropout
            tf.keras.layers.Dropout(0.3),
            
            # Output layer for 3 sentiment classes
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        logger.info(f"Built LSTM model with vocabulary size {self.vocab_size}")
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with appropriate loss and optimizer."""
        if self.model is None:
            self.build_model()
            
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=['accuracy']
        )
        logger.info("Model compiled with categorical_crossentropy loss and Adam optimizer")
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=16):
        """Train the model with early stopping."""
        if self.model is None:
            self.compile_model()
            
        # Check if labels are already one-hot encoded
        if len(y_train.shape) == 1 or y_train.shape[1] == 1:
            logger.info("Converting labels to one-hot encoding")
            # Convert integer labels to one-hot encoding
            num_classes = 3  # negative, neutral, positive
            y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
            y_val = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)
        
        # Create directory for checkpoints
        os.makedirs(MODELS_DIR / "checkpoints", exist_ok=True)
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=2,  # Stop after 2 epochs without improvement
            restore_best_weights=True,
            verbose=1
        )
        
        # Checkpoint callback to save best model
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, "checkpoints", "lstm_model_{epoch:02d}_{val_accuracy:.2f}.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        # Train the model
        logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=3,  # Force max 3 epochs regardless of input
            batch_size=2,  # Force small batch size regardless of input
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        logger.info("Model training completed")
        return self.history
    
    def predict(self, X):
        """Generate sentiment predictions."""
        if self.model is None:
            raise ValueError("Model is not trained yet")
        
        return self.model.predict(X)
    
    def predict_classes(self, X):
        """Get class predictions from probabilities."""
        predictions = self.predict(X)
        return np.argmax(predictions, axis=1)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data."""
        if self.model is None:
            raise ValueError("Model is not trained yet")
            
        # Check if labels are already one-hot encoded
        if len(y_test.shape) == 1 or y_test.shape[1] == 1:
            logger.info("Converting test labels to one-hot encoding")
            # Convert integer labels to one-hot encoding
            num_classes = 3  # negative, neutral, positive
            y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
        
        loss, accuracy = self.model.evaluate(X_test, y_test)
        logger.info(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
        
        # Get predictions
        y_pred = self.predict_classes(X_test)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        report = classification_report(y_true, y_pred)
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
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, "lstm_training_history.png"))
        plt.close()
        
        logger.info("Training history plot saved to model directory")
    
    def save(self, filepath=None):
        """Save the model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "lstm_sentiment_model.h5")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
        return filepath

class SimpleLSTMModel(LSTMSentimentModel):
    """A simplified version of the LSTM model."""
    
    def build_model(self):
        """Build an even simpler LSTM model for sentiment analysis."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size + 1, 16, input_length=self.input_shape),
            tf.keras.layers.GlobalAveragePooling1D(),  # Use pooling instead of LSTM for simplicity
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        logger.info(f"Built Simple LSTM model with vocabulary size {self.vocab_size}")
        return self.model