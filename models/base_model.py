"""
Base model class for sentiment analysis.
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from datetime import datetime
from pathlib import Path
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, EPOCHS, BATCH_SIZE, SENTIMENT_CLASSES
from utils.logger import setup_logger

logger = setup_logger("model")

class BaseSentimentModel:
    def __init__(self, name, input_shape=None):
        self.name = name
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
        # Create model directory
        self.model_dir = MODELS_DIR / self.name
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def build_model(self):
        """Build the model architecture. To be implemented by subclasses."""
        raise NotImplementedError
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with appropriate loss and optimizer."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model.summary()
    
    def train(self, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE):
        """Train the model."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        
        # Define callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint_path = self.model_dir / f"best_model_{timestamp}.h5"
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        tensorboard_log_dir = self.model_dir / "logs" / timestamp
        tensorboard_callback = TensorBoard(
            log_dir=tensorboard_log_dir,
            histogram_freq=1
        )
        
        callbacks = [checkpoint, early_stopping, tensorboard_callback]
        
        # Train the model
        logger.info(f"Training model {self.name}...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info(f"Model training complete. Best model saved to {checkpoint_path}")
        
        # Save training history
        history_path = self.model_dir / f"history_{timestamp}.json"
        with open(history_path, 'w') as f:
            json.dump({k: list(map(float, v)) for k, v in self.history.history.items()}, f)
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        
        logger.info(f"Evaluating model {self.name}...")
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=1)
        
        logger.info(f"Test Loss: {loss:.4f}")
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        
        return loss, accuracy
    
    def predict(self, X):
        """Make predictions with the model."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        
        predictions = self.model.predict(X)
        return predictions
    
    def predict_classes(self, X):
        """Predict sentiment classes."""
        predictions = self.predict(X)
        return np.argmax(predictions, axis=1)
    
    def save_model(self, file_path=None):
        """Save the model to a file."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.model_dir / f"{self.name}_{timestamp}.h5"
        
        self.model.save(file_path)
        logger.info(f"Model saved to {file_path}")
        
        # Save model configuration
        config_path = file_path.with_suffix('.json')
        with open(config_path, 'w') as f:
            json.dump({
                'name': self.name,
                'input_shape': self.input_shape,
                'classes': SENTIMENT_CLASSES
            }, f)
    
    def load_model(self, file_path):
        """Load the model from a file."""
        self.model = tf.keras.models.load_model(file_path)
        logger.info(f"Model loaded from {file_path}")
        
        # Load model configuration
        config_path = file_path.with_suffix('.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.name = config.get('name', self.name)
                self.input_shape = config.get('input_shape', self.input_shape)