"""
BERT model for sentiment analysis of French financial news.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from transformers import TFAutoModel, AutoTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from config import MODELS_DIR
from utils.logger import setup_logger

logger = setup_logger("bert_model")

class BERTSentimentModel:
    def __init__(self, max_length=128):
        """Initialize the BERT model."""
        self.max_length = max_length
        self.model = None
        self.history = None
        self.tokenizer = None
        
    def build_model(self):
        """Build a BERT model for sentiment analysis."""
        # Load multilingual BERT tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        base_model = TFAutoModel.from_pretrained("bert-base-multilingual-cased")
        
        # Define inputs
        input_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32, name="input_ids")
        attention_mask = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32, name="attention_mask")
        
        # Get BERT embeddings
        bert_output = base_model(input_ids, attention_mask=attention_mask)[0]
        
        # Use CLS token for classification
        cls_output = bert_output[:, 0, :]
        
        # Add dropout and classification layers
        x = tf.keras.layers.Dropout(0.2)(cls_output)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
        
        # Create the model
        self.model = tf.keras.Model(
            inputs=[input_ids, attention_mask], 
            outputs=outputs
        )
        
        # Freeze the BERT layers to prevent overfitting with small dataset
        for layer in base_model.layers:
            layer.trainable = False
            
        logger.info("Built multilingual BERT model for French financial sentiment analysis")
        return self.model
    
    def compile_model(self, learning_rate=2e-5):
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
    
    def prepare_inputs(self, texts):
        """Convert text to BERT inputs."""
        encodings = self.tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='tf'
        )
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }
    
    def train(self, X_train, y_train, X_val, y_val, epochs=3, batch_size=8):
        """Train the model with early stopping."""
        if self.model is None:
            self.compile_model()
            
        # Check if labels are already one-hot encoded
        if len(y_train.shape) == 1 or y_train.shape[1] == 1:
            logger.info("Converting labels to one-hot encoding")
            num_classes = 3  # negative, neutral, positive
            y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
            y_val = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)
        
        # Prepare inputs
        train_inputs = self.prepare_inputs(X_train)
        val_inputs = self.prepare_inputs(X_val)
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train the model
        logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
        
        self.history = self.model.fit(
            train_inputs,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_inputs, y_val),
            callbacks=[early_stopping],
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
        
        # Get predictions
        return self.model.predict(inputs)
    
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
            num_classes = 3  # negative, neutral, positive
            y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
        else:
            y_test_onehot = y_test
        
        # Prepare inputs
        test_inputs = self.prepare_inputs(X_test)
        
        # Evaluate
        loss, accuracy = self.model.evaluate(test_inputs, y_test_onehot)
        logger.info(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
        
        # Get predictions
        y_pred = self.predict_classes(X_test)
        y_true = np.argmax(y_test_onehot, axis=1) if len(y_test_onehot.shape) > 1 else y_test
        
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
            self.model.save(model_path)
            
            # Save the tokenizer
            tokenizer_path = os.path.join(model_dir, "tokenizer")
            self.tokenizer.save_pretrained(tokenizer_path)
            
            filepath = model_path
            
        logger.info(f"Model saved to {filepath}")
        return filepath