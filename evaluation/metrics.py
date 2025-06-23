"""
Metrics for evaluating sentiment analysis models.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging

from utils.logger import setup_logger

logger = setup_logger("metrics")

class SentimentMetrics:
    """Calculate and store metrics for sentiment analysis evaluation."""
    
    def __init__(self, y_true, y_pred, class_names=None):
        """Initialize with true and predicted labels."""
        self.y_true = y_true
        self.y_pred = y_pred
        
        # Determine unique classes in the data
        self.unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        
        # Use provided class names or default to numeric labels
        if class_names is not None:
            self.class_names = class_names
        else:
            # Default class names for sentiment analysis
            sentiment_classes = ['negative', 'neutral', 'positive']
            if len(self.unique_classes) <= len(sentiment_classes):
                self.class_names = [sentiment_classes[i] for i in self.unique_classes]
            else:
                self.class_names = [f"Class {i}" for i in self.unique_classes]
        
        # Calculate metrics
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate all metrics."""
        # Handle the case with very few samples
        if len(self.y_true) == 0:
            logger.warning("No samples to evaluate")
            self.accuracy = 0
            self.precision = 0
            self.recall = 0
            self.f1 = 0
            self.confusion_mat = np.zeros((len(self.unique_classes), len(self.unique_classes)))
            self.class_metrics = {cls_name: {'precision': 0, 'recall': 0, 'f1': 0} 
                                 for cls_name in self.class_names}
            return
        
        try:
            # Overall metrics
            self.accuracy = accuracy_score(self.y_true, self.y_pred)
            
            # Handle case with only one class
            if len(self.unique_classes) <= 1:
                logger.warning("Only one class present, metrics may not be meaningful")
                self.precision = 1.0 if self.accuracy == 1.0 else 0.0
                self.recall = 1.0 if self.accuracy == 1.0 else 0.0
                self.f1 = 1.0 if self.accuracy == 1.0 else 0.0
            else:
                # Multi-class metrics with handling for classes not present in predictions
                self.precision = precision_score(self.y_true, self.y_pred, 
                                               average='weighted', zero_division=0)
                self.recall = recall_score(self.y_true, self.y_pred, 
                                         average='weighted', zero_division=0)
                self.f1 = f1_score(self.y_true, self.y_pred, 
                                 average='weighted', zero_division=0)
            
            # Confusion matrix
            self.confusion_mat = confusion_matrix(self.y_true, self.y_pred, 
                                                labels=self.unique_classes)
            
            # Per-class metrics
            self.class_metrics = {}
            for i, cls in enumerate(self.unique_classes):
                cls_name = self.class_names[i]
                
                # For per-class metrics, use a binary approach (one-vs-rest)
                y_true_bin = (self.y_true == cls).astype(int)
                y_pred_bin = (self.y_pred == cls).astype(int)
                
                # Calculate metrics, handling division by zero
                if sum(y_true_bin) == 0 and sum(y_pred_bin) == 0:
                    precision = 1.0  # Both true and predicted have no positives
                elif sum(y_pred_bin) == 0:
                    precision = 0.0  # Predicted no positives but there were some
                else:
                    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
                
                if sum(y_true_bin) == 0:
                    recall = 1.0 if sum(y_pred_bin) == 0 else 0.0
                else:
                    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
                
                if precision == 0 and recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                self.class_metrics[cls_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            # Set default values if calculation fails
            self.accuracy = 0
            self.precision = 0
            self.recall = 0
            self.f1 = 0
            self.confusion_mat = np.zeros((len(self.unique_classes), len(self.unique_classes)))
            self.class_metrics = {cls_name: {'precision': 0, 'recall': 0, 'f1': 0} 
                                 for i, cls_name in enumerate(self.class_names)}
    
    def print_metrics(self):
        """Print all metrics in a readable format."""
        print(f"Accuracy: {self.accuracy:.4f}")
        print(f"Precision: {self.precision:.4f}")
        print(f"Recall: {self.recall:.4f}")
        print(f"F1 Score: {self.f1:.4f}")
        print("\nConfusion Matrix:")
        print(self.confusion_mat)
        print("\nMetrics by Class:")
        for cls_name, metrics in self.class_metrics.items():
            print(f"{cls_name}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")
    
    def get_metrics_dict(self):
        """Return all metrics as a dictionary."""
        metrics_dict = {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'confusion_matrix': self.confusion_mat.tolist(),
            'class_metrics': self.class_metrics
        }
        return metrics_dict