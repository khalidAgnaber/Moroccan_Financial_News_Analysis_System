"""
Visualization utilities for model evaluation and results.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SENTIMENT_CLASSES
from utils.logger import setup_logger

logger = setup_logger("visualization")

def plot_training_history(history, figsize=(15, 5)):
    """
    Plot training history metrics.
    
    Args:
        history: Training history object from model.fit()
        figsize: Figure size (width, height) in inches
    """
    plt.figure(figsize=figsize)
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    return plt.gcf()

def plot_sentiment_distribution(df, sentiment_column='sentiment', figsize=(10, 6)):
    """
    Plot distribution of sentiment classes.
    
    Args:
        df: DataFrame containing sentiment data
        sentiment_column: Column name with sentiment labels
        figsize: Figure size (width, height) in inches
    """
    plt.figure(figsize=figsize)
    
    # Count sentiment classes
    sentiment_counts = df[sentiment_column].value_counts()
    
    # Plot bar chart
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    
    # Add count labels on top of bars
    for i, count in enumerate(sentiment_counts.values):
        plt.text(i, count + 5, str(count), ha='center')
    
    plt.tight_layout()
    return plt.gcf()

def plot_roc_curve_multiclass(y_true, y_score, class_labels=SENTIMENT_CLASSES, figsize=(10, 8)):
    """
    Plot ROC curve for multi-class classification.
    
    Args:
        y_true: True labels (one-hot encoded or integer class indices)
        y_score: Predicted scores (probabilities) for each class
        class_labels: Class names for display
        figsize: Figure size (width, height) in inches
    """
    # Convert y_true to one-hot encoding if it's not already
    if len(y_true.shape) == 1:
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=np.arange(len(class_labels)))
    else:
        y_true_bin = y_true
    
    plt.figure(figsize=figsize)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(len(class_labels)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'ROC curve (class: {class_labels[i]}, area = {roc_auc[i]:.2f})')
    
    # Plot the diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    return plt.gcf()

def plot_sentiment_by_company(df, company_column='companies', sentiment_column='sentiment', top_n=10, figsize=(12, 8)):
    """
    Plot sentiment distribution by company.
    
    Args:
        df: DataFrame containing sentiment and company data
        company_column: Column name with company names
        sentiment_column: Column name with sentiment labels
        top_n: Number of top companies to display
        figsize: Figure size (width, height) in inches
    """
    # Handle case where companies are in a comma-separated string
    if df[company_column].dtype == 'object' and df[company_column].str.contains(',').any():
        # Explode the dataframe so each company gets its own row
        df = df.copy()
        df[company_column] = df[company_column].str.split(',')
        df = df.explode(company_column)
        df[company_column] = df[company_column].str.strip()
    
    # Count mentions by company
    company_counts = df[company_column].value_counts().head(top_n)
    top_companies = company_counts.index.tolist()
    
    # Filter for top companies
    df_top = df[df[company_column].isin(top_companies)]
    
    # Create pivot table for sentiment by company
    pivot = pd.crosstab(df_top[company_column], df_top[sentiment_column])
    pivot = pivot.reindex(top_companies)
    
    # Plot stacked bar chart
    plt.figure(figsize=figsize)
    pivot.plot(kind='bar', stacked=True, colormap='viridis')
    
    plt.title('Sentiment Distribution by Company')
    plt.xlabel('Company')
    plt.ylabel('Count')
    plt.legend(title='Sentiment')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return plt.gcf()

def plot_sentiment_over_time(df, date_column='date', sentiment_column='sentiment', freq='M', figsize=(15, 6)):
    """
    Plot sentiment trends over time.
    
    Args:
        df: DataFrame containing sentiment and date data
        date_column: Column name with date information
        sentiment_column: Column name with sentiment labels
        freq: Frequency for resampling ('D' for daily, 'W' for weekly, 'M' for monthly)
        figsize: Figure size (width, height) in inches
    """
    # Ensure date column is datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Set date as index
    df.set_index(date_column, inplace=True)
    
    # Group by date and sentiment, count occurrences
    sentiment_over_time = df.groupby([pd.Grouper(freq=freq), sentiment_column]).size().unstack(fill_value=0)
    
    # Plot
    plt.figure(figsize=figsize)
    sentiment_over_time.plot(kind='line', marker='o')
    
    plt.title('Sentiment Trends Over Time')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Sentiment')
    
    plt.tight_layout()
    return plt.gcf()

def save_visualization(fig, file_path):
    """
    Save visualization to file.
    
    Args:
        fig: Matplotlib figure object
        file_path: Path to save the figure
    """
    fig.savefig(file_path, bbox_inches='tight', dpi=300)
    logger.info(f"Visualization saved to {file_path}")
    plt.close(fig)