"""
Optimized BERT sentiment analysis training script with significant accuracy improvements.
Addresses low accuracy issues with better hyperparameters and training strategies.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, AutoConfig
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import seaborn as sns
import re
import random
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ========== OPTIMIZED CONFIG ==========
MODEL_NAME = "camembert-base"
MAX_LENGTH = 384  
EPOCHS = 10  
BATCH_SIZE = 4  
LEARNING_RATE = 3e-5 
WARMUP_EPOCHS = 2
PATIENCE = 7  
DROPOUT_RATE = 0.1 
MIN_TEXT_LENGTH = 30  
AUGMENTATION_FACTOR = 2.0 
PROCESSED_DATA_FILE = "data/processed/labeled_news_multi.csv"
MODEL_DIR = "models/optimized_bert_sentiment_model"
PLOT_PATH = os.path.join(MODEL_DIR, "training_history.png")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_mapping.txt")

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)

# Set seeds for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seeds(42)

# ========== ENHANCED PREPROCESSING ==========

def advanced_text_cleaning(text):
    """Enhanced text cleaning for financial news."""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove URLs and emails
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Normalize financial terms (keep more context)
    text = re.sub(r'\b\d+[.,]\d+\s*[€$£¥]\b', ' <MONTANT> ', text)
    text = re.sub(r'\b\d+[.,]\d+\s*%\b', ' <POURCENTAGE> ', text)
    text = re.sub(r'\b\d+[.,]\d+\b', ' <NOMBRE> ', text)
    
    # Clean excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def create_financial_augmentations(text, sentiment):
    """Create multiple augmentations for financial texts."""
    augmentations = []
    
    # Synonym replacements for French financial terms
    financial_synonyms = {
        'augmentation': ['hausse', 'progression', 'croissance', 'montée'],
        'baisse': ['chute', 'diminution', 'recul', 'déclin'],
        'bénéfice': ['profit', 'gain', 'résultat positif'],
        'perte': ['déficit', 'résultat négatif', 'moins-value'],
        'entreprise': ['société', 'compagnie', 'firme', 'groupe'],
        'marché': ['bourse', 'secteur', 'domaine'],
        'investissement': ['placement', 'mise de fonds'],
        'croissance': ['développement', 'expansion', 'essor'],
        'chiffre d\'affaires': ['CA', 'revenus', 'recettes'],
        'action': ['titre', 'valeur mobilière'],
        'performance': ['résultat', 'rendement'],
        'économie': ['secteur économique', 'marché'],
    }
    
    words = text.split()
    if len(words) < 5:
        return [text]
    
    # Create 2-3 augmentations per text
    for _ in range(min(3, max(1, len(words) // 20))):
        aug_words = words.copy()
        changes_made = 0
        
        for i, word in enumerate(aug_words):
            word_lower = word.lower()
            if word_lower in financial_synonyms and random.random() < 0.4:
                aug_words[i] = random.choice(financial_synonyms[word_lower])
                changes_made += 1
            elif changes_made < 3 and random.random() < 0.1 and len(aug_words) > 10:
                # Random word dropout
                if i < len(aug_words) - 1:
                    aug_words.pop(i)
                    changes_made += 1
        
        if changes_made > 0:
            augmented_text = ' '.join(aug_words)
            augmentations.append(augmented_text)
    
    return augmentations if augmentations else [text]

def load_and_prepare_data(file_path):
    """Enhanced data loading with aggressive augmentation."""
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Enhanced text combination and cleaning
    df['title'] = df['title'].fillna('').apply(advanced_text_cleaning)
    df['text'] = df['text'].fillna('').apply(advanced_text_cleaning)
    df['combined_text'] = df['title'] + ' [SEP] ' + df['text']
    
    # Filter out short texts more aggressively
    df = df[df['combined_text'].str.len() > MIN_TEXT_LENGTH]
    df = df.dropna(subset=['sentiment', 'combined_text'])
    df = df.drop_duplicates(subset=['combined_text'])
    
    print(f"After cleaning: {len(df)} samples")
    
    # Aggressive data augmentation
    df = aggressive_augmentation(df)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['sentiment'])
    
    print(f"Final dataset: {len(df)} samples")
    print(f"Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
    
    return df['combined_text'].values, y, label_encoder

def aggressive_augmentation(df):
    """More aggressive augmentation to increase dataset size."""
    original_size = len(df)
    sentiment_counts = df['sentiment'].value_counts()
    target_size = int(sentiment_counts.max() * AUGMENTATION_FACTOR)
    
    augmented_data = []
    
    for sentiment in sentiment_counts.index:
        sentiment_df = df[df['sentiment'] == sentiment].copy()
        current_count = len(sentiment_df)
        
        if current_count < target_size:
            needed = target_size - current_count
            
            # Create augmentations
            for _ in range(needed):
                sample = sentiment_df.sample(1).iloc[0]
                augmentations = create_financial_augmentations(
                    sample['combined_text'], sentiment
                )
                
                for aug_text in augmentations[:1]:  # Take first augmentation
                    augmented_data.append({
                        'combined_text': aug_text,
                        'sentiment': sentiment
                    })
                    if len(augmented_data) >= needed:
                        break
                
                if len(augmented_data) >= needed:
                    break
    
    if augmented_data:
        augmented_df = pd.DataFrame(augmented_data)
        df = pd.concat([df, augmented_df], ignore_index=True)
        print(f"Added {len(augmented_data)} augmented samples")
    
    return df

# ========== OPTIMIZED MODEL BUILDING ==========

def build_optimized_model(num_labels):
    """Build optimized BERT model."""
    print(f"Loading {MODEL_NAME} model and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Optimized model configuration
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = num_labels
    config.hidden_dropout_prob = DROPOUT_RATE
    config.attention_probs_dropout_prob = DROPOUT_RATE
    
    model = TFAutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        config=config
    )
    
    # Use higher learning rate for better training
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=LEARNING_RATE,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    print("Optimized model built and compiled.")
    return model, tokenizer

def prepare_inputs(tokenizer, texts):
    """Prepare inputs with enhanced tokenization."""
    texts = [str(text) if text else "" for text in texts]
    
    encoded = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='tf',
        add_special_tokens=True,
        return_attention_mask=True
    )
    
    return {
        'input_ids': encoded['input_ids'], 
        'attention_mask': encoded['attention_mask']
    }

# ========== OPTIMIZED TRAINING ==========

def create_optimized_lr_scheduler():
    """Create optimized learning rate scheduler."""
    def lr_schedule(epoch, lr):
        if epoch < WARMUP_EPOCHS:
            # Warmup: gradually increase LR
            return LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
        elif epoch < 8:
            # Stable phase
            return LEARNING_RATE
        elif epoch < 15:
            # First decay
            return LEARNING_RATE * 0.5
        else:
            # Final decay
            return LEARNING_RATE * 0.1
    
    return tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)

def plot_training_history(history, plot_path):
    """Plot comprehensive training history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning Rate
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'], 'g-', linewidth=2)
        axes[1, 0].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy difference (overfitting indicator)
    train_acc = np.array(history.history['accuracy'])
    val_acc = np.array(history.history['val_accuracy'])
    acc_diff = train_acc - val_acc
    
    axes[1, 1].plot(acc_diff, 'purple', linewidth=2)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Train-Val Accuracy Gap', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Accuracy Difference')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {plot_path}")

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def evaluate_model(model, test_inputs, y_test, label_encoder):
    """Comprehensive model evaluation."""
    print("=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    
    # Evaluation
    loss, accuracy = model.evaluate(test_inputs, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Predictions
    logits = model.predict(test_inputs, verbose=0).logits
    y_pred_proba = tf.nn.softmax(logits, axis=1).numpy()
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Confidence analysis
    confidence_scores = np.max(y_pred_proba, axis=1)
    print(f"\nPrediction Confidence Statistics:")
    print(f"Mean confidence: {confidence_scores.mean():.4f}")
    print(f"Std confidence: {confidence_scores.std():.4f}")
    print(f"Min confidence: {confidence_scores.min():.4f}")
    print(f"Max confidence: {confidence_scores.max():.4f}")
    
    # High confidence predictions (> 0.7)
    high_conf_mask = confidence_scores > 0.7
    if high_conf_mask.sum() > 0:
        high_conf_acc = (y_test[high_conf_mask] == y_pred[high_conf_mask]).mean()
        print(f"High confidence predictions: {high_conf_mask.sum()}/{len(y_test)} ({high_conf_mask.sum()/len(y_test)*100:.1f}%)")
        print(f"High confidence accuracy: {high_conf_acc:.4f}")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Save confusion matrix
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plot_confusion_matrix(y_test, y_pred, label_encoder.classes_, cm_path)
    
    return accuracy, y_pred, y_pred_proba

def save_label_mapping(label_encoder, path):
    """Save label mapping."""
    with open(path, "w", encoding='utf-8') as f:
        f.write("Label Mapping:\n")
        f.write("-" * 20 + "\n")
        for idx, label in enumerate(label_encoder.classes_):
            f.write(f"{idx}: {label}\n")
    print(f"Label mapping saved to {path}")

def main():
    print("=" * 70)
    print("OPTIMIZED BERT SENTIMENT ANALYSIS TRAINING")
    print("=" * 70)
    
    # Load and prepare data
    X, y, label_encoder = load_and_prepare_data(PROCESSED_DATA_FILE)
    
    # Better stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nDataset splits:")
    print(f"Train: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples") 
    print(f"Test: {len(X_test)} samples")
    
    # Class weights
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"\nClass weights: {class_weight_dict}")
    
    # Build model
    model, tokenizer = build_optimized_model(num_labels=len(label_encoder.classes_))
    
    # Prepare inputs
    print("\nTokenizing inputs...")
    train_inputs = prepare_inputs(tokenizer, X_train)
    val_inputs = prepare_inputs(tokenizer, X_val)
    test_inputs = prepare_inputs(tokenizer, X_test)
    
    # Optimized callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "best_model"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        create_optimized_lr_scheduler()
    ]
    
    # Train
    print(f"\nStarting training:")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Max length: {MAX_LENGTH}")
    print("-" * 50)
    
    history = model.fit(
        train_inputs,
        y_train,
        validation_data=(val_inputs, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history, PLOT_PATH)
    
    # Evaluate
    test_accuracy, y_pred, y_pred_proba = evaluate_model(
        model, test_inputs, y_test, label_encoder
    )
    
    # Save model and artifacts
    print(f"\nSaving model...")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    save_label_mapping(label_encoder, LABEL_MAP_PATH)
    
    # Final summary
    print(f"\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Model saved to: {MODEL_DIR}")
    
    return model, tokenizer, label_encoder, history

if __name__ == "__main__":
    model, tokenizer, label_encoder, history = main()