"""
Compare performance of different model architectures.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np

from config import MODELS_DIR

# Find all model directories
model_dirs = [d for d in MODELS_DIR.iterdir() if d.is_dir() and 'sentiment_model' in d.name]

# Create dataframe to store results
results = []

for model_dir in model_dirs:
    # Extract model type from directory name
    model_type = model_dir.name.split('_')[0]
    
    # Load metrics if available
    metrics_file = model_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            
        # Add to results
        results.append({
            'model_type': model_type,
            'accuracy': metrics.get('accuracy', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1': metrics.get('f1', 0)
        })

# Create dataframe
if results:
    df = pd.DataFrame(results)
    
    # Print results
    print("Model Performance Comparison:")
    print(df.set_index('model_type'))
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    # Set width of bars
    barWidth = 0.2
    
    # Set positions of bars on X axis
    r1 = np.arange(len(df))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    
    # Create bars
    plt.bar(r1, df['accuracy'], width=barWidth, label='Accuracy')
    plt.bar(r2, df['precision'], width=barWidth, label='Precision')
    plt.bar(r3, df['recall'], width=barWidth, label='Recall')
    plt.bar(r4, df['f1'], width=barWidth, label='F1 Score')
    
    # Add labels and title
    plt.xlabel('Model Type')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks([r + barWidth*1.5 for r in range(len(df))], df['model_type'])
    plt.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "model_comparison.png"))
    plt.close()
    
    print(f"Comparison chart saved to {os.path.join(MODELS_DIR, 'model_comparison.png')}")
else:
    print("No model metrics found for comparison.")