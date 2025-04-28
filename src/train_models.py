"""Fit KNN, LDA, and LR models and save pickles.

This module orchestrates the training pipeline for the fraud detection system. It:
1. Loads preprocessed data from the data/processed/ directory
2. Trains three different models:
   - K-Nearest Neighbors (KNN)
   - Linear Discriminant Analysis (LDA)
   - Linear Regression (LR)
3. Saves the trained models as pickled files in the artifacts/ directory

Usage:
    Run this directly (e.g., `python src/train_models.py`) or via the Makefile (`make train`)

Note:
    - Models are defined in separate modules within the models/ directory
    - Each model should be tuned for high recall (≥0.93) as per project requirements
    - The saved models will be used by the ensemble.py module for final predictions
    - Train models on the same preprocessed data to ensure consistency
"""

import os
import pickle
import logging
import sys
from typing import Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

# Add the project root to the Python path to allow imports when run directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import model classes from respective modules
from src import ARTIFACTS_DIR, PROC_DIR, MODEL_NAMES, MODEL_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_data(balanced=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load preprocessed training and test data.
    
    Args:
        balanced: If True, load the balanced version of the dataset.
                 If False, load the original unbalanced dataset.
    
    Returns:
        Tuple containing:
        - X_train: Training features
        - X_test: Test features
        - y_train: Training labels
        - y_test: Test labels
    """
    prefix = "balanced_" if balanced else ""
    
    print("\n" + "="*20)
    print(f"LOADING DATASET")
    print("="*20)
    
    # Load X_train, X_test, y_train, y_test from CSV files
    X_train = pd.read_csv(os.path.join(PROC_DIR, f"{prefix}X_train.csv")).values
    X_test = pd.read_csv(os.path.join(PROC_DIR, f"{prefix}X_test.csv")).values
    y_train = pd.read_csv(os.path.join(PROC_DIR, f"{prefix}y_train.csv"))["isFraud"].values
    y_test = pd.read_csv(os.path.join(PROC_DIR, f"{prefix}y_test.csv"))["isFraud"].values
    
    # Calculate fraud percentages
    fraud_percent_train = np.mean(y_train) * 100
    fraud_percent_test = np.mean(y_test) * 100
    
    print(f"\nTraining set: {X_train.shape[0]:,} samples ({fraud_percent_train:.2f}% fraud)")
    print(f"Testing set:  {X_test.shape[0]:,} samples ({fraud_percent_test:.2f}% fraud)")
    
    return X_train, X_test, y_train, y_test


def train_and_save_models(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Train KNN, LDA and Linear Regression models and save them to disk.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary mapping model names to their recall scores on test data
    """
    recall_scores = {}
    
    # Ensure artifacts directory exists
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)
        print(f"\nCreated artifacts directory: {ARTIFACTS_DIR}")
    
    print("\n" + "="*18)
    print(f"TRAINING MODELS")
    print("="*18)
    
    # Train and save each model
    for name, model in MODEL_CLASSES.items():
        model_name = MODEL_NAMES.get(name, name.upper())
        
        print(f"\nTraining {model_name}...")
        
        # Initialize model
        model = model()
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Save model
        model_path = os.path.join(ARTIFACTS_DIR, f"{name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved model to {os.path.basename(model_path)}")
        
        # Evaluate model recall
        y_pred = model.predict(X_test)
        recall = recall_score(y_test, y_pred)
        recall_scores[name] = recall
        print(f"{model_name} recall: {recall:.4f}")
    
    return recall_scores


def main():
    """Main entry point for training models."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train fraud detection models')
    parser.add_argument('--balanced', action='store_true',
                        help='Use balanced dataset instead of original dataset')
    args = parser.parse_args()
    
    print("\n" + "="*45)
    print("FRAUD DETECTION SYSTEM - MODEL TRAINING")
    print("="*45)
    print("\nThis script will train three models for fraud detection:")
    print("  - K-Nearest Neighbors (KNN)")
    print("  - Linear Discriminant Analysis (LDA)")
    print("  - Linear Regression (LR)")
    print("\nAll models will be saved to the artifacts/ directory for later use.")
    
    # Load preprocessed data (balanced or unbalanced)
    X_train, X_test, y_train, y_test = load_data(balanced=args.balanced)
    
    # Train models and save them to artifacts directory
    recall_scores = train_and_save_models(X_train, y_train, X_test, y_test)
    
    # Save ground truth (y_test) for later evaluation
    y_true_path = os.path.join(ARTIFACTS_DIR, "y_true.npy")
    np.save(y_true_path, y_test)
    print(f"\nSaved ground truth labels to {os.path.basename(y_true_path)}")
    print("    (These will be used for evaluating model performance later)")

    # Print recall scores summary
    print("\n" + "="*50)
    print("TRAINING COMPLETE - RECALL SCORES SUMMARY")
    print("="*50)
    
    # Sort models by recall score (highest first)
    sorted_models = sorted(recall_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\nModel Recall Scores:")
    for i, (model_name, recall) in enumerate(sorted_models):
        full_name = MODEL_NAMES.get(model_name, model_name.upper())

        print(f"{i+1}. {full_name}: {recall:.4f}")

if __name__ == "__main__":
    main()