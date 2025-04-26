"""Fit KNN, LDA, and LR models and save pickles.

This module orchestrates the training pipeline for the fraud detection system. It:
1. Loads preprocessed data from the data/processed/ directory
2. Trains three different models:
   - K-Nearest Neighbors (KNN)
   - Linear Discriminant Analysis (LDA)
   - Logistic Regression (LR)
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
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import recall_score

# Import model classes from respective modules
from src.models.knn import KNNModel
from src.models.lda import LDAModel
from src.models.linreg import LogisticRegressionModel
from src.__init__ import ARTIFACTS_DIR, PROC_DIR

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load preprocessed training and test data.
    
    Returns:
        Tuple containing:
        - X_train: Training features
        - X_test: Test features
        - y_train: Training labels
        - y_test: Test labels
    """
    # TODO: Implement data loading from processed directory
    # This should load the preprocessed data created by preprocessing.py
    
    logger.info("Loading preprocessed data")
    
    #TODO: return X_train, X_test, y_train, y_test


def train_and_save_models(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Train KNN, LDA and Logistic Regression models and save them to disk.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary mapping model names to their recall scores on test data
    """
    # Dictionary to hold model instances
    models = {
        'knn': KNNModel(),  # Instantiate KNN model
        'lda': LDAModel(),  # Instantiate LDA model
        'lr': LogisticRegressionModel()  # Instantiate Logistic Regression model
    }
    
    # Dictionary to hold recall scores
    recall_scores = {}
    
    # Ensure artifacts directory exists
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)
        logger.info(f"Created artifacts directory at {ARTIFACTS_DIR}")
    
    # Train and save each model
    for name, model in models.items():
        logger.info(f"Training {name.upper()} model")
        
        # TODO: Train the model
        # model.fit(X_train, y_train)
        
        # Save model to disk
        model_path = os.path.join(ARTIFACTS_DIR, f"{name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved {name.upper()} model to {model_path}")
        
        # Evaluate model recall
        # y_pred = model.predict(X_test)
        # recall = recall_score(y_test, y_pred)
        # recall_scores[name] = recall
        # logger.info(f"{name.upper()} recall: {recall:.4f}")
    
    return recall_scores


def main():
    """Main entry point for training models."""
    # TODO: - save models to artifacts/ as pkl files
    #       - save y_test to artifacts/y_true.npy        


if __name__ == "__main__":
    main()