"""Apply Algorithm 1 (rule layer)

This module implements the ensemble voting logic (Algorithm 1 / rule layer) that:
1. Loads the trained models (KNN, LDA, LR) from the artifacts directory
2. Loads test data from the processed directory
3. Applies each model to generate individual predictions
4. Combines these predictions using a voting mechanism
5. Saves the final ensemble predictions to artifacts/y_pred.npy

Usage:
    Run directly (e.g., `python src/ensemble.py`) or via Makefile (`make predict`)
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Import necessary constants from __init__
from src import ARTIFACTS_DIR, PROC_DIR, MODEL_NAMES

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_models():
    """
    Load trained models from artifacts directory.
    
    Returns:
        dict: Dictionary mapping model names to loaded model objects
    """
    models = {}
    for model_name, _ in MODEL_NAMES.items():
        model_path = os.path.join(ARTIFACTS_DIR, f"{model_name}.pkl")
        with open(model_path, 'rb') as f:
            models[model_name] = pickle.load(f)
    return models


def load_test_data(balanced=False):
    """
    Load test data from processed directory.
    
    Returns:
        tuple: X_test features array
    """
    prefix = "balanced_" if balanced else ""
    
    X_test = pd.read_csv(os.path.join(PROC_DIR, f"{prefix}X_test.csv")).values
    return X_test

def apply_rule_layer(predictions: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Implement Algorithm 1 (rule layer) to combine model predictions.
    
    This function takes individual model predictions and applies
    a voting mechanism to determine the final ensemble prediction.
    
    The exact implementation would follow the paper's Algorithm 1:
    - If all models predict fraud (1), the ensemble predicts fraud
    - If any model predicts non-fraud (0), the ensemble predicts non-fraud
    - This favors high recall at the expense of precision
    
    Args:
        predictions: Dictionary mapping model names to their predictions
        
    Returns:
        np.ndarray: Final ensemble predictions
    """
    # Initialize an array to store the final predictions
    ensemble_predictions = np.ones(len(predictions[list(predictions.keys())[0]]), dtype=int)
    
    # Iterate through each prediction
    for i in range(len(ensemble_predictions)):
        # Count the number of predictions for each class
        for _, model_predictions in predictions.items():
            ensemble_predictions[i] = ensemble_predictions[i] and model_predictions[i]

    return ensemble_predictions


def main():
    """Main entry point for generating ensemble predictions."""
    logger.info("Starting ensemble prediction process")
    
    # Ensure artifacts directory exists
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # Load trained models
    models = load_models()
    
    # Load test data
    X_test = load_test_data()
    
    # Generate predictions from each model
    predictions = {}
    for model_name, model in models.items():
        logger.info(f"Generating predictions with {model_name} model")
        predictions[model_name] = model.predict(X_test)
    
    # Apply rule layer to combine predictions
    ensemble_predictions = apply_rule_layer(predictions)
    
    # Save ensemble predictions
    output_path = os.path.join(ARTIFACTS_DIR, "y_pred.npy")
    np.save(output_path, ensemble_predictions)
    logger.info(f"Saved ensemble predictions to {output_path}")
    
    # Summary statistics
    logger.info("Ensemble prediction complete")
    logger.info(f"Total predictions: {len(ensemble_predictions)}")
    logger.info(f"Predicted frauds: {np.sum(ensemble_predictions)}")
    logger.info(f"Predicted non-frauds: {len(ensemble_predictions) - np.sum(ensemble_predictions)}")


if __name__ == "__main__":
    main()