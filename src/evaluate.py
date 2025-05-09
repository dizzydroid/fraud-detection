"""Compute metrics and dump to JSON

This module evaluates the performance of the ensemble fraud detection model by:
1. Loading the ground truth labels (y_true) from artifacts directory
2. Loading the ensemble predictions (y_pred) from artifacts directory
3. Computing key performance metrics (especially recall) / Confusion Matrix
   - True Positive Rate (TPR)
   - False Positive Rate (FPR)
   - Precision
   - Accuracy
   - Recall
   - F1 Score
   - ROC AUC
4. Saving these metrics to results/metrics.json
5. Generating optional visualization plots in results/figures/

Usage:
    Run directly (e.g., `python src/evaluate.py`) or via Makefile (`make evaluate`)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, recall_score, precision_score, f1_score
import seaborn as sns
import logging

from src import ARTIFACTS_DIR, RESULTS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """
    Load ground truth and predictions from artifacts directory.
    
    Returns:
        tuple: y_true, y_pred
    """
    y_true = np.load(os.path.join(ARTIFACTS_DIR, "y_true.npy"))
    y_pred = np.load(os.path.join(ARTIFACTS_DIR, "y_pred.npy"))
    return y_true, y_pred

def compute_metrics(y_true, y_pred):
    """
    Compute key performance metrics, ensuring all outputs are JSON serializable and confusion matrix is not inverted.
    Args:
        y_true: Ground truth labels (isFraud: 1=Fraud, 0=Non-Fraud)
        y_pred: Predicted labels
    Returns:
        dict: Performance metrics
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=0)
    roc_auc = auc(fpr, tpr)
    recall = float(recall_score(y_true, y_pred, pos_label=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, pos_label=0))

    return {
        "recall": recall,
        "f1": f1,
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "accuracy": accuracy,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist()
    }

def plot_metrics(metrics):
    """
    Plot confusion matrix as a heatmap, following notebook convention.
    Args:
        metrics: Performance metrics
    """
    cm = np.array(metrics["confusion_matrix"])
    labels = ['Non-Fraud', 'Fraud']

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, linewidths=.5, linecolor='black')
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.title('Ensemble Confusion Matrix')

    # Ensure directory exists
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "figures", "ensemble_confusion_matrix.png"))
    plt.close()

def main():
    """Main entry point for evaluating the ensemble model."""
    logger.info("Evaluating ensemble model")
    
    # Load data
    y_true, y_pred = load_data()
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    
    # Plot metrics
    plot_metrics(metrics)
    
    # Save metrics to JSON
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()



