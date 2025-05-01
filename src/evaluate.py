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
<<<<<<< HEAD
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
=======
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, recall_score, precision_score, f1_score
>>>>>>> origin/soloPlayer
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
    Compute key performance metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        dict: Performance metrics
    """
    cm = confusion_matrix(y_true, y_pred)
<<<<<<< HEAD
    tn, fp, fn, tp = cm.ravel()
=======
    # tn, fp, fn, tp = cm.ravel()
>>>>>>> origin/soloPlayer
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Compute accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
<<<<<<< HEAD
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,   
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "accuracy": accuracy,
        "fpr": fpr,
        "tpr": tpr
    }


=======
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),  # already converted
        "accuracy": float(accuracy),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist()
    }

>>>>>>> origin/soloPlayer
def plot_metrics(metrics):
    """
    Plot key performance metrics.

    Args:
        metrics: Performance metrics
    """
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot precision, recall, and F1 score
    ax1.plot(metrics["precision"], metrics["recall"], label="Precision-Recall Curve")
    ax1.set_xlabel("Precision")
    ax1.set_ylabel("Recall")
    ax1.set_title("Precision-Recall Curve")
    ax1.legend()
    
    # Plot ROC curve
    ax2.plot(metrics["fpr"], metrics["tpr"], label="ROC Curve (AUC = {:.2f})".format(metrics["roc_auc"]))
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()
    
    # Plot accuracy
    ax3.plot(metrics["accuracy"], label="Accuracy")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy")
    ax3.set_title("Accuracy")
    ax3.legend()
    
    # Plot confusion matrix
<<<<<<< HEAD
    ax3.imshow(metrics["confusion_matrix"], cmap=plt.cm.Blues)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("True")
    ax3.set_title("Confusion Matrix")
    ax3.colorbar()
=======
    im = ax3.imshow(metrics["confusion_matrix"], cmap=plt.cm.Blues)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("True")
    ax3.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax3)  
    
    # Ensure directory exists
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
>>>>>>> origin/soloPlayer
    
    # Save plots
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "figures", "metrics.png"))
    plt.close()


<<<<<<< HEAD
=======
def compute_recall():
    pass
    
    
>>>>>>> origin/soloPlayer
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



