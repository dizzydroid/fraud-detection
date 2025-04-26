"""Compute metrics and dump to JSON

This module evaluates the performance of the ensemble fraud detection model by:
1. Loading the ground truth labels (y_true) from artifacts directory
2. Loading the ensemble predictions (y_pred) from artifacts directory
3. Computing key performance metrics (especially recall) / Confusion Matrix
   - True Positive Rate (TPR)
   - False Positive Rate (FPR)
   - Precision
   - Recall
   - F1 Score
   - ROC AUC
4. Saving these metrics to results/metrics.json
5. Generating optional visualization plots in results/figures/

Usage:
    Run directly (e.g., `python src/evaluate.py`) or via Makefile (`make evaluate`)
"""
