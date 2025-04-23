"""Tiny smoke test to ensure ensemble recall ≥ 0.93 (TODO: add fixtures)."""
import numpy as np
from sklearn.metrics import recall_score
def test_recall():
    y_true = np.load("artifacts/y_true.npy")
    y_pred = np.load("artifacts/y_pred.npy")
    assert recall_score(y_true, y_pred) >= 0.93