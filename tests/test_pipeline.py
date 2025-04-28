import pytest
import json
import numpy as np
from sklearn.metrics import recall_score

@pytest.fixture
def load_metrics() -> dict[str, float]:
    """ Loads model output's metrics

    Returns:
        dict[str, float]: holds the metric key and values
    """

    file_path = "../results/metrics.json"
    with open (file_path, "r") as f:
        metrics = json.load(f)
        
    return metrics

@pytest.fixture
def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Load model output for metric tests.

    Returns:
        tuple: (y_true, y_pred)
            - y_true (numpy.ndarray): The true labeled test values from the dataset.
            - y_pred (numpy.ndarray): The predicted values from running the models ensemble.
    """
    y_true = np.load('artifacts/y_true.npy')
    y_pred = np.load('artifacts/y_pred.npy')
    return y_true, y_pred

def test_recall(load_metrics: dict[str, float]) -> None:
    """ Test function to ensure ensemble prediction recall to be 
        greater than or equal to 1 (worst case in reference paper)"""
    metrics = load_metrics
    assert metrics['recall'] >= 1, f"Recall too low: {metrics['recall']}"

def test_precision(load_metrics: dict[str, float]) -> None:
    """ Test function to ensure ensemble prediction recall to be 
        greater than or equal to 1 (worst case in reference paper)"""
    metrics = load_metrics
    assert metrics['precision'] >= 0.0656, f"Recall too low: {metrics['precision']}"

def test_accuracy(load_metrics: dict[str, float]) -> None:
    """ Test function to ensure ensemble prediction recall to be 
        greater than or equal to 1 (worst case in reference paper)"""
    metrics = load_metrics
    assert metrics['accuracy'] >= 0.9989, f"Recall too low: {metrics['accuracy']}"

