import pytest
import numpy as np
from sklearn.metrics import recall_score

@pytest.fixture
def loaded_data() -> tuple[np.ndarray, np.ndarray]:
    """Load model output for metric tests.

    Returns:
        tuple: (y_true, y_pred)
            - y_true (numpy.ndarray): The true labeled test values from the dataset.
            - y_pred (numpy.ndarray): The predicted values from running the models ensemble.
    """
    y_true = np.load('artifacts/y_true.npy')
    y_pred = np.load('artifacts/y_pred.npy')
    return y_true, y_pred

def test_recall(loaded_data: tuple[np.ndarray, np.ndarray]) -> None:
    """ Test function to ensure ensemble prediction recall to be 
        greater than or equal to 0.93 (worst case in reference paper)"""
    y_true, y_pred = loaded_data
    recall = recall_score(y_true, y_pred)
    assert recall >= 0.93, f"Recall too low: {recall}"
