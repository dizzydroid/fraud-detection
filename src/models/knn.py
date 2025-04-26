import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

class KNNModel:
    """KNN model optimized for fraud detection."""
    
    def __init__(self, n_neighbors = 5, weights = 'distance'):
        pass

    def fit(self,x,y):
        """Train the KNN model on the given data.
        
        Args:
            X: Training features
            y: Target labels
        
        Returns:
            self: The trained model instance
        """
        pass

    def predict(self, x):
        """Make predictions using the trained KNN model.
        
        Args:
            X: Test features
        
        Returns:
            y_pred: Predicted labels
        """
        pass

    def predict_proba(self, x):
        """Predict probabilities using the trained KNN model.
        
        Args:
            X: Test features
        
        Returns:
            y_proba: Predicted probabilities
        """
        pass
