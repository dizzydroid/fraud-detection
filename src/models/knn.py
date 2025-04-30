import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class KNNModel:
    """KNN model optimized for fraud detection."""

    def __init__(self, n_neighbors=5, weights="uniform"):
        """Initialize KNN model

        According to the paper :
        - algorithm: auto
        - leaf_size: 30
        - metric: minkowski
        - metric_params: None
        - n_jobs: -1 (use all processors)
        - n_neighbors: 5
        - p: 2 (Euclidean distance)
        - weights: uniform
        """
        self.scaler = StandardScaler()
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm="auto",
            leaf_size=30, # la kant m3mola
            p=2,
            metric="minkowski",
            metric_params=None,
            n_jobs=-1,
        )

    def fit(self, x, y):
        """Train the KNN model on the given data.

        Args:
            X: Training features
            y: Target labels

        Returns:
            self: The trained model instance
        """
        # Scale features first
        x_scaled = self.scaler.fit_transform(x)
        # Train KNN model
        self.model.fit(x_scaled, y)
        return self

    def predict(self, x):
        """Make predictions using the trained KNN model.

        Args:
            X: Test features

        Returns:
            y_pred: Predicted labels
        """
        x_scaled = self.scaler.transform(x)
        return self.model.predict(x_scaled)

    def predict_proba(self, x):
        """Predict probabilities using the trained KNN model.

        Args:
            X: Test features

        Returns:
            y_proba: Predicted probabilities
        """
        x_scaled = self.scaler.transform(x)
        return self.model.predict_proba(x_scaled)
