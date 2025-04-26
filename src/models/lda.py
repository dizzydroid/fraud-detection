import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

class LDAModel:
    """LDA model optimized for fraud detection."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LinearDiscriminantAnalysis()

    def fit(self, x, y):
        """Train the LDA model on the given data.
        
        Args:
            X: Training features
            y: Target labels
        
        Returns:
            self: The trained model instance
        """
        self.class_labels = np.unique(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = x.shape[1]
        self.n_samples_ = x.shape[0]
        x_scaled = self.scaler.fit_transform(x)
        self.model.fit(x_scaled, y)

        return self

    def predict(self, x):
        """Make predictions using the trained LDA model.
        
        Args:
            X: Test features
        
        Returns:
            y_pred: Predicted labels
        """
        x_scaled = self.scaler.transform(x)
        y_pred = self.model.predict(x_scaled)
        return y_pred

    def predict_proba(self, x):
        """Predict probabilities using the trained LDA model.
        
        Args:
            X: Test features
        
        Returns:
            y_proba: Predicted probabilities
        """
        x_scaled = self.scaler.transform(x)
        y_proba = self.model.predict_proba(x_scaled)
        return y_proba
