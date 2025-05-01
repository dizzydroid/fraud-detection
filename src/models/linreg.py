import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class LinearRegressionModel:
    """Linear Regression model optimized for fraud detection.
    
    According to the paper :
    - retain all default parameters
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LinearRegression()
        
    def fit(self, X_train, y_train):
        """Train the Linear Regression model on the given data.
        
        Args:
            X_train: Training features
            y_train: Target labels
            
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        return self
    
    def predict(self, X_test):
        """Make predictions using the trained Linear Regression model.
        
        Args:
            X_test: Test features
            
        Returns:
            y_pred: Predicted labels
        """
        scores = self.decision_function(X_test)
        thresh = scores.mean()
        return (scores > thresh).astype(int)
    
    def predict_proba(self, X_test):
        """Predict probabilities using the trained Linear Regression model.
        
        Note: LinearRegression doesn't have predict_proba natively,
        so we're implementing a custom version that converts regression
        scores to pseudo-probabilities.
        
        Args:
            X_test: Test features
            
        Returns:
            y_proba: Predicted probabilities
        """
        X_test_scaled = self.scaler.transform(X_test)
        raw_predictions = self.model.predict(X_test_scaled)
        
        # Scale predictions to [0, 1] range
        min_val = raw_predictions.min()
        max_val = raw_predictions.max()
        
        if max_val > min_val:
            scaled = (raw_predictions - min_val) / (max_val - min_val)
        else:
            scaled = np.zeros_like(raw_predictions)
        
        # Format as 2D array with probabilities for class 0 and class 1
        return np.vstack([1-scaled, scaled]).T

    def decision_function(self, X):
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs)    

