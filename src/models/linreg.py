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
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_pred = np.where(y_pred > y_pred.mean(), 1, 0)
        return y_pred
    
    def predict_proba(self, X_test):
        """Predict probabilities using the trained Linear Regression model.
        
        Args:
            X_test: Test features
            
        Returns:
            y_proba: Predicted probabilities
        """
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_test_scaled)
        
