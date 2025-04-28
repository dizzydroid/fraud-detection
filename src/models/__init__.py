from .knn import KNNModel
from .lda import LDAModel
from .linreg import LinearRegressionModel

MODEL_CLASSES = {
    'knn': KNNModel,      # Note: No parentheses
    'lda': LDAModel,      # Just the class reference
    'lr': LinearRegressionModel
}
