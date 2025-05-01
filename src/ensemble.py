"""
Ensemble inference – Algorithm 1 (“rule layer” in Chung & Lee, 2023)

Steps
-----
1. Load the three trained models from artifacts/
2. Load the test split prepared in preprocessing
3. Produce:
   • knn_pred  (0 / 1)
   • lda_pred  (0 / 1)
   • lr_score  (continuous)
4. Apply the rule layer:
        IF  (knn==0 or lda==0)  AND  lr_score < mean(lr_score)  -> 0
        IF  (knn==1 or lda==1)  AND  lr_score > mean(lr_score)  -> 1
        ELSE knn                                                -> tie-breaker
5. Save `y_pred.npy` to artifacts/
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Project-level constants collected in src/__init__.py
# ----------------------------------------------------------------------
from src import ARTIFACTS_DIR, PROC_DIR, MODEL_NAMES    #  MODEL_NAMES = {'knn':'KNN', ...}

# ----------------------------------------------------------------------
# 1. helpers
# ----------------------------------------------------------------------
def load_models():
    """Return {'knn': knn_model, 'lda': lda_model, 'linreg': lr_model}"""
    models = {}
    for short_name in MODEL_NAMES:           # 'knn', 'lda', 'linreg'
        with open(Path(ARTIFACTS_DIR, f"{short_name}.pkl"), "rb") as f:
            models[short_name] = pickle.load(f)
    return models


def load_test_features(balanced: bool = False) -> np.ndarray:
    prefix = "balanced_" if balanced else ""
    return pd.read_csv(Path(PROC_DIR, f"{prefix}X_test.csv")).values


def apply_rule_layer(knn_pred: np.ndarray,
                     lda_pred: np.ndarray,
                     lr_score: np.ndarray) -> np.ndarray:
    """
    Vectorised implementation of Algorithm 1.
    """
    lr_mean = lr_score.mean()
    ensemble = knn_pred.copy()                    # initialise with the tie-breaker

    for i in range(len(knn_pred)):
        # If "non-fraud" is indicated by either KNN or LDA
        if knn_pred[i] == 0 or lda_pred[i] == 0:
            if lr_score[i] == 0:
                ensemble[i] = 0  # Predict non-fraud
        # If "fraud" is indicated by either KNN or LDA
        elif knn_pred[i] == 1 or lda_pred[i] == 1:
            if lr_score[i] == 1:
                ensemble[i] = 1  # Predict fraud
        # Otherwise, allocate predicted values from KNN to remaining cases
        else:
            ensemble[i] = knn_pred[i]

    # ensemble[non_fraud] = 0
    # ensemble[fraud]     = 1
    return ensemble


# ----------------------------------------------------------------------
# 2. main entry point
# ----------------------------------------------------------------------
def main():
    # Load assets -------------------------------------------------------
    models  = load_models()
    X_test  = load_test_features()

    # 3. individual model outputs --------------------------------------
    knn_pred = models["knn"].predict(X_test)
    lda_pred = models["lda"].predict(X_test)

    # **new** – keep LR continuous
    # -----------------------------------------
    # → Add `.decision_function()` to LinearRegressionModel
    lr_score = models["lr"].predict(X_test)

    # 4. ensemble -------------------------------------------------------
    y_pred = apply_rule_layer(knn_pred, lda_pred, lr_score)

    # 5. save -----------------------------------------------------------
    Path(ARTIFACTS_DIR).mkdir(exist_ok=True)
    np.save(Path(ARTIFACTS_DIR, "y_pred.npy"), y_pred)

    # quick sanity print
    n_fraud = y_pred.sum()
    print(f"[ensemble] predictions saved → artifacts/y_pred.npy")
    print(f"[ensemble] total: {len(y_pred):,} | fraud flagged: {n_fraud:,}")


if __name__ == "__main__":
    main()