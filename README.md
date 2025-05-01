# Credit-Card Fraud Detection (Recall-First, Chung & Lee 2023, PaySim)

This repository implements and extends the high-recall ensemble approach for fraud detection from **Chung & Lee (2023, Sensors 23-7788)** using the [PaySim](https://www.kaggle.com/datasets/ntnu-testimon/paysim1) dataset. The solution is optimized for **perfect or near-perfect recall** (≥0.93), aiming to catch every fraudulent transaction, following the principle that missing fraud is much more costly than a false alarm.

## 🚀 Getting Started

Clone the repo and ensure your environment has Python 3.9+ and the required packages (see `requirements.txt`).  
Typical workflow:

```bash
make setup         # install dependencies
make preprocess    # preprocess data (encoding, split)
make train         # fit key models and save them
make ensemble      # apply ensemble voting (Algorithm 1)
make evaluate      # compute metrics, visualize and save results
```

Artifacts will be saved in `artifacts/`, processed data in `data/processed/`, and results (metrics, confusion matrix) in `results/`.

You can run `make clean` to wipe all outputs and start fresh.

## 📊 Label Convention

> **Note:**  
> In this repo, we follow the convention used in Chung & Lee (2023) where:
>
> - `1` = **Non-Fraud**
> - `0` = **Fraud**
>
> This is the opposite of the standard for PaySim (`isFraud`), so please interpret all results accordingly.

## 🗂️ Project Structure

- **notebooks/**: Main notebook (`fraud-detection.ipynb`) with code, analysis, and visualizations
- **data/**: Raw and processed PaySim data
- **artifacts/**: Saved models (e.g., `knn.pkl`, `lda.pkl`, `lr.pkl`)
- **results/**: Metrics, figures, confusion matrices, etc.
- **docs/**: Literature notes
- **slides/**: Presentation slides

## 🏆 Methodology

- **Dataset:** [PaySim](https://www.kaggle.com/datasets/ntnu-testimon/paysim1) (6.3M mobile money transactions, highly imbalanced)
- **Models:**  
  - K-Nearest Neighbors (KNN)
  - Linear Discriminant Analysis (LDA)
  - Linear Regression (thresholded)
  - Logistic Regression, Decision Tree, Random Forest, Naive Bayes (for comparison)
- **Ensemble Logic:**  
  - Inspired by Chung & Lee (2023)
  - Prioritizes **recall** (fraud detection), combining KNN, LDA, and Linear Regression predictions using a voting/thresholding strategy
- **Metrics:**  
  - **Primary:** Recall (for fraud, label=0)
  - **Also reported:** Precision, Accuracy, Confusion Matrix (visualized as a heatmap)

## 📈 Results

Summary of model performance (see notebook for details):

| Model              | Recall   | Precision | Accuracy  |
|--------------------|----------|-----------|-----------|
| Decision Tree      | 0.9998   | 0.9998    | 0.9997    |
| Naive Bayes        | 0.9971   | 0.9988    | 0.9960    |
| **Ensemble**       | 0.9998   | 0.7508    | 0.9991    |

> The ensemble achieves nearly perfect recall with competitive precision and accuracy, validating the approach.

## 📚 References

- **Chung, H., & Lee, J. (2023).** “A High-Recall Ensemble Approach for Fraud Detection in Financial Transactions.” [Sensors 23(18), 7788](https://www.mdpi.com/1424-8220/23/18/7788)
- [PaySim Dataset on Kaggle](https://www.kaggle.com/datasets/ntnu-testimon/paysim1)
- Scikit-learn documentation: https://scikit-learn.org/stable/documentation.html

## 👥 Contributors

- Shehab Mahmoud Salah
- Abdelrahman Hany Mohamed
- Youssef Ahmed Mohamed
- Omar Mamon Hamed
- Seif El Din Tamer Shawky
- Seif Eldeen Ahmed Abdulaziz
- Habiba El-sayed Mowafy
- Aya Tarek Salem
- Moaz Ragab
- Ahmed Ashraf Ali

---

For details, see the [notebook](fraud-detection.ipynb) and [docs/](docs/).